import os
from collections import defaultdict
from functools import cache
from glob import glob
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from gptnl_dataloader.utils import (
    ALL_BUCKETS_DEFAULT,
    BUCKETS_OUTPUTS_DEFAULT_NAME,
    DATASET_DEFAULT_PATH,
)

PRIMARY_GROUPS_DEFAULT = {
    "en": [
        "english_high_quality",
        "english_low_medium_quality",
    ],
    "nl": [
        "dutch_high_quality",
        "dutch_low_medium_quality",
    ],
    # epoch 2: nl_new getting a different factor in __main__.py
    "nl_new": [
        "new_high_quality",
        "new_low_medium_quality",
    ],
    "code": ["code"],
    "other": ["other_languages"],
}

ANNEALING_GROUPS_DEFAULT = {
    "en": [
        "english_high_quality",
    ],
    "nl": [
        "dutch_high_quality",
        # epoch 2: for annealing, the new buckets do not need a different factor
        "new_high_quality",
    ],
    "code": ["code"],
    "other": ["other_languages"],
}


def get_actual_counts(
    df: pd.DataFrame, column_mapping: Dict[str, str]
) -> Dict[str, float]:
    result = {}
    for key in column_mapping.keys():
        column_name = column_mapping[key]
        if key == "code":
            available = float(df[df["is_code"] == True][column_name].sum())  # noqa: E712
        else:
            available = float(df[column_name].sum())
        result[key] = available
    return result


def calculate_scaling_factors_nuanced(
    available_counts: Dict[str, float] | Dict[str, int],
    target_dist: Dict[str, float],
    max_upsample_ratio: Dict[str, float] | None = None,
    allow_downsample: Dict[str, bool] | None = None,
    target_total_ratio: float | None = None,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Advanced scaling factor calculation with constraints.

    Args:
        available_counts: Current data amounts per group
        target_dist: Desired distribution (should sum to 1.0)
        max_upsample_ratio: Maximum allowed upsampling per group (e.g., {"en": 2.0, "code": 1.0})
        allow_downsample: Which groups can be downsampled (e.g., {"code": True, "en": False})
        target_total_ratio: Target ratio of final size to original size (e.g., 1.2 for 20% increase)
        debug: Whether to print debug information

    Returns:
        Dict of multiplication factors per group
    """
    # Normalize target distribution
    sum_targets = sum(target_dist.values())
    if abs(sum_targets - 1.0) > 1e-6:
        if debug:
            print(f"WARN: sum_targets is {sum_targets:.6f}, renormalizing target_dist")
        target_dist = to_relative(target_dist)

    # Set defaults for constraints
    if max_upsample_ratio is None:
        max_upsample_ratio = {k: float("inf") for k in available_counts.keys()}
    if allow_downsample is None:
        allow_downsample = {k: True for k in available_counts.keys()}

    # Validate inputs
    for k, v in available_counts.items():
        if v == 0:
            raise ValueError(f"No data available for {k}")

    if debug:
        print("target_dist:", target_dist)
        print("max_upsample_ratio:", max_upsample_ratio)
        print("allow_downsample:", allow_downsample)
        if target_total_ratio:
            print("target_total_ratio:", target_total_ratio)

    available_dist = to_relative(available_counts)
    total_available = sum(available_counts.values())

    if debug:
        print("available_dist:", available_dist)

    # Calculate naive scaling factors (what we'd need without constraints)
    naive_factors = {
        k: target_dist[k] / available_dist[k] for k in available_counts.keys()
    }

    if debug:
        print("naive_factors (target/available):", naive_factors)

    # Apply constraints iteratively
    mul_factors = {}

    # First pass: apply constraints
    for group in available_counts.keys():
        naive_factor = naive_factors[group]

        if naive_factor > 1.0:  # Would need upsampling
            max_allowed = max_upsample_ratio.get(group, float("inf"))
            mul_factors[group] = min(naive_factor, max_allowed)
        elif naive_factor < 1.0:  # Would need downsampling
            if allow_downsample.get(group, True):
                mul_factors[group] = naive_factor
            else:
                mul_factors[group] = 1.0  # No downsampling allowed
        else:
            mul_factors[group] = 1.0

    if debug:
        print("mul_factors after constraints:", mul_factors)

    # Calculate resulting distribution and total size
    constrained_counts = {
        k: available_counts[k] * mul_factors[k] for k in available_counts.keys()
    }
    constrained_total = sum(constrained_counts.values())
    constrained_dist = to_relative(constrained_counts)

    if debug:
        print(
            "constrained_counts:",
            {k: f"{int(v) >> 30}GB" for k, v in constrained_counts.items()},
        )
        print("constrained_dist:", {k: f"{v:.4f}" for k, v in constrained_dist.items()})
        print(
            f"size change: {constrained_total / total_available:.3f}x (from {int(total_available) >> 30}GB to {int(constrained_total) >> 30}GB)"
        )

    # If target_total_ratio is specified, scale everything proportionally
    if target_total_ratio is not None:
        current_ratio = constrained_total / total_available
        global_scale = target_total_ratio / current_ratio
        mul_factors = {k: v * global_scale for k, v in mul_factors.items()}

        if debug:
            print(
                f"Applying global scale {global_scale:.3f} to reach target ratio {target_total_ratio:.3f}"
            )
            final_counts = {
                k: available_counts[k] * mul_factors[k] for k in available_counts.keys()
            }
            final_total = sum(final_counts.values())
            print(
                f"Final size: {int(final_total) >> 30}GB (ratio: {final_total / total_available:.3f}x)"
            )

    # Final validation - check constraints are still respected after global scaling
    if target_total_ratio is not None:
        violated_constraints = []
        for group, factor in mul_factors.items():
            max_allowed = max_upsample_ratio.get(group, float("inf"))
            min_allowed = 1.0 if not allow_downsample.get(group, True) else 0.0

            if factor > max_allowed or factor < min_allowed:
                violated_constraints.append(
                    f"{group}: {factor:.3f} (limits: {min_allowed:.1f}-{max_allowed:.1f})"
                )

        if violated_constraints and debug:
            print(
                "WARNING: Constraints violated after global scaling:",
                violated_constraints,
            )

    if debug:
        print("final mul_factors:", {k: f"{v:.3f}" for k, v in mul_factors.items()})
        # Show final distribution
        final_counts = {
            k: available_counts[k] * mul_factors[k] for k in available_counts.keys()
        }
        final_dist = to_relative(final_counts)
        print("final_dist:", {k: f"{v:.4f}" for k, v in final_dist.items()})

    return mul_factors


def calculate_scaling_factors(
    available_counts: Dict[str, float] | Dict[str, int],
    target_dist: Dict[str, float],
    upsample: bool = True,
    debug: bool = False,
) -> Dict[str, float]:
    """Computes scaling factors from a dict of statistics (`available_counts`) to a target distribution (`target_dist`). Operates in upsample and in downsample mode.

    - upsample (`upsample=True`): upsamples groups that are less represented than the corresponding target. Returned scaling factors will be all >= 1.0.
    - downsample (`upsample=False`): downsamples groups that are more represented than the corresponding target. Returned scaling factors will be all <= 1.0.
    """
    sum_targets = sum(target_dist.values())
    if sum_targets != 1.0:
        print("WARN: sum_targets is not 1.0, renormalizing target_dist")
        target_dist = to_relative(target_dist)
    if debug:
        print("target_dist", target_dist)

    for k, v in available_counts.items():
        if v == 0:
            raise ValueError(f"WARN: no data available for {k}")

    available_dist = to_relative(available_counts)
    if debug:
        print("available_dist", available_dist)
    # compared to the target. 2.5 means that it's currently 2.5 bigger than desired
    comparison_factors = {k: target_dist[k] / v for k, v in available_dist.items()}
    if debug:
        print("comparison_factors", comparison_factors)
    if upsample:
        # upsample the ones that are less than desired (overpopulated stays the same)
        reference_value = min(comparison_factors.values())
    else:
        # downsample the ones that are more than desired (underpopulated stays the same)
        reference_value = max(comparison_factors.values())
    if debug:
        print("reference_value", reference_value)
    # mul_factors will have 1.0 at the reference value
    mul_factors = {k: v / reference_value for k, v in comparison_factors.items()}
    if debug:
        print("mul_factors", mul_factors)
    # emulate scaling to see if it works
    if debug:
        balanced_counts = {k: v * mul_factors[k] for k, v in available_counts.items()}
        print("balanced_counts", balanced_counts)
        balanced_dist = to_relative(balanced_counts)
        print("balanced_dist", balanced_dist)
    return mul_factors


def to_relative(stats: dict[str, float] | Dict[str, int]) -> dict[str, float]:
    """Scales a dict of numeric values to percentages"""
    total = sum(stats.values())
    return {k: v / total for k, v in stats.items()}


def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)


@cache
def count_tokens(file_path: str, dtype=np.uint32):
    array = np.memmap(file_path, dtype=dtype, mode="r")
    len = array.shape[0]
    return len


def find_bucket_files(
    buckets_root: str = str(Path(DATASET_DEFAULT_PATH) / BUCKETS_OUTPUTS_DEFAULT_NAME),
    buckets_names: List[str] = ALL_BUCKETS_DEFAULT,
    file_name_pattern: str = "*.parquet",
) -> Dict[str, List[str]]:
    files_paths_by_bucket: Dict[str, List[str]] = {}
    for bucket in buckets_names:
        full_path = f"{buckets_root}/{bucket}/{file_name_pattern}"
        files_paths = sorted(glob(full_path))
        files_paths_by_bucket[bucket] = files_paths
        if not files_paths:
            print(f"No files found for {full_path} {len(files_paths)}")

    return files_paths_by_bucket


def analyse_files(
    files_paths_by_bucket: Dict[str, List[str]], mode=Literal["size", "tokens"]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Analyze the size or token count of files, returning:

    1. total value for each bucket
    2. value for each file"""
    tot_value_by_bucket: Dict[str, int] = {}
    values_by_path: Dict[str, int] = {}

    for bucket, files_paths in files_paths_by_bucket.items():
        values = []
        for el in files_paths:
            if mode == "size":
                value = get_file_size(el)
            elif mode == "tokens":
                value = count_tokens(el)
            else:
                raise ValueError(mode)
            values.append(value)
            values_by_path[el] = value
        files_paths_by_bucket[bucket] = files_paths
        tot_value_by_bucket[bucket] = sum(values)

    return tot_value_by_bucket, values_by_path


def split_train_validation(
    files_paths_by_bucket: Dict[str, List[str]],
    files_size_by_path: Dict[str, int],
    validation_size: float = 0.05,
    file_size_aware: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    train_paths_by_bucket: Dict[str, List[str]] = {}
    validation_paths_by_bucket: Dict[str, List[str]] = {}
    for bucket, files_paths in files_paths_by_bucket.items():
        train = sample_file_paths(
            files_paths=files_paths,
            mul_factor=1.0 - validation_size,
            file_size_by_path=files_size_by_path,
            file_size_aware=file_size_aware,
        )
        # ensure that at least 1 file is kept for validation
        if len(train) == len(files_paths):
            train = train[:-1]
        validation = [el for el in files_paths if el not in train]
        train_paths_by_bucket[bucket] = train
        validation_paths_by_bucket[bucket] = validation

    return train_paths_by_bucket, validation_paths_by_bucket


def aggregate_groups(
    stat_by_bucket: Dict[str, int],
    groups: Dict[str, List[str]],
) -> Dict[str, int]:
    result = defaultdict(int)
    for group, group_bucket_names in groups.items():
        for bucket in group_bucket_names:
            assert bucket in stat_by_bucket, (
                f"bucket {bucket} not found in {stat_by_bucket.keys()}"
            )
            val = stat_by_bucket[bucket]
            result[group] += val
    return result


def sample_file_paths(
    files_paths: List[str],
    mul_factor: float,
    file_size_by_path: Dict[str, int],
    file_size_aware: bool = True,
    debug: bool = False,
) -> List[str]:
    n_files = len(files_paths)
    # checking shuffling: Which ones get repeated 1 more time? (e.g. mul_factor=4.93 --> 4 vs 5)
    full_repetitions = int(mul_factor // 1)
    partial_repetition_factor = mul_factor - full_repetitions

    file_sizes_in_bucket = [file_size_by_path[path] for path in files_paths]
    # [0, f1, f1+f2, f1+f2+f3, ..., all] --> length(f) + 1
    progressive_cumulative_size_in_bucket = [
        sum(file_sizes_in_bucket[:max_i]) for max_i in range(len(files_paths) + 1)
    ]
    if file_size_aware:
        total_input_size = sum(file_sizes_in_bucket)
        desired_bucket_size = mul_factor * total_input_size
        missing_bucket_size = partial_repetition_factor * total_input_size
        # interval [0,length(f)]: will only be 0 if we need to take 0 files
        n_files_partial_high = next(
            i
            for i, val in enumerate(progressive_cumulative_size_in_bucket)
            if val >= missing_bucket_size
        )
        higher = progressive_cumulative_size_in_bucket[n_files_partial_high]
        if n_files_partial_high == 0:
            n_files_partial_repetition = 0
        else:
            lower = progressive_cumulative_size_in_bucket[n_files_partial_high - 1]
            if (missing_bucket_size - lower) > (higher - missing_bucket_size):
                # with one file less, we arrive closer to target
                n_files_partial_repetition = n_files_partial_high
                side = "lower"
            else:
                # otherwise we also take the next file
                n_files_partial_repetition = n_files_partial_high + 1
                side = "higher"
            if debug:
                print(
                    f"      n_files_partial_repetition({side})={n_files_partial_repetition} lower={lower / 1e9:5.2f}B higher={higher / 1e9:5.2f}B target={missing_bucket_size / 1e9:5.2f}B"
                )
    else:
        n_files_partial_repetition = round(
            partial_repetition_factor * n_files
        )  # threshold 0.5 for rounding
    partial_repetition_paths = files_paths[:n_files_partial_repetition]
    oversampling_paths = files_paths * full_repetitions + partial_repetition_paths
    oversampling_sizes = [file_size_by_path[path] for path in oversampling_paths]
    oversampled_size = sum(oversampling_sizes)
    if debug:
        print(
            f"      mul_factor={mul_factor:7.3f} n_files={n_files:5.0f} full_repetitions: {full_repetitions:5.0f} partial repetition: {n_files_partial_repetition:5.0f} oversampling_paths: {len(oversampling_paths):5.0f}"
        )
        print(
            f"      SIZE: total_input_size={total_input_size / 1e9:5.2f}B desired_bucket_size={desired_bucket_size / 1e9:5.2f}B oversampled_size={oversampled_size / 1e9:5.2f}B"
        )
    return oversampling_paths


def get_file_based_sampling(
    groups: Dict[str, List[str]],
    mul_factors: Dict[str, float],
    files_paths_by_bucket: Dict[str, List[str]],
    file_size_by_path: Dict[str, int],
    debug: bool = False,
) -> Dict[str, List[str]]:
    result = defaultdict(list)
    for group, group_bucket_names in groups.items():
        for bucket in group_bucket_names:
            if debug:
                print(f"{group}:{bucket}:")
            mul_factor = mul_factors[group]
            files_paths = files_paths_by_bucket[bucket]
            oversampling_paths = sample_file_paths(
                files_paths=files_paths,
                mul_factor=mul_factor,
                file_size_by_path=file_size_by_path,
                debug=debug,
            )
            result[group] += oversampling_paths

    return result


def aggregate_size_groups(
    paths_by_group: Dict[str, List[str]],
    file_size_by_path: Dict[str, int],
) -> Dict[str, int]:
    result = defaultdict(int)
    for group, paths in paths_by_group.items():
        for path in paths:
            assert path in file_size_by_path, (
                f"{path} not found in {file_size_by_path.keys()}"
            )
            size = file_size_by_path[path]
            result[group] += size

    return result


def compare_file_based_sampling(
    target_dist: Dict[str, float],
    mul_factors: Dict[str, float],
    file_based_sampling_paths: Dict[str, List[str]],
    file_size_by_path: Dict[str, int],
    debug: bool = False,
) -> Tuple[int, Dict[str, int]]:
    actual_sizes = aggregate_size_groups(file_based_sampling_paths, file_size_by_path)
    actual_dist = to_relative(actual_sizes)
    total_size = sum(actual_sizes.values())
    total_tokens = sum(
        count_tokens(path)
        for group_paths in file_based_sampling_paths.values()
        for path in group_paths
    )
    total_b_tokens = total_tokens / 1e9
    if debug:
        print(f"total size: {total_size >> 30}GB n_tokens: {total_b_tokens:5.2f}B")
    total_size_by_group = {}
    for group in target_dist.keys():
        target_value = target_dist[group]
        assert group in mul_factors, f"{group} not in mul_factors: {mul_factors.keys()}"
        mul_factor = mul_factors[group]
        # assert group in actual_dist, f"{group} not in actual_dist: {actual_dist.keys()}"
        actual_size = actual_sizes.get(group, 0)
        actual = actual_dist.get(group, 0)
        total_size_by_group[group] = actual_size
        size_gb = int(actual_size) >> 30
        total_tokens_group = sum(
            count_tokens(path) for path in file_based_sampling_paths[group]
        )
        b_tokens = total_tokens_group / 1e9
        if debug:
            print(
                f"{group: >5}: size_gb: {size_gb:5.0f}GB \t n_tokens: {b_tokens:5.2f}B \t mul_factor: {mul_factor:7.3f} \t target: {target_value} \t actual_dist: {actual:.4f}"
            )
    return total_size, total_size_by_group
