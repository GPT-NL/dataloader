import json
import os
import shutil
from pathlib import Path

from typer import Typer

from gptnl_dataloader.sampling.functions import (
    ALL_BUCKETS_DEFAULT,
    ANNEALING_GROUPS_DEFAULT,
    PRIMARY_GROUPS_DEFAULT,
    aggregate_groups,
    analyse_files,
    calculate_scaling_factors,
    calculate_scaling_factors_nuanced,
    compare_file_based_sampling,
    find_bucket_files,
    get_file_based_sampling,
    to_relative,
)
from gptnl_dataloader.utils import (
    TOKENIZED_DATASET_ROOT_PATH,
    Tee,
)

my_path = Path(os.path.dirname(os.path.abspath(__file__)))
LOGGING_PATH = str(my_path / "logs/v2")  # versioned in GIT here
OUTPUT_PATH = (
    Path(TOKENIZED_DATASET_ROOT_PATH) / "train_datamixtures"
)  # will also contain a copy of the LOGGING_PATH (for easier versioning of dataset)

app = Typer(pretty_exceptions_enable=False)

# Constants for sampling
VALIDATION_RATIO = 0.05
ANNEALING_RATIO = 0.30  # of remaining data after validation (15% for 2 epochs --> 30%)

# Epoch 1
# PRIMARY_LANGUAGE_DIST = {
#     "en": 0.50,
#     "nl": 0.25,
#     "code": 0.15,
#     "other": 0.10,
# }

# desired proportions with the nl_new added (epoch 1: nl_new=0.25, here we want nl_new to have 2x nl)
PRIMARY_LANGUAGE_DIST = {
    "en": 0.5,
    "nl": 0.0833,
    "nl_new": 0.1667,
    "code": 0.15,
    "other": 0.10,
}

PRIMARY_UPSAMPLING_MAX = {
    "en": 2.0,  # English can be upsampled up to 2x
    "nl": 1.0,  # Dutch can be upsampled up to 3x
    "nl_new": 4.0,  # Dutch new can be upsampled up to 3x
    "code": 1.0,  # Code cannot be upsampled (forces downsampling)
    "other": 2.0,  # Other can be upsampled up to 2x
}

PRIMARY_ALLOW_DOWN_SAMPLE = {
    "en": True,  # Allow English to be downsampled
    "nl": False,  # Protect Dutch from downsampling
    "nl_new": False,  # Protect Dutch new from downsampling
    "code": True,  # Allow code to be downsampled
    "other": True,  # Allow other to be downsampled
}

PRIMARY_TARGET_TOTAL_RATIO = 1.0  # Allow up to 5% size increase


ANNEALING_LANGUAGE_DIST = {
    "en": 0.40,
    "nl": 0.35,
    # "nl": 0.11667,
    # "nl_new": 0.2333,
    "code": 0.20,
    "other": 0.05,
}


@app.command()
def main(
    logging_path: str = LOGGING_PATH,
    output_path: Path = OUTPUT_PATH,
    dataset_root: str = TOKENIZED_DATASET_ROOT_PATH,
    buckets_train_subfolder: str = str("buckets_split_train"),
    buckets_validate_subfolder: str = str("buckets_split_validate"),
    file_name_pattern: str = "*.npy",
):
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    logname = f"{logging_path}/output_log.text"

    logger = Tee(logname, mode="w")

    buckets_train_full_path = Path(dataset_root) / buckets_train_subfolder
    buckets_validate_full_path = Path(dataset_root) / buckets_validate_subfolder
    # general file stats
    train_files_paths_by_bucket = find_bucket_files(
        buckets_root=str(buckets_train_full_path),
        file_name_pattern=file_name_pattern,
    )
    validate_files_paths_by_bucket = find_bucket_files(
        buckets_root=str(buckets_validate_full_path),
        file_name_pattern=file_name_pattern,
    )
    # train
    train_files_size_by_bucket, train_files_size_by_path = analyse_files(
        train_files_paths_by_bucket, mode="size"
    )
    train_tokens_by_bucket, train_tokens_by_path = analyse_files(
        train_files_paths_by_bucket, mode="tokens"
    )
    # validate
    validate_files_size_by_bucket, validate_files_size_by_path = analyse_files(
        validate_files_paths_by_bucket, mode="size"
    )
    validate_tokens_by_bucket, validate_tokens_by_path = analyse_files(
        validate_files_paths_by_bucket, mode="tokens"
    )

    print("pre-sampling")
    print(
        f"total TRAIN:\t {sum(train_files_size_by_bucket.values()) / (2**30):8.3f} GB {sum(train_tokens_by_bucket.values()) / 1e9:7.3f} B tokens"
    )
    print(
        f"total VALIDATE:\t {sum(validate_files_size_by_bucket.values()) / (2**30):8.3f} GB {sum(validate_tokens_by_bucket.values()) / 1e9:7.3f} B tokens"
    )

    for bucket in ALL_BUCKETS_DEFAULT:
        print(bucket)
        print(
            f"     TRAIN    {len(train_files_paths_by_bucket[bucket]): >5} files, total {train_files_size_by_bucket[bucket] / (2**30):7.3f} GB {train_tokens_by_bucket[bucket] / 1e9:7.3f} B tokens"
        )
        print(
            f"     VALIDATE {len(validate_files_paths_by_bucket[bucket]): >5} files, total {validate_files_size_by_bucket[bucket] / (2**30):7.3f} GB {validate_tokens_by_bucket[bucket] / 1e9:7.3f} B tokens"
        )

    # primary
    print("\nPRIMARY")
    primary_available_file_based = aggregate_groups(
        train_files_size_by_bucket, PRIMARY_GROUPS_DEFAULT
    )
    print(
        "available:",
        {k: f"{v >> 30} GB" for k, v in primary_available_file_based.items()},
    )
    print(
        "dist available:",
        {k: f"{v:.3f}" for k, v in to_relative(primary_available_file_based).items()},
    )
    mul_factors_primary_file_based = calculate_scaling_factors_nuanced(
        primary_available_file_based,
        PRIMARY_LANGUAGE_DIST,
        max_upsample_ratio=PRIMARY_UPSAMPLING_MAX,
        allow_downsample=PRIMARY_ALLOW_DOWN_SAMPLE,
        target_total_ratio=PRIMARY_TARGET_TOTAL_RATIO,
        debug=True,
    )

    print("multiplication factors:", mul_factors_primary_file_based)

    # file-based oversampling
    print("primary upscaling file-based")
    primary_oversampling_paths = get_file_based_sampling(
        PRIMARY_GROUPS_DEFAULT,
        mul_factors=mul_factors_primary_file_based,
        files_paths_by_bucket=train_files_paths_by_bucket,
        file_size_by_path=train_tokens_by_path,
        debug=True,
    )

    # verify actual distribution
    print("primary actual distribution")
    primary_tot_size, primary_tot_size_by_group = compare_file_based_sampling(
        PRIMARY_LANGUAGE_DIST,
        mul_factors_primary_file_based,
        primary_oversampling_paths,
        train_files_size_by_path,
        debug=True,
    )

    # annealing
    print("\nANNEALING")
    annealing_available_file_based = aggregate_groups(
        train_files_size_by_bucket, ANNEALING_GROUPS_DEFAULT
    )
    print(
        "available:",
        {k: f"{int(v) >> 30} GB" for k, v in annealing_available_file_based.items()},
    )
    print(
        "dist available:",
        {k: f"{v:.3f}" for k, v in to_relative(annealing_available_file_based).items()},
    )
    mul_factors_annealing_file_based_down = calculate_scaling_factors(
        annealing_available_file_based, ANNEALING_LANGUAGE_DIST, upsample=False
    )
    print("multiplication factors:", mul_factors_annealing_file_based_down)

    annealing_oversampling_paths_down = get_file_based_sampling(
        ANNEALING_GROUPS_DEFAULT,
        mul_factors=mul_factors_annealing_file_based_down,
        files_paths_by_bucket=train_files_paths_by_bucket,
        file_size_by_path=train_tokens_by_path,
    )

    annealing_tot_size, annealing_tot_size_by_group = compare_file_based_sampling(
        ANNEALING_LANGUAGE_DIST,
        mul_factors_annealing_file_based_down,
        annealing_oversampling_paths_down,
        train_files_size_by_path,
    )
    print(f"with downsampling now {annealing_tot_size >> 30}GB")

    # want annealing to have a certain size
    target_annealing_size = primary_tot_size * ANNEALING_RATIO
    print(
        f"target_annealing_size: {int(target_annealing_size) >> 30}GB (considering ANNEALING_RATIO={ANNEALING_RATIO})"
    )
    scaling_factor_global_annealing = target_annealing_size / annealing_tot_size
    print(f"scaling_factor_global_annealing: {scaling_factor_global_annealing}")
    mul_factors_annealing_file_based_scaled_to_ratio = {
        k: v * scaling_factor_global_annealing
        for k, v in mul_factors_annealing_file_based_down.items()
    }
    print(
        "multiplication factors (scaled):",
        mul_factors_annealing_file_based_scaled_to_ratio,
    )

    # final resized annealing with certain size
    print("annealing upscaling file-based")
    annealing_oversampling_paths_scaled = get_file_based_sampling(
        ANNEALING_GROUPS_DEFAULT,
        mul_factors=mul_factors_annealing_file_based_scaled_to_ratio,
        files_paths_by_bucket=train_files_paths_by_bucket,
        file_size_by_path=train_tokens_by_path,
        debug=True,
    )
    print("annealing actual distribution")
    annealing_tot_size_scaled, annealing_tot_size_by_group_scaled = (
        compare_file_based_sampling(
            ANNEALING_LANGUAGE_DIST,
            mul_factors_annealing_file_based_scaled_to_ratio,
            annealing_oversampling_paths_scaled,
            train_files_size_by_path,
            debug=True,
        )
    )

    # save paths by buckets
    with open(f"{logging_path}/train_paths.json", "w") as f:
        json.dump(train_files_paths_by_bucket, f, indent=2)
    with open(f"{logging_path}/validate_paths.json", "w") as f:
        json.dump(validate_files_paths_by_bucket, f, indent=2)

    # save sampling paths
    with open(f"{logging_path}/train_mix_primary.json", "w") as f:
        json.dump(primary_oversampling_paths, f, indent=2)
    with open(f"{logging_path}/train_mix_annealing.json", "w") as f:
        json.dump(annealing_oversampling_paths_scaled, f, indent=2)

    # stats (now only in stdout not structured)
    with open(f"{logging_path}/stats.json", "w") as f:
        json.dump(
            {
                "train": {
                    "mixtures": {
                        "primary": {
                            "total_size": primary_tot_size,
                            "by_group": primary_tot_size_by_group,
                        },
                        "annealing": {
                            "total_size": annealing_tot_size_scaled,
                            "by_group": annealing_tot_size_by_group_scaled,
                        },
                    }
                }
            },
            f,
            indent=2,
        )

    # now create the symlinks for the train mixtures
    primary_output_dir = output_path / "primary"
    if os.path.exists(primary_output_dir):
        # cleanup existing symlinks
        shutil.rmtree(primary_output_dir)
    os.makedirs(primary_output_dir)
    for group, paths in primary_oversampling_paths.items():
        # use index to keep multiple symlinks to the same file (repetitions in oversampling)
        for i, file_path in enumerate(paths):
            file_name = file_path.split("/")[-1]
            output_file_path = primary_output_dir / f"{i:05}_{file_name}"
            os.symlink(file_path, output_file_path)

    annealing_output_dir = output_path / "annealing"
    if os.path.exists(annealing_output_dir):
        shutil.rmtree(annealing_output_dir)
    os.makedirs(annealing_output_dir)
    for group, paths in annealing_oversampling_paths_scaled.items():
        # use index to keep multiple symlinks to the same file (repetitions in oversampling)
        for i, file_path in enumerate(paths):
            file_name = file_path.split("/")[-1]
            output_file_path = annealing_output_dir / f"{i:05}_{file_name}"
            os.symlink(file_path, output_file_path)

    # copy logging_path to output_path
    output_logs = output_path / "logs"

    # flush the log
    logger.flush()
    shutil.copytree(logging_path, output_logs, dirs_exist_ok=True)


if __name__ == "__main__":
    app()
