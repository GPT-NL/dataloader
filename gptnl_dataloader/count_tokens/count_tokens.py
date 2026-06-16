import numpy as np
from pyarrow import parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from collections import defaultdict
import os
import json
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

EPOCH_2_FOLDERS = {
    "train": os.path.join(os.environ["GPTNL_2_TOKENIZED"], "buckets_split_train"),
    "validate": os.path.join(os.environ["GPTNL_2_TOKENIZED"], "buckets_split_validate"),
}

UNTOKENIZED_EPOCH_2_FOLDERS = {
    "train": os.path.join(
        os.environ["GPTNL_SCRATCH"], "gpt_nl_dataset_v2.0_bucketed/buckets_split_train"
    ),
    "validate": os.path.join(
        os.environ["GPTNL_SCRATCH"],
        "gpt_nl_dataset_v2.0_bucketed/buckets_split_validate",
    ),
}

buckets = [
    "code",
    "dutch_high_quality",
    "dutch_low_medium_quality",
    "english_high_quality",
    "english_low_medium_quality",
    "new_high_quality",
    "new_low_medium_quality",
    "other_languages",
]

# Restructure to track progress: {train_valid: {bucket: {dataset_source: count}}}
counts_per_dataset_source = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def load_counts(filename="counts_per_dataset_source.json"):
    """Load counts dictionary from JSON file if it exists"""
    save_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            regular_dict = json.load(f)
        logger.info(f"Loaded existing counts from {save_path}")
        # Reconstruct nested defaultdicts
        result = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for train_valid, buckets_data in regular_dict.items():
            for bucket, dataset_counts in buckets_data.items():
                for dataset_source, count in dataset_counts.items():
                    result[train_valid][bucket][dataset_source] = count
        return result
    return defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def get_completed_buckets(counts_dict, train_valid):
    """Get list of buckets that have already been processed for a given train_valid split"""
    return list(counts_dict.get(train_valid, {}).keys())


def get_doc_lengths(tokens, eos_token_id: int) -> list[int]:
    # tokens = np.memmap(npy_path, dtype=np.uint32, mode='r')
    eos_positions = np.where(tokens == eos_token_id)[0]
    if len(eos_positions) == 0:
        return [len(tokens)]

    doc_lengths = [eos_positions[0] + 1]  # +1 to include the EOS token itself

    if len(eos_positions) > 1:
        doc_lengths.extend(np.diff(eos_positions).tolist())

    if eos_positions[-1] < len(tokens) - 1:
        remaining_length = len(tokens) - eos_positions[-1] - 1
        doc_lengths.append(remaining_length)

    return doc_lengths


def save_counts(counts_dict, filename="counts_per_dataset_source.json"):
    """Save counts dictionary to JSON file"""
    save_path = os.path.join(os.path.dirname(__file__), filename)
    # Convert nested defaultdicts to regular dicts for JSON serialization
    regular_dict = {}
    for train_valid, buckets_data in counts_dict.items():
        regular_dict[train_valid] = {}
        for bucket, dataset_counts in buckets_data.items():
            regular_dict[train_valid][bucket] = dict(dataset_counts)

    with open(save_path, "w") as f:
        json.dump(regular_dict, f, indent=2)
    logger.info(f"Saved counts to {save_path}")


# Load existing counts if available
counts_per_dataset_source = load_counts("counts_intermediate.json")

for train_valid, tokenized_folder, untokenized_folder in zip(
    ["train", "validate"],
    EPOCH_2_FOLDERS.values(),
    UNTOKENIZED_EPOCH_2_FOLDERS.values(),
):
    logger.info(f"Processing {train_valid}...")
    logger.info(f"  Tokenized folder: {tokenized_folder}")
    logger.info(f"  Untokenized folder: {untokenized_folder}")

    # Check which buckets are already completed
    completed_buckets = get_completed_buckets(counts_per_dataset_source, train_valid)
    logger.info(f"Already completed buckets for {train_valid}: {completed_buckets}")

    for bucket in reversed(buckets):
        if bucket == "code":
            continue

        # Skip if this bucket is already completed
        if bucket in completed_buckets:
            logger.info(f"Skipping already completed bucket: {bucket}")
            continue

        logger.info(f"Processing bucket: {bucket}")

        tokenized_path = os.path.join(tokenized_folder, bucket)
        untokenized_path = os.path.join(untokenized_folder, bucket)

        for i, npy_file in enumerate(os.listdir(tokenized_path)):
            pqt_file = npy_file.replace(".npy", ".parquet").replace("_tokenized", "")

            # print(pqt_file.split("/")[-1].replace(".parquet", ""))

            pqt_fp = os.path.join(untokenized_path, pqt_file)
            npy_fp = os.path.join(tokenized_path, npy_file)

            if not os.path.exists(pqt_fp):
                logger.warning(
                    f"  {pqt_file} does not exist in {untokenized_path}, but {npy_fp} exists in {tokenized_path}."
                )

            array = np.memmap(npy_fp, dtype=np.uint32)
            n_tokens = array.shape[0]
            doc_lengths = get_doc_lengths(
                array, eos_token_id=2
            )  # Assuming EOS token ID is 2

            # Read parquet metadata to get number of rows
            parquet_file = pq.ParquetFile(pqt_fp)
            num_rows_parquet = parquet_file.metadata.num_rows
            sub_pqt = parquet_file.read(
                ["n_non_symbol_words", "dataset_name", "source"]
            )

            # 2. Group by dataset_name + source combination efficiently
            # Create a mask for rows where dataset_name is "NDP"
            is_ndp = pc.equal(sub_pqt["dataset_name"], "NDP")

            # For NDP rows, combine dataset_name and source with ---
            # For non-NDP rows, just use dataset_name
            dataset_source = pc.if_else(
                is_ndp,
                pc.binary_join_element_wise(
                    sub_pqt["dataset_name"], sub_pqt["source"], "---"
                ),
                sub_pqt["dataset_name"],
            )
            sub_pqt = sub_pqt.add_column(0, "dataset_source", dataset_source)

            # We can infer the tokens exactly by using the document lengths
            if len(doc_lengths) == num_rows_parquet:
                sub_pqt = sub_pqt.add_column(
                    0, "token_count", pa.array(doc_lengths, type=pa.int64())
                )
                grouped = sub_pqt.select(["token_count", "dataset_source"])
                result = grouped.group_by("dataset_source").aggregate(
                    [("token_count", "sum")]
                )

                for i in range(len(result)):
                    combo = result["dataset_source"][i].as_py()
                    token_count = result["token_count_sum"][i].as_py()
                    counts_per_dataset_source[train_valid][bucket][combo] += token_count

            # Else: We need to make an estimate based on the fraction of n_non_symbol_words or n_rows per dataset_name + source
            else:
                # 1. Total number of n_non_symbol_words (vectorized sum)
                total_words = pc.sum(sub_pqt["n_non_symbol_words"]).as_py()

                # If we have this information use the word fraction (word_count/total_words) to attribute tokens to dataset_name + source groups
                if total_words:
                    grouped = sub_pqt.select(["n_non_symbol_words", "dataset_source"])
                    result = grouped.group_by("dataset_source").aggregate(
                        [("n_non_symbol_words", "sum")]
                    )

                    # Calculate fractions based on n_non_symbol_words
                    for i in range(len(result)):
                        combo = result["dataset_source"][i].as_py()
                        word_count = result["n_non_symbol_words_sum"][i].as_py()
                        fraction = word_count / total_words
                        counts_per_dataset_source[train_valid][bucket][combo] += (
                            fraction * n_tokens
                        )

                # If we don't have word counts, we can only estimate based on the number of rows
                else:
                    logger.warning(
                        "Last resort, using row counts to estimate token counts."
                    )
                    # Calculate fractions based on number of rows since total_words is 0
                    result = sub_pqt.group_by("dataset_source").aggregate(
                        [("dataset_source", "count")]
                    )

                    for i in range(len(result)):
                        combo = result["dataset_source"][i].as_py()
                        row_count = result["dataset_source_count"][i].as_py()
                        fraction = row_count / num_rows_parquet
                        counts_per_dataset_source[train_valid][bucket][combo] += (
                            fraction * n_tokens
                        )

        # Save intermediate progress after each bucket
        save_counts(counts_per_dataset_source, "counts_intermediate.json")
        logger.info(f"Completed bucket {bucket} for {train_valid}")

# Save final counts after processing everything
save_counts(counts_per_dataset_source, "counts_final_all.json")
