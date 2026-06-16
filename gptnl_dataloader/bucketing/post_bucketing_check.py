import os
import dotenv
import json
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Loads PIPELINE_INPUT, OUTPUT_TAGGING
dotenv.load_dotenv()
config_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.env")
dotenv.load_dotenv(config_env)

OUTPUT_TAGGING = os.environ.get("OUTPUT_TAGGING")
OUTPUT_BUCKETING = os.environ.get("OUTPUT_BUCKETING")
METADATA_FILE = os.environ.get("QUALITY_METADATA_FILE")

assert OUTPUT_BUCKETING, "OUTPUT_BUCKETING is not set in config.env"
assert OUTPUT_TAGGING, "OUTPUT_TAGGING is not set in config.env"
assert METADATA_FILE, "QUALITY_METADATA_FILE is not set in config.env"

REGISTRY_FILE = os.path.join(OUTPUT_TAGGING, "document_ids_registry.json")

# Path to the YAML metadata file
METADATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    METADATA_FILE,
)

assert os.path.exists(METADATA_FILE), (
    f"Metadata file not found at {METADATA_FILE}. Please check the path, or run the `notebooks/explore-curated-data.ipynb` notebook."
)

assert os.path.exists(REGISTRY_FILE), f"Registry file not found at {REGISTRY_FILE}. "

# Bucket definitions (matching the bucketing.py structure)
BUCKETS = {
    "new_high_quality": os.path.join(OUTPUT_BUCKETING, "new_high_quality"),
    "new_low_medium_quality": os.path.join(OUTPUT_BUCKETING, "new_low_medium_quality"),
    "dutch_high_quality": os.path.join(OUTPUT_BUCKETING, "dutch_high_quality"),
    "dutch_low_medium_quality": os.path.join(
        OUTPUT_BUCKETING, "dutch_low_medium_quality"
    ),
    "english_high_quality": os.path.join(OUTPUT_BUCKETING, "english_high_quality"),
    "english_low_medium_quality": os.path.join(
        OUTPUT_BUCKETING, "english_low_medium_quality"
    ),
    "other_languages": os.path.join(OUTPUT_BUCKETING, "other_languages"),
    "code": os.path.join(OUTPUT_BUCKETING, "code"),
}


def load_registry():
    """Load the document IDs registry."""
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)


def get_doc_ids_from_parquet(file_path):
    """Efficiently read only doc_id column from parquet file."""
    try:
        # Read only the doc_id column
        table = pq.read_table(file_path, columns=["doc_id"])
        return set(table["doc_id"].to_pylist())
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return set()


def extract_dataset_from_path(original_path):
    """Extract dataset name from original filepath."""
    dataset_name = original_path.split("/")[-2]
    return dataset_name


def perform_post_bucketing_check():
    """
    Check if all document IDs from the registry are present in bucketed files.
    Returns validation results including missing IDs and duplicates.
    """
    logger.info("Starting post-bucketing validation check...")

    # Load registry and initialize tracking structures
    registry = load_registry()

    # Track IDs found per dataset (using sorted sets for efficiency)
    found_ids_per_dataset = defaultdict(set)
    duplicate_ids = set()
    all_found_ids = set()

    # Expected IDs per dataset from registry
    expected_ids_per_dataset = defaultdict(set)
    total_expected_ids = 0

    # Build expected IDs from registry
    logger.info("Building expected IDs from registry...")
    for file_info in registry["files"].values():
        dataset = file_info["dataset"]
        count = file_info["count"]
        start_id = file_info["start_id"]

        # Generate expected IDs for this file
        # Extract the base pattern and generate all IDs
        base_parts = start_id.rsplit("_", 1)[0]  # Remove the last numeric part
        for i in range(count):
            doc_id = f"{base_parts}_{i:08d}"
            expected_ids_per_dataset[dataset].add(doc_id)
            total_expected_ids += 1

    logger.info(
        f"Expected {total_expected_ids} total IDs across {len(expected_ids_per_dataset)} datasets"
    )

    # Process each bucket
    logger.info("Scanning bucket files for document IDs...")
    total_files_processed = 0

    for bucket_name, bucket_path in BUCKETS.items():
        if not os.path.exists(bucket_path):
            logger.warning(f"Bucket directory not found: {bucket_path}")
            continue

        parquet_files = list(Path(bucket_path).glob("*.parquet"))
        if not parquet_files:
            logger.info(f"No parquet files in bucket: {bucket_name}")
            continue

        logger.info(f"Processing {len(parquet_files)} files in bucket: {bucket_name}")

        for file_path in parquet_files:
            try:
                # Read only necessary columns efficiently
                table = pq.read_table(
                    file_path, columns=["doc_id", "original_file_path"]
                )
                doc_ids = table["doc_id"].to_pylist()
                original_paths = table["original_file_path"].to_pylist()

                # Process each document
                for doc_id, original_path in zip(doc_ids, original_paths):
                    # Extract dataset from original path
                    dataset = extract_dataset_from_path(original_path)

                    # Check for duplicates
                    if doc_id in all_found_ids:
                        duplicate_ids.add(doc_id)
                        logger.warning(
                            f"Duplicate ID found: {doc_id} in {bucket_name}/{file_path.name}"
                        )
                    else:
                        all_found_ids.add(doc_id)

                    # Track by dataset
                    found_ids_per_dataset[dataset].add(doc_id)

                total_files_processed += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

    logger.info(f"Processed {total_files_processed} parquet files across all buckets")
    logger.info(f"Found {len(all_found_ids)} unique document IDs")

    # Validate completeness per dataset
    validation_results = {
        "validation_passed": True,
        "total_expected": total_expected_ids,
        "total_found": len(all_found_ids),
        "duplicates_count": len(duplicate_ids),
        "duplicate_ids": list(duplicate_ids),
        "dataset_results": {},
        "missing_ids_by_dataset": {},
        "unexpected_ids_by_dataset": {},
    }

    logger.info("Validating completeness per dataset...")

    for dataset in expected_ids_per_dataset:
        expected_ids = expected_ids_per_dataset[dataset]
        found_ids = found_ids_per_dataset[dataset]

        missing_ids = expected_ids - found_ids
        unexpected_ids = found_ids - expected_ids

        validation_results["dataset_results"][dataset] = {
            "expected_count": len(expected_ids),
            "found_count": len(found_ids),
            "missing_count": len(missing_ids),
            "unexpected_count": len(unexpected_ids),
            "complete": len(missing_ids) == 0 and len(unexpected_ids) == 0,
        }

        if missing_ids:
            validation_results["missing_ids_by_dataset"][dataset] = sorted(
                list(missing_ids)
            )
            validation_results["validation_passed"] = False
            logger.error(f"Dataset {dataset}: {len(missing_ids)} missing IDs")

        if unexpected_ids:
            validation_results["unexpected_ids_by_dataset"][dataset] = sorted(
                list(unexpected_ids)
            )
            validation_results["validation_passed"] = False
            logger.warning(f"Dataset {dataset}: {len(unexpected_ids)} unexpected IDs")

        logger.info(
            f"Dataset {dataset}: {len(found_ids)}/{len(expected_ids)} IDs found"
        )

    # Check for datasets found but not expected
    unexpected_datasets = set(found_ids_per_dataset.keys()) - set(
        expected_ids_per_dataset.keys()
    )
    if unexpected_datasets:
        logger.warning(f"Found unexpected datasets: {unexpected_datasets}")
        for dataset in unexpected_datasets:
            validation_results["unexpected_ids_by_dataset"][dataset] = sorted(
                list(found_ids_per_dataset[dataset])
            )

    # Final validation summary
    if duplicate_ids:
        validation_results["validation_passed"] = False
        logger.error(f"Found {len(duplicate_ids)} duplicate document IDs")

    total_missing = sum(
        len(missing)
        for missing in validation_results["missing_ids_by_dataset"].values()
    )
    total_unexpected = sum(
        len(unexpected)
        for unexpected in validation_results["unexpected_ids_by_dataset"].values()
    )

    if total_missing > 0:
        logger.error(f"Total missing IDs: {total_missing}")

    if total_unexpected > 0:
        logger.warning(f"Total unexpected IDs: {total_unexpected}")

    logger.info(
        f"Validation complete. Passed: {validation_results['validation_passed']}"
    )

    return validation_results


if __name__ == "__main__":
    results = perform_post_bucketing_check()

    if results["validation_passed"]:
        logger.info("🎉 All validation checks passed!")
    else:
        logger.error("💥 Validation failed!")
