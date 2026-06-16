"""This script processes and buckets data into different categories based on quality and language.

The buckets:

- new_high_quality: New high quality data (assumed to be all dutch)
- new_low_medium_quality: New low-medium quality data (assumed to be all dutch)
- dutch_high_quality: Dutch data with high quality (high quality label)
- dutch_low_medium_quality: Dutch data with low-medium quality (not high quality label)
- english_high_quality: English data with high quality (high quality label)
- english_low_medium_quality: English data with low-medium quality (not high quality label)
- other_languages: Data in other languages
- code: Data identified as code

The script uses multiprocessing to process files in parallel, and it handles large datasets efficiently
by using PyArrow for batch processing. It also includes a merge phase to combine temporary files into uniform-sized output files.
"""

import argparse
import gc
import glob
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import dotenv
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Loads OUTPUT_TAGGING
dotenv.load_dotenv()
config_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.env")
dotenv.load_dotenv(config_env)

OUTPUT_BUCKETING = os.environ.get("OUTPUT_BUCKETING")
assert OUTPUT_BUCKETING, "OUTPUT_BUCKETING is not set in config.env"

OUTPUT_TAGGING = os.environ.get("OUTPUT_TAGGING")
assert OUTPUT_TAGGING, "OUTPUT_TAGGING is not set in config.env"

# Path to input registry from tagging step
INPUT_REGISTRY_FILE = os.path.join(OUTPUT_TAGGING, "document_ids_registry.json")

TRACKING_FILE = os.path.join(OUTPUT_BUCKETING, "bucketing_status.json")
TEMP_DIR = os.path.join(OUTPUT_BUCKETING, "temp")

# Default configuration
MAX_ROWS_PER_FILE = None
MAX_FILE_SIZE_MB = 2000  # Maximum size of each output file in MB
MAX_WORKERS = 4
BATCH_SIZE = 20
MERGE_CHUNK_SIZE = 50000  # Chunk size for merge phase

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_BUCKETING, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Bucket definitions
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

# Create bucket directories if they don't exist
for bucket_path in BUCKETS.values():
    os.makedirs(bucket_path, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(OUTPUT_BUCKETING, "bucketing.log"),
    filemode="a",
)

# Add console handler to also log to standard output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)


def load_tracking_data() -> Dict[str, Any]:
    """Load tracking data from the tracking file, or create it if it doesn't exist."""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "processed_batches": [],
            "failed_batches": [],
            "in_progress_batches": [],
            "merged_status": False,
            "bucket_counts": {bucket: 0 for bucket in BUCKETS.keys()},
            "temp_files": {bucket: [] for bucket in BUCKETS.keys()},
        }


def save_tracking_data(tracking_data: Dict[str, Any]) -> None:
    """Save tracking data to the tracking file."""
    # Create a temporary file first to avoid corruption
    temp_file = f"{TRACKING_FILE}.tmp"
    with open(temp_file, "w") as f:
        json.dump(tracking_data, f, indent=4)

    # Atomically replace the original file
    os.replace(temp_file, TRACKING_FILE)


def get_parquet_files() -> List[str]:
    """Get all parquet files in the processed data directory and its subdirectories."""
    parquet_files = []

    for subdir in os.listdir(OUTPUT_TAGGING):
        subdir_path = os.path.join(OUTPUT_TAGGING, subdir)  # type: ignore
        if os.path.isdir(subdir_path):
            files = glob.glob(os.path.join(subdir_path, "*.parquet"))
            parquet_files.extend(files)

    return parquet_files


def sample_all_schemas() -> Dict[str, str]:
    """We assume consistent schema per subdir (dataset), so we sample one file
    per subdir to determine all schemas."""
    schemas = {}
    for subdir in os.listdir(OUTPUT_TAGGING):
        subdir_path = os.path.join(OUTPUT_TAGGING, subdir)  # type: ignore
        if os.path.isdir(subdir_path):
            files = glob.glob(os.path.join(subdir_path, "*.parquet"))
            if files:
                schemas[subdir] = files[0]
    return schemas


def create_master_schema() -> pa.Schema:
    # Collect all field names and their PyArrow types
    logger.info(
        "Creating master schema from, by sampling one file per subdir and reading its schema."
    )
    all_fields = {}

    schemas = sample_all_schemas()

    for subdir, file_path in schemas.items():
        try:
            schema = pq.read_schema(file_path)
            for field in schema:
                # Skip PII field since we're dropping it anyway
                if field.name == "PII":
                    continue

                if field.name in all_fields:
                    # Check if types match
                    existing_type = all_fields[field.name]["type"]
                    if str(existing_type) != str(field.type):
                        raise ValueError(
                            f"Type mismatch for field '{field.name}':\n"
                            f"  - File: {all_fields[field.name]['source']} has type: {existing_type}\n"
                            f"  - File: {file_path} has type: {field.type}"
                        )
                else:
                    # Store the type and source file for debugging
                    all_fields[field.name] = {"type": field.type, "source": file_path}
        except Exception as e:
            logger.error(f"Error reading schema from {file_path}: {str(e)}")
            raise

    # Create master schema
    master_schema_fields = []
    for field_name, field_info in all_fields.items():
        master_schema_fields.append(
            pa.field(field_name, field_info["type"], nullable=True)
        )

    master_schema = pa.schema(master_schema_fields)
    logger.info(f"Created master schema with {len(master_schema_fields)} fields")

    for field in master_schema_fields:
        logger.info(f"  - {field.name}: {field.type}")

    return master_schema


def get_file_info(file_path: str) -> Dict[str, str]:
    """Extract subdir and filename from a full path"""
    subdir = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    return {"subdir": subdir, "filename": filename}


def create_temp_bucket_dirs(worker_id: Union[str, int]) -> Dict[str, str]:
    """Create temporary bucket directories for a worker."""
    temp_buckets = {}
    for bucket_name in BUCKETS.keys():
        temp_bucket_path = os.path.join(TEMP_DIR, f"worker_{worker_id}", bucket_name)
        os.makedirs(temp_bucket_path, exist_ok=True)
        temp_buckets[bucket_name] = temp_bucket_path
    return temp_buckets


def process_rows(
    batch: pa.Table,
) -> Tuple[Dict[str, List[pa.Table]], Dict[str, int]]:
    """Process a subset of a parquet file (batch of rows) and distribute rows into buckets.

    The rules for bucketing are:
    - Code: is_code = True
    - New high quality: is_new_epoch_2 = True and dataset_quality = high (assumed to be all dutch)
    - New low-medium quality: is_new_epoch_2 = True and dataset_quality != high
    - Dutch high quality: is_new_epoch_2 = False and language = nl and dataset_quality = high
    - Dutch low-medium quality: is_new_epoch_2 = False and language = nl and dataset_quality != high
    - English high quality: is_new_epoch_2 = False and language = en and dataset_quality = high
    - English low-medium quality: is_new_epoch_2 = False and language = en and dataset_quality != high
    - Other languages: all other languages (all qualities)

    The function returns two dictionaries:
    - buckets_data: mapping of bucket names to lists of PyArrow tables
    - buckets_counts: mapping of bucket names to the number of rows in each bucket
    """

    # Initialize bucket arrays
    buckets_data = {bucket: [] for bucket in BUCKETS.keys()}
    buckets_counts = {bucket: 0 for bucket in BUCKETS.keys()}

    # Validate required columns
    assert "is_code" in batch.column_names, "is_code must be present in the batch."
    assert "language" in batch.column_names, "language must be present in the batch."
    assert "dataset_quality" in batch.column_names, (
        "dataset_quality must be present in the batch."
    )

    remaining_batch = batch

    # Step 1: Filter out code data first
    if len(remaining_batch) > 0:
        code_mask = pc.equal(remaining_batch["is_code"], True)
        code_indices = pc.filter(pa.array(range(len(remaining_batch))), code_mask)

        if len(code_indices) > 0:
            code_rows = remaining_batch.take(code_indices)
            buckets_data["code"].append(code_rows)
            buckets_counts["code"] += len(code_rows)

            # Remove code rows from further processing
            non_code_mask = pc.invert(code_mask)
            remaining_batch = remaining_batch.filter(non_code_mask)

    # Step 2: Filter out new data (if is_new_epoch_2 column exists)
    if len(remaining_batch) > 0 and "is_new_epoch_2" in remaining_batch.column_names:
        new_mask = pc.equal(remaining_batch["is_new_epoch_2"], True)
        new_indices = pc.filter(pa.array(range(len(remaining_batch))), new_mask)

        if len(new_indices) > 0:
            new_rows = remaining_batch.take(new_indices)

            # Divide new data by quality
            high_quality_mask = pc.equal(new_rows["dataset_quality"], "high")
            high_quality_indices = pc.filter(
                pa.array(range(len(new_rows))), high_quality_mask
            )

            if len(high_quality_indices) > 0:
                new_high_quality = new_rows.take(high_quality_indices)
                buckets_data["new_high_quality"].append(new_high_quality)
                buckets_counts["new_high_quality"] += len(new_high_quality)

            # Low-medium quality new
            low_med_mask = pc.invert(high_quality_mask)
            low_med_indices = pc.filter(pa.array(range(len(new_rows))), low_med_mask)
            if len(low_med_indices) > 0:
                new_low_med_quality = new_rows.take(low_med_indices)
                buckets_data["new_low_medium_quality"].append(new_low_med_quality)
                buckets_counts["new_low_medium_quality"] += len(new_low_med_quality)

            # Remove new rows from further processing
            non_new_mask = pc.invert(new_mask)
            remaining_batch = remaining_batch.filter(non_new_mask)

    # Step 3: Filter Dutch data
    if len(remaining_batch) > 0:
        dutch_mask = pc.equal(remaining_batch["language"], "nl")
        dutch_indices = pc.filter(pa.array(range(len(remaining_batch))), dutch_mask)

        if len(dutch_indices) > 0:
            dutch_rows = remaining_batch.take(dutch_indices)

            # Divide Dutch data by quality
            high_quality_mask = pc.equal(dutch_rows["dataset_quality"], "high")
            high_quality_indices = pc.filter(
                pa.array(range(len(dutch_rows))), high_quality_mask
            )

            if len(high_quality_indices) > 0:
                dutch_high = dutch_rows.take(high_quality_indices)
                buckets_data["dutch_high_quality"].append(dutch_high)
                buckets_counts["dutch_high_quality"] += len(dutch_high)

            # Low-medium quality Dutch
            low_med_mask = pc.invert(high_quality_mask)
            low_med_indices = pc.filter(pa.array(range(len(dutch_rows))), low_med_mask)
            if len(low_med_indices) > 0:
                dutch_low_med = dutch_rows.take(low_med_indices)
                buckets_data["dutch_low_medium_quality"].append(dutch_low_med)
                buckets_counts["dutch_low_medium_quality"] += len(dutch_low_med)

            # Remove Dutch rows from further processing
            non_dutch_mask = pc.invert(dutch_mask)
            remaining_batch = remaining_batch.filter(non_dutch_mask)

    # Step 4: Filter English data
    if len(remaining_batch) > 0:
        english_mask = pc.equal(remaining_batch["language"], "en")
        english_indices = pc.filter(pa.array(range(len(remaining_batch))), english_mask)

        if len(english_indices) > 0:
            english_rows = remaining_batch.take(english_indices)

            # Divide English data by quality
            high_quality_mask = pc.equal(english_rows["dataset_quality"], "high")
            high_quality_indices = pc.filter(
                pa.array(range(len(english_rows))), high_quality_mask
            )

            if len(high_quality_indices) > 0:
                english_high = english_rows.take(high_quality_indices)
                buckets_data["english_high_quality"].append(english_high)
                buckets_counts["english_high_quality"] += len(english_high)

            # Low-medium quality English
            low_med_mask = pc.invert(high_quality_mask)
            low_med_indices = pc.filter(
                pa.array(range(len(english_rows))), low_med_mask
            )
            if len(low_med_indices) > 0:
                english_low_med = english_rows.take(low_med_indices)
                buckets_data["english_low_medium_quality"].append(english_low_med)
                buckets_counts["english_low_medium_quality"] += len(english_low_med)

            # Remove English rows from further processing
            non_english_mask = pc.invert(english_mask)
            remaining_batch = remaining_batch.filter(non_english_mask)

    # Step 5: Everything else goes to other languages
    if len(remaining_batch) > 0:
        buckets_data["other_languages"].append(remaining_batch)
        buckets_counts["other_languages"] += len(remaining_batch)

    return buckets_data, buckets_counts


def write_temp_bucket_files(
    buckets_data: Dict[str, List[pa.Table]],
    buckets_counts: Dict[str, int],
    temp_buckets: Dict[str, str],
    worker_id: Union[str, int],
    batch_id: int,
) -> Dict[str, Dict[str, Union[str, int]]]:
    """Write data to temporary bucket files.

    This function takes processed data tables for each bucket and writes them to
    temporary parquet files with a unique naming pattern. If multiple tables exist
    for a single bucket, they are combined into one table before writing.

    Args:
        buckets_data: Dictionary mapping bucket names to lists of PyArrow tables
        buckets_counts: Dictionary tracking row counts for each bucket
        temp_buckets: Dictionary mapping bucket names to temporary directory paths
        worker_id: Unique identifier for the worker process
        batch_id: Current batch identifier for file naming

    Returns:
        Dictionary containing paths and row counts of created temporary files
    """
    temp_files = {}

    for bucket_name in BUCKETS.keys():
        if buckets_data[bucket_name] and buckets_counts[bucket_name] > 0:
            # Combine tables if there are multiple
            if len(buckets_data[bucket_name]) > 1:
                table = pa.concat_tables(buckets_data[bucket_name])
            elif len(buckets_data[bucket_name]) == 1:
                table = buckets_data[bucket_name][0]
            else:
                continue

            # Create a unique filename
            timestamp = int(time.time())
            filename = f"{bucket_name}_worker{worker_id}_batch{batch_id}_{timestamp}_{buckets_counts[bucket_name]}_rows.parquet"
            output_path = os.path.join(temp_buckets[bucket_name], filename)

            # Write the table
            pq.write_table(table, output_path, compression="snappy")
            temp_files[bucket_name] = {
                "path": output_path,
                "rows": buckets_counts[bucket_name],
            }

            # Clean up to free memory
            del table

    return temp_files


def process_file(
    file_path: str,
    worker_id: Union[str, int],
) -> Dict[str, Any]:
    """
    Process a single parquet file and distribute rows to temporary buckets.

    This function reads data from a Parquet file in batches to manage memory efficiently.
    Each batch is processed and rows are distributed across multiple temporary bucket files
    based on a partitioning strategy (some simple value rules).

    The temporary bucket files serve as intermediate storage during the bucketing process:
    - They allow processing of datasets larger than available memory
    - Each temp bucket will contain a subset of rows that belong together
    - These temp files can later be merged into final output buckets

    Batching approach:
    - Data is read in chunks rather than loading the entire file at once
    - This controls memory usage for large datasets
    - Each batch is processed and immediately written to temp files
    - Batch size can be configured based on available system resources
    """
    try:
        # Create temp bucket directories for this worker
        temp_buckets = create_temp_bucket_dirs(worker_id)

        # Process file in batches using PyArrow for memory efficiency
        file_results = {
            "processed": True,
            "error": None,
            "temp_files": {bucket: [] for bucket in BUCKETS.keys()},
            "bucket_counts": {bucket: 0 for bucket in BUCKETS.keys()},
        }

        # Read the parquet file using PyArrow - process in batches
        parquet_file = pq.ParquetFile(file_path)
        batch_id = 0

        for batch in parquet_file.iter_batches(batch_size=10000):
            table = pa.Table.from_batches([batch])

            # Process the batch
            buckets_data, buckets_counts = process_rows(table)

            # Write to temp files
            temp_files = write_temp_bucket_files(
                buckets_data, buckets_counts, temp_buckets, worker_id, batch_id
            )

            # Update file results
            for bucket_name in BUCKETS.keys():
                if bucket_name in temp_files:
                    file_results["temp_files"][bucket_name].append(
                        temp_files[bucket_name]
                    )
                    file_results["bucket_counts"][bucket_name] += temp_files[
                        bucket_name
                    ]["rows"]

            # Clean up memory
            del table, buckets_data
            gc.collect()

            batch_id += 1

        return file_results

    except Exception as e:
        # Return error information
        return {
            "processed": False,
            "error": str(e),
            "temp_files": {},
            "bucket_counts": {},
        }


def process_batch(
    batch_files: List[str],
    batch_id: int,
    max_workers: int,
) -> Optional[Dict[str, Any]]:
    """Process a batch of files in parallel."""
    batch_results = {
        "batch_id": batch_id,
        "files": batch_files,
        "processed_count": 0,
        "failed_count": 0,
        "temp_files": {bucket: [] for bucket in BUCKETS.keys()},
        "bucket_counts": {bucket: 0 for bucket in BUCKETS.keys()},
        "errors": [],
    }

    # Mark batch as in progress
    tracking_data = load_tracking_data()
    tracking_data["in_progress_batches"].append(batch_id)
    save_tracking_data(tracking_data)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, file_path in enumerate(batch_files):
                worker_id = f"{batch_id}_{i}"
                future = executor.submit(
                    process_file,
                    file_path,
                    worker_id,
                )
                futures[future] = file_path

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result["processed"]:
                        # Successfully processed
                        batch_results["processed_count"] += 1

                        # Collect temp files and counts
                        for bucket_name in BUCKETS.keys():
                            if bucket_name in result["temp_files"]:
                                batch_results["temp_files"][bucket_name].extend(
                                    result["temp_files"][bucket_name]
                                )
                                batch_results["bucket_counts"][bucket_name] += result[
                                    "bucket_counts"
                                ][bucket_name]
                    else:
                        # Failed to process
                        batch_results["failed_count"] += 1
                        batch_results["errors"].append(
                            {"file": file_path, "error": result["error"]}
                        )
                        logger.error(
                            f"Failed to process file {file_path}: {result['error']}"
                        )
                except Exception as e:
                    # Exception in future
                    batch_results["failed_count"] += 1
                    batch_results["errors"].append({"file": file_path, "error": str(e)})
                    logger.error(f"Exception processing file {file_path}: {str(e)}")

        # Update tracking data
        tracking_data = load_tracking_data()
        if batch_id in tracking_data["in_progress_batches"]:
            tracking_data["in_progress_batches"].remove(batch_id)

        # Update temp files and counts
        for bucket_name in BUCKETS.keys():
            tracking_data["temp_files"][bucket_name].extend(
                batch_results["temp_files"][bucket_name]
            )
            tracking_data["bucket_counts"][bucket_name] += batch_results[
                "bucket_counts"
            ][bucket_name]

        # Add to processed batches
        tracking_data["processed_batches"].append(batch_id)
        save_tracking_data(tracking_data)

        logger.info(
            f"Batch {batch_id} completed: {batch_results['processed_count']} processed, {batch_results['failed_count']} failed"
        )
        for bucket_name in BUCKETS.keys():
            if batch_results["bucket_counts"][bucket_name] > 0:
                logger.info(
                    f"  - {bucket_name}: {batch_results['bucket_counts'][bucket_name]} rows"
                )

        return batch_results

    except Exception as e:
        # Update tracking data for batch failure
        tracking_data = load_tracking_data()
        if batch_id in tracking_data["in_progress_batches"]:
            tracking_data["in_progress_batches"].remove(batch_id)
        tracking_data["failed_batches"].append({"batch_id": batch_id, "error": str(e)})
        save_tracking_data(tracking_data)

        logger.error(f"Failed to process batch {batch_id}: {str(e)}")
        return None


def harmonize_schema(table: pa.Table, master_schema: pa.Schema) -> pa.Table:
    """
    Transform a table to conform to the master schema.
    - Adds missing columns with null values
    - Casts existing columns to the correct type
    - Drops PII column if present (since it's not in master schema)
    """
    # Drop PII column first if it exists, since it's not in master schema
    if "PII" in table.column_names:
        table = table.drop(["PII"])

    # Get the field names from both schemas
    table_fields = {field.name: field for field in table.schema}
    master_fields = {field.name: field for field in master_schema}

    # Create arrays for the new table
    new_arrays = []

    # Process each field in the master schema
    for field_name, master_field in master_fields.items():
        if field_name in table_fields:
            # Field exists in the original table
            orig_array = table[field_name]
            orig_type = table_fields[field_name].type
            target_type = master_field.type

            if str(orig_type) != str(target_type):
                # Types don't match, cast to target type
                try:
                    new_array = orig_array.cast(target_type, safe=False)
                except pa.ArrowInvalid:
                    # If casting fails, create a column of nulls with the right type
                    logger.warning(
                        f"Cannot cast column {field_name} from {orig_type} to {target_type}. Using nulls."
                    )
                    new_array = pa.nulls(len(table), type=target_type)
            else:
                # Types match, use as is
                new_array = orig_array
        else:
            # Field doesn't exist in the original table, create a null array
            new_array = pa.nulls(len(table), type=master_field.type)

        new_arrays.append(new_array)

    # Create harmonized table
    result_table = pa.Table.from_arrays(new_arrays, schema=master_schema)

    return result_table


def merge_temp_files(
    temp_files: List[Dict[str, Union[str, int]]],
    bucket_name: str,
    master_schema: pa.Schema,
    max_rows: int | None = None,
    max_size_mb: int | None = None,
) -> List[str]:
    """Merge temporary files into uniform-sized final files using the master schema."""
    # Validate input parameters - exactly one sizing parameter must be set
    if (max_rows is None and max_size_mb is None) or (
        max_rows is not None and max_size_mb is not None
    ):
        raise ValueError("Exactly one of max_rows or max_size_mb must be specified")

    if not temp_files:
        logger.info(f"No temp files to merge for {bucket_name}")
        return []

    output_files = []
    total_rows = sum(file_info["rows"] for file_info in temp_files)  # type: ignore
    logger.info(
        f"Merging {len(temp_files)} temp files with {total_rows} total rows for {bucket_name}"
    )

    # Read all temp files and create output tables
    timestamp = int(time.time())
    current_output_idx = 0
    current_output_rows = 0
    current_batch_tables = []

    # For size-based splitting
    current_estimated_size_mb = 0
    using_size_based = max_size_mb is not None

    target_size_mb = max_size_mb

    # Compression ratio estimation
    compression_ratio = 0.4  # Initial estimate

    # If we have temp files, use the first small batch as a calibration sample
    if using_size_based and temp_files:
        try:
            # Use the first file as a sample to estimate compression ratio
            sample_file_path = temp_files[0]["path"]
            sample_file = pq.ParquetFile(sample_file_path)

            # Read a sample batch (limited to avoid memory issues)
            sample_batch_size = min(10000, sample_file.metadata.num_rows)
            sample_table = pa.Table.from_batches(
                [next(sample_file.iter_batches(batch_size=sample_batch_size))]
            )
            sample_table = harmonize_schema(sample_table, master_schema)

            # Measure memory size before compression
            memory_size_bytes = sample_table.nbytes

            # Write to a temporary file with compression
            temp_output = os.path.join(
                TEMP_DIR, f"{bucket_name}_compression_test.parquet"
            )
            pq.write_table(sample_table, temp_output, compression="snappy")

            # Measure size after compression
            disk_size_bytes = os.path.getsize(temp_output)

            # Calculate actual compression ratio
            actual_ratio = disk_size_bytes / memory_size_bytes

            # Use this ratio with slight safety margin
            compression_ratio = actual_ratio * 1.05  # Add 5% margin for variations

            logger.info(
                f"Bucket {bucket_name}: Calibrated compression ratio to {compression_ratio:.4f} based on test sample"
            )

            # Clean up the test file
            os.remove(temp_output)
        except Exception as e:
            logger.warning(
                f"Failed to calibrate compression ratio: {str(e)}. Using default ratio of 0.4"
            )
            compression_ratio = 0.4

    # For row-based splitting
    rows_per_file = None
    if not using_size_based:
        # Calculate number of output files needed for row-based splitting
        num_output_files = (total_rows + max_rows - 1) // max_rows  # Ceiling division

        if num_output_files == 0:
            logger.warning(
                f"No output files needed for {bucket_name} (total rows: {total_rows})"
            )
            return []

        # Target rows per file (may be less than max_rows for the last file)
        rows_per_file = [
            (total_rows // num_output_files) for _ in range(num_output_files)
        ]
        remainder = total_rows % num_output_files
        for i in range(remainder):
            rows_per_file[i] += 1

        logger.info(
            f"Creating {num_output_files} output files for {bucket_name} (row-based)"
        )
        for i, target_rows in enumerate(rows_per_file):
            logger.info(f"  - File {i + 1}: target ~{target_rows} rows")
    else:
        logger.info(
            f"Creating output files for {bucket_name} (size-based, max {max_size_mb}MB, target {target_size_mb:.2f}MB)"
        )

    # Track actual compression ratios observed to improve future estimates
    observed_ratios = []

    for file_info in temp_files:
        file_path = file_info["path"]

        if not os.path.exists(file_path):
            logger.warning(f"Temp file not found: {file_path}")
            continue

        # Read the file in chunks to avoid memory issues
        parquet_file = pq.ParquetFile(file_path)

        for batch in parquet_file.iter_batches(batch_size=MERGE_CHUNK_SIZE):
            table = pa.Table.from_batches([batch])

            # Harmonize the schema before merging
            table = harmonize_schema(table, master_schema)

            batch_rows = table.num_rows

            if using_size_based:
                # Size-based splitting using nbytes as a proxy for file size
                batch_size_bytes = table.nbytes

                # Apply current compression ratio estimate
                batch_size_mb = (batch_size_bytes * compression_ratio) / (1024 * 1024)

                if current_batch_tables:
                    # Estimate total size if we add this batch
                    new_total_size_mb = current_estimated_size_mb + batch_size_mb

                    # If adding this batch would exceed target size, write current output and start new file
                    if new_total_size_mb > target_size_mb:
                        # Write current output without the new batch
                        output_table = pa.concat_tables(current_batch_tables)
                        output_filename = f"{bucket_name}_{timestamp}_{current_output_idx}_{current_output_rows}_rows.parquet"
                        output_path = os.path.join(
                            BUCKETS[bucket_name], output_filename
                        )
                        pq.write_table(output_table, output_path, compression="snappy")

                        # Calculate actual compression ratio for this file and update our estimate
                        output_memory_size = output_table.nbytes
                        actual_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        actual_compression = (
                            actual_size_mb * 1024 * 1024
                        ) / output_memory_size

                        # Store observed ratio
                        observed_ratios.append(actual_compression)

                        # Update compression ratio if we have enough observations (weighted average)
                        if len(observed_ratios) >= 1:
                            # More weight to recent observations (last 3 files)
                            recent_ratios = (
                                observed_ratios[-3:]
                                if len(observed_ratios) >= 3
                                else observed_ratios
                            )
                            compression_ratio = sum(recent_ratios) / len(recent_ratios)

                        logger.info(
                            f"  - File {current_output_idx}: {current_output_rows} rows, "
                            f"estimated {current_estimated_size_mb:.2f}MB, actual {actual_size_mb:.2f}MB, "
                            f"compression ratio: {actual_compression:.4f}, next ratio: {compression_ratio:.4f}"
                        )

                        output_files.append(output_path)

                        # Clean up and start new output
                        del output_table, current_batch_tables
                        current_batch_tables = [table]  # Start with current batch
                        current_output_idx += 1
                        current_output_rows = batch_rows
                        current_estimated_size_mb = batch_size_mb
                        gc.collect()
                    else:
                        # Batch fits in current output
                        current_batch_tables.append(table)
                        current_output_rows += batch_rows
                        current_estimated_size_mb = new_total_size_mb
                else:
                    # First batch for this output
                    current_batch_tables.append(table)
                    current_output_rows += batch_rows
                    current_estimated_size_mb = batch_size_mb
            else:
                # Row-based splitting (original logic)
                remaining_rows = rows_per_file[current_output_idx] - current_output_rows  # type: ignore

                if batch_rows <= remaining_rows:
                    # Entire batch fits in current output
                    current_batch_tables.append(table)
                    current_output_rows += batch_rows

                    # Check if current output is complete
                    if current_output_rows == rows_per_file[current_output_idx]:  # type: ignore
                        # Write current output
                        if current_batch_tables:
                            output_table = pa.concat_tables(current_batch_tables)
                            output_filename = f"{bucket_name}_{timestamp}_{current_output_idx}_{current_output_rows}_rows.parquet"
                            output_path = os.path.join(
                                BUCKETS[bucket_name], output_filename
                            )
                            pq.write_table(
                                output_table, output_path, compression="snappy"
                            )
                            output_files.append(output_path)

                            # Clean up
                            del output_table, current_batch_tables
                            current_batch_tables = []
                            gc.collect()

                        # Move to next output
                        current_output_idx += 1
                        current_output_rows = 0

                        # Exit if we've created all outputs
                        if current_output_idx >= len(rows_per_file):  # type: ignore
                            break
                else:
                    # Batch needs to be split
                    # First part goes to current output
                    first_part = table.slice(0, remaining_rows)
                    current_batch_tables.append(first_part)

                    # Write current output
                    output_table = pa.concat_tables(current_batch_tables)
                    output_filename = f"{bucket_name}_{timestamp}_{current_output_idx}_{rows_per_file[current_output_idx]}_rows.parquet"  # type: ignore
                    output_path = os.path.join(BUCKETS[bucket_name], output_filename)
                    pq.write_table(output_table, output_path, compression="snappy")
                    output_files.append(output_path)

                    # Clean up
                    del output_table, current_batch_tables, first_part
                    current_batch_tables = []

                    # Move to next output
                    current_output_idx += 1

                    # Exit if we've created all outputs
                    if current_output_idx >= len(rows_per_file):  # type: ignore
                        break

                    # Remaining part goes to next output
                    second_part = table.slice(remaining_rows)
                    current_batch_tables.append(second_part)
                    current_output_rows = second_part.num_rows

        # Clean up after processing each file
        try:
            os.remove(file_path)  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to remove temp file {file_path}: {str(e)}")

    # Handle any remaining data
    if current_batch_tables:
        output_table = pa.concat_tables(current_batch_tables)
        output_filename = f"{bucket_name}_{timestamp}_{current_output_idx}_{current_output_rows}_rows.parquet"
        output_path = os.path.join(BUCKETS[bucket_name], output_filename)
        pq.write_table(output_table, output_path, compression="snappy")
        output_files.append(output_path)

        # For size-based splitting, log the actual file size
        if using_size_based:
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(
                f"  - File {current_output_idx}: {current_output_rows} rows, estimated {current_estimated_size_mb:.2f}MB, actual {file_size_mb:.2f}MB"
            )

    # Log summary of compression ratios
    if observed_ratios:
        avg_ratio = sum(observed_ratios) / len(observed_ratios)
        min_ratio = min(observed_ratios)
        max_ratio = max(observed_ratios)
        logger.info(
            f"Bucket {bucket_name}: Compression ratio stats - avg: {avg_ratio:.4f}, min: {min_ratio:.4f}, max: {max_ratio:.4f}"
        )

    return output_files


def merge_all_temp_files(
    max_rows_per_file: int | None = None, max_size_mb: int | None = None
) -> None:
    """Merge all temporary files into final output files."""

    assert max_rows_per_file or max_size_mb, (
        "Either max_rows_per_file or max_size_mb must be specified"
    )
    assert not (max_rows_per_file and max_size_mb), (
        "Only one of max_rows_per_file or max_size_mb can be specified"
    )

    tracking_data = load_tracking_data()

    # Check if merging has already been done
    if tracking_data["merged_status"]:
        logger.info("Files have already been merged. Skipping merge phase.")
        return

    logger.info("Starting merge phase")

    # Create master schema that will be used for all buckets
    master_schema = create_master_schema()
    logger.info(f"Created master schema with {len(master_schema)} fields")

    # Calculate total number of rows in temp files for verification
    temp_rows_total = 0
    for bucket_name in BUCKETS.keys():
        temp_files = tracking_data["temp_files"][bucket_name]
        if temp_files:
            bucket_rows = sum(file_info["rows"] for file_info in temp_files)
            temp_rows_total += bucket_rows
            logger.info(f"Bucket {bucket_name}: {bucket_rows} rows in temp files")

    logger.info(f"Total rows across all temp files: {temp_rows_total}")
    input_total = tracking_data.get("input_row_count", 0)
    if input_total > 0:
        if temp_rows_total == input_total:
            logger.info(
                f"Pre-merge validation PASSED: all {input_total} input rows accounted for in temp files"
            )
        else:
            logger.warning(
                f"Pre-merge validation WARNING: {input_total} input rows vs {temp_rows_total} rows in temp files (difference: {input_total - temp_rows_total})"
            )

    # Process each bucket
    for bucket_name in BUCKETS.keys():
        temp_files = tracking_data["temp_files"][bucket_name]
        if not temp_files:
            logger.info(f"No files to merge for bucket {bucket_name}")
            continue

        logger.info(f"Processing {len(temp_files)} files in {bucket_name}")

        try:
            # Use the harmonized schema-aware merge
            output_files = merge_temp_files(
                temp_files=temp_files,
                bucket_name=bucket_name,
                master_schema=master_schema,
                max_rows=max_rows_per_file,
                max_size_mb=max_size_mb,
            )
            logger.info(
                f"Created {len(output_files)} uniform output files for {bucket_name}"
            )
        except Exception as e:
            logger.error(f"Error merging files for bucket {bucket_name}: {str(e)}")
            continue

    # Clean up temp directories
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logger.info(f"Removed temporary directory: {TEMP_DIR}")
    except Exception as e:
        logger.warning(f"Failed to remove temp directory {TEMP_DIR}: {str(e)}")

    # Update tracking data
    tracking_data["merged_status"] = True
    tracking_data["temp_files"] = {bucket: [] for bucket in BUCKETS.keys()}
    save_tracking_data(tracking_data)
    logger.info("Merge phase completed successfully")


def count_rows_in_parquet_file(file_path: str) -> int:
    """Count the number of rows in a parquet file without loading it all into memory."""
    try:
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows
    except Exception as e:
        logger.error(f"Error counting rows in {file_path}: {str(e)}")
        return 0


def get_row_counts(file_paths: List[str]) -> Dict[str, int]:
    """Get row counts for multiple parquet files."""
    counts = {}
    total = 0

    for file_path in file_paths:
        num_rows = count_rows_in_parquet_file(file_path)
        counts[file_path] = num_rows
        total += num_rows

    counts["total"] = total
    return counts


def verify_row_counts(
    input_counts: Dict[str, int], output_counts: Dict[str, int]
) -> bool:
    """Verify that the total number of rows is preserved."""
    input_total = input_counts.get("total", 0)
    output_total = output_counts.get("total", 0)

    if input_total == output_total:
        logger.info(
            f"Row count verification PASSED: {input_total} input rows = {output_total} output rows"
        )
        return True
    else:
        logger.error(
            f"Row count verification FAILED: {input_total} input rows ≠ {output_total} output rows (difference: {input_total - output_total})"
        )
        if input_total > output_total:
            logger.error(f"  Missing {input_total - output_total} rows in output")
        else:
            logger.error(f"  Extra {output_total - input_total} rows in output")
        return False


def load_input_id_registry():
    """Load document ID registry from the tagging step."""
    if not os.path.exists(INPUT_REGISTRY_FILE):
        logger.warning(f"Input ID registry not found: {INPUT_REGISTRY_FILE}")
        return None

    with open(INPUT_REGISTRY_FILE, "r") as f:
        return json.load(f)


def extract_all_output_ids():
    """Extract all doc_id values from final bucket files."""
    logger.info("Extracting all document IDs from final bucket files...")
    all_output_ids = set()

    for bucket_name, bucket_path in BUCKETS.items():
        bucket_files = glob.glob(os.path.join(bucket_path, "*.parquet"))
        bucket_ids = set()

        for file_path in bucket_files:
            try:
                # Read only the doc_id column
                table = pq.read_table(file_path, columns=["doc_id"])
                doc_ids = table["doc_id"].to_pylist()
                bucket_ids.update(doc_ids)
            except Exception as e:
                logger.error(f"Error reading doc_ids from {file_path}: {str(e)}")
                continue

        logger.info(f"Bucket {bucket_name}: {len(bucket_ids)} unique document IDs")
        all_output_ids.update(bucket_ids)

    logger.info(f"Total unique document IDs in output: {len(all_output_ids)}")
    return all_output_ids


def get_expected_ids_from_registry(registry):
    """Extract expected document IDs from the tagging registry."""
    if not registry or "files" not in registry:
        return set()

    expected_ids = set()
    for file_key, file_info in registry["files"].items():
        start_id = file_info["start_id"]
        count = file_info["count"]

        # Extract base pattern from start_id (dataset_filename_)
        parts = start_id.split("_")
        if len(parts) >= 3:
            base_pattern = "_".join(parts[:-1]) + "_"

            # Generate expected IDs for this file
            for i in range(count):
                doc_id = f"{base_pattern}{i:08d}"
                expected_ids.add(doc_id)

    return expected_ids


def validate_document_ids():
    """Validate that all expected document IDs are present in output files."""
    logger.info("Starting document ID validation...")

    # Load input registry
    input_registry = load_input_id_registry()
    if not input_registry:
        logger.warning("Cannot validate document IDs: input registry not available")
        return False

    # Get expected IDs from registry
    expected_ids = get_expected_ids_from_registry(input_registry)
    expected_count = len(expected_ids)
    logger.info(f"Expected {expected_count} document IDs from input registry")

    if expected_count == 0:
        logger.warning("No expected IDs found in registry")
        return False

    # Extract actual output IDs
    output_ids = extract_all_output_ids()
    output_count = len(output_ids)

    # Compare sets
    missing_ids = expected_ids - output_ids
    extra_ids = output_ids - expected_ids

    # Log results
    validation_passed = len(missing_ids) == 0 and len(extra_ids) == 0

    if validation_passed:
        logger.info(
            f"✅ Document ID validation PASSED: All {expected_count} expected IDs present in output"
        )
    else:
        logger.error("❌ Document ID validation FAILED:")
        logger.error(f"  Expected IDs: {expected_count}")
        logger.error(f"  Output IDs: {output_count}")
        logger.error(f"  Missing IDs: {len(missing_ids)}")
        logger.error(f"  Extra IDs: {len(extra_ids)}")

        # Log details about missing IDs (limit to first 10)
        if missing_ids:
            sample_missing = sorted(list(missing_ids))[:10]
            logger.error(f"  Sample missing IDs: {sample_missing}")
            if len(missing_ids) > 10:
                logger.error(f"  ... and {len(missing_ids) - 10} more missing IDs")

        # Log details about extra IDs (limit to first 10)
        if extra_ids:
            sample_extra = sorted(list(extra_ids))[:10]
            logger.error(f"  Sample extra IDs: {sample_extra}")
            if len(extra_ids) > 10:
                logger.error(f"  ... and {len(extra_ids) - 10} more extra IDs")

    # Update tracking data with validation results
    tracking_data = load_tracking_data()
    tracking_data["id_validation"] = {
        "passed": validation_passed,
        "expected_count": expected_count,
        "output_count": output_count,
        "missing_count": len(missing_ids),
        "extra_count": len(extra_ids),
    }
    save_tracking_data(tracking_data)

    return validation_passed


def process_files(args: argparse.Namespace) -> None:
    """Process all parquet files and distribute data into buckets."""
    try:
        # Get tracking data and files
        tracking_data = load_tracking_data()
        parquet_files = get_parquet_files()

        # Count input rows at the beginning for verification
        logger.info("Counting rows in input files...")
        input_row_counts = get_row_counts(parquet_files)
        logger.info(f"Total input rows across all files: {input_row_counts['total']}")

        # Store row counts in tracking data
        tracking_data["input_row_count"] = input_row_counts["total"]
        save_tracking_data(tracking_data)

        # Filter out files from already processed batches
        processed_files = []
        for batch_id in tracking_data["processed_batches"]:
            batch_start = batch_id * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(parquet_files))
            processed_files.extend(parquet_files[batch_start:batch_end])

        # Filter out files from in-progress batches
        in_progress_files = []
        for batch_id in tracking_data["in_progress_batches"]:
            batch_start = batch_id * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(parquet_files))
            in_progress_files.extend(parquet_files[batch_start:batch_end])

        # Remaining files to process
        remaining_files = [
            f
            for f in parquet_files
            if f not in processed_files and f not in in_progress_files
        ]

        logger.info(f"Found {len(parquet_files)} total parquet files")
        logger.info(f"Already processed in batches: {len(processed_files)}")
        logger.info(f"In progress in batches: {len(in_progress_files)}")
        logger.info(f"Remaining to process: {len(remaining_files)}")

        # Calculate batch count
        total_batch_count = (
            len(parquet_files) + args.batch_size - 1
        ) // args.batch_size
        processed_batch_count = len(tracking_data["processed_batches"])
        in_progress_batch_count = len(tracking_data["in_progress_batches"])
        remaining_batch_count = (
            total_batch_count - processed_batch_count - in_progress_batch_count
        )

        logger.info(f"Total batches: {total_batch_count}")
        logger.info(f"Processed batches: {processed_batch_count}")
        logger.info(f"In progress batches: {in_progress_batch_count}")
        logger.info(f"Remaining batches: {remaining_batch_count}")

        # Process in debug mode (only one batch) or full mode
        if args.debug:
            if remaining_files:
                batch_id = max(tracking_data["processed_batches"] + [-1]) + 1
                files_to_process = remaining_files[: args.batch_size]
                logger.info(
                    f"DEBUG MODE: Processing only batch {batch_id} with {len(files_to_process)} files"
                )
                process_batch(files_to_process, batch_id, args.workers)
            else:
                logger.info("No files to process in debug mode")
        else:
            # Process all remaining batches
            for i in range(remaining_batch_count):
                batch_id = processed_batch_count + i
                batch_start = batch_id * args.batch_size
                batch_end = min(batch_start + args.batch_size, len(parquet_files))

                # Skip if batch is already in progress
                if batch_id in tracking_data["in_progress_batches"]:
                    logger.info(f"Skipping batch {batch_id} (already in progress)")
                    continue

                # Skip if batch is already processed
                if batch_id in tracking_data["processed_batches"]:
                    logger.info(f"Skipping batch {batch_id} (already processed)")
                    continue

                batch_files = parquet_files[batch_start:batch_end]
                logger.info(
                    f"Processing batch {batch_id} ({i + 1}/{remaining_batch_count}) with {len(batch_files)} files"
                )
                _batch_result = process_batch(
                    batch_files,
                    batch_id,
                    args.workers,
                )

                # Log progress
                tracking_data = load_tracking_data()
                logger.info(
                    f"Overall progress: {len(tracking_data['processed_batches'])}/{total_batch_count} batches"
                )
                logger.info(f"Current bucket counts: {tracking_data['bucket_counts']}")

        # Merge phase if not in debug mode or explicitly requested
        if not args.debug or args.merge:
            merge_all_temp_files(
                max_rows_per_file=args.max_rows, max_size_mb=args.max_size_mb
            )

        # Final report
        tracking_data = load_tracking_data()
        logger.info("Bucketing complete!")
        logger.info(
            f"Processed {len(tracking_data['processed_batches'])}/{total_batch_count} batches"
        )
        logger.info(f"Failed batches: {len(tracking_data['failed_batches'])}")
        logger.info(f"Final bucket counts: {tracking_data['bucket_counts']}")

        # Verify row counts
        if tracking_data.get("merged_status", False):
            # Get all output files from all buckets
            output_files = []
            for bucket_name, bucket_path in BUCKETS.items():
                bucket_files = glob.glob(os.path.join(bucket_path, "*.parquet"))
                output_files.extend(bucket_files)

            # Count output rows
            logger.info("Counting rows in output files...")
            output_row_counts = get_row_counts(output_files)
            output_total = output_row_counts["total"]

            # Bucket-specific counts for detailed reporting
            bucket_totals = {}
            for bucket_name, bucket_path in BUCKETS.items():
                bucket_files = glob.glob(os.path.join(bucket_path, "*.parquet"))
                bucket_count = sum(output_row_counts.get(f, 0) for f in bucket_files)
                bucket_totals[bucket_name] = bucket_count
                logger.info(f"Bucket {bucket_name}: {bucket_count} rows")

            # Compare with tracking data counts
            for bucket_name, count in bucket_totals.items():
                tracked_count = tracking_data["bucket_counts"].get(bucket_name, 0)
                if count != tracked_count:
                    logger.warning(
                        f"Bucket {bucket_name}: Count mismatch - {count} rows counted, but {tracked_count} tracked"
                    )

            # Compare total input and output counts
            input_total = tracking_data.get("input_row_count", 0)
            if input_total > 0:
                verify_row_counts({"total": input_total}, {"total": output_total})

                # Validate document IDs
                logger.info("Performing document ID validation...")
                id_validation_passed = validate_document_ids()

                if id_validation_passed:
                    logger.info(
                        "🎉 All validations passed: row counts and document IDs match"
                    )
                else:
                    logger.error(
                        "⚠️  Document ID validation failed - data integrity issue detected"
                    )

                # Save final counts to tracking data
                tracking_data["output_row_count"] = output_total
                tracking_data["output_bucket_counts"] = bucket_totals
                tracking_data["row_count_verified"] = input_total == output_total
                save_tracking_data(tracking_data)
        else:
            logger.warning("Merge phase not completed, skipping validation")

        # Print failed batches details
        for failed in tracking_data["failed_batches"]:
            logger.info(
                f"Failed batch: {failed['batch_id']} - Error: {failed['error']}"
            )

    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        print(f"Error: {str(e)}")


def main() -> None:
    """Main function to run the bucketing process."""
    # Declare globals at the beginning of the function
    global MAX_WORKERS, BATCH_SIZE, MAX_ROWS_PER_FILE, MAX_FILE_SIZE_MB

    parser = argparse.ArgumentParser(
        description="Optimized script to bucket processed data by quality and language."
    )

    # Basic operation modes
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (process only one batch)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only perform the merge phase (skip processing)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Force merge phase after processing (even in debug mode)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of files to process in each batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        required=False,
        default=MAX_ROWS_PER_FILE,
        help=f"Maximum rows per output file (default: {MAX_ROWS_PER_FILE})",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        required=False,
        default=MAX_FILE_SIZE_MB,
        help=f"Maximum file size (default: {MAX_FILE_SIZE_MB} MB)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})",
    )

    args = parser.parse_args()

    # Update globals based on args
    MAX_WORKERS = args.workers
    BATCH_SIZE = args.batch_size
    MAX_ROWS_PER_FILE = args.max_rows
    MAX_FILE_SIZE_MB = args.max_size_mb

    if args.max_rows and args.max_size_mb:
        raise ValueError("Either --max-rows or --max-size-mb must be set, not both.")

    assert MAX_ROWS_PER_FILE or MAX_FILE_SIZE_MB, (
        "Either --max-rows or --max-size-mb must be set, or one of their default values."
    )

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        MAX_WORKERS = max(1, int(int(slurm_cpus) * 0.6))
        logger.info(
            f"Detected {slurm_cpus} CPUs in Slurm environment, using (75%) workers = {MAX_WORKERS}"
        )
        if MAX_WORKERS > args.workers:
            logger.warning(
                f"SLURM_CPUS_PER_TASK is set to {slurm_cpus}, but --workers is set to {args.workers}. Using {MAX_WORKERS} workers."
            )
        args.workers = MAX_WORKERS

    print(f"Starting bucketing process with {MAX_WORKERS} workers")
    print(f"Batch size: {BATCH_SIZE} files")
    print(f"Max rows per output file: {MAX_ROWS_PER_FILE}")
    print(f"Max file size: {MAX_FILE_SIZE_MB} MB")

    if args.debug:
        print("Running in debug mode - will process only one batch")

    if args.merge_only:
        # Only perform the merge phase
        print("Performing merge phase only")
        merge_all_temp_files(
            max_rows_per_file=args.max_rows, max_size_mb=args.max_size_mb
        )
    else:
        # Normal processing
        process_files(args)

    print("Bucketing complete. Check the log file for details.")


if __name__ == "__main__":
    main()
