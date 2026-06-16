import argparse
import gc
import glob
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import dotenv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Loads PIPELINE_INPUT, OUTPUT_TAGGING
dotenv.load_dotenv()
config_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.env")
dotenv.load_dotenv(config_env)

PIPELINE_INPUT = os.environ.get("PIPELINE_INPUT")
OUTPUT_TAGGING = os.environ.get("OUTPUT_TAGGING")
METADATA_FILE = os.environ.get("QUALITY_METADATA_FILE")

assert PIPELINE_INPUT, "PIPELINE_INPUT is not set in config.env"
assert OUTPUT_TAGGING, "OUTPUT_TAGGING is not set in config.env"
assert METADATA_FILE, "QUALITY_METADATA_FILE is not set in config.env"

# Path to the YAML metadata file
METADATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    METADATA_FILE,
)

assert os.path.exists(METADATA_FILE), (
    f"Metadata file not found at {METADATA_FILE}. Please check the path, or run the `notebooks/explore-curated-data.ipynb` notebook."
)

if not os.path.exists(OUTPUT_TAGGING):
    os.makedirs(OUTPUT_TAGGING, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(OUTPUT_TAGGING, "processing.log"),
    filemode="a",
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)

TRACKING_FILE = os.path.join(OUTPUT_TAGGING, "processing_status.json")
REGISTRY_FILE = os.path.join(OUTPUT_TAGGING, "document_ids_registry.json")
BATCH_SIZE = 10  # Number of files to process before updating tracking file

# Determine max workers based on environment
if os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK"):
    # Running on SLURM, use 75% of available CPUs
    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK"))  # type: ignore
    logger.info(f"Detected {cpu_count} CPUs on SLURM")
    MAX_WORKERS = max(1, int(cpu_count * 0.5))  # type: ignore
    logger.info(
        f"Running on SLURM with {MAX_WORKERS} workers ({cpu_count} CPUs detected)"
    )
else:
    # Not running on SLURM, use fixed number
    MAX_WORKERS = 16
    logger.info(f"Running locally with {MAX_WORKERS} workers")

# Load dataset metadata
with open(METADATA_FILE, "r") as f:
    DATASET_METADATA = yaml.safe_load(f)
    logger.info(DATASET_METADATA)


def get_datasets():
    return [d for d in os.listdir(PIPELINE_INPUT)]


def get_dataset_quality(dataset_name: str) -> str:
    """Get quality label for a dataset from metadata YAML"""
    if dataset_name in DATASET_METADATA:
        return DATASET_METADATA[dataset_name]["quality"]
    else:
        logger.error(
            f"No quality metadata found for {dataset_name}, defaulting to 'medium'"
        )
        raise ValueError(
            f"Dataset {dataset_name} not found in metadata. Please check the meta data to be complete."
        )


def get_is_new_epoch_2(dataset_name: str) -> bool:
    """Get is_new_epoch_2 value for a dataset from metadata YAML"""
    if dataset_name in DATASET_METADATA:
        return DATASET_METADATA[dataset_name].get("is_new_epoch_2", False)
    else:
        logger.error(
            f"No is_new_epoch_2 metadata found for {dataset_name}, defaulting to False"
        )
        raise ValueError(
            f"Dataset {dataset_name} not found in metadata. Please check the meta data to be complete."
        )


def get_dataset_is_code(dataset_name):
    """Get is_code value for a dataset from metadata YAML"""
    if dataset_name in DATASET_METADATA:
        return DATASET_METADATA[dataset_name]["is_code"]
    else:
        logger.error(
            f"No is_code metadata found for {dataset_name}. Please check the meta data to be complete."
        )
        raise ValueError(
            f"Dataset {dataset_name} not found in metadata. Please check the meta data to be complete."
        )


def get_parquet_files(dataset_path):
    """Get all parquet files in the subdirectories of the given path."""
    parquet_files = []

    # Only look at immediate subdirectories (one level nesting)
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if os.path.isdir(subdir_path):
            files = glob.glob(os.path.join(subdir_path, "*.parquet"))
            parquet_files.extend(files)

    return parquet_files


def get_file_info(filepath):
    """Extract subdir and filename from a full path"""
    subdir = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    return {"subdir": subdir, "filename": filename}


def get_full_path(file_info, base_dir=PIPELINE_INPUT):
    """Reconstruct full path from file info"""
    return os.path.join(base_dir, file_info["subdir"], file_info["filename"])


def load_tracking_data():
    """Load tracking data from the tracking file, or create it if it doesn't exist."""
    if not os.path.exists(OUTPUT_TAGGING):  # type: ignore
        os.makedirs(OUTPUT_TAGGING)  # type: ignore

    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "processed_files": [],
            "failed_files": [],
            "in_progress": [],
            "processed_count": 0,
        }


def save_tracking_data(tracking_data):
    """Save tracking data to the tracking file."""
    with open(TRACKING_FILE, "w") as f:
        json.dump(tracking_data, f, indent=4)


def create_default_registry():
    """Create a default registry structure."""
    return {
        "files": {},  # {dataset_filename: {start_id: int, end_id: int, count: int}}
        "total_ids": 0,
        "last_updated": None,
    }


def save_id_registry(registry_data):
    """Save document ID registry to file with proper file locking."""
    import datetime

    registry_data["last_updated"] = datetime.datetime.now().isoformat()

    # Save document ID registry to file
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry_data, f, indent=4)


def generate_document_ids(dataset_name, filename, n_rows):
    """Generate unique document IDs for a file."""
    filename_base = os.path.splitext(filename)[0]
    ids = []
    for i in range(n_rows):
        doc_id = f"{dataset_name}_{filename_base}_{i:08d}"
        ids.append(doc_id)
    return ids


def process_parquet_file(
    file_path, quality_label, is_new_label, tracking_update_queue=None
):
    """Process a single parquet file with the required transformations using PyArrow."""
    file_info = get_file_info(file_path)
    dataset_name = file_info["subdir"]
    filename = file_info["filename"]

    try:
        # Define the schema to read only required columns
        schema = pa.schema(
            [("language", pa.string()), ("language_score", pa.float64())]
        )

        # Read the parquet file with PyArrow
        table = pq.read_table(file_path, schema=schema)

        # Get necessary column data
        language_col = table.column("language").to_numpy()

        # Create new columns
        n_rows = len(language_col)

        # Generate document IDs
        doc_ids = generate_document_ids(dataset_name, filename, n_rows)

        # Get is_code from YAML metadata instead of calculating from language_score
        is_code_value = get_dataset_is_code(dataset_name)
        is_code = np.full(n_rows, is_code_value)
        is_new = np.full(n_rows, is_new_label)

        # Create dictionary for new columns
        new_cols = {
            "doc_id": pa.array(doc_ids),
            "dataset_quality": pa.array([quality_label] * n_rows),
            "is_code": pa.array(is_code),
            "is_new_epoch_2": pa.array(is_new),
            "original_file_path": pa.array([file_path] * n_rows),
        }

        # Read the original table again with all columns
        full_table = pq.read_table(file_path)

        # Add new columns to the original table
        for col_name, col_data in new_cols.items():
            full_table = full_table.append_column(col_name, col_data)

        # Save the transformed file
        output_subdir = os.path.join(OUTPUT_TAGGING, file_info["subdir"])  # type: ignore
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)

        output_file = os.path.join(output_subdir, file_info["filename"])
        pq.write_table(full_table, output_file, compression="snappy")

        # Free memory
        del table, full_table, new_cols
        gc.collect()

        if tracking_update_queue is not None:
            tracking_update_queue.append({"status": "success", "file_info": file_info})

        logger.info(
            f"Successfully processed: {file_info['subdir']}/{file_info['filename']}"
        )
        return True

    except Exception as e:
        if tracking_update_queue is not None:
            tracking_update_queue.append(
                {
                    "status": "failed",
                    "file_info": file_info,
                    "error": str(e),
                }
            )

        logger.error(
            f"Failed to process {file_info['subdir']}/{file_info['filename']}: {str(e)}, {str(e.__traceback__)}"
        )
        return False


def batch_update_tracking(tracking_update_queue, tracking_data):
    """Update tracking data with a batch of updates"""
    for update in tracking_update_queue:
        if update["status"] == "success":
            tracking_data["processed_files"].append(update["file_info"])
            tracking_data["processed_count"] += 1
        else:
            tracking_data["failed_files"].append(
                {
                    "file_info": update["file_info"],
                    "error": update["error"],
                }
            )

    # Clear in-progress for completed files
    for update in tracking_update_queue:
        file_info = update["file_info"]
        if file_info in tracking_data["in_progress"]:
            tracking_data["in_progress"].remove(file_info)

    # Save updated tracking data
    save_tracking_data(tracking_data)

    # Clear the queue
    tracking_update_queue.clear()


def worker_process_file(file_path, quality_label, is_new_label):
    """Worker function for parallel processing"""
    try:
        return process_parquet_file(file_path, quality_label, is_new_label)
    except Exception as e:
        file_info = get_file_info(file_path)
        logger.error(
            f"Worker error processing {file_info['subdir']}/{file_info['filename']}: {str(e)}"
        )
        return False


def process_datasets(debug_mode=False):
    """Process all parquet files in the subdirectories of PIPELINE_INPUT."""
    tracking_data = load_tracking_data()
    parquet_files = get_parquet_files(PIPELINE_INPUT)

    # Convert processed_files to the new format if needed (backward compatibility)
    if tracking_data["processed_files"] and isinstance(
        tracking_data["processed_files"][0], str
    ):
        logger.info("Converting tracking data to new format...")
        tracking_data["processed_files"] = [
            get_file_info(f) for f in tracking_data["processed_files"]
        ]
        tracking_data["failed_files"] = [
            {
                "file_info": get_file_info(f["file"]),
                "error": f["error"],
            }
            for f in tracking_data["failed_files"]
        ]
        tracking_data["processed_count"] = len(tracking_data["processed_files"])
        if tracking_data.get("in_progress") and isinstance(
            tracking_data["in_progress"], str
        ):
            tracking_data["in_progress"] = [get_file_info(tracking_data["in_progress"])]
        else:
            tracking_data["in_progress"] = []

    # Get processed file infos
    processed_file_infos = tracking_data["processed_files"]

    # Convert the list of parquet_files to file_infos
    all_file_infos = [get_file_info(f) for f in parquet_files]

    # Filter out already processed files
    def is_file_processed(file_info):
        return any(
            p["subdir"] == file_info["subdir"]
            and p["filename"] == file_info["filename"]
            for p in processed_file_infos
        )

    remaining_file_infos = [f for f in all_file_infos if not is_file_processed(f)]

    logger.info(f"Found {len(all_file_infos)} total parquet files")
    logger.info(f"Already processed: {len(processed_file_infos)}")
    logger.info(f"Remaining to process: {len(remaining_file_infos)}")

    # Convert file_infos back to full paths for processing
    remaining_files = [get_full_path(file_info) for file_info in remaining_file_infos]

    # Process in debug mode (only one file) or full mode
    if debug_mode:
        if remaining_files:
            file_to_process = remaining_files[0]
            file_info = get_file_info(file_to_process)
            quality_label = get_dataset_quality(file_info["subdir"])
            is_new_label = get_is_new_epoch_2(file_info["subdir"])
            logger.info(
                f"DEBUG MODE: Processing only {file_info['subdir']}/{file_info['filename']} with quality '{quality_label}'"
            )

            # Add to in_progress
            tracking_data["in_progress"].append(file_info)
            save_tracking_data(tracking_data)

            # Process the file
            tracking_update_queue = []
            process_parquet_file(
                file_to_process, quality_label, is_new_label, tracking_update_queue
            )
            batch_update_tracking(tracking_update_queue, tracking_data)
        else:
            logger.info("No files to process in debug mode")
    else:
        # Group files by subdirectory to use the same quality label
        files_by_subdir = {}
        file_info_by_path = {get_full_path(info): info for info in remaining_file_infos}

        for file_path in remaining_files:
            file_info = file_info_by_path[file_path]
            subdir = file_info["subdir"]
            if subdir not in files_by_subdir:
                files_by_subdir[subdir] = []
            files_by_subdir[subdir].append(file_path)

        # Process each subdirectory with its quality label from YAML
        for subdir, files in files_by_subdir.items():
            # Read dataset-level labels from metadata YAML.
            quality_label = get_dataset_quality(subdir)
            is_new_label = get_is_new_epoch_2(subdir)
            logger.info(
                f"Processing {len(files)} files in {subdir} with quality '{quality_label}' and is_new_epoch_2={is_new_label}"
            )

            # Mark files as in-progress
            for file_path in files:
                tracking_data["in_progress"].append(file_info_by_path[file_path])
            save_tracking_data(tracking_data)

            # Batch processing with progress bar
            tracking_update_queue = []

            if len(files) > 1 and not debug_mode:
                # Parallel processing
                process_func = partial(
                    worker_process_file,
                    is_new_label=is_new_label,
                    quality_label=quality_label,
                )

                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(process_func, file_path): file_path
                        for file_path in files
                    }

                    # Process with progress logging appropriate for Slurm
                logger.info(
                    f"Starting parallel processing of {len(files)} files in {subdir}"
                )
                processed = 0

                for i, future in enumerate(as_completed(futures)):
                    file_path = futures[future]
                    file_info = file_info_by_path[file_path]
                    try:
                        success = future.result()
                        if success:
                            tracking_update_queue.append(
                                {"status": "success", "file_info": file_info}
                            )
                        else:
                            tracking_update_queue.append(
                                {
                                    "status": "failed",
                                    "file_info": file_info,
                                    "error": "Worker reported failure",
                                }
                            )
                    except Exception as e:
                        tracking_update_queue.append(
                            {
                                "status": "failed",
                                "file_info": file_info,
                                "error": str(e),
                            }
                        )

                    processed += 1
                    if processed % 10 == 0:
                        logger.info(
                            f"Progress: {processed}/{len(files)} files processed in {subdir}"
                        )

                        # Update tracking data in batches
                        if (i + 1) % BATCH_SIZE == 0 or i == len(files) - 1:
                            batch_update_tracking(tracking_update_queue, tracking_data)
            else:
                # Sequential processing for small batches with Slurm-friendly logging
                logger.info(
                    f"Starting sequential processing of {len(files)} files in {subdir}"
                )
                for i, file_path in enumerate(files):
                    file_info = file_info_by_path[file_path]
                    process_parquet_file(
                        file_path, quality_label, is_new_label, tracking_update_queue
                    )

                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        logger.info(
                            f"Progress: {i + 1}/{len(files)} files processed in {subdir}"
                        )

                    # Update tracking data in batches
                    if (i + 1) % BATCH_SIZE == 0 or i == len(files) - 1:
                        batch_update_tracking(tracking_update_queue, tracking_data)

    # Final report
    tracking_data = load_tracking_data()
    logger.info(
        f"Processing complete. Processed {tracking_data['processed_count']} files successfully."
    )
    logger.info(f"Failed files: {len(tracking_data['failed_files'])}")

    # Print failed files details
    for failed in tracking_data["failed_files"]:
        file_info = failed["file_info"]
        logger.info(
            f"Failed: {file_info['subdir']}/{file_info['filename']} - Error: {failed['error']}"
        )


def build_up_registry():
    logger.info("Building up document ID registry from existing files...")
    registry = create_default_registry()
    for dataset_name in os.listdir(OUTPUT_TAGGING):
        subdir_path = os.path.join(OUTPUT_TAGGING, dataset_name)  # type: ignore
        if os.path.isdir(subdir_path):
            files = glob.glob(os.path.join(subdir_path, "*.parquet"))
            for file in files:
                file_info = get_file_info(file)
                # Read only the doc_id column to save memory
                table = pq.read_table(file, columns=["doc_id"])
                doc_ids_array = table["doc_id"].to_numpy()
                start_id = doc_ids_array[0]
                end_id = doc_ids_array[-1]
                assert start_id and end_id, "Start and end IDs must not be None"
                count = len(doc_ids_array)
                registry["files"][f"{dataset_name}_{file_info['filename']}"] = {
                    "dataset": dataset_name,
                    "filename": file_info["filename"],
                    "start_id": start_id,
                    "end_id": end_id,
                    "count": count,
                }
                registry["total_ids"] += count
                # Free memory explicitly
                del table, doc_ids_array
                gc.collect()
    save_id_registry(registry)
    logger.info(
        f"Document ID registry built with {len(registry['files'])} files and {registry['total_ids']} total IDs."
    )


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Process parquet files with transformations."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode (process only one file)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of files to process before updating tracking (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--dataset-metadata",
        type=str,
        default=METADATA_FILE,
        help=f"Path to the dataset metadata file (default: {METADATA_FILE})",
    )
    args = parser.parse_args()

    # Update global settings from args
    MAX_WORKERS = args.workers
    BATCH_SIZE = args.batch_size
    METADATA_FILE = args.dataset_metadata

    datasets = get_datasets()

    print("Curated datasets found on the system:")
    for i, d in enumerate(datasets):
        print(f"{i + 1}. {d}")

    # Use the command line flag for debug mode
    debug_mode = args.debug

    if debug_mode:
        print("Running in debug mode - will process only one file")
    else:
        print(
            f"Running in full mode with {MAX_WORKERS} workers - will process all files"
        )

    # Process the datasets
    process_datasets(debug_mode)
    build_up_registry()

    print("Processing complete. Check the log file for details.")
