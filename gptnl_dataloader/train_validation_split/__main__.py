import os
from pathlib import Path
from typing import List

from typer import Typer

from gptnl_dataloader.train_validation_split.functions import (
    analyse_datasets_for_bucket,
    split_bucket,
)
from gptnl_dataloader.utils import (
    ALL_BUCKETS_DEFAULT,
    BUCKETS_OUTPUTS_DEFAULT_NAME,
    BUCKETS_TRAIN_SPLIT_DEFAULT_NAME,
    BUCKETS_VALIDATE_SPLIT_DEFAULT_NAME,
    DATASET_DEFAULT_PATH,
    Tee,
)

app = Typer(pretty_exceptions_enable=False)

my_path = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_PATH = str(my_path / "outputs")


@app.command("i")
def inspect_buckets(
    output_dir: str = DEFAULT_OUTPUT_PATH,
    buckets_path: str = str(Path(DATASET_DEFAULT_PATH) / BUCKETS_OUTPUTS_DEFAULT_NAME),
    bucket_names: List[str] = ALL_BUCKETS_DEFAULT,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logname = f"{output_dir}/output_inspect_buckets.text"

    _logger = Tee(logname, mode="w")
    for bucket in bucket_names:
        print(f"BUCKET: {bucket}")
        bucket_path = str(Path(buckets_path) / bucket)
        bucket_datasets_by_file_name = analyse_datasets_for_bucket(
            bucket_path=bucket_path
        )
        for k, v in bucket_datasets_by_file_name.items():
            print(f"\t{k}: {v}")


@app.command("r")
def run_split(
    output_dir: str = DEFAULT_OUTPUT_PATH,
    buckets_path: str = str(Path(DATASET_DEFAULT_PATH) / BUCKETS_OUTPUTS_DEFAULT_NAME),
    buckets_train_path: str = str(
        Path(DATASET_DEFAULT_PATH) / BUCKETS_TRAIN_SPLIT_DEFAULT_NAME
    ),
    buckets_validate_path: str = str(
        Path(DATASET_DEFAULT_PATH) / BUCKETS_VALIDATE_SPLIT_DEFAULT_NAME
    ),
    bucket_names: List[str] = ALL_BUCKETS_DEFAULT,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logname = f"{output_dir}/output_split_buckets.text"

    _logger = Tee(logname, mode="w")
    print(f"buckets_path: {buckets_path}")
    print(f"buckets_train_path: {buckets_train_path}")
    print(f"buckets_validate_path: {buckets_validate_path}")
    print(f"bucket_names: {bucket_names}")
    for bucket in bucket_names:
        print(f"BUCKET: {bucket}")
        bucket_path = str(Path(buckets_path) / bucket)
        bucket_train_path = str(Path(buckets_train_path) / bucket)
        bucket_validate_path = str(Path(buckets_validate_path) / bucket)
        if not os.path.exists(bucket_train_path):
            os.makedirs(bucket_train_path)
        if not os.path.exists(bucket_validate_path):
            os.makedirs(bucket_validate_path)
        split_bucket(
            bucket_path=bucket_path,
            train_path=bucket_train_path,
            validate_path=bucket_validate_path,
        )


if __name__ == "__main__":
    app()
