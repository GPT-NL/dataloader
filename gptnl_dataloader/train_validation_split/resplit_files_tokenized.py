# - Problem: `GPTNL_1_TOKENIZED/buckets_split_train/` and `GPTNL_1_TOKENIZED/buckets_split_validate/` are flat directories
# - Solution: run `python -m train_validation_split.resplit_files_tokenized`
# - Outputs: `GPTNL_1_TOKENIZED/splitted_buckets_train` and `GPTNL_1_TOKENIZED/splitted_buckets_validate`

import os
import re
from glob import glob
from pathlib import Path

from tqdm import tqdm
from typer import Typer

from gptnl_dataloader.utils import TOKENIZED_DATASET_ROOT_PATH

app = Typer(pretty_exceptions_enable=False)

my_path = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_PATH = str(my_path / "outputs")

INPUT_DIRS = [
    "buckets_split_train",  # needed for file-based sampling
    "buckets_split_validate",  # needed for evaluation metric separate
]

OUTPUT_DIRS = [
    "splitted_buckets_train",
    "splitted_buckets_validate",
]


def split_files_in_directories(input_path: Path, output_path: Path):
    files_paths = sorted(glob(str(input_path / "*.npy")))
    for file_path in tqdm(files_paths):
        copy_file_in_subdirectory(file_path=file_path, output_path=output_path)


def copy_file_wrap(args):
    copy_file_in_subdirectory(*args)


def copy_file_in_subdirectory(file_path: str, output_path: Path):
    file_name = file_path.split("/")[-1]
    pattern = r"^([a-zA-Z_]+?)_\d"
    match = re.match(pattern, file_name)
    if not match:
        raise ValueError(file_path)
    prefix = match.group(1)
    output_folder = output_path / prefix
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = output_folder / file_name
    os.symlink(file_path, output_file_path)


@app.command()
def main():
    for input_dir, output_dir in zip(INPUT_DIRS, OUTPUT_DIRS):
        input_path = Path(TOKENIZED_DATASET_ROOT_PATH) / input_dir
        output_path = Path(TOKENIZED_DATASET_ROOT_PATH) / output_dir
        split_files_in_directories(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    app()
