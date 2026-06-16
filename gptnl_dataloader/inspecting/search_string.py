import json
import os
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd
from tqdm import tqdm
from typer import Typer

from gptnl_dataloader.train_validation_split.functions import get_dataset_folder
from gptnl_dataloader.utils import ALL_BUCKETS_DEFAULT, DATASET_DEFAULT_PATH, Tee


def find_matches(
    glob_path: str | Path,
    substring: str,
    limit: int | None = 3,
    context_len: int = 20,
    post_bucketing: bool = True,
):
    # original_file_path only after bucketing
    query = f"""
    SELECT text, title, filename{", original_file_path" if post_bucketing else ""}
    FROM read_parquet('{glob_path}', filename = true)
    WHERE '{substring}' in text
    """
    # query = f"""
    # SELECT text, title, filename{', original_file_path' if post_bucketing else ''}
    # FROM read_parquet('{glob_path}', filename = true)
    # WHERE REGEXP_MATCHES(text, '{regex}')
    # """
    if limit:
        query += f"LIMIT {limit}"
        batch_size = min([100, limit])
    else:
        batch_size = 100
    pbar = tqdm()
    handle = duckdb.query(query=query)
    results_nice = []
    while batch := handle.fetchmany(batch_size):
        for row in batch:
            pbar.update(1)
            if post_bucketing:
                text, title, filename, original_file_path = row
                bucket = get_dataset_folder(filename)
                dataset_folder = get_dataset_folder(original_file_path)
            else:
                text, title, filename = row
                original_file_path = None
                bucket = None
                dataset_folder = get_dataset_folder(filename)
            context, index = find_substring_context(
                text, substring=substring, context_len=context_len
            )
            results_nice.append(
                {
                    "match_context": context,
                    "match_index": index,
                    "title": title,
                    "filename": filename,
                    "original_file_path": original_file_path,
                    "bucket": bucket,
                    "dataset_folder": dataset_folder,
                }
            )
    pbar.close()
    print(pbar)
    return results_nice


def find_substring_context(text: str, substring: str, context_len: int = 20):
    index = text.find(substring)
    start = max([index - context_len, 0])
    end = min([index + context_len, len(text)])
    return text[start:end], index


def aggregate_matches(df: pd.DataFrame):
    dataset_counts = df["dataset_folder"].value_counts()
    return dataset_counts


substring = "   \n"

default_paths_to_search = [
    Path(DATASET_DEFAULT_PATH) / "bucketing_output" / bucket / "*.parquet"
    for bucket in ALL_BUCKETS_DEFAULT
    if bucket not in ["code"]  # don't search in code
]

app = Typer(pretty_exceptions_enable=False)

my_path = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_PATH = str(my_path / "outputs")


@app.command()
def main(
    substring: str = substring,
    paths_to_search: List[Path] = default_paths_to_search,
    post_bucketing: bool = True,
    output_dir: str = DEFAULT_OUTPUT_PATH,
    limit: Optional[int] = None,
    context_len: int = 40,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logname = f"{output_dir}/output_search_string.text"

    _logger = Tee(logname, mode="w")
    for path in tqdm(paths_to_search, desc="paths"):
        print(f"Searching {repr(substring)} in {path}")
        matches = find_matches(
            glob_path=path,
            substring=substring,
            post_bucketing=post_bucketing,
            limit=limit,
            context_len=context_len,
        )
        matches_df = pd.DataFrame(matches)
        bucket = path.parent.name
        with open(f"{output_dir}/{bucket}.json", "w") as f:
            json.dump(matches, f, indent=2)
        aggregated = aggregate_matches(matches_df)
        print(aggregated)


if __name__ == "__main__":
    app()
