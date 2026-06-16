import gc
import os
from glob import glob
from multiprocessing import Pool
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_max_workers():
    # Determine max workers based on environment
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if os.environ.get("SLURM_JOB_ID") and slurm_cpus_per_task:
        # Running on SLURM, use 75% of available CPUs
        cpu_count = int(slurm_cpus_per_task)
        print(f"Detected {cpu_count} CPUs on SLURM")
        max_workers = max(1, int(cpu_count * 0.5))
        print(
            f"Running on SLURM with {max_workers} workers ({cpu_count} CPUs detected)"
        )
    else:
        # Not running on SLURM, use fixed number
        max_workers = 16
        print(f"Running locally with {max_workers} workers")
    return max_workers


def get_dataset_folder(file_path: str):
    return file_path.split("/")[-2]


def analyse_file_dataset(file_path: str) -> List[str]:
    table = pq.read_table(file_path)
    # datasets = table["dataset_name"].unique()
    paths = table["original_file_path"].unique()
    datasets = sorted(set([get_dataset_folder(str(el)) for el in paths]))
    del table, paths
    gc.collect()
    return datasets


def analyse_datasets_for_bucket(
    bucket_path: str,
    max_workers: int = get_max_workers(),
) -> Dict[str, List[str]]:
    files = sorted(glob(f"{bucket_path}/*.parquet"))

    with Pool(max_workers, maxtasksperchild=1) as pool:
        datasets_lists = [
            el for el in tqdm(pool.imap(analyse_file_dataset, files), total=len(files))
        ]

    results = {}
    for i, file_path in enumerate(tqdm(files)):
        # for file_path in files:
        file_name = file_path.split("/")[-1]
        # print(file_path, datasets)
        results[file_name] = datasets_lists[i]
    return results


def split_bucket(
    bucket_path: str,
    train_path: str,
    validate_path: str,
    max_workers: int = get_max_workers(),
):
    files = sorted(glob(f"{bucket_path}/*.parquet"))

    args_list = [(file_path, train_path, validate_path) for file_path in files]

    # maxtasksperchild reduces memory consumption
    with Pool(max_workers, maxtasksperchild=1) as pool:
        list(tqdm(pool.imap(split_file_wrap, args_list), total=len(args_list)))


def split_file_wrap(args):
    split_file(*args)


def split_file(file_path: str, train_path: str, validate_path: str):
    file_name = file_path.split("/")[-1]
    file_name_base = file_name.replace(".parquet", "")
    df = pd.read_parquet(
        file_path, engine="fastparquet"
    )  # less memory leak with fastparquet
    stratify_col = "dataset_folder"
    df[stratify_col] = df["original_file_path"].apply(get_dataset_folder)
    # Perform stratified train-test split
    # If there's not enough data in each class
    if df[stratify_col].value_counts().min() < 2:
        counts = df[stratify_col].value_counts()
        single_value_categories = counts[counts == 1].index
        print(
            f"⚠️ non-stratified for {file_name}: insufficient data for stratification for columns: \n{single_value_categories}"
        )
        df_single = df[df[stratify_col].isin(single_value_categories)]
        df_rest = df[~df[stratify_col].isin(single_value_categories)]
        stratify_col_values = df_rest[stratify_col]
        # only stratify the ones without single values
        train_df, validate_df = train_test_split(
            df_rest,
            test_size=0.02,
            random_state=42,
            stratify=stratify_col_values,  # ensures proportional distribution of 'target'
        )
        # the ones with not enough rows, go into training
        train_df = pd.concat([train_df, df_single])
    else:
        train_df, validate_df = train_test_split(
            df,
            test_size=0.02,
            random_state=42,
            stratify=df[stratify_col],  # ensures proportional distribution of 'target'
        )
    print(
        f"    {file_name: >60}: total rows: {len(df):6.0f} train rows: {len(train_df):6.0f} validate rows: {len(validate_df):6.0f}"
    )
    train_df.to_parquet(f"{train_path}/{file_name_base}_train.parquet", index=False)
    validate_df.to_parquet(
        f"{validate_path}/{file_name_base}_validate.parquet", index=False
    )
    del df, train_df, validate_df
    gc.collect()
