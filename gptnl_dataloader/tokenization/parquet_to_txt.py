import glob
import os
import random
import concurrent.futures
import pandas as pd
import argparse

random.seed(47)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to the base directory containing bucketed files.",
    )
    parser.add_argument(
        "--output_txt", type=str, required=True, help="Path to the output text file."
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="text",
        help="Name of the column containing the text data.",
    )

    return parser.parse_args()


def clean_text(text):
    if not isinstance(text, str):
        return None
    # Remove null bytes and sanitize
    return text.replace("\x00", "").strip().replace("\n", " ")


def read_and_clean_file(file_path, text_column):
    try:
        df = pd.read_parquet(file_path, columns=[text_column])
        df[text_column] = df[text_column].dropna().astype(str)
        lines = df[text_column].map(clean_text)
        return [line for line in lines if line]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def sample_files_from_folder(folder_path, sample_ratio, text_column):
    all_files = glob.glob(f"{folder_path}/*.parquet")
    num_files_to_sample = max(1, int(len(all_files) * sample_ratio))
    sampled_files = random.sample(all_files, num_files_to_sample)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(
            executor.map(lambda f: read_and_clean_file(f, text_column), sampled_files)
        )

    all_sampled_lines = [line for file_lines in results for line in file_lines]
    return all_sampled_lines, sampled_files


def sample_and_convert_folders(base_dir, output_path, text_column):
    quality_ratios = {
        "code": 0.06,
        "dutch_high_quality": 0.75,
        "dutch_low_medium_quality": 0.25,
        "english_high_quality": 0.24975,
        "english_low_medium_quality": 0.009,
        "other_languages": 0.0375,
    }

    print(f"Quality ratios: {quality_ratios}")

    all_sampled_lines = []
    all_sampled_files = []
    for folder_path, sample_ratio in quality_ratios.items():
        folder_samples, sampled_files = sample_files_from_folder(
            os.path.join(base_dir, folder_path), sample_ratio, text_column
        )
        all_sampled_lines.extend(folder_samples)
        all_sampled_files.extend(sampled_files)

    # Write to single txt file and file txt file
    print(f"Outputting to {output_path}")
    with open(output_path, "w", encoding="utf-8") as out_file:
        for line in all_sampled_lines:
            out_file.write(line + "\n")

    print(f"Writing files to {output_path}.files")
    with open(f"{output_path}.files", "w", encoding="utf-8") as out_file:
        for line in all_sampled_files:
            out_file.write(line + "\n")


def main(args):
    sample_and_convert_folders(args.base_dir, args.output_txt, args.column_name)


if __name__ == "__main__":
    main(parse_args())
