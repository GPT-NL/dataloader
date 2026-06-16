"""
Script for preparing data for fine-tuning an OLMo model.
"""

import logging
import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.eos_token_id = args.eos
    file_pattern = args.file_pattern  # e.g. "cultura_x-train-0000*-of-00521.arrow"
    log.info(
        f"Number of files to be tokenized: {len(list(Path(args.data_path).glob(file_pattern)))}"
    )
    file_extension = "." + file_pattern.split(".")[-1]
    for data_filepath in Path(args.data_path).glob(file_pattern):
        data_filename = os.path.basename(data_filepath)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = (
            str(os.path.join(output_dir, data_filename.split(file_extension)[0]))
            + "_tokenized.npy"
        )
        if os.path.exists(output_path):
            log.info(f"File {data_filename} is already tokenized, skipping...")
            continue
        dataset = ds.load_dataset(
            args.data_path,
            cache_dir=args.cache_dir,
            data_files=data_filename,
        )
        log.info(f"Tokenizing dataset {data_filename}...")
        # we preprocess for pretraining not for instruction
        # dataset["train"] = dataset["train"].select(range(12000))
        dataset = dataset.map(
            partial(
                preprocess_instruct
                if args.data_type == "instruct"
                else preprocess_pretraining,
                tokenizer=tokenizer,
                max_seq_len=args.seq_len,
            ),
            batched=False,
            remove_columns=args.remove_columns,
            num_proc=args.num_proc,  # type: ignore
        )
        if args.data_type == "instruct":
            log.info("Filtering dataset...")
            n = len(dataset)  # type: ignore
            dataset = dataset.filter(filter, batched=False, num_proc=args.num_proc)  # type: ignore
            log.info(f"Filtered out {n - len(dataset):,d} examples")

        log.info("Counting tokens...")
        total_tokens = 0
        for ex in track(dataset["train"]):
            total_tokens += len(ex["input_ids"])  # type: ignore
        log.info(f"Total tokens: {total_tokens:,d}")

        log.info(f"Saving results to {output_path}...")
        input_ids_file = np.memmap(
            output_path, dtype=np.uint32, mode="w+", shape=(total_tokens,)
        )

        if args.data_type == "instruct":
            label_mask_file = np.memmap(
                str(output_dir / "label_mask.npy"),
                dtype=np.bool_,
                mode="w+",
                shape=(total_tokens,),
            )
        offset = 0
        for ex in track(dataset["train"]):
            ex_len = len(ex["input_ids"])
            input_ids_file[offset : offset + ex_len] = ex["input_ids"]
            if args.data_type == "instruct":
                label_mask_file[offset : offset + ex_len] = ex["label_mask"]  # type: ignore
            offset += ex_len

        input_ids_file.flush()
        if args.data_type == "instruct":
            label_mask_file.flush()

        log.info("Done!")


def filter(example):
    return example["n_labels"] > 0


def encode(
    tokenizer, text, add_special_tokens=False, add_end_of_seq=False, use_tiktoken=False
):
    if use_tiktoken:
        try:
            text = text + tokenizer.eot_token + "\n" if add_end_of_seq else text
            return tokenizer.encode(text)
        except ValueError:
            text = text.replace("<|endoftext|>", "")
            text = text + tokenizer.eot_token + "\n" if add_end_of_seq else text
            return tokenizer.encode(text)
    else:
        text = text + tokenizer.eos_token + "\n" if add_end_of_seq else text
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)


def preprocess_pretraining(example, tokenizer: AutoTokenizer, max_seq_len: int) -> dict:
    tokens = encode(tokenizer, example["text"].strip())
    tokens.append(tokenizer.eos_token_id)
    # tokens = tokens[:max_seq_len]
    # For Llama tokenizer manually add eos_token
    return {"input_ids": tokens}


def preprocess_instruct(example, tokenizer: AutoTokenizer, max_seq_len: int) -> dict:
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    for msg in example["messages"]:
        role_tokens = encode(tokenizer, f"<|{msg['role']}|>\n")
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = encode(
                tokenizer, msg["content"].strip(), add_end_of_seq=True
            )
            label_mask += [True] * len(content_tokens)
            # mask out the last '\n'
            assert content_tokens[-2] == tokenizer.eos_token_id
            label_mask[-1] = False
        else:
            content_tokens = encode(tokenizer, msg["content"].strip() + "\n")
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Tokenize GPT-NL dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        help="""Directory where the data shard files are.""",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="""Directory to save the results to.""",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        help="""File pattern to match on the shard file names of the dataset that needs to be loaded.""",
    )
    parser.add_argument(
        "--cache_dir", type=str, help="""Cache directory to save dataset cache"""
    )
    parser.add_argument(
        "--use_tiktoken",
        type=bool,
        help="""If true, use tiktoken config to load tokenizer. Else, load from HuggingFace .json config.""",
    )

    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default=Path(__file__).parent
        / "tokenizers"
        / "allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
    )
    parser.add_argument(
        "--remove_columns",
        nargs="+",
        help="the names of the columns that need to be dropped",
        required=True,
    )

    parser.add_argument(
        "-s", "--seq-len", type=int, help="""Max sequence length.""", default=4096
    )
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=50279)
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=1)
    parser.add_argument(
        "-j", "--num-proc", type=int, help="""Number of workers.""", default=8
    )
    parser.add_argument(
        "--data_type",
        default="pretraining",
        type=str,
        help="Which type of data: instruction-based or pretraining.",
    )

    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
