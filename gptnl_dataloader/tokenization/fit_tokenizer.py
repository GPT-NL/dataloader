import argparse
import json
import os
import shutil
import yaml

import sentencepiece as spm
from spm_to_hf_tokenizer import spm_to_hf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the to be trained tokenizer",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Config file with training options (YAML)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_control_symbols_file(config):
    if not os.path.exists(config["user_defined_symbols_file"]):
        config.pop("control_symbols_file", None)
        return config
    with open(config["control_symbols_file"]) as f:
        lines = f.readlines()
    config["control_symbols"] += lines
    config.pop("control_symbols_file", None)
    return config


def add_user_symbols_file(config):
    if not os.path.exists(config["user_defined_symbols_file"]):
        config.pop("user_defined_symbols_file", None)
        return config
    with open(config["user_defined_symbols_file"]) as f:
        lines = f.readlines()
    config["user_defined_symbols"] += lines
    config.pop("user_defined_symbols_file", None)
    return config


def add_whitespace_tokens(config):
    # Do not add "▁" as separate token as SentencePiece uses this internally for whitespace, so only "▁▁" and more
    unicode_space_tokens = [
        "▁" * i for i in range(2, config["unicode_space_tokens"] + 2)
    ]
    tabulation_tokens = ["\t" * i for i in range(1, config["tabulation_tokens"] + 1)]
    newline_tokens = ["\n" * i for i in range(1, config["newline_tokens"] + 1)]
    config["user_defined_symbols"] += (
        unicode_space_tokens + tabulation_tokens + newline_tokens
    )
    config.pop("unicode_space_tokens", None)
    config.pop("tabulation_tokens", None)
    config.pop("newline_tokens", None)
    return config


def add_digits(config):
    if config["add_digits"]:
        config["user_defined_symbols"] += [str(i) for i in range(10)]
        config.pop("add_digits", None)
    return config


def add_reserved_tokens(config):
    reserved_tokens = [
        config["reserved_tokens_template"].replace("{}", str(i))
        for i in range(1, config["reserved_tokens"] + 1)
    ]
    config["user_defined_symbols"] += reserved_tokens
    config.pop("reserved_tokens_template", None)
    config.pop("reserved_tokens", None)
    return config


def train_sentencepiece(args):
    config = load_config(args.config_path)

    if config["num_threads"] == -1:
        config["num_threads"] = os.cpu_count() // 2

    os.makedirs(args.output_path, exist_ok=True)

    spm_config = config
    spm_config["model_prefix"] = os.path.join(args.output_path, config["model_prefix"])
    spm_config = add_whitespace_tokens(spm_config)
    spm_config = add_reserved_tokens(spm_config)
    spm_config = add_control_symbols_file(spm_config)
    spm_config = add_user_symbols_file(spm_config)
    # Skip digits as 0-9 are already represented in the bytes
    # spm_config = add_digits(spm_config)
    print(spm_config)
    spm.SentencePieceTrainer.train(**spm_config)
    print(
        f"Done training. Saving Sentencepiece tokenizer in {spm_config['model_prefix']}"
    )
    shutil.copy(args.config_path, os.path.join(args.output_path, "config.yml"))
    tokenizer = spm.SentencePieceProcessor(model_file=f"{config['model_prefix']}.model")
    vocabulary = {
        id_: tokenizer.id_to_piece(id_) for id_ in range(tokenizer.vocab_size())
    }
    with open(os.path.join(args.output_path, "vocab.json"), "w") as v_f:
        json.dump(vocabulary, v_f, indent=4)

    tokenizer_model = f"{spm_config['model_prefix']}.model"
    save_folder = os.path.join(args.output_path, "hf-test")
    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving HuggingFace tokenizer to {save_folder}")
    spm_to_hf(tokenizer_model, save_folder)


def main(args):
    train_sentencepiece(args)


if __name__ == "__main__":
    main(parse_args())
