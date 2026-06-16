import os
import argparse

from transformers import LlamaTokenizer, LlamaTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        required=True,
        help="Path to tokenizer.model file",
    )
    return parser.parse_args()


def spm_to_hf(tokenizer_model, save_folder):
    # First convert to slow tokenizer and ultimately save in FastTokenizer format
    slow_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_model)
    slow_tokenizer.save_pretrained(save_folder)
    fast_tokenizer = LlamaTokenizerFast.from_pretrained(save_folder)

    fast_tokenizer.save_pretrained(save_folder)


def main():
    args = parse_args()

    save_folder = os.path.join(os.path.dirname(args.tokenizer_model), "hf")
    os.makedirs(save_folder)
    spm_to_hf(args.tokenizer_model, save_folder)


if __name__ == "__main__":
    main()
