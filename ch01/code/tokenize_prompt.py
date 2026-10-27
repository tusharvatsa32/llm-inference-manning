"""Show how a prompt becomes token IDs.

Reproduces section 1.2.1: the Llama 3 BPE tokenizer turns
"The capital of France is" into [791, 5765, 315, 9822, 374]. Pass your own
text to see how words map to tokens; code, numbers, rare words, and other
languages usually need more tokens than plain English.

The tokenizer is downloaded from the Hugging Face Hub the first time you run
the script. The default repository is the MLX build used in this chapter,
which does not require accepting the Llama license.

Examples:
    python tokenize_prompt.py
    python tokenize_prompt.py "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"
    python tokenize_prompt.py --file my_prompt.txt
"""

import argparse

from tokenizers import Tokenizer

DEFAULT_TOKENIZER = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
DEFAULT_TEXT = "The capital of France is"


def main(args):
    if args.tokenizer.endswith(".json"):
        tokenizer = Tokenizer.from_file(args.tokenizer)
    else:
        tokenizer = Tokenizer.from_pretrained(args.tokenizer)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    # Skip special tokens such as <|begin_of_text|> so the IDs match the
    # prompt text alone.
    encoding = tokenizer.encode(text, add_special_tokens=False)

    if len(encoding.ids) <= args.max_rows:
        for token_id in encoding.ids:
            piece = tokenizer.decode([token_id])
            print(f"{piece!r:<20} -> {token_id}")
        print()

    words = len(text.split())
    tokens = len(encoding.ids)
    print(f"Token IDs: {encoding.ids if tokens <= args.max_rows else '...'}")
    print(f"Words:     {words:,}")
    print(f"Tokens:    {tokens:,}")
    if words:
        print(f"Tokens per word: {tokens / words:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT,
                        help=f"text to tokenize (default: {DEFAULT_TEXT!r})")
    parser.add_argument("--file", help="read the text from a file instead")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="Hugging Face repo ID or path to a tokenizer.json")
    parser.add_argument("--max-rows", type=int, default=50,
                        help="print each token when there are at most this many")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
