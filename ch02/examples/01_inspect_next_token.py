"""Inspect the vocabulary distribution produced by one prompt prefill."""

import argparse

import torch

from _common import DEFAULT_MODEL, best_device, encode_prompt, load_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default=best_device()
    )
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")

    tokenizer, runner, _ = load_stack(args.model, args.device)
    prompt_token_ids = encode_prompt(tokenizer, args.prompt)
    output = runner.prefill(prompt_token_ids)
    probabilities = torch.softmax(output.logits, dim=-1)
    top = min(args.top, probabilities.numel())
    top_probabilities, top_token_ids = torch.topk(probabilities, top)

    print(f"device: {args.device}")
    print(f"prompt tokens: {len(prompt_token_ids)}")
    print(f"logits shape: {tuple(output.logits.shape)}")
    print("\nrank  token_id  probability  decoded token")
    for rank, (token_id, probability) in enumerate(
        zip(top_token_ids.tolist(), top_probabilities.tolist()), start=1
    ):
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        print(f"{rank:>4}  {token_id:>8}  {probability:>11.6f}  {piece!r}")


if __name__ == "__main__":
    main()
