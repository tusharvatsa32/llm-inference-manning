"""Compare deterministic greedy decoding with seeded sampling."""

import argparse

from mini_inference.sampling import SamplingParams

from _common import (
    DEFAULT_MODEL,
    best_device,
    encode_prompt,
    generate_token_ids,
    load_stack,
)


def render(tokenizer, prompt: str, token_ids: list[int]) -> str:
    continuation = tokenizer.decode(token_ids, skip_special_tokens=True)
    return prompt + continuation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Once upon a time, a small robot")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default=best_device()
    )
    args = parser.parse_args()

    tokenizer, _, decoder = load_stack(args.model, args.device)
    prompt_token_ids = encode_prompt(tokenizer, args.prompt)

    print("greedy (temperature=0):")
    for run in range(2):
        token_ids = generate_token_ids(
            decoder,
            prompt_token_ids,
            SamplingParams(temperature=0),
            args.max_new_tokens,
        )
        print(f"  run {run + 1}: {render(tokenizer, args.prompt, token_ids)!r}")

    print("\nsampling (temperature=0.8, top_p=0.9):")
    for seed in (7, 17, 27):
        token_ids = generate_token_ids(
            decoder,
            prompt_token_ids,
            SamplingParams(temperature=0.8, top_p=0.9, seed=seed),
            args.max_new_tokens,
        )
        print(f"  seed {seed}: {render(tokenizer, args.prompt, token_ids)!r}")


if __name__ == "__main__":
    main()
