"""Compare candidate sets and outputs from common sampling strategies."""

import argparse

import torch

from mini_inference.sampling import Sampler, SamplingParams

from _common import (
    DEFAULT_MODEL,
    best_device,
    encode_prompt,
    generate_token_ids,
    load_stack,
)


STRATEGIES = {
    "temperature": SamplingParams(temperature=0.7, seed=11),
    "top-k": SamplingParams(temperature=0.8, top_k=20, seed=11),
    "top-p": SamplingParams(temperature=0.8, top_p=0.9, seed=11),
    "min-p": SamplingParams(temperature=0.8, min_p=0.08, seed=11),
    "typical": SamplingParams(temperature=0.8, typical_p=0.9, seed=11),
    "combined": SamplingParams(
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        seed=11,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="A fast inference system must")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default=best_device()
    )
    args = parser.parse_args()

    tokenizer, runner, decoder = load_stack(args.model, args.device)
    prompt_token_ids = encode_prompt(tokenizer, args.prompt)
    first_step = runner.prefill(prompt_token_ids)
    sampler = Sampler()

    print(f"vocabulary size: {first_step.logits.numel()}")
    for name, params in STRATEGIES.items():
        prepared = sampler.prepare_logits(first_step.logits, params)
        candidate_count = int(torch.isfinite(prepared).sum().item())
        token_ids = generate_token_ids(
            decoder, prompt_token_ids, params, args.max_new_tokens
        )
        continuation = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"\n{name}: {candidate_count} first-step candidates")
        print(f"  {args.prompt + continuation!r}")


if __name__ == "__main__":
    main()
