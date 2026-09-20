"""A compact beam search with a separate pool for finished hypotheses.

For clarity this optional example recomputes each complete prefix. The main
Chapter 2 decoder uses a KV cache; efficient cache branching is a separate
systems problem and should not be hidden inside a beam-search listing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


@dataclass(frozen=True)
class Beam:
    token_ids: tuple[int, ...]
    score: float


@torch.inference_mode()
def beam_search(model, prompt_token_ids, eos_token_ids, width, max_new_tokens):
    active = [Beam(tuple(prompt_token_ids), 0.0)]
    finished = []
    prompt_length = len(prompt_token_ids)

    for _ in range(max_new_tokens):
        candidates = []
        for beam in active:
            input_ids = torch.tensor([beam.token_ids], device=model.device)
            logits = model(input_ids=input_ids).logits[0, -1]
            log_probabilities = torch.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probabilities, width)
            for log_probability, token_id in zip(values.tolist(), indices.tolist()):
                candidate = Beam(
                    beam.token_ids + (token_id,), beam.score + log_probability
                )
                if token_id in eos_token_ids:
                    finished.append(candidate)
                else:
                    candidates.append(candidate)
        if not candidates:
            break
        active = sorted(candidates, key=lambda beam: beam.score, reverse=True)[:width]

    pool = finished or active
    best = max(pool, key=lambda beam: beam.score)
    return best.token_ids[prompt_length:], best.score, len(finished)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    prompt_token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    raw_eos = model.generation_config.eos_token_id
    eos_token_ids = {raw_eos} if isinstance(raw_eos, int) else set(raw_eos or ())
    output_ids, score, finished_count = beam_search(
        model,
        prompt_token_ids,
        eos_token_ids,
        args.width,
        args.max_new_tokens,
    )
    print(f"finished hypotheses: {finished_count}")
    print(f"best cumulative log probability: {score:.3f}")
    print(args.prompt + tokenizer.decode(output_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
