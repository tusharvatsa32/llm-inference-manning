"""A compact beam search, compared with greedy decoding.

Section 2.4: beam search keeps several candidate sequences (beams), extends
each one at every step, and keeps the highest-scoring candidates by their
cumulative log probability. Greedy decoding is beam search with one beam.

For clarity, every step reruns the model on each full beam. A real engine
reuses each beam's KV cache instead, which is why beam search costs extra
memory as well as extra computation.

Examples:
    python beam_search.py
    python beam_search.py --num-beams 5 --max-new-tokens 30
"""

import argparse
from dataclasses import dataclass

import torch

from decoder import add_model_args, load_model


@dataclass
class Beam:
    ids: list[int]
    score: float  # cumulative log probability of the generated tokens


@torch.no_grad()
def beam_search(model, tokenizer, prompt, num_beams, max_new_tokens):
    prompt_ids = tokenizer.encode(prompt)
    beams = [Beam(prompt_ids, 0.0)]
    finished = []

    for _ in range(max_new_tokens):
        candidates = []
        for beam in beams:
            ids = torch.tensor([beam.ids], device=model.device)
            logits = model(ids).logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            # Each beam proposes its num_beams best continuations.
            top = torch.topk(log_probs, num_beams)
            for lp, tid in zip(top.values.tolist(), top.indices.tolist()):
                candidates.append(Beam(beam.ids + [tid], beam.score + lp))

        # Keep the num_beams best candidates across all beams.
        candidates.sort(key=lambda b: b.score, reverse=True)
        beams = []
        for candidate in candidates[:num_beams]:
            if candidate.ids[-1] == tokenizer.eos_token_id:
                finished.append(candidate)
            else:
                beams.append(candidate)

        if not beams or len(finished) >= num_beams:
            break

    best = max(finished + beams, key=lambda b: b.score)
    text = tokenizer.decode(best.ids[len(prompt_ids):], skip_special_tokens=True)
    return text, best.score


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--prompt", default="The best way to learn a new language is")
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)

    greedy_text, greedy_score = beam_search(
        model, tokenizer, args.prompt, 1, args.max_new_tokens)
    beam_text, beam_score = beam_search(
        model, tokenizer, args.prompt, args.num_beams, args.max_new_tokens)

    print(f"Prompt: {args.prompt!r}\n")
    print(f"Greedy, 1 beam (log prob {greedy_score:.2f}):\n  {greedy_text.strip()!r}\n")
    print(f"Beam search, {args.num_beams} beams (log prob {beam_score:.2f}):\n"
          f"  {beam_text.strip()!r}")


if __name__ == "__main__":
    main()
