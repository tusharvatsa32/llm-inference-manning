"""Compare how each sampling strategy changes the candidate set.

Runs the decoder from Listing 2.10 once per strategy on the same prompt. At
chosen decode steps it prints how many tokens survived truncation and the
five most likely survivors, then prints each strategy's full output.

Top-k always keeps the same number of candidates. Top-p, min-p, and locally
typical sampling keep more candidates when the model is unsure and fewer when
it is confident.

Examples:
    python compare_strategies.py
    python compare_strategies.py --prompt "Write a poem about the ocean." --steps 1 10 50
    python compare_strategies.py --repetition-penalty 1.1
"""

import argparse

import torch

from decoder import Decoder, DecoderConfig, add_model_args, load_model

# Truncation strategies use the same temperature so that the filter is the
# only thing that changes. "combined" uses the Qwen3 defaults from 2.3.5.
STRATEGIES = {
    "sampling": dict(temperature=1.0),
    "top-k": dict(temperature=1.0, top_k=50),
    "top-p": dict(temperature=1.0, top_p=0.9),
    "min-p": dict(temperature=1.0, min_p=0.1),
    "typical": dict(temperature=1.0, typical_p=0.9),
    "combined": dict(temperature=0.6, top_k=20, top_p=0.95),
}


def candidate_logits(decoder, logits, generated):
    """Apply the same steps as Decoder.select_next_token, without sampling."""
    c = decoder.config
    if c.repetition_penalty != 1.0:
        logits = decoder.apply_repetition_penalty(logits, generated)
    logits = logits / c.temperature
    if c.top_k is not None:
        logits = decoder.top_k_filter(logits)
    if c.top_p is not None:
        logits = decoder.top_p_filter(logits)
    if c.min_p is not None:
        logits = decoder.min_p_filter(logits)
    if c.typical_p is not None:
        logits = decoder.typical_filter(logits)
    return logits


def describe_candidates(tokenizer, logits):
    survivors = int(torch.isfinite(logits).sum())
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, min(5, survivors))
    shown = ", ".join(
        f"{tokenizer.decode([i])!r} {p:.2f}"
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    )
    return survivors, shown


@torch.no_grad()
def run(decoder, prompt, steps, seed):
    """Generate like Decoder.generate, reporting candidates at `steps`."""
    torch.manual_seed(seed)
    tokenizer = decoder.tokenizer
    ids = tokenizer.encode(prompt, return_tensors="pt").to(decoder.model.device)
    start = ids.shape[1]

    for step in range(1, decoder.config.max_new_tokens + 1):
        logits = decoder.model(ids).logits[0, -1, :]
        generated = ids[0, start:]

        if step in steps:
            filtered = candidate_logits(decoder, logits, generated)
            survivors, shown = describe_candidates(tokenizer, filtered)
            print(f"  step {step:>3}: {survivors:>6,} candidates | {shown}")

        next_id = decoder.select_next_token(logits, generated)
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            print(f"  stopped at step {step} (end-of-sequence token)")
            break

    return tokenizer.decode(ids[0, start:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--prompt", default="Write a short story about a lighthouse keeper.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 10, 50, 100, 200],
                        help="decode steps at which to print the candidate set")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=list(STRATEGIES))
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    vocab = model.get_output_embeddings().weight.shape[0]
    print(f"Model: {args.model} ({vocab:,} vocabulary entries)")
    print(f"Prompt: {args.prompt!r}\n")

    outputs = {}
    for name in args.strategies:
        config = DecoderConfig(
            **STRATEGIES[name],
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
        settings = ", ".join(f"{k}={v}" for k, v in STRATEGIES[name].items())
        print(f"== {name} ({settings})")
        decoder = Decoder(model, tokenizer, config)
        outputs[name] = run(decoder, args.prompt, set(args.steps), args.seed)
        print()

    for name, text in outputs.items():
        print(f"== {name} output\n{text.strip()}\n")


if __name__ == "__main__":
    main()
