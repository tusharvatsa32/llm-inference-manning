"""Split an inference bill into prompt (prefill) and completion (decode) cost.

Reproduces the intent-routing example from section 1.3.3 and figure 1.6:
382.63K requests, 2.67 billion prompt tokens, and 6.51 million completion
tokens, priced at $0.15 per 1M prompt tokens and $0.60 per 1M completion
tokens. More than 99% of that bill is prompt tokens, so prefix caching saves
far more than any decode optimization.

Examples:
    python cost_breakdown.py
    python cost_breakdown.py --requests 10000 --prompt-tokens 2e6 --completion-tokens 5e6
    python cost_breakdown.py --prompt-price 3.00 --completion-price 15.00
"""

import argparse

MILLION = 1_000_000


def main(args):
    prompt_cost = args.prompt_tokens / MILLION * args.prompt_price
    completion_cost = args.completion_tokens / MILLION * args.completion_price
    total_cost = prompt_cost + completion_cost
    total_tokens = args.prompt_tokens + args.completion_tokens

    print(f"Requests:          {args.requests:>16,.0f}")
    print(f"Prompt tokens:     {args.prompt_tokens:>16,.0f}"
          f"  ({args.prompt_tokens / args.requests:,.0f} per request)")
    print(f"Completion tokens: {args.completion_tokens:>16,.0f}"
          f"  ({args.completion_tokens / args.requests:,.0f} per request)")
    print(f"Input:output ratio: {args.prompt_tokens / args.completion_tokens:,.0f} to 1")
    print(f"Prompt share of tokens: {args.prompt_tokens / total_tokens:.2%}\n")

    print(f"Prompt cost:     ${prompt_cost:>12,.2f}  "
          f"({prompt_cost / total_cost:.1%} of the bill)")
    print(f"Completion cost: ${completion_cost:>12,.2f}  "
          f"({completion_cost / total_cost:.1%} of the bill)")
    print(f"Total:           ${total_cost:>12,.2f}")
    print(f"Per 1K requests: ${total_cost / args.requests * 1000:>12,.4f}")

    if args.cached_fraction:
        # Treat cached prompt tokens as free; real providers often charge a
        # discounted rate instead.
        saved = prompt_cost * args.cached_fraction
        print(f"\nCaching {args.cached_fraction:.0%} of prompt tokens saves "
              f"${saved:,.2f} ({saved / total_cost:.1%} of the bill).")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=float, default=382_630)
    parser.add_argument("--prompt-tokens", type=float, default=2.67e9)
    parser.add_argument("--completion-tokens", type=float, default=6.51e6)
    parser.add_argument("--prompt-price", type=float, default=0.15,
                        help="dollars per 1M prompt tokens (default: 0.15)")
    parser.add_argument("--completion-price", type=float, default=0.60,
                        help="dollars per 1M completion tokens (default: 0.60)")
    parser.add_argument("--cached-fraction", type=float, default=0.0,
                        help="share of prompt tokens served from a prefix "
                             "cache, from 0 to 1 (default: 0)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
