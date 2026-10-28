"""Chunked prefill: sharing the token budget between prefill and decode.

Sections 4.1.3 and 4.4.3, and the Pause and Think in section 4.1.4. Eight
requests are streaming, each expecting its next token within 50 ms, when a
request with a 4,000-token prompt arrives. Figure 4.4 shows the three ways to
fill a step's token budget, and this script runs each one:

    hold back     spend the budget on decodes only; the new prompt waits
                  for a free slot (max batch size 8)
    one pass      prefill the whole prompt in one forward pass
                  (chunked prefill off)
    chunked       slice the prompt so each chunk rides alongside the decodes;
                  the token budget (max_num_batched_tokens) sets the chunk size

Examples:
    python chunked_prefill.py
    python chunked_prefill.py --prompt 8000 --budgets 4096 1024 256
"""

import argparse

from scheduler import Request, Scheduler, run_continuous


def run(prompt_len, budget, chunked, max_batch_size, arrival_ms):
    streaming = [Request(f"S{i}", prompt_len=100, output_len=150) for i in range(8)]
    new = Request("NEW", prompt_len=prompt_len, output_len=50, arrival=arrival_ms)
    scheduler = Scheduler(streaming + [new], token_budget=budget, chunked_prefill=chunked)
    run_continuous(scheduler, max_batch_size)

    # How the streams felt while the new prompt was being prefilled.
    window_end = new.token_times[0]
    gaps = [b - a for r in streaming for a, b in zip(r.token_times, r.token_times[1:])
            if arrival_ms <= b <= window_end]
    return new.ttft, max(gaps, default=0.0), gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", type=int, default=4000,
                        help="tokens in the new request's prompt (default: 4000)")
    parser.add_argument("--budgets", type=int, nargs="+", default=[2048, 512, 128],
                        help="token budgets to try with chunked prefill")
    parser.add_argument("--slo", type=float, default=50.0,
                        help="target time between tokens for streaming requests, ms")
    args = parser.parse_args()

    arrival = 100.0  # the streams are well under way
    one_pass_budget = args.prompt + 8
    runs = [
        ("hold back (decodes only)", 2048, True, 8),
        (f"one pass (budget {one_pass_budget:,}, unchunked)", one_pass_budget, False, 9),
    ] + [(f"chunked, budget {b:,}", b, True, 9) for b in args.budgets]

    print(f"8 requests streaming; a {args.prompt:,}-token prompt arrives at {arrival:.0f} ms\n")
    print(f"  {'Policy':<36} | New request TTFT | Longest stream gap | Gaps over {args.slo:.0f} ms")
    for label, budget, chunked, max_batch in runs:
        ttft, longest, gaps = run(args.prompt, budget, chunked, max_batch, arrival)
        late = sum(g > args.slo for g in gaps)
        print(f"  {label:<36} | {ttft:>12,.0f} ms | {longest:>15,.1f} ms | {late:>6} of {len(gaps)}")
    print()
    print("  One pass gives the new request its first token soonest, but every stream")
    print("  stalls for the whole prefill. Holding the prompt back keeps the streams")
    print("  smooth but makes the new request wait. Smaller chunks protect the streams")
    print("  and delay the new request's first token: a dial instead of a switch.")


if __name__ == "__main__":
    main()
