"""Static batching versus continuous batching.

Sections 4.2.4 and 4.3. Runs the same requests through run_static
(Listing 4.4) and run_continuous (Listing 4.5) and compares them:

    figure 4.7   a four-slot batch where B and D finish after one pass while
                 E and F wait
    figure 4.6   eight requests needing 15 to 600 output tokens, with eight
                 more waiting behind them
    try it       section 4.2.4's experiment in simulation: 32 requests, half
                 needing 16 tokens and half 512

Examples:
    python static_vs_continuous.py
    python static_vs_continuous.py --max-batch-size 4
"""

import argparse
import statistics

from scheduler import Request, Scheduler, print_summary, run_continuous, run_static, summarize

POLICIES = {"static": run_static, "continuous": run_continuous}


def slot_utilization(scheduler, max_batch_size):
    """Share of batch slots that produced a token, averaged over passes."""
    used = sum(len(step.tokens) for step in scheduler.steps)
    return used / (len(scheduler.steps) * max_batch_size)


def figure_4_7():
    print("== Figure 4.7: four slots; B and D finish after one pass, E and F wait")
    for policy, run in POLICIES.items():
        requests = [Request(name, prompt_len=10, output_len=n)
                    for name, n in (("A", 3), ("B", 1), ("C", 3), ("D", 1), ("E", 2), ("F", 2))]
        scheduler = Scheduler(requests)
        run(scheduler, max_batch_size=4)
        print(f"  {policy}:")
        for i, step in enumerate(scheduler.steps, 1):
            names = [r.name for r in step.tokens]
            slots = names + ["."] * (4 - len(names))
            print(f"    pass {i}: {' '.join(slots)}")
    print("  '.' is a slot that produced nothing in that pass.\n")


def figure_4_6(max_batch_size):
    print(f"== Figure 4.6: mixed output lengths, max batch size {max_batch_size}")
    lengths = [15, 40, 90, 160, 250, 350, 470, 600]
    for policy, run in POLICIES.items():
        requests = [Request(f"R{i}", prompt_len=200, output_len=n)
                    for i, n in enumerate(lengths, 1)]
        # Eight more requests waiting behind the first batch.
        requests += [Request(f"W{i}", prompt_len=200, output_len=n)
                     for i, n in enumerate(reversed(lengths), 1)]
        scheduler = Scheduler(requests)
        run(scheduler, max_batch_size)
        print_summary(policy, summarize(scheduler))
        waiting = [r for r in scheduler.finished if r.name.startswith("W")]
        print(f"    slots doing useful work: {slot_utilization(scheduler, max_batch_size):.0%}; "
              f"waiting requests got their first token after "
              f"{statistics.mean(r.ttft for r in waiting):,.0f} ms on average")
    print()


def try_it(max_batch_size):
    print(f"== Section 4.2.4 try it, simulated: 32 requests, half need 16 tokens "
          f"and half 512, max batch size {max_batch_size}")
    for policy, run in POLICIES.items():
        requests = [Request(f"R{i}", prompt_len=100, output_len=16 if i % 2 else 512)
                    for i in range(32)]
        scheduler = Scheduler(requests)
        run(scheduler, max_batch_size)
        s = summarize(scheduler)
        short = [r.e2e for r in scheduler.finished if r.output_len == 16]
        print(f"  {policy:>10}: {s.requests / s.makespan_ms * 1000:5.1f} requests/s, "
              f"short requests complete after {statistics.mean(short):7,.0f} ms on average")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-batch-size", type=int, default=8)
    args = parser.parse_args()

    figure_4_7()
    figure_4_6(args.max_batch_size)
    try_it(max(args.max_batch_size, 16))


if __name__ == "__main__":
    main()
