"""Admission and preemption under KV cache memory pressure.

Section 4.4. A request is admitted only when the ledger has enough free KV
cache blocks for its prompt (Listing 4.6), so admission can stop even when
the batch has open slots. Admission is a bet: requests keep growing as they
decode, and when a running request needs a block and none are free, the
scheduler preempts the newest running request (Listing 4.7). The preempted
request rebuilds its cache by recomputation when it resumes, which repeats
prefill work.

The script runs one workload under different KV cache sizes and batch
limits and reports preemptions, recomputed tokens, and latency. vLLM's advice
for frequent preemption is to lower max_num_seqs or max_num_batched_tokens, or
to give the KV cache more memory.

Examples:
    python memory_pressure.py
    python memory_pressure.py --blocks 100 200 400 --max-batch-size 16
"""

import argparse

from scheduler import BLOCK_SIZE, Request, Scheduler, run_continuous, summarize


def workload():
    """16 requests with 256-token prompts and 100 to 400 output tokens."""
    return [Request(f"R{i}", prompt_len=256, output_len=100 + (i * 97) % 300)
            for i in range(16)]


def run(blocks, max_batch_size):
    scheduler = Scheduler(workload(), total_blocks=blocks)
    scheduler.admit_from_waiting(up_to=max_batch_size)  # the first admission decision
    admitted = len(scheduler.running)
    run_continuous(scheduler, max_batch_size)
    recomputed = sum(r.recomputed for r in scheduler.finished)
    return admitted, recomputed, summarize(scheduler)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blocks", type=int, nargs="+", default=[180, 260, 400, 1000],
                        help=f"KV cache sizes to try, in {BLOCK_SIZE}-token blocks")
    parser.add_argument("--max-batch-size", type=int, nargs="+", default=[16, 8],
                        help="batch limits to try, vLLM's max_num_seqs (default: 16 8)")
    args = parser.parse_args()

    prompt_blocks = workload()[0].prompt_len // BLOCK_SIZE
    print(f"16 requests, 256-token prompts ({prompt_blocks} blocks each), "
          f"100 to 400 output tokens, {BLOCK_SIZE}-token blocks\n")
    print("  KV blocks | Max batch | Admitted at start | Preemptions | Recomputed tokens "
          "| Throughput   | Mean TTFT")
    for max_batch_size in args.max_batch_size:
        for blocks in args.blocks:
            admitted, recomputed, s = run(blocks, max_batch_size)
            print(f"  {blocks:>9} | {max_batch_size:>9} | {admitted:>17} | {s.preemptions:>11} "
                  f"| {recomputed:>17,} | {s.throughput_tok_s:>6,.0f} tok/s "
                  f"| {s.mean_ttft_ms:>7,.0f} ms")
    print()
    print("  With a small KV cache, admission stops before the batch is full, and")
    print("  requests that keep growing force preemptions whose recomputation repeats")
    print("  prefill work. A smaller batch limit admits fewer requests, so fewer of")
    print("  them have to be preempted.")


if __name__ == "__main__":
    main()
