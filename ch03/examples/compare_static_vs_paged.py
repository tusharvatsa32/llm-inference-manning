#!/usr/bin/env python3
"""Compare static vs. paged KV cache allocation across a simulated request batch
(Chapter 3, Section 3.3).

Usage:
    python3 ch03/examples/compare_static_vs_paged.py
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mini_inference.memory import ModelProfile, ServingConfig, bytes_per_token  # noqa: E402

NUM_REQUESTS = 32
MIN_CONTEXT_TOKENS = 32
MAX_SEQ_CAP = 2048
BLOCK_SIZE = 16
BAR_WIDTH = 40
RANDOM_SEED = 3  # fixed so the printed numbers are reproducible


def static_reserved_slots(_current_tokens: int, max_tokens: int) -> int:
    return max_tokens


def paged_reserved_slots(current_tokens: int, block_size: int = BLOCK_SIZE) -> int:
    return math.ceil(current_tokens / block_size) * block_size


def ascii_bar(value: float, max_value: float, width: int = BAR_WIDTH) -> str:
    filled = round(width * value / max_value) if max_value else 0
    return "#" * filled + "-" * (width - filled)


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    context_lengths = [
        rng.randint(MIN_CONTEXT_TOKENS, MAX_SEQ_CAP) for _ in range(NUM_REQUESTS)
    ]

    static_slots = [static_reserved_slots(n, MAX_SEQ_CAP) for n in context_lengths]
    paged_slots = [paged_reserved_slots(n) for n in context_lengths]
    ideal_slots = context_lengths  # the tokens actually stored, no allocation waste

    total_ideal = sum(ideal_slots)
    total_static = sum(static_slots)
    total_paged = sum(paged_slots)

    model = ModelProfile(layers=32, kv_heads=8, head_dim=128)
    config = ServingConfig(kv_dtype_bytes=2)
    per_token_bytes = bytes_per_token(model, config)

    def to_gb(slots: int) -> float:
        return (slots * per_token_bytes) / (1024 ** 3)

    print(f"Simulated {NUM_REQUESTS} requests, context lengths in "
          f"[{MIN_CONTEXT_TOKENS}, {MAX_SEQ_CAP}] tokens (Llama-3-8B-sized cache)\n")

    header = f"{'Request':>8} {'Tokens':>7} {'Static':>7} {'Paged':>7} {'Waste (static)':>15}"
    print(header)
    print("-" * len(header))
    for i, (tokens, static, paged) in enumerate(zip(context_lengths, static_slots, paged_slots)):
        print(f"{i:>8} {tokens:>7} {static:>7} {paged:>7} {static - tokens:>15}")

    print()
    print("Totals across all requests")
    print("-" * 60)
    print(f"{'Ideal (tokens stored)':<28}: {total_ideal:>8} slots  ({to_gb(total_ideal):.2f} GB)")
    print(f"{'Static allocation':<28}: {total_static:>8} slots  ({to_gb(total_static):.2f} GB)")
    print(f"{'Paged allocation (block=16)':<28}: {total_paged:>8} slots  ({to_gb(total_paged):.2f} GB)")

    memory_saved_pct = 100 * (total_static - total_paged) / total_static
    static_waste = total_static - total_ideal
    paged_waste = total_paged - total_ideal
    fragmentation_reduction_pct = 100 * (static_waste - paged_waste) / static_waste

    print()
    print(f"Memory saved by paging vs. static : {memory_saved_pct:.1f}%")
    print(f"Internal fragmentation reduction  : {fragmentation_reduction_pct:.1f}% "
          f"(wasted slots: static={static_waste}, paged={paged_waste})")

    print()
    print("Reserved memory, static vs. paged (relative to static total)")
    print(f"  static : {ascii_bar(total_static, total_static)} {to_gb(total_static):6.2f} GB")
    print(f"  paged  : {ascii_bar(total_paged, total_static)} {to_gb(total_paged):6.2f} GB")
    print(f"  ideal  : {ascii_bar(total_ideal, total_static)} {to_gb(total_ideal):6.2f} GB")


if __name__ == "__main__":
    main()
