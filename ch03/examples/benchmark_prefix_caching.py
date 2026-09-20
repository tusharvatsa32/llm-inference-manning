#!/usr/bin/env python3
"""Benchmark prefix caching against a shared system prompt (Chapter 3, Section 3.5).

Simulates a 2,048-token shared system prompt followed by 10 user requests with
unique tails, using the real BlockAllocator / PrefixCache from mini_inference to
show how many tokens are recomputed vs. reused from cache, and how much KV cache
memory block sharing avoids.

Usage:
    python3 ch03/examples/benchmark_prefix_caching.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mini_inference.memory import (  # noqa: E402
    BlockAllocator,
    BlockTable,
    ModelProfile,
    PrefixCache,
    ServingConfig,
    bytes_per_token,
)

BLOCK_SIZE = 16
SYSTEM_PROMPT_TOKENS = 2048
NUM_REQUESTS = 10
MIN_TAIL_TOKENS = 20
MAX_TAIL_TOKENS = 200
RANDOM_SEED = 7


def build_tail(rng: random.Random, request_index: int) -> list[int]:
    # Offset tail token ids well clear of the shared prompt's id range (and each
    # other's) so a tail can never accidentally hash-collide with cached content.
    tail_len = rng.randint(MIN_TAIL_TOKENS, MAX_TAIL_TOKENS)
    base = 1_000_000 + request_index * 10_000
    return list(range(base, base + tail_len))


def prefill_without_cache(prompt_and_tail: list[int]) -> int:
    """No sharing: every request recomputes the full prompt and its own tail."""
    return len(prompt_and_tail)


def prefill_with_cache(
    allocator: BlockAllocator, prefix_cache: PrefixCache, tokens: list[int]
) -> tuple[BlockTable, int, int]:
    """Reuse cached prefix blocks; only compute and cache the uncached remainder."""
    matched_blocks, remaining_tokens = prefix_cache.match_prefix(tokens, BLOCK_SIZE)

    table = BlockTable(block_size=BLOCK_SIZE)
    table.blocks.extend(matched_blocks)

    tail_start = len(tokens) - len(remaining_tokens)
    parent_hash = matched_blocks[-1].hash_key if matched_blocks else None

    remaining_table = allocator.allocate_sequence(len(remaining_tokens))
    table.blocks.extend(remaining_table.blocks)

    # Cache only full blocks that fall within the shared system prompt: tails are
    # unique per request, so caching them would never produce a hit.
    offset = 0
    for block in remaining_table.blocks:
        block_start = tail_start + offset
        block_end = block_start + block.num_tokens
        if block.is_full and block_end <= SYSTEM_PROMPT_TOKENS:
            chunk = tokens[block_start:block_end]
            parent_hash = prefix_cache.insert_block(chunk, block, parent_hash)
        offset += block.num_tokens

    tokens_from_cache = len(matched_blocks) * BLOCK_SIZE
    tokens_computed = len(remaining_tokens)
    return table, tokens_from_cache, tokens_computed


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    system_prompt = list(range(SYSTEM_PROMPT_TOKENS))

    allocator = BlockAllocator(num_blocks=4096, block_size=BLOCK_SIZE)
    prefix_cache = PrefixCache()

    model = ModelProfile(layers=32, kv_heads=8, head_dim=128)
    config = ServingConfig(kv_dtype_bytes=2)
    per_token_bytes = bytes_per_token(model, config)

    total_tokens_naive = 0
    total_tokens_computed_cached = 0
    total_tokens_from_cache = 0
    private_prompt_blocks_avoided = 0

    header = f"{'Request':>8} {'Tail':>6} {'Naive prefill':>14} {'Cached: computed':>17} {'Cached: from cache':>19}"
    print(f"Shared system prompt: {SYSTEM_PROMPT_TOKENS} tokens, {NUM_REQUESTS} requests\n")
    print(header)
    print("-" * len(header))

    for i in range(NUM_REQUESTS):
        tail = build_tail(rng, i)
        full_tokens = system_prompt + tail

        naive_tokens = prefill_without_cache(full_tokens)
        _, tokens_from_cache, tokens_computed = prefill_with_cache(
            allocator, prefix_cache, full_tokens
        )

        total_tokens_naive += naive_tokens
        total_tokens_computed_cached += tokens_computed
        total_tokens_from_cache += tokens_from_cache
        if i > 0:
            private_prompt_blocks_avoided += SYSTEM_PROMPT_TOKENS // BLOCK_SIZE

        print(f"{i:>8} {len(tail):>6} {naive_tokens:>14} {tokens_computed:>17} {tokens_from_cache:>19}")

    memory_saved_gb = (private_prompt_blocks_avoided * BLOCK_SIZE * per_token_bytes) / (1024 ** 3)
    tokens_saved_pct = 100 * (total_tokens_naive - total_tokens_computed_cached) / total_tokens_naive

    print()
    print("Totals")
    print("-" * 60)
    print(f"{'Tokens computed, no prefix caching':<38}: {total_tokens_naive:>10}")
    print(f"{'Tokens computed, with prefix caching':<38}: {total_tokens_computed_cached:>10}")
    print(f"{'Tokens served from cache':<38}: {total_tokens_from_cache:>10}")
    print(f"{'Prefill work avoided':<38}: {tokens_saved_pct:>9.1f}%")
    print(f"{'KV cache memory saved by sharing':<38}: {memory_saved_gb:>9.3f} GB "
          f"({private_prompt_blocks_avoided} private prompt blocks avoided)")


if __name__ == "__main__":
    main()
