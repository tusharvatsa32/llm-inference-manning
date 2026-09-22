"""Reproduces the KV-cache examples from Chapter 1, Section 1.3.2.

Run it from the repository root after installing the package:

    python3 ch01/supplementary/kv_cache_memory_calculator.py

Try It Now: change MODEL, CONTEXT_TOKENS, or NUM_CONCURRENT_USERS below and
see how the numbers move. What context length would make 50 concurrent users
exceed an 80 GB GPU?
"""

from mini_inference.memory import (
    LLAMA2_7B,
    bytes_to_gb,
    bytes_to_mb,
    kv_cache_bytes_for_concurrent_requests,
    kv_cache_bytes_for_request,
    kv_cache_bytes_per_token,
)

MODEL = LLAMA2_7B
CONTEXT_TOKENS = 2_000
LONG_CONTEXT_TOKENS = 32_000
NUM_CONCURRENT_USERS = 50


def main() -> None:
    per_token = kv_cache_bytes_per_token(MODEL)
    print(f"KV cache per token:              {bytes_to_mb(per_token):.3f} MB   (book: ~0.5 MB)")

    one_user = kv_cache_bytes_for_request(MODEL, CONTEXT_TOKENS)
    print(
        f"One user, {CONTEXT_TOKENS:,} tokens:       {bytes_to_gb(one_user):.3f} GB   (book: roughly 1 GB)"
    )

    concurrent = kv_cache_bytes_for_concurrent_requests(
        MODEL, CONTEXT_TOKENS, NUM_CONCURRENT_USERS
    )
    print(
        f"{NUM_CONCURRENT_USERS} users, {CONTEXT_TOKENS:,} tokens each:  "
        f"{bytes_to_gb(concurrent):.3f} GB   (book: around 50 GB)"
    )

    long_context = kv_cache_bytes_for_request(MODEL, LONG_CONTEXT_TOKENS)
    print(
        f"One user, {LONG_CONTEXT_TOKENS:,} tokens:      "
        f"{bytes_to_gb(long_context):.3f} GB   (book: on the order of 16 GB)"
    )


if __name__ == "__main__":
    main()