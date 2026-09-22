"""KV-cache memory sizing.

Grounded in Chapter 1, Section 1.3.2 ("Memory as the limiting factor"), which
gives this formula for a Llama-2-style model:

    2 (keys + values) x layers x heads x head_dim x dtype_bytes

and works it through three examples: ~0.5 MB per token, ~1 GB for a single
2,000-token request, and ~50 GB for 50 concurrent 2,000-token requests.

Chapter 4's scheduler reuses these functions for admission control: it cannot
admit a new request once the KV cache they describe would exceed available
GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    """The handful of dimensions that determine KV-cache size per token."""

    num_layers: int
    num_heads: int
    head_dim: int
    # GQA models read fewer KV heads than query heads (Section 1.5.3).
    # Defaults to num_heads, i.e. standard multi-head attention.
    num_kv_heads: int | None = None

    @property
    def effective_kv_heads(self) -> int:
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads


# The book's own worked example: a Llama-2-style 7B model.
LLAMA2_7B = ModelShape(num_layers=32, num_heads=32, head_dim=128)


def kv_cache_bytes_per_token(shape: ModelShape, dtype_bytes: int = 2) -> float:
    """Bytes of KV cache one token adds, per Section 1.3.2's formula."""
    return 2 * shape.num_layers * shape.effective_kv_heads * shape.head_dim * dtype_bytes


def kv_cache_bytes_for_request(shape: ModelShape, num_tokens: int, dtype_bytes: int = 2) -> float:
    """KV-cache size in bytes for one request holding `num_tokens` of context."""
    return kv_cache_bytes_per_token(shape, dtype_bytes) * num_tokens


def kv_cache_bytes_for_concurrent_requests(
    shape: ModelShape, num_tokens: int, num_requests: int, dtype_bytes: int = 2
) -> float:
    """Total KV-cache bytes for `num_requests` concurrent requests, each
    holding `num_tokens` of context (Section 1.3.2's "50 concurrent users" case).
    """
    return kv_cache_bytes_for_request(shape, num_tokens, dtype_bytes) * num_requests


def bytes_to_mb(num_bytes: float) -> float:
    """Convert bytes to decimal megabytes (1 MB = 10**6 bytes)."""
    return num_bytes / 1e6


def bytes_to_gb(num_bytes: float) -> float:
    """Convert bytes to decimal gigabytes (1 GB = 10**9 bytes)."""
    return num_bytes / 1e9
