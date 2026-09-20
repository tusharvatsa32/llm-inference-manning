"""KV cache sizing and capacity-planning models (Chapter 3, Sections 3.2, 3.4, 3.7).

Two memory units show up in capacity planning and they are not interchangeable:
decimal GB (10**9 bytes), which is how GPU vendors advertise HBM capacity, and
binary GiB (1024**3 bytes), which is how allocators and frameworks actually carve
up that memory. This module keeps GPU memory buffers (weights, KV budget, token
pool) in binary GiB via `bytes_to_gib`/`gib_to_bytes`, and only touches decimal
values when converting a parameter count in billions to a raw byte count.
"""

import math
from dataclasses import dataclass
from typing import Literal

GIB = 1024 ** 3


def bytes_to_gib(num_bytes: float) -> float:
    return num_bytes / GIB


def gib_to_bytes(num_gib: float) -> float:
    return num_gib * GIB


@dataclass
class ModelProfile:
    layers: int
    kv_heads: int
    head_dim: int
    params_b: float = 0.0
    sliding_window: int | None = None


@dataclass
class WorkloadProfile:
    current_tokens_per_request: int
    max_tokens_per_request: int
    concurrency: int


@dataclass
class ServingConfig:
    allocation: Literal["static", "paged"] = "paged"
    block_size: int = 16
    kv_dtype_bytes: int = 2
    weight_dtype_bytes: int = 2
    gpu_memory_gib: float = 80.0
    gpu_utilization: float = 0.90


@dataclass
class CapacityReport:
    weight_gb: float
    kv_budget_gb: float
    per_token_bytes: int
    token_pool: int
    max_concurrency: int


def active_tokens(model: ModelProfile, workload: WorkloadProfile) -> int:
    """Cap the active sequence length at the model's sliding window, if any."""
    if model.sliding_window is None:
        return workload.current_tokens_per_request
    return min(workload.current_tokens_per_request, model.sliding_window)


def bytes_per_token(model: ModelProfile, config: ServingConfig) -> int:
    """Bytes needed to cache one token's keys and values across every layer."""
    return int(
        2
        * model.layers
        * model.kv_heads
        * model.head_dim
        * config.kv_dtype_bytes
    )


def reserved_token_slots(workload: WorkloadProfile, config: ServingConfig) -> int:
    """Token slots the allocator sets aside under the configured allocation strategy."""
    if config.allocation == "static":
        return workload.concurrency * workload.max_tokens_per_request

    if config.allocation == "paged":
        blocks_per_request = math.ceil(
            workload.current_tokens_per_request / config.block_size
        )
        return workload.concurrency * blocks_per_request * config.block_size

    raise ValueError(f"unknown allocation: {config.allocation}")


def estimate_capacity(
    model: ModelProfile,
    workload: WorkloadProfile,
    config: ServingConfig,
    weight_gb: float | None = None,
    available_kv_gb: float | None = None,
) -> CapacityReport:
    """Estimate the token pool and concurrency ceiling a GPU can support (Listing 3.10).

    `weight_gb` and `available_kv_gb` let a caller override the first-pass
    parameter-count estimate with a measured value from the serving stack.

    `token_pool == 0` means the model's weights alone don't fit in
    `config.gpu_memory_gib * config.gpu_utilization` -- this GPU can't serve
    this model at this precision at all, before a single request arrives.
    """
    if weight_gb is None:
        weight_bytes = model.params_b * 1e9 * config.weight_dtype_bytes
        weight_gb = bytes_to_gib(weight_bytes)

    if available_kv_gb is None:
        usable_gb = config.gpu_memory_gib * config.gpu_utilization
        # Clamped at 0: a model whose weights alone exceed the usable budget
        # doesn't fit on this GPU at all, not "fit a negative number of
        # tokens" -- see estimate_capacity's docstring for how to read that.
        available_kv_gb = max(0.0, usable_gb - weight_gb)

    per_token_bytes = bytes_per_token(model, config)
    token_pool = int(gib_to_bytes(available_kv_gb) / per_token_bytes)

    active = active_tokens(model, workload)
    max_concurrency = token_pool // active if active > 0 else 0

    return CapacityReport(
        weight_gb=round(weight_gb, 1),
        kv_budget_gb=round(available_kv_gb, 1),
        per_token_bytes=per_token_bytes,
        token_pool=token_pool,
        max_concurrency=max_concurrency,
    )
