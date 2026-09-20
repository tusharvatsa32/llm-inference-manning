"""KV cache sizing, PagedAttention block allocation, and prefix caching (Chapter 3)."""

from .block_allocator import Block, BlockAllocator, BlockTable
from .capacity_planner import (
    CapacityReport,
    ModelProfile,
    ServingConfig,
    WorkloadProfile,
    active_tokens,
    bytes_per_token,
    bytes_to_gib,
    estimate_capacity,
    gib_to_bytes,
    reserved_token_slots,
)
from .prefix_cache import PrefixCache, hash_block

__all__ = [
    "Block",
    "BlockTable",
    "BlockAllocator",
    "PrefixCache",
    "hash_block",
    "ModelProfile",
    "WorkloadProfile",
    "ServingConfig",
    "CapacityReport",
    "estimate_capacity",
    "active_tokens",
    "bytes_per_token",
    "reserved_token_slots",
    "bytes_to_gib",
    "gib_to_bytes",
]
