"""PagedAttention block allocation (Chapter 3, Section 3.3).

`Block.ref_count` counts how many BlockTables currently point at that block.
`allocate()` hands out a block with ref_count = 1 (the caller is its only
owner). A second table can become a co-owner of the same block via
`BlockTable.append_block()`, which bumps ref_count -- that's how prefix-cache
sharing (Section 3.5) and copy-on-write work.
"""

import collections
import math
from dataclasses import dataclass


@dataclass
class Block:
    block_id: int
    block_size: int
    ref_count: int = 0
    num_tokens: int = 0
    hash_key: int | None = None

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    @property
    def available_slots(self) -> int:
        return self.block_size - self.num_tokens

    def reset(self) -> None:
        self.ref_count = 0
        self.num_tokens = 0
        self.hash_key = None


def _needs_prefix_cache_to_free(block: Block) -> bool:
    """True if freeing `block` right now would be its last reference while
    it's still registered in a PrefixCache -- i.e. free() needs a
    `prefix_cache` to evict it with, or it will raise.
    """
    return block.ref_count == 1 and block.hash_key is not None


def _prefix_cache_error(block: Block) -> str:
    return (
        f"block {block.block_id} is still registered in a PrefixCache -- "
        "pass prefix_cache=... to evict it, or evict it yourself first"
    )


class BlockTable:
    """The logical sequence of physical blocks assigned to one request."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = []

    def append_block(self, block: Block) -> None:
        """Attach a block this table doesn't already own, becoming a co-owner."""
        self.blocks.append(block)
        block.ref_count += 1

    def get_physical_block_ids(self) -> list[int]:
        return [block.block_id for block in self.blocks]


class BlockAllocator:
    """Fixed pool of `num_blocks` physical KV-cache blocks of size `block_size`."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.all_blocks: dict[int, Block] = {
            block_id: Block(block_id=block_id, block_size=block_size)
            for block_id in range(num_blocks)
        }
        self.free_blocks: collections.deque[Block] = collections.deque(
            self.all_blocks.values()
        )

    def allocate(self) -> Block:
        if not self.free_blocks:
            raise MemoryError("Out of KV Cache blocks")
        block = self.free_blocks.popleft()
        block.ref_count = 1
        return block

    def free(self, block: Block, prefix_cache=None) -> None:
        """Release one reference to `block`.

        Once ref_count reaches zero, the block is reset and returned to the
        pool. Freeing an already-free block is a no-op, not a double-free.

        If `block` is registered in a PrefixCache (its hash_key is set),
        pass that cache as `prefix_cache` so the stale entry is evicted
        before the block is reused -- otherwise a later request could get a
        cache "hit" pointing at a block someone else now owns.
        """
        if block.ref_count == 0:
            return
        if _needs_prefix_cache_to_free(block) and prefix_cache is None:
            raise ValueError(_prefix_cache_error(block))

        block.ref_count -= 1
        if block.ref_count == 0:
            if block.hash_key is not None:
                prefix_cache.evict(block)
            block.reset()
            self.free_blocks.append(block)

    def free_sequence(self, block_table: BlockTable, prefix_cache=None) -> None:
        """Release every block a finished request's table owns.

        Checks every block up front when `prefix_cache` is omitted, so this
        either frees the whole table or raises without freeing any of it --
        never half of it.
        """
        if prefix_cache is None:
            for block in block_table.blocks:
                if _needs_prefix_cache_to_free(block):
                    raise ValueError(_prefix_cache_error(block))

        for block in block_table.blocks:
            self.free(block, prefix_cache=prefix_cache)
        block_table.blocks = []

    def allocate_sequence(self, num_tokens: int) -> BlockTable:
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        table = BlockTable(self.block_size)
        remaining = num_tokens
        for _ in range(num_blocks_needed):
            block = self.allocate()
            block.num_tokens = min(self.block_size, remaining)
            remaining -= block.num_tokens
            table.blocks.append(block)
        return table

    def append_token(self, block_table: BlockTable) -> Block | None:
        """Append one token's slot, allocating a new block if the tail is full
        or shared (ref_count > 1 -- writing into it would corrupt another
        request's cache, so this is a copy-on-write).
        """
        tail = block_table.blocks[-1] if block_table.blocks else None
        if tail is None or tail.is_full or tail.ref_count > 1:
            new_block = self.allocate()
            new_block.num_tokens = 1
            block_table.blocks.append(new_block)
            return new_block

        tail.num_tokens += 1
        return None

    def can_allocate(self, num_tokens: int) -> bool:
        needed = math.ceil(num_tokens / self.block_size)
        return len(self.free_blocks) >= needed
