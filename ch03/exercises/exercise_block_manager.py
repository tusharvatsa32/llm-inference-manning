#!/usr/bin/env python3
"""Exercise: LRU eviction for BlockAllocator (Chapter 3, Section 3.3).

BlockAllocator.allocate() raises MemoryError the instant the free-block pool is
empty. But many freed blocks are not truly done: a block just released by a
finished request may still hold a system prompt another request is about to
reuse. Instead of forgetting them the moment ref_count hits 0, keep them
"warm" in an LRU structure, and only wipe the least-recently-used one when a
new allocation would otherwise fail.

Fill in the TODOs below, then run this file directly:
    python3 ch03/exercises/exercise_block_manager.py
All asserts should pass.
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mini_inference.memory import Block, BlockAllocator  # noqa: E402


class LRUBlockAllocator(BlockAllocator):
    """A BlockAllocator that evicts the least-recently-used freed block instead
    of raising MemoryError, as long as at least one freed-but-not-yet-wiped
    block is available.
    """

    def __init__(self, num_blocks: int, block_size: int):
        super().__init__(num_blocks, block_size)
        # TODO: create the structure that tracks freed-but-still-warm blocks in
        # least-recently-used order. An `collections.OrderedDict[int, Block]`
        # keyed by block_id works well: inserting/touching a key and moving it
        # to the end marks it "most recently used".
        self.evictable_blocks = ...  # TODO

    def free(self, block: Block) -> None:
        """Decrement ref_count. Once it hits zero, do NOT reset/reclaim the
        block immediately -- instead, park it in `evictable_blocks` as the
        most-recently-used entry, so it can be reused as-is (e.g. by a prefix
        cache hit) before it gets wiped.
        """
        # TODO: decrement block.ref_count (guard against going below zero).
        # TODO: if ref_count == 0, add/move `block` to the MRU end of
        # `evictable_blocks` instead of calling block.reset() or touching
        # `self.free_blocks`.
        raise NotImplementedError

    def touch(self, block: Block) -> None:
        """Mark a still-evictable block as freshly used (e.g. on a cache hit)."""
        # TODO: if block.block_id is in `evictable_blocks`, move it to the MRU end.
        raise NotImplementedError

    def allocate(self) -> Block:
        """Pop a free block as usual. If none are free, evict the
        least-recently-used entry from `evictable_blocks` (oldest = LRU),
        reset it, and use it -- only raise MemoryError if both are empty.
        """
        # TODO: if `self.free_blocks` is empty:
        #   - if `evictable_blocks` is non-empty, pop its LRU entry (the OLDEST
        #     one), reset() it, and push it onto `self.free_blocks`.
        #   - otherwise, let the MemoryError from the parent class propagate.
        return super().allocate()


def run_checks() -> None:
    allocator = LRUBlockAllocator(num_blocks=4, block_size=16)

    blocks = [allocator.allocate() for _ in range(4)]
    assert len(allocator.free_blocks) == 0

    # Freeing a block should not immediately return it to free_blocks -- it
    # should become "evictable" instead.
    allocator.free(blocks[0])
    assert len(allocator.free_blocks) == 0
    assert blocks[0].block_id in allocator.evictable_blocks

    # The pool has no free blocks, but one evictable block: allocate() should
    # reclaim it via LRU eviction rather than raising MemoryError.
    reused = allocator.allocate()
    assert reused.block_id == blocks[0].block_id
    assert reused.ref_count == 1
    assert reused.num_tokens == 0
    assert len(allocator.evictable_blocks) == 0

    # With nothing free and nothing evictable, allocate() must still raise.
    try:
        allocator.allocate()
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError when the pool is truly exhausted")

    # LRU order: the oldest freed block should be evicted first.
    allocator.free(blocks[1])
    allocator.free(blocks[2])
    allocator.touch(blocks[1])  # blocks[2] is now the least-recently-used
    evicted = allocator.allocate()
    assert evicted.block_id == blocks[2].block_id

    print("All checks passed.")


if __name__ == "__main__":
    run_checks()
