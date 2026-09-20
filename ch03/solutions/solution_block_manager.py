#!/usr/bin/env python3
"""Solution: LRU eviction for BlockAllocator (Chapter 3, Section 3.3).

See ch03/exercises/exercise_block_manager.py for the problem statement.
"""

import collections
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mini_inference.memory import Block, BlockAllocator, PrefixCache  # noqa: E402


class LRUBlockAllocator(BlockAllocator):
    """A BlockAllocator that evicts the least-recently-used freed block instead
    of raising MemoryError, as long as at least one freed-but-not-yet-wiped
    block is available.

    A block that hits ref_count 0 is kept warm in `evictable_blocks` rather
    than reset immediately, so it stays valid for a PrefixCache hit until LRU
    pressure actually reclaims it. `reuse()` is the only sanctioned way to
    bring a warm block back into service: it must remove the block from
    `evictable_blocks` in the same step that grants it a new owner, or
    allocate() could still hand that "in use" block to someone else.
    """

    def __init__(self, num_blocks: int, block_size: int):
        super().__init__(num_blocks, block_size)
        self.evictable_blocks: "collections.OrderedDict[int, Block]" = collections.OrderedDict()

    def free(self, block: Block, prefix_cache=None) -> None:
        # prefix_cache is accepted only so the inherited free_sequence() can
        # call this with the base class's signature. A block freed here
        # isn't reset yet -- see allocate() -- so there's nothing to evict
        # from a cache until it's actually reclaimed.
        if block.ref_count > 0:
            block.ref_count -= 1
        if block.ref_count == 0:
            self.evictable_blocks[block.block_id] = block
            self.evictable_blocks.move_to_end(block.block_id)

    def reuse(self, block: Block) -> None:
        """Remove a still-warm block from the eviction pool because it just
        gained a new owner (e.g. a PrefixCache hit already bumped its
        ref_count). Must be called before that reference is used, or
        allocate() could reclaim -- and reset -- the block out from under it.
        """
        self.evictable_blocks.pop(block.block_id, None)

    def allocate(self, prefix_cache=None) -> Block:
        if not self.free_blocks and self.evictable_blocks:
            victim = next(iter(self.evictable_blocks.values()))  # oldest = LRU
            assert victim.ref_count == 0, (
                f"block {victim.block_id} is in evictable_blocks but still "
                "referenced -- reuse() should have removed it"
            )
            if victim.hash_key is not None and prefix_cache is None:
                raise ValueError(
                    f"block {victim.block_id} is the LRU victim but still "
                    "registered in a PrefixCache -- call allocate(prefix_cache=...) "
                    "so it can be evicted before reuse"
                )
            del self.evictable_blocks[victim.block_id]
            if victim.hash_key is not None:
                prefix_cache.evict(victim)
            victim.reset()
            self.free_blocks.append(victim)
        return super().allocate()

    def can_allocate(self, num_tokens: int) -> bool:
        needed = math.ceil(num_tokens / self.block_size)
        return len(self.free_blocks) + len(self.evictable_blocks) >= needed


def run_checks() -> None:
    allocator = LRUBlockAllocator(num_blocks=4, block_size=16)
    prefix_cache = PrefixCache()

    blocks = [allocator.allocate() for _ in range(4)]
    blocks[0].num_tokens = 16  # pretend this block got filled with a prompt
    assert len(allocator.free_blocks) == 0

    prefix_cache.insert_block(list(range(16)), blocks[0])
    allocator.free(blocks[0])
    assert blocks[0].block_id in allocator.evictable_blocks
    assert blocks[0].ref_count == 0

    # A later request's prefix matches block 0 while it's still warm.
    matched, _ = prefix_cache.match_prefix(list(range(16)), block_size=16)
    assert matched == [blocks[0]]
    assert blocks[0].ref_count == 1

    # reuse() must remove it from the eviction pool in the same step --
    # otherwise, with all 4 blocks now "in use" (0 reused, 1/2/3 never
    # freed), allocate() could still hand block 0 out a second time.
    allocator.reuse(blocks[0])
    assert blocks[0].block_id not in allocator.evictable_blocks
    assert blocks[0].num_tokens == 16  # content preserved, not reset

    # Proof: every block is now live and nothing is evictable, so the pool
    # is genuinely exhausted -- allocate() must raise, not silently hand
    # block 0 back out to a second owner.
    try:
        allocator.allocate()
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError -- every block is in use")

    # Free block 0 again -- warm once more -- and free block 1 too.
    allocator.free(blocks[0])
    allocator.free(blocks[1])
    assert len(allocator.evictable_blocks) == 2

    # No free blocks, but two evictable ones: allocate() reclaims the LRU
    # victim (block 0, freed first) instead of raising, and evicts its
    # stale PrefixCache entry before handing it back out.
    reused = allocator.allocate(prefix_cache=prefix_cache)
    assert reused.block_id == blocks[0].block_id
    assert reused.ref_count == 1
    assert reused.num_tokens == 0  # actually reclaimed this time, content gone
    matched_after, _ = prefix_cache.match_prefix(list(range(16)), block_size=16)
    assert matched_after == []  # evicted, no longer a valid hit

    # Block 1 (never cached) is still evictable: one more allocate() clears
    # the pool completely.
    allocator.allocate()
    assert len(allocator.evictable_blocks) == 0
    assert len(allocator.free_blocks) == 0

    # Nothing free, nothing evictable: allocate() must still raise.
    try:
        allocator.allocate()
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError when the pool is truly exhausted")

    print("All checks passed.")


if __name__ == "__main__":
    run_checks()
