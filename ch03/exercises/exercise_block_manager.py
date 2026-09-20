#!/usr/bin/env python3
"""Exercise (advanced, optional): LRU eviction for BlockAllocator (Chapter 3, Section 3.3).

This one is harder than a typical chapter exercise -- it asks you to reason
about reference-counting invariants across two collaborating structures
(the eviction pool and a PrefixCache), not just implement one self-contained
method. If you're working through this chapter for the first time, it's
fine to read the solution instead of solving it from scratch; the value is
in understanding *why* each piece is needed, particularly the aliasing bug
described below, which is a real and common class of bug in production
KV-cache allocators.

BlockAllocator.allocate() raises MemoryError the instant the free-block pool is
empty. But many freed blocks are not truly done: a block just released by a
finished request may still hold a system prompt another request is about to
reuse. Instead of forgetting them the moment ref_count hits 0, keep them
"warm" in an LRU structure, and only wipe the least-recently-used one when a
new allocation would otherwise fail.

The dangerous case this design has to get right: a warm block gets a new
owner (e.g. a PrefixCache hit bumps its ref_count), but nothing has told the
LRU pool that block is no longer eligible for eviction. If allocate() doesn't
check before reclaiming, it can hand that "in use" block to a second owner
while the first is still using it -- two requests silently sharing one
physical block with no ref_count reflecting it.

Fill in the TODOs below, then run this file directly:
    python3 ch03/exercises/exercise_block_manager.py
All asserts should pass.
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
    """

    def __init__(self, num_blocks: int, block_size: int):
        super().__init__(num_blocks, block_size)
        # TODO: create the structure that tracks freed-but-still-warm blocks in
        # least-recently-used order. An `collections.OrderedDict[int, Block]`
        # keyed by block_id works well: inserting/moving a key to the end
        # marks it "most recently used" (so the LRU victim is always the
        # first item in iteration order).
        self.evictable_blocks = ...  # TODO

    def free(self, block: Block, prefix_cache=None) -> None:
        """Decrement ref_count. Once it hits zero, do NOT reset/reclaim the
        block immediately -- instead, park it in `evictable_blocks` as the
        most-recently-used entry, so it can be reused as-is (e.g. by a prefix
        cache hit) before it gets wiped.

        `prefix_cache` is accepted only so the inherited free_sequence() can
        call this with the base class's signature -- a block freed here isn't
        reset yet, so there's nothing to evict from a cache until allocate()
        actually reclaims it.
        """
        # TODO: decrement block.ref_count (guard against going below zero).
        # TODO: if ref_count == 0, add/move `block` to the MRU end of
        # `evictable_blocks` instead of calling block.reset() or touching
        # `self.free_blocks`.
        raise NotImplementedError

    def reuse(self, block: Block) -> None:
        """Remove a still-warm block from the eviction pool because it just
        gained a new owner (e.g. a PrefixCache hit already bumped its
        ref_count). Must be called before that reference is used, or
        allocate() could reclaim -- and reset -- the block out from under it.
        """
        # TODO: remove block.block_id from `evictable_blocks` if present.
        # Do NOT touch block.ref_count here -- whatever gave this block a new
        # owner (e.g. PrefixCache.match_prefix()) already did that.
        raise NotImplementedError

    def allocate(self, prefix_cache=None) -> Block:
        """Pop a free block as usual. If none are free, evict the
        least-recently-used entry from `evictable_blocks` (oldest = LRU),
        reset it, and use it -- only raise MemoryError if both are empty.

        If the LRU victim is still registered in a PrefixCache (its
        hash_key is set), pass `prefix_cache` so the stale entry gets
        evicted before the block is reused.
        """
        # TODO: if `self.free_blocks` is empty and `evictable_blocks` is not:
        #   1. Peek at the LRU victim (the OLDEST entry) WITHOUT removing it
        #      yet -- `next(iter(self.evictable_blocks.values()))` works.
        #   2. Sanity-check `victim.ref_count == 0` (an `assert` is fine here:
        #      this should be impossible if reuse() is used correctly).
        #   3. If `victim.hash_key is not None` and `prefix_cache is None`,
        #      raise ValueError instead of silently going stale -- mirror the
        #      base class's free() error message.
        #   4. Only now remove the victim from `evictable_blocks`, evict it
        #      from `prefix_cache` if it had a hash_key, reset() it, and push
        #      it onto `self.free_blocks`.
        # Otherwise, let the MemoryError from the parent class propagate.
        return super().allocate()

    def can_allocate(self, num_tokens: int) -> bool:
        # TODO: like the base class, but count evictable_blocks as available
        # too -- they can be reclaimed just like free_blocks can.
        needed = math.ceil(num_tokens / self.block_size)
        return len(self.free_blocks) >= needed


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
