#!/usr/bin/env python3
"""Solution: LRU eviction for BlockAllocator (Chapter 3, Section 3.3).

See ch03/exercises/exercise_block_manager.py for the problem statement.
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
        self.evictable_blocks: "collections.OrderedDict[int, Block]" = collections.OrderedDict()

    def free(self, block: Block) -> None:
        if block.ref_count > 0:
            block.ref_count -= 1
        if block.ref_count == 0:
            self.evictable_blocks[block.block_id] = block
            self.evictable_blocks.move_to_end(block.block_id)

    def touch(self, block: Block) -> None:
        if block.block_id in self.evictable_blocks:
            self.evictable_blocks.move_to_end(block.block_id)

    def allocate(self) -> Block:
        if not self.free_blocks and self.evictable_blocks:
            _, victim = self.evictable_blocks.popitem(last=False)  # oldest = LRU
            victim.reset()
            self.free_blocks.append(victim)
        return super().allocate()


def run_checks() -> None:
    allocator = LRUBlockAllocator(num_blocks=4, block_size=16)

    blocks = [allocator.allocate() for _ in range(4)]
    assert len(allocator.free_blocks) == 0

    allocator.free(blocks[0])
    assert len(allocator.free_blocks) == 0
    assert blocks[0].block_id in allocator.evictable_blocks

    reused = allocator.allocate()
    assert reused.block_id == blocks[0].block_id
    assert reused.ref_count == 1
    assert reused.num_tokens == 0
    assert len(allocator.evictable_blocks) == 0

    try:
        allocator.allocate()
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError when the pool is truly exhausted")

    allocator.free(blocks[1])
    allocator.free(blocks[2])
    allocator.touch(blocks[1])  # blocks[2] is now the least-recently-used
    evicted = allocator.allocate()
    assert evicted.block_id == blocks[2].block_id

    print("All checks passed.")


if __name__ == "__main__":
    run_checks()
