"""PagedAttention block allocation (Chapter 3, Section 3.3).

Ownership model used throughout this module: `Block.ref_count` counts how many
`BlockTable`s currently point at that physical block.

- `BlockAllocator.allocate()` mints a block with `ref_count = 1`: the caller is
  its sole owner. Building a private sequence (`allocate_sequence`,
  `append_token`) attaches these freshly minted blocks straight into a
  table's block list, since that "1" already accounts for the new owner.
- `BlockTable.append_block()` is for the sharing path: attaching a block that
  some other table already owns (for example, a prefix-cache hit), which
  makes this table an additional owner and bumps `ref_count`.
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


class BlockTable:
    """The logical sequence of physical blocks assigned to one request."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = []

    def append_block(self, block: Block) -> None:
        self.blocks.append(block)
        block.ref_count += 1

    def release_all(self):
        """Decrement ref_count on every tracked block, yielding those that hit zero.

        This is a generator: it must be fully consumed (e.g. `list(...)`) for
        the table's block list to be cleared.
        """
        for block in self.blocks:
            block.ref_count -= 1
            if block.ref_count == 0:
                yield block
        self.blocks = []

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

    def free(self, block: Block) -> None:
        # Guarded rather than unconditional so this also serves as the reclaim
        # step for blocks BlockTable.release_all() already decremented to zero.
        if block.ref_count > 0:
            block.ref_count -= 1
        if block.ref_count == 0:
            block.reset()
            self.free_blocks.append(block)

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
        or shared (ref_count > 1, meaning writing into it would corrupt another
        request's cache and a copy-on-write is required).
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
