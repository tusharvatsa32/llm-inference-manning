"""Block-level prefix caching (Chapter 3, Section 3.5).

Blocks are chained by hashing each block's tokens together with its parent
block's hash, so a cache hit on block N implies every block before it also
matched. This mirrors the radix/prefix-tree lookup described in the text
without needing an actual tree structure.
"""

from .block_allocator import Block

_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK_64 = 0xFFFFFFFFFFFFFFFF


def hash_block(token_ids: list[int] | tuple[int, ...], parent_hash: int | None = None) -> int:
    """Deterministic 64-bit FNV-1a hash over token_ids, chained with parent_hash."""
    h = _FNV_OFFSET_BASIS
    if parent_hash is not None:
        h = ((h ^ (parent_hash & _MASK_64)) * _FNV_PRIME) & _MASK_64
    for token_id in token_ids:
        h = ((h ^ (token_id & _MASK_64)) * _FNV_PRIME) & _MASK_64
    return h


class PrefixCache:
    """Tracks which physical blocks hold a previously-seen token prefix."""

    def __init__(self):
        self.cached_blocks: dict[int, Block] = {}

    def match_prefix(
        self, token_ids: list[int], block_size: int
    ) -> tuple[list[Block], list[int]]:
        """Find the longest cached prefix of `token_ids`, in `block_size` chunks.

        Every matched block already has its ref_count bumped by 1 for the
        caller -- attach the returned blocks straight to a BlockTable's
        `.blocks` list (`table.blocks.extend(matched)`), not through
        `BlockTable.append_block()`, which would bump ref_count a second time.
        """
        matched_blocks: list[Block] = []
        parent_hash: int | None = None
        idx = 0
        num_tokens = len(token_ids)

        while idx + block_size <= num_tokens:
            chunk = tuple(token_ids[idx : idx + block_size])
            block_hash = hash_block(chunk, parent_hash)
            block = self.cached_blocks.get(block_hash)
            if block is None:
                break
            block.ref_count += 1
            matched_blocks.append(block)
            parent_hash = block_hash
            idx += block_size

        remaining_tokens = list(token_ids[idx:])
        return matched_blocks, remaining_tokens

    def insert_block(
        self,
        token_ids: list[int] | tuple[int, ...],
        block: Block,
        parent_hash: int | None = None,
    ) -> int:
        block_hash = hash_block(tuple(token_ids), parent_hash)
        block.hash_key = block_hash
        self.cached_blocks[block_hash] = block
        return block_hash

    def evict(self, block: Block) -> None:
        if block.hash_key is not None:
            # Identity-checked: if a later insert_block() for the same
            # content overwrote this hash with a different, still-live
            # block, evicting `block` must not delete that other block's
            # entry out from under it.
            if self.cached_blocks.get(block.hash_key) is block:
                del self.cached_blocks[block.hash_key]
            block.hash_key = None
