"""A toy block allocator: how PagedAttention removes allocation waste.

Section 3.3. SequenceBlocks (Listing 3.3) tracks which physical blocks belong
to one request and claims a new block only when the current one fills.
block_stats (Listing 3.4) counts the blocks a request needs and the slots
left empty in its final block.

The script walks through the chapter's examples:
    - one request growing past block boundaries (section 3.3.2)
    - two requests growing at once, so their blocks interleave (figure 3.10)
    - waste and block-table size for different block sizes (section 3.3.3)
    - static versus paged reservation (figure 3.8 and section 3.3.4)

Examples:
    python block_allocator.py
    python block_allocator.py --block-size 8 --tokens 33
"""

import argparse
import math

from capacity_planner import ServingConfig, WorkloadProfile, reserved_token_slots


# Listing 3.3: memory allocation in PagedAttention.
class SequenceBlocks:
    def __init__(self, block_size):
        self.block_size = block_size
        self.blocks = []
        self.length = 0

    def append_token(self, free_blocks):
        if self.length % self.block_size == 0:
            self.blocks.append(free_blocks.pop())

        self.length += 1


# Listing 3.4: block usage and wasted slots.
def block_stats(num_tokens: int, block_size: int):
    blocks = math.ceil(num_tokens / block_size)
    reserved_slots = blocks * block_size
    wasted_slots = reserved_slots - num_tokens
    return blocks, wasted_slots


def grow_one_request(block_size, num_tokens, pool_size):
    print(f"== One request growing to {num_tokens} tokens, block size {block_size}")
    free_blocks = list(range(pool_size))
    seq = SequenceBlocks(block_size)
    for token in range(num_tokens):
        before = len(seq.blocks)
        seq.append_token(free_blocks)
        if len(seq.blocks) > before:
            print(f"  token {token:>3}: claims physical block #{seq.blocks[-1]} "
                  f"-> block table {seq.blocks}")
    blocks, wasted = block_stats(num_tokens, block_size)
    print(f"  {seq.length} tokens in {blocks} blocks, {wasted} empty slots "
          f"in the last block\n")


def grow_two_requests(block_size, pool_size):
    print("== Two requests growing at the same time (figure 3.10)")
    # Shuffle the free list so physical blocks are scattered, as they are
    # once requests of different lengths have come and gone.
    free_blocks = [7, 2, 11, 4, 9, 0, 5, 13, 1, 8][:pool_size]
    free_blocks.reverse()  # pop() takes from the end
    a, b = SequenceBlocks(block_size), SequenceBlocks(block_size)
    for _ in range(40):
        a.append_token(free_blocks)
    for _ in range(20):
        b.append_token(free_blocks)
    for _ in range(10):
        a.append_token(free_blocks)
    for name, seq in (("A", a), ("B", b)):
        mapping = ", ".join(f"logical {i} -> #{p}" for i, p in enumerate(seq.blocks))
        print(f"  request {name}: {seq.length} tokens, {mapping}")
    print("  Each request's tokens stay in order through its block table, even")
    print("  though its physical blocks are not next to each other.\n")


def compare_block_sizes(num_tokens):
    print(f"== Waste for a {num_tokens}-token request (section 3.3.3)")
    for size in (8, 16, 32):
        blocks, wasted = block_stats(num_tokens, size)
        print(f"  block size {size:>2}: {blocks} blocks, {wasted:>2} wasted slots")
    print("\n== Block-table entries for an 8,192-token sequence")
    for size in (1, 16, 128, 512):
        blocks, wasted = block_stats(8192, size)
        print(f"  block size {size:>3}: {blocks:>5} entries")
    print()


def static_vs_paged():
    print("== Figure 3.8: request A, 20-token prompt, allowed to grow to 64")
    for used in (20, 30):
        workload = WorkloadProfile(used, 64, concurrency=1)
        static = reserved_token_slots(workload, ServingConfig(allocation="static"))
        paged = reserved_token_slots(workload, ServingConfig(allocation="paged"))
        print(f"  {used} tokens used: static reserves {static} slots "
              f"({static - used} empty), paged reserves {paged} ({paged - used} empty)")

    print("\n== Section 3.3.4: 32 requests at 201 tokens, allowed to grow to 2,048")
    workload = WorkloadProfile(201, 2048, concurrency=32)
    ideal = workload.concurrency * workload.current_tokens_per_request
    print(f"  ideal:  {ideal:>6,} token slots")
    for allocation in ("static", "paged"):
        slots = reserved_token_slots(workload, ServingConfig(allocation=allocation))
        print(f"  {allocation}: {slots:>6,} token slots reserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=33,
                        help="tokens for the growing request (default: 33)")
    args = parser.parse_args()

    pool_size = math.ceil(args.tokens / args.block_size) + 8
    grow_one_request(args.block_size, args.tokens, pool_size)
    grow_two_requests(16, 10)
    compare_block_sizes(args.tokens)
    static_vs_paged()


if __name__ == "__main__":
    main()
