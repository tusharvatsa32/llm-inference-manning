"""A toy prefix cache built on the block allocator.

Section 3.5. The cache follows the design the chapter describes for vLLM:

    - The KV cache is split into fixed-size blocks (section 3.3).
    - Each full block is identified by a hash of its tokens chained with the
      previous block's hash, so a block's hash stands for the whole prefix
      up to and including it.
    - A new request walks its full blocks from the start of the prompt and
      reuses every block whose hash is already cached. The match stops at
      the first miss and at the last full block.
    - When a request finishes, its blocks stay cached. When the pool runs
      out, cached blocks that no running request uses are evicted, least
      recently used first, and the end of a prefix before its beginning.

Like the allocator, it tracks blocks only and stores no real keys or values.
Tokens are words, so the prompts stay readable.

Examples:
    python prefix_cache.py
    python prefix_cache.py --block-size 4
"""

import argparse
from collections import OrderedDict

from capacity_planner import PRESETS, ServingConfig, bytes_per_token


class PrefixCache:
    def __init__(self, num_blocks, block_size=16):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.cached = {}               # prefix hash -> physical block
        self.block_hash = {}           # physical block -> prefix hash
        self.ref_count = {}            # physical block -> running requests using it
        self.evictable = OrderedDict() # cached blocks no request uses, oldest first

    def prefix_hashes(self, tokens):
        """One chained hash per full block of tokens."""
        hashes, parent = [], None
        for start in range(0, len(tokens) - self.block_size + 1, self.block_size):
            parent = hash((parent, tuple(tokens[start:start + self.block_size])))
            hashes.append(parent)
        return hashes

    def admit(self, tokens):
        """Build a block table for a new request, reusing cached blocks.

        Returns the block table and how many prompt tokens were served
        from the cache. Every other token must be computed by prefill.
        """
        hashes = self.prefix_hashes(tokens)
        table = []

        # 1. Reuse the longest cached prefix, one full block at a time.
        for h in hashes:
            block = self.cached.get(h)
            if block is None:
                break
            self.evictable.pop(block, None)  # in use again
            self.ref_count[block] = self.ref_count.get(block, 0) + 1
            table.append(block)
        cached_tokens = len(table) * self.block_size

        # 2. Allocate blocks for the rest of the prompt. Full blocks are
        #    registered so later requests can reuse them.
        needed = -(-len(tokens) // self.block_size)  # ceiling division
        for i in range(len(table), needed):
            block = self.allocate()
            self.ref_count[block] = 1
            if i < len(hashes):
                self.cached[hashes[i]] = block
                self.block_hash[block] = hashes[i]
            table.append(block)

        return table, cached_tokens

    def allocate(self):
        if self.free_blocks:
            return self.free_blocks.pop()
        if self.evictable:
            block, _ = self.evictable.popitem(last=False)  # least recently used
            del self.cached[self.block_hash.pop(block)]
            return block
        raise MemoryError("no free or evictable KV cache blocks")

    def release(self, table):
        """A request finished. Keep its full blocks cached for reuse."""
        # Release the last block first, as vLLM does, so the end of a prefix
        # is evicted before its beginning. A prefix match must start at the
        # first block, so losing the first block would waste all the others.
        for block in reversed(table):
            self.ref_count[block] -= 1
            if self.ref_count[block] > 0:
                continue
            del self.ref_count[block]
            if block in self.block_hash:
                self.evictable[block] = None  # newest at the end
            else:
                self.free_blocks.append(block)  # partial blocks are not cached


def tokenize(text, vocab):
    """Word-level stand-in for a real tokenizer: one token per word."""
    return [vocab.setdefault(word, len(vocab)) for word in text.split()]


def run(cache, name, tokens, per_token_kb):
    table, cached = cache.admit(tokens)
    computed = len(tokens) - cached
    print(f"  {name}: {len(tokens):,} tokens, {cached:,} reused, {computed:,} computed "
          f"({computed * per_token_kb / 1024:,.1f} MB of KV cache built)")
    cache.release(table)
    return cached


def shared_system_prompt(block_size, per_token_kb):
    print("== Figure 3.15: three requests share a 2,000-token system prompt")
    vocab = {}
    system = " ".join(f"rule{i}" for i in range(2000))
    cache = PrefixCache(num_blocks=1000, block_size=block_size)
    for name, question in (("A", "What is our refund policy?"),
                           ("B", "How do I reset my password?"),
                           ("C", "Can I change my delivery address?")):
        run(cache, f"request {name}", tokenize(f"{system} {question}", vocab), per_token_kb)
    print()


def prompt_layout(block_size, per_token_kb):
    print("== Figure 3.16: put stable content first and variable content last")
    vocab = {}
    stable = " ".join(f"instruction{i}" for i in range(500))
    stable += " " + " ".join(f"tool{i}" for i in range(300))
    stable += " " + " ".join(f"document{i}" for i in range(1200))
    cache = PrefixCache(num_blocks=1000, block_size=block_size)

    print(" Good layout: stable prefix, then the question")
    for question in ("Summarize section two.", "Who signed the contract?"):
        run(cache, "request", tokenize(f"{stable} {question}", vocab), per_token_kb)

    print(" Bad layout: a timestamp at the top of the system prompt")
    for time in ("09:14:02", "09:14:07"):
        run(cache, "request", tokenize(f"Current time: {time} {stable} Who signed?", vocab),
            per_token_kb)
    print()


def eviction(block_size):
    print("== Eviction: cached blocks no request uses go first, oldest first")
    vocab = {}
    cache = PrefixCache(num_blocks=8, block_size=block_size)
    prompt_a = tokenize(" ".join(f"a{i}" for i in range(4 * block_size)), vocab)
    prompt_b = tokenize(" ".join(f"b{i}" for i in range(6 * block_size)), vocab)

    table, _ = cache.admit(prompt_a)
    cache.release(table)
    print(f"  A ran and finished: {len(cache.evictable)} cached blocks kept, "
          f"{len(cache.free_blocks)} free")
    table, _ = cache.admit(prompt_b)
    print(f"  B needs 6 blocks: takes the {8 - 4} free blocks and evicts "
          f"{6 - 4} of A's cached blocks")
    cache.release(table)
    table, cached = cache.admit(prompt_a)
    print(f"  A again: {cached} of {len(prompt_a)} tokens reused. B evicted A's last "
          f"blocks, so A's first blocks still match.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    per_token_kb = bytes_per_token(PRESETS["llama-3-8b"], ServingConfig()) / 1024
    print(f"KV cache sizes below use Llama 3 8B at FP16 ({per_token_kb:.0f} KB per token).\n")
    shared_system_prompt(args.block_size, per_token_kb)
    prompt_layout(args.block_size, per_token_kb)
    eviction(args.block_size)


if __name__ == "__main__":
    main()
