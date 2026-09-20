import math

from mini_inference.memory import (
    Block,
    BlockAllocator,
    BlockTable,
    CapacityReport,
    ModelProfile,
    PrefixCache,
    ServingConfig,
    WorkloadProfile,
    active_tokens,
    bytes_per_token,
    estimate_capacity,
    reserved_token_slots,
)


def test_bytes_per_token_exact_formula():
    llama3_8b = ModelProfile(layers=32, kv_heads=8, head_dim=128)
    config = ServingConfig(kv_dtype_bytes=2)

    per_token = bytes_per_token(llama3_8b, config)

    assert per_token == 131_072
    assert per_token == 128 * 1024


def test_paged_allocation_eliminates_waste():
    allocator = BlockAllocator(num_blocks=256, block_size=16)

    table = allocator.allocate_sequence(num_tokens=33)

    assert len(table.blocks) == 3
    reserved_slots = len(table.blocks) * allocator.block_size
    assert reserved_slots == 48
    wasted_slots = reserved_slots - 33
    assert wasted_slots == 15

    workload = WorkloadProfile(
        current_tokens_per_request=33, max_tokens_per_request=2048, concurrency=1
    )
    static_config = ServingConfig(allocation="static")
    paged_config = ServingConfig(allocation="paged", block_size=16)

    assert reserved_token_slots(workload, static_config) == 2048
    assert reserved_token_slots(workload, paged_config) == 48
    assert reserved_token_slots(workload, paged_config) < reserved_token_slots(
        workload, static_config
    )


def test_block_table_ref_counting_and_free():
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    free_at_start = len(allocator.free_blocks)

    table_a = allocator.allocate_sequence(num_tokens=20)
    assert len(table_a.blocks) == 2
    assert all(block.ref_count == 1 for block in table_a.blocks)
    assert len(allocator.free_blocks) == free_at_start - 2

    shared_block = table_a.blocks[0]
    table_b = BlockTable(block_size=16)
    table_b.append_block(shared_block)
    assert shared_block.ref_count == 2

    allocator.free_sequence(table_b)
    assert shared_block.ref_count == 1
    assert table_b.blocks == []
    assert len(allocator.free_blocks) == free_at_start - 2  # still in use by table_a

    allocator.free_sequence(table_a)
    assert table_a.blocks == []
    assert len(allocator.free_blocks) == free_at_start


def test_free_is_safe_against_double_free():
    allocator = BlockAllocator(num_blocks=2, block_size=16)
    block = allocator.allocate()
    assert len(allocator.free_blocks) == 1  # the other block, never allocated

    allocator.free(block)
    assert len(allocator.free_blocks) == 2

    # Freeing an already-free block must be a no-op, not a second reclaim --
    # otherwise the same physical block ends up duplicated in free_blocks and
    # can be handed out to two different owners at once.
    allocator.free(block)
    assert len(allocator.free_blocks) == 2

    first = allocator.allocate()
    second = allocator.allocate()
    assert first.block_id != second.block_id


def test_free_without_prefix_cache_raises_for_a_cached_block():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    prefix_cache = PrefixCache()

    table = allocator.allocate_sequence(num_tokens=16)
    block = table.blocks[0]
    prefix_cache.insert_block(list(range(16)), block)

    # Forgetting prefix_cache=... here must fail loudly, not silently corrupt
    # the cache -- see test_free_evicts_stale_prefix_cache_entry for the fix.
    try:
        allocator.free(block)
    except ValueError:
        pass
    else:
        raise AssertionError("expected free() to reject a still-cached block")

    # The rejected call must not have mutated anything: the block is still
    # exactly as it was, so evicting by hand and retrying must now succeed.
    assert block.ref_count == 1
    assert block not in allocator.free_blocks
    prefix_cache.evict(block)
    allocator.free(block)
    assert block in allocator.free_blocks


def test_free_sequence_is_all_or_nothing():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    prefix_cache = PrefixCache()

    table = allocator.allocate_sequence(num_tokens=32)  # two blocks
    uncached_block, cached_block = table.blocks
    prefix_cache.insert_block(list(range(16, 32)), cached_block)

    # One block in the table needs prefix_cache=... and it wasn't given.
    # free_sequence() must reject the whole call up front rather than
    # freeing uncached_block and then raising on cached_block -- otherwise
    # the table would keep a stale reference to a block the pool already
    # handed to someone else.
    try:
        allocator.free_sequence(table)
    except ValueError:
        pass
    else:
        raise AssertionError("expected free_sequence() to reject a still-cached block")

    assert uncached_block.ref_count == 1
    assert uncached_block not in allocator.free_blocks
    assert table.blocks == [uncached_block, cached_block]

    allocator.free_sequence(table, prefix_cache=prefix_cache)
    assert table.blocks == []
    assert uncached_block in allocator.free_blocks
    assert cached_block in allocator.free_blocks


def test_free_evicts_stale_prefix_cache_entry():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    prefix_cache = PrefixCache()

    table = allocator.allocate_sequence(num_tokens=16)
    block = table.blocks[0]
    block_hash = prefix_cache.insert_block(list(range(16)), block)
    assert block_hash in prefix_cache.cached_blocks

    allocator.free_sequence(table, prefix_cache=prefix_cache)

    # The freed block must no longer be reachable through the prefix cache --
    # otherwise a later request could get a "hit" on a block now owned by,
    # and possibly already overwritten by, an unrelated request.
    assert block_hash not in prefix_cache.cached_blocks
    matched, remaining = prefix_cache.match_prefix(list(range(16)), block_size=16)
    assert matched == []
    assert remaining == list(range(16))


def test_prefix_caching_hit_and_cow():
    allocator = BlockAllocator(num_blocks=64, block_size=16)
    prefix_cache = PrefixCache()
    shared_prefix = list(range(32))

    table_a = allocator.allocate_sequence(num_tokens=32)
    parent_hash = None
    for i, block in enumerate(table_a.blocks):
        chunk = shared_prefix[i * 16 : (i + 1) * 16]
        parent_hash = prefix_cache.insert_block(chunk, block, parent_hash)

    matched_blocks, remaining_tokens = prefix_cache.match_prefix(
        shared_prefix, block_size=16
    )
    assert remaining_tokens == []
    assert matched_blocks == table_a.blocks
    assert all(block.ref_count == 2 for block in matched_blocks)

    table_b = BlockTable(block_size=16)
    table_b.blocks.extend(matched_blocks)

    new_block = allocator.append_token(table_a)

    assert new_block is not None
    assert len(table_a.blocks) == 3
    assert table_a.blocks[-1] is new_block
    assert new_block.ref_count == 1
    assert table_a.blocks[1].ref_count == 2


def test_sliding_window_capping():
    workload_short = WorkloadProfile(
        current_tokens_per_request=2000, max_tokens_per_request=4096, concurrency=1
    )
    workload_long = WorkloadProfile(
        current_tokens_per_request=10_000, max_tokens_per_request=32_000, concurrency=1
    )

    full_attention_model = ModelProfile(layers=32, kv_heads=8, head_dim=128)
    sliding_window_model = ModelProfile(
        layers=32, kv_heads=8, head_dim=128, sliding_window=4096
    )

    assert active_tokens(full_attention_model, workload_long) == 10_000
    assert active_tokens(sliding_window_model, workload_short) == 2000
    assert active_tokens(sliding_window_model, workload_long) == 4096


def test_capacity_report_token_pool():
    llama3_8b = ModelProfile(layers=32, kv_heads=8, head_dim=128, params_b=8.03)
    workload = WorkloadProfile(
        current_tokens_per_request=8192, max_tokens_per_request=8192, concurrency=1
    )
    config = ServingConfig(gpu_memory_gb=80.0, gpu_utilization=0.90)

    report = estimate_capacity(llama3_8b, workload, config)

    assert isinstance(report, CapacityReport)
    assert 14.0 <= report.weight_gb <= 16.0
    assert 55.0 <= report.kv_budget_gb <= 58.0
    assert report.per_token_bytes == 131_072
    # ~467k tokens as computed by hand in Section 3.7.1
    assert 400_000 <= report.token_pool <= 500_000
    # At an 8K active context, roughly 56 concurrent requests fit.
    assert 40 <= report.max_concurrency <= 70
