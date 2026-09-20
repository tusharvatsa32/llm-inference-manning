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

    released_from_b = list(table_b.release_all())
    assert released_from_b == []
    assert shared_block.ref_count == 1
    assert table_b.blocks == []

    released_from_a = list(table_a.release_all())
    assert len(released_from_a) == 2
    assert all(block.ref_count == 0 for block in released_from_a)

    for block in released_from_a:
        allocator.free(block)

    assert len(allocator.free_blocks) == free_at_start


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
