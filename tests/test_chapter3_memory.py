import importlib.util
import math
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    config = ServingConfig(gpu_memory_gib=80.0, gpu_utilization=0.90)

    report = estimate_capacity(llama3_8b, workload, config)

    assert isinstance(report, CapacityReport)
    assert 14.0 <= report.weight_gb <= 16.0
    assert 55.0 <= report.kv_budget_gb <= 58.0
    assert report.per_token_bytes == 131_072
    # ~467k tokens as computed by hand in Section 3.7.1
    assert 400_000 <= report.token_pool <= 500_000
    # At an 8K active context, roughly 56 concurrent requests fit.
    assert 40 <= report.max_concurrency <= 70


def test_estimate_capacity_accepts_measured_overrides():
    model = ModelProfile(layers=32, kv_heads=8, head_dim=128)
    workload = WorkloadProfile(
        current_tokens_per_request=8192, max_tokens_per_request=8192, concurrency=1
    )
    config = ServingConfig()

    # A caller with real numbers from the serving stack should be able to
    # skip the params_b-based first-pass estimate entirely.
    report = estimate_capacity(
        model, workload, config, weight_gb=20.0, available_kv_gb=40.0
    )

    assert report.weight_gb == 20.0
    assert report.kv_budget_gb == 40.0
    per_token = bytes_per_token(model, config)
    assert report.token_pool == int(40.0 * 1024**3 / per_token)


def test_reserved_token_slots_rejects_unknown_allocation():
    workload = WorkloadProfile(
        current_tokens_per_request=100, max_tokens_per_request=100, concurrency=1
    )
    config = ServingConfig(allocation="bogus")

    try:
        reserved_token_slots(workload, config)
    except ValueError:
        pass
    else:
        raise AssertionError("expected reserved_token_slots to reject a bad allocation")


def test_can_allocate_reflects_free_pool_size():
    allocator = BlockAllocator(num_blocks=4, block_size=16)

    assert allocator.can_allocate(64) is True  # exactly 4 blocks
    assert allocator.can_allocate(65) is False  # needs a 5th block

    allocator.allocate()
    assert allocator.can_allocate(48) is True  # exactly the 3 remaining
    assert allocator.can_allocate(49) is False


def test_allocate_raises_when_pool_exhausted():
    allocator = BlockAllocator(num_blocks=1, block_size=16)
    allocator.allocate()

    try:
        allocator.allocate()
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError once the pool is empty")


def test_allocate_sequence_does_not_leak_on_partial_failure():
    allocator = BlockAllocator(num_blocks=2, block_size=16)

    # 3 blocks needed, only 2 exist: must raise before claiming any of them,
    # not claim 2 and then raise with no table to return them through.
    try:
        allocator.allocate_sequence(num_tokens=33)
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError for a request that can't fit")

    assert len(allocator.free_blocks) == 2  # nothing claimed, nothing leaked


def test_append_token_without_cow_extends_the_same_block():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    table = allocator.allocate_sequence(num_tokens=1)
    tail = table.blocks[0]

    result = allocator.append_token(table)

    # Tail has room and isn't shared: append_token should just grow it in
    # place, not allocate a new block.
    assert result is None
    assert len(table.blocks) == 1
    assert tail.num_tokens == 2


def test_match_prefix_stops_at_the_first_divergent_block():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    prefix_cache = PrefixCache()

    cached_prefix = list(range(32))  # two blocks
    table = allocator.allocate_sequence(num_tokens=32)
    parent_hash = None
    for i, block in enumerate(table.blocks):
        chunk = cached_prefix[i * 16 : (i + 1) * 16]
        parent_hash = prefix_cache.insert_block(chunk, block, parent_hash)

    # A request whose first block matches but second block diverges should
    # get exactly one matched block back, with everything from there on
    # treated as uncached -- not a total miss, not a total hit.
    requested = cached_prefix[:16] + list(range(1000, 1016)) + [9999]
    matched, remaining = prefix_cache.match_prefix(requested, block_size=16)

    assert matched == [table.blocks[0]]
    assert remaining == requested[16:]


def test_estimate_capacity_clamps_when_weights_exceed_budget():
    # Llama-3-70B-sized weights on a single 80 GiB GPU at 90% utilization:
    # 70.6B params x 2 bytes = ~131.5 GiB of weights alone, more than the
    # ~72 GiB usable -- this must report "doesn't fit" cleanly, not a
    # negative token pool and negative concurrency.
    llama3_70b = ModelProfile(layers=80, kv_heads=8, head_dim=128, params_b=70.6)
    workload = WorkloadProfile(
        current_tokens_per_request=4096, max_tokens_per_request=4096, concurrency=32
    )
    config = ServingConfig(gpu_memory_gib=80.0, gpu_utilization=0.90)

    report = estimate_capacity(llama3_70b, workload, config)

    assert report.kv_budget_gb == 0.0
    assert report.token_pool == 0
    assert report.max_concurrency == 0


def test_evict_does_not_remove_a_different_blocks_entry():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    prefix_cache = PrefixCache()
    tokens = list(range(16))

    block_a = allocator.allocate()
    block_a.num_tokens = 16
    prefix_cache.insert_block(tokens, block_a)

    # Same content, a second block: insert_block() overwrites the hash's
    # entry to point at block_b instead of block_a.
    block_b = allocator.allocate()
    block_b.num_tokens = 16
    prefix_cache.insert_block(tokens, block_b)
    assert prefix_cache.cached_blocks[block_a.hash_key] is block_b

    # Evicting block_a (whose entry was already overwritten) must not
    # delete block_b's still-live entry out from under it.
    prefix_cache.evict(block_a)
    assert block_a.hash_key is None
    matched, _ = prefix_cache.match_prefix(tokens, block_size=16)
    assert matched == [block_b]


def test_solution_block_manager_run_checks():
    # The answer-key file has no import path of its own into the test suite,
    # which is exactly how the two Critical bugs in it shipped unnoticed --
    # run its own self-check under pytest so a regression here fails CI.
    solution = _load_module(
        REPO_ROOT / "ch03" / "solutions" / "solution_block_manager.py"
    )
    solution.run_checks()
