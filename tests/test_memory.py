"""Checks mini_inference.memory against the numbers printed in Chapter 1,
Section 1.3.2 - if these fail, the book and the code have drifted apart.
"""

import pytest

from mini_inference.memory import (
    LLAMA2_7B,
    bytes_to_gb,
    bytes_to_mb,
    kv_cache_bytes_for_concurrent_requests,
    kv_cache_bytes_for_request,
    kv_cache_bytes_per_token,
)


def test_per_token_is_about_half_a_megabyte():
    assert bytes_to_mb(kv_cache_bytes_per_token(LLAMA2_7B)) == pytest.approx(0.5, abs=0.03)


def test_one_user_2000_tokens_is_about_1gb():
    size = kv_cache_bytes_for_request(LLAMA2_7B, num_tokens=2_000)
    assert bytes_to_gb(size) == pytest.approx(1.0, abs=0.1)


def test_fifty_concurrent_users_is_about_50gb():
    size = kv_cache_bytes_for_concurrent_requests(LLAMA2_7B, num_tokens=2_000, num_requests=50)
    assert bytes_to_gb(size) == pytest.approx(50.0, abs=5.0)


def test_one_user_32k_context_is_about_16gb():
    size = kv_cache_bytes_for_request(LLAMA2_7B, num_tokens=32_000)
    assert bytes_to_gb(size) == pytest.approx(16.0, abs=1.0)


def test_gqa_model_uses_fewer_kv_heads_than_query_heads():
    from mini_inference.memory import ModelShape

    mha = ModelShape(num_layers=32, num_heads=32, head_dim=128)
    gqa = ModelShape(num_layers=32, num_heads=32, head_dim=128, num_kv_heads=8)

    assert kv_cache_bytes_per_token(gqa) == kv_cache_bytes_per_token(mha) / 4
