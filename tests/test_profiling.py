import pytest

from mini_inference.profiling import (
    InferenceProfile,
    ProfileResult,
    TransformerFlops,
    estimate_decode_flops,
    estimate_prefill_flops,
)


def test_profile_reports_first_token_latency_tpot_and_output_throughput():
    profile = InferenceProfile(
        prompt_tokens=10,
        output_tokens=5,
        prefill_to_first_token_seconds=0.2,
        decode_seconds=0.4,
    )

    assert profile.prefill_to_first_token_seconds == 0.2
    assert profile.tpot_seconds == pytest.approx(0.1)
    assert profile.total_seconds == pytest.approx(0.6)
    assert profile.output_tokens_per_second == pytest.approx(5 / 0.6)


def test_single_output_token_has_no_decode_time_per_token():
    profile = InferenceProfile(
        10, 1, prefill_to_first_token_seconds=0.2, decode_seconds=0.0
    )

    assert profile.tpot_seconds is None


def test_profile_result_groups_measurements_for_later_engine_stages():
    profile = InferenceProfile(
        10, 2, prefill_to_first_token_seconds=0.2, decode_seconds=0.1
    )
    result = ProfileResult(
        profile=profile,
        prefill_flops=TransformerFlops(100, 20),
        decode_flops=TransformerFlops(10, 2),
        kv_cache_bytes=1_024,
        generated_text="done",
    )

    assert result.profile is profile
    assert result.prefill_flops.total == 120
    assert result.generated_text == "done"


def test_doubling_prompt_length_quadruples_prefill_attention_flops():
    short = estimate_prefill_flops(
        num_parameters=100, num_layers=2, hidden_size=8, prompt_tokens=10
    )
    long = estimate_prefill_flops(
        num_parameters=100, num_layers=2, hidden_size=8, prompt_tokens=20
    )

    assert long.parameter_flops == 2 * short.parameter_flops
    assert long.attention_flops == 4 * short.attention_flops


def test_prefill_flops_match_a_worked_example():
    estimate = estimate_prefill_flops(
        num_parameters=100, num_layers=2, hidden_size=8, prompt_tokens=10
    )

    assert estimate.parameter_flops == 2_000
    assert estimate.attention_flops == 6_400
    assert estimate.total == 8_400


def test_decode_flops_match_a_worked_example():
    estimate = estimate_decode_flops(
        num_parameters=100,
        num_layers=2,
        hidden_size=8,
        prompt_tokens=10,
        output_tokens=3,
    )

    assert estimate.parameter_flops == 400
    assert estimate.attention_flops == 1_472
    assert estimate.total == 1_872


def test_decode_estimate_excludes_first_token_produced_by_prefill():
    estimate = estimate_decode_flops(
        num_parameters=100,
        num_layers=2,
        hidden_size=8,
        prompt_tokens=10,
        output_tokens=1,
    )

    assert estimate.total == 0