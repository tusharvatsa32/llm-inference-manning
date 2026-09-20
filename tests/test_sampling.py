import pytest
import torch

from mini_inference.sampling import (
    Sampler,
    SamplingParams,
    apply_repetition_penalty,
    min_p_filter,
    top_k_filter,
    top_p_filter,
    typical_filter,
)


def finite_token_ids(logits: torch.Tensor) -> set[int]:
    return set(torch.nonzero(torch.isfinite(logits), as_tuple=False).flatten().tolist())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"min_p": 1.1}, "min_p"),
        ({"typical_p": 0.0}, "typical_p"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
        ({"seed": -1}, "seed"),
    ],
)
def test_sampling_params_reject_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SamplingParams(**kwargs)


@pytest.mark.parametrize("kwargs", [{"top_k": 2.5}, {"seed": 1.5}])
def test_sampling_params_reject_non_integer_discrete_values(kwargs):
    with pytest.raises(TypeError):
        SamplingParams(**kwargs)


def test_top_k_clamps_to_the_vocabulary_size():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])

    assert finite_token_ids(top_k_filter(logits, top_k=99)) == {0, 1, 2, 3}
    assert finite_token_ids(top_k_filter(logits, top_k=2)) == {0, 1}


def test_top_k_keeps_exactly_k_candidates_when_boundary_scores_tie():
    logits = torch.tensor([3.0, 2.0, 2.0, 1.0])

    assert len(finite_token_ids(top_k_filter(logits, top_k=2))) == 2


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    logits = torch.log(torch.tensor([0.50, 0.30, 0.15, 0.05]))

    filtered = top_p_filter(logits, top_p=0.70)

    assert finite_token_ids(filtered) == {0, 1}


def test_min_p_uses_a_threshold_relative_to_the_best_token():
    logits = torch.log(torch.tensor([0.60, 0.07, 0.05, 0.01]))

    filtered = min_p_filter(logits, min_p=0.10)

    assert finite_token_ids(filtered) == {0, 1}


def test_typical_filter_keeps_at_least_one_token_and_removes_tail():
    logits = torch.tensor([3.0, 2.0, 1.0, -2.0, -4.0])

    filtered = typical_filter(logits, typical_p=0.60)
    kept = finite_token_ids(filtered)

    assert kept
    assert len(kept) < logits.numel()


def test_repetition_penalty_moves_seen_logits_away_from_selection():
    logits = torch.tensor([2.0, 1.0, -1.0])

    penalized = apply_repetition_penalty(logits, token_ids=[0, 2], penalty=2.0)

    assert penalized.tolist() == [1.0, 1.0, -2.0]
    assert logits.tolist() == [2.0, 1.0, -1.0]


def test_greedy_respects_repetition_penalty_and_allowed_mask():
    logits = torch.tensor([5.0, 4.0, 3.0])
    allowed = torch.tensor([True, True, False])

    token_id = Sampler().select(
        logits,
        SamplingParams(temperature=0.0, repetition_penalty=2.0),
        generated_token_ids=[0],
        allowed_token_mask=allowed,
    )

    assert token_id == 1


def test_seeded_sampling_is_reproducible_for_the_same_execution_path():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    params = SamplingParams(seed=17)
    first_generator = torch.Generator().manual_seed(params.seed)
    second_generator = torch.Generator().manual_seed(params.seed)
    sampler = Sampler()

    first = [
        sampler.select(logits, params, generator=first_generator) for _ in range(20)
    ]
    second = [
        sampler.select(logits, params, generator=second_generator) for _ in range(20)
    ]

    assert first == second


def test_sampler_requires_one_dimensional_logits():
    with pytest.raises(ValueError, match="vocab_size"):
        Sampler().select(torch.zeros(1, 3), SamplingParams(temperature=0.0))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_sampler_rejects_invalid_model_logits(invalid):
    with pytest.raises(ValueError, match="NaN or positive infinity"):
        Sampler().select(
            torch.tensor([0.0, invalid]), SamplingParams(temperature=0.0)
        )
