from types import SimpleNamespace

import pytest
import torch

from mini_inference.runner import HuggingFaceModelRunner


class TinyCachedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=[6, 7])
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache,
        past_key_values=None,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "use_cache": use_cache,
                "past_key_values": past_key_values,
            }
        )
        vocab_size = 10
        logits = torch.zeros((*input_ids.shape, vocab_size), device=input_ids.device)
        logits[..., int(input_ids[0, -1]) % vocab_size] = 4.0
        new_cache = ("cache", len(self.calls))
        return SimpleNamespace(logits=logits, past_key_values=new_cache)


def test_hugging_face_runner_prefills_then_decodes_with_cache():
    model = TinyCachedModel()
    runner = HuggingFaceModelRunner(model)

    prefill = runner.prefill([1, 2, 3])
    decoded = runner.decode(4, prefill.state)

    assert prefill.logits.shape == (10,)
    assert decoded.logits.shape == (10,)
    assert model.calls[0]["input_ids"].tolist() == [[1, 2, 3]]
    assert model.calls[0]["attention_mask"].tolist() == [[1, 1, 1]]
    assert model.calls[0]["past_key_values"] is None
    assert model.calls[1]["input_ids"].tolist() == [[4]]
    assert model.calls[1]["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert model.calls[1]["past_key_values"] == ("cache", 1)
    assert all(call["use_cache"] for call in model.calls)
    assert runner.eos_token_ids == frozenset({6, 7})


def test_runner_rejects_empty_prefill():
    with pytest.raises(ValueError, match="at least one"):
        HuggingFaceModelRunner(TinyCachedModel()).prefill([])


def test_explicit_eos_configuration_overrides_model_configuration():
    runner = HuggingFaceModelRunner(TinyCachedModel(), eos_token_ids=9)

    assert runner.eos_token_ids == frozenset({9})
