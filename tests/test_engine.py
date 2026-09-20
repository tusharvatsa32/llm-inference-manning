from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from mini_inference.engine import (
    Decoder,
    InferenceRequest,
    RequestFinished,
    TokenOutput,
)
from mini_inference.runner import ModelStepOutput
from mini_inference.sampling import SamplingParams


def peaked_logits(token_id: int, vocab_size: int = 8) -> torch.Tensor:
    logits = torch.full((vocab_size,), -10.0)
    logits[token_id] = 10.0
    return logits


@dataclass(frozen=True)
class FakeState:
    position: int


class ScriptedRunner:
    device = torch.device("cpu")

    def __init__(self, token_ids: list[int], eos_token_ids=()):
        self.logits = [peaked_logits(token_id) for token_id in token_ids]
        self.eos_token_ids = frozenset(eos_token_ids)
        self.prefill_calls = []
        self.decode_calls = []

    def prefill(self, prompt_token_ids):
        self.prefill_calls.append(tuple(prompt_token_ids))
        return ModelStepOutput(self.logits[0], FakeState(position=0))

    def decode(self, token_id, state):
        self.decode_calls.append((token_id, state))
        position = state.position + 1
        return ModelStepOutput(self.logits[position], FakeState(position=position))


class RecordingSampler:
    def __init__(self):
        self.generators = []

    def select(self, logits, params, generated_token_ids, generator):
        self.generators.append(generator)
        return len(self.generators)


def make_request(**overrides):
    values = {
        "request_id": "request-1",
        "prompt_token_ids": (3, 4),
        "sampling": SamplingParams(temperature=0),
        "max_new_tokens": 3,
    }
    values.update(overrides)
    return InferenceRequest(**values)


def test_length_limited_stream_emits_tokens_then_one_terminal_event():
    runner = ScriptedRunner([1, 2, 3])

    events = list(Decoder(runner).generate(make_request()))

    assert events == [
        TokenOutput("request-1", 1),
        TokenOutput("request-1", 2),
        TokenOutput("request-1", 3),
        RequestFinished("request-1", "length"),
    ]
    assert runner.prefill_calls == [(3, 4)]
    assert [token_id for token_id, _ in runner.decode_calls] == [1, 2]


def test_eos_token_is_emitted_before_eos_terminal_event():
    runner = ScriptedRunner([1, 7, 3], eos_token_ids={7})

    events = list(Decoder(runner).generate(make_request()))

    assert events[-2:] == [
        TokenOutput("request-1", 7),
        RequestFinished("request-1", "eos"),
    ]
    assert [
        event.token_id for event in events if isinstance(event, TokenOutput)
    ] == [1, 7]
    assert len(runner.decode_calls) == 1


def test_request_stop_token_is_distinct_from_model_eos():
    runner = ScriptedRunner([5, 2, 3], eos_token_ids={7})

    events = list(
        Decoder(runner).generate(make_request(stop_token_ids=(5,), max_new_tokens=3))
    )

    assert events == [TokenOutput("request-1", 5), RequestFinished("request-1", "stop")]
    assert runner.decode_calls == []


def test_eos_precedes_stop_when_token_appears_in_both_sets():
    runner = ScriptedRunner([5], eos_token_ids={5})

    events = list(
        Decoder(runner).generate(make_request(stop_token_ids=(5,), max_new_tokens=1))
    )

    assert events[-1] == RequestFinished("request-1", "eos")


def test_seed_creates_one_generator_for_the_whole_request():
    runner = ScriptedRunner([1, 2, 3])
    sampler = RecordingSampler()
    request = make_request(sampling=SamplingParams(temperature=1, seed=19))

    list(Decoder(runner, sampler=sampler).generate(request))

    assert sampler.generators[0] is not None
    assert len({id(generator) for generator in sampler.generators}) == 1


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"request_id": ""}, ValueError, "request_id"),
        ({"request_id": 1}, TypeError, "request_id"),
        ({"prompt_token_ids": ()}, ValueError, "prompt_token_ids"),
        ({"prompt_token_ids": [1]}, TypeError, "prompt_token_ids"),
        ({"prompt_token_ids": (-1,)}, ValueError, "prompt_token_ids"),
        ({"prompt_token_ids": (1.5,)}, TypeError, "prompt_token_ids"),
        ({"sampling": None}, TypeError, "sampling"),
        ({"max_new_tokens": 0}, ValueError, "max_new_tokens"),
        ({"max_new_tokens": 1.5}, TypeError, "max_new_tokens"),
        ({"stop_token_ids": [-1]}, TypeError, "stop_token_ids"),
        ({"stop_token_ids": (-1,)}, ValueError, "stop_token_ids"),
    ],
)
def test_request_validation(overrides, exception, message):
    with pytest.raises(exception, match=message):
        make_request(**overrides)


def test_engine_source_has_no_text_or_tokenizer_boundary():
    source = Path("src/mini_inference/engine.py").read_text()

    assert "tokenizer" not in source
    assert "prompt: str" not in source
