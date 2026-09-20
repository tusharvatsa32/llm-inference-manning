"""The Chapter 2 single-request inference engine.

Text conversion is deliberately outside this module.  The engine consumes
token IDs, emits token events as generation proceeds, and ends every stream
with one explicit finish event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Protocol, TypeAlias

import torch

from .runner import ModelRunner
from .sampling import Sampler, SamplingParams

FinishReason = Literal["eos", "length", "stop", "cancelled"]


@dataclass(frozen=True)
class InferenceRequest:
    """One tokenized request and its per-request generation controls."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams = field(default_factory=SamplingParams)
    max_new_tokens: int = 256
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError("request_id must be a string")
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not isinstance(self.prompt_token_ids, tuple):
            raise TypeError("prompt_token_ids must be a tuple")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must contain at least one token")
        _validate_token_ids("prompt_token_ids", self.prompt_token_ids)
        if not isinstance(self.sampling, SamplingParams):
            raise TypeError("sampling must be SamplingParams")
        if not isinstance(self.max_new_tokens, int) or isinstance(
            self.max_new_tokens, bool
        ):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if not isinstance(self.stop_token_ids, tuple):
            raise TypeError("stop_token_ids must be a tuple")
        _validate_token_ids("stop_token_ids", self.stop_token_ids)


@dataclass(frozen=True)
class TokenOutput:
    request_id: str
    token_id: int
    logprob: float | None = None


@dataclass(frozen=True)
class RequestFinished:
    request_id: str
    reason: FinishReason


EngineOutput: TypeAlias = TokenOutput | RequestFinished


class InferenceEngine(Protocol):
    """Chapter 2's synchronous streaming facade.

    Chapters 3 and 4 retain the request and event vocabulary while deciding
    the concurrent scheduling interface from concrete requirements.
    """

    def generate(self, request: InferenceRequest) -> Iterator[EngineOutput]: ...


class Decoder:
    """Generate one cached request at a time; no batching or scheduler yet."""

    def __init__(self, runner: ModelRunner, sampler: Sampler | None = None) -> None:
        self.runner = runner
        self.sampler = sampler or Sampler()

    def generate(self, request: InferenceRequest) -> Iterator[EngineOutput]:
        generator = _make_generator(
            request.sampling.seed,
            self.runner.device,
        )
        output_token_ids: list[int] = []
        runner_output = self.runner.prefill(request.prompt_token_ids)

        while True:
            token_id = self.sampler.select(
                runner_output.logits,
                request.sampling,
                generated_token_ids=output_token_ids,
                generator=generator,
            )
            output_token_ids.append(token_id)
            yield TokenOutput(request_id=request.request_id, token_id=token_id)

            finish_reason = self._finish_reason(
                request,
                generated_tokens=len(output_token_ids),
                latest_token_id=token_id,
            )
            if finish_reason is not None:
                yield RequestFinished(
                    request_id=request.request_id,
                    reason=finish_reason,
                )
                return

            runner_output = self.runner.decode(token_id, runner_output.state)

    def _finish_reason(
        self,
        request: InferenceRequest,
        generated_tokens: int,
        latest_token_id: int,
    ) -> FinishReason | None:
        # Model EOS takes precedence if a token also appears in the request's
        # stop set. This keeps model termination distinguishable at the API.
        if latest_token_id in self.runner.eos_token_ids:
            return "eos"
        if latest_token_id in request.stop_token_ids:
            return "stop"
        if generated_tokens >= request.max_new_tokens:
            return "length"
        return None


def _make_generator(
    seed: int | None, device: torch.device
) -> torch.Generator | None:
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _validate_token_ids(name: str, token_ids: tuple[int, ...]) -> None:
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise TypeError(f"{name} must contain integers")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError(f"{name} must be greater than or equal to 0")
