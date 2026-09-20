"""Model execution behind the Chapter 2 decoder.

The decoder only needs the next-token logits and an opaque model state.  The
Hugging Face implementation below owns ``past_key_values`` so Chapter 3 can
replace cache management without changing requests, sampling, or output
events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModelStepOutput:
    """Next-token logits plus the opaque state needed by the next step."""

    logits: Tensor
    state: object


class ModelRunner(Protocol):
    """The smallest execution seam needed by the single-request decoder."""

    device: torch.device
    eos_token_ids: frozenset[int]

    def prefill(self, prompt_token_ids: Sequence[int]) -> ModelStepOutput:
        """Process the whole prompt and return logits for its next token."""

    def decode(self, token_id: int, state: object) -> ModelStepOutput:
        """Process one new token using state returned by the previous call."""


@dataclass(frozen=True)
class _HuggingFaceState:
    past_key_values: Any
    attention_mask: Tensor


class HuggingFaceModelRunner:
    """Run a causal Hugging Face model with its built-in KV cache enabled.

    The tokenizer stays outside this class.  Callers may therefore use chat
    templates, plain prompts, or pre-tokenized workloads without changing the
    engine boundary.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device | None = None,
        eos_token_ids: int | Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.device = (
            torch.device(device) if device is not None else _model_device(model)
        )
        if device is not None:
            self.model.to(self.device)
        self.model.eval()
        self.eos_token_ids = _normalize_eos_token_ids(
            eos_token_ids if eos_token_ids is not None else _model_eos_token_ids(model)
        )

    def prefill(self, prompt_token_ids: Sequence[int]) -> ModelStepOutput:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must contain at least one token")

        input_ids = torch.tensor(
            [list(prompt_token_ids)], dtype=torch.long, device=self.device
        )
        attention_mask = torch.ones_like(input_ids)
        output = self._forward(input_ids=input_ids, attention_mask=attention_mask)
        return ModelStepOutput(
            logits=_last_token_logits(output.logits),
            state=_HuggingFaceState(
                past_key_values=output.past_key_values,
                attention_mask=attention_mask,
            ),
        )

    def decode(self, token_id: int, state: object) -> ModelStepOutput:
        if not isinstance(state, _HuggingFaceState):
            raise TypeError("state was not created by HuggingFaceModelRunner")
        if token_id < 0:
            raise ValueError("token_id must be greater than or equal to 0")

        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        one = torch.ones(
            (state.attention_mask.shape[0], 1),
            dtype=state.attention_mask.dtype,
            device=self.device,
        )
        attention_mask = torch.cat((state.attention_mask, one), dim=-1)
        output = self._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
        )
        return ModelStepOutput(
            logits=_last_token_logits(output.logits),
            state=_HuggingFaceState(
                past_key_values=output.past_key_values,
                attention_mask=attention_mask,
            ),
        )

    def _forward(self, **kwargs: object) -> Any:
        with torch.inference_mode():
            output = self.model(use_cache=True, **kwargs)
        if getattr(output, "past_key_values", None) is None:
            raise RuntimeError(
                "model did not return past_key_values with use_cache=True"
            )
        return output


def _last_token_logits(logits: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(
            "model logits must have shape [1, sequence_length, vocab_size]"
        )
    return logits[0, -1].float()


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _model_eos_token_ids(model: torch.nn.Module) -> int | Sequence[int] | None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        value = getattr(generation_config, "eos_token_id", None)
        if value is not None:
            return value
    config = getattr(model, "config", None)
    return getattr(config, "eos_token_id", None)


def _normalize_eos_token_ids(
    eos_token_ids: int | Sequence[int] | None,
) -> frozenset[int]:
    if eos_token_ids is None:
        return frozenset()
    values = (eos_token_ids,) if isinstance(eos_token_ids, int) else eos_token_ids
    normalized = frozenset(int(token_id) for token_id in values)
    if any(token_id < 0 for token_id in normalized):
        raise ValueError("eos_token_ids must be greater than or equal to 0")
    return normalized
