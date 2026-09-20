"""Token selection for the Chapter 2 inference engine.

The model produces one logit per vocabulary token. This module owns every
transformation between those logits and the selected token ID. Keeping that
work in one place makes sampling behavior consistent when later chapters add
batching and speculative decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    """Per-request controls for selecting the next token.

    Generation length and stopping conditions belong to ``InferenceRequest``;
    they are intentionally absent here.
    """

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    typical_p: float | None = None
    repetition_penalty: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError(
                "temperature must be a finite number greater than or equal to 0"
            )
        if self.top_k is not None:
            if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
                raise TypeError("top_k must be an integer when enabled")
            if self.top_k < 1:
                raise ValueError("top_k must be at least 1 when enabled")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if self.typical_p is not None and not 0 < self.typical_p <= 1:
            raise ValueError("typical_p must be in (0, 1]")
        if (
            not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise ValueError(
                "repetition_penalty must be a finite number greater than 0"
            )
        if self.seed is not None:
            if not isinstance(self.seed, int) or isinstance(self.seed, bool):
                raise TypeError("seed must be an integer when enabled")
            if self.seed < 0:
                raise ValueError("seed must be greater than or equal to 0")


def apply_repetition_penalty(
    logits: Tensor, token_ids: Sequence[int], penalty: float
) -> Tensor:
    """Penalize tokens that have already appeared in the generated output."""
    if penalty == 1.0 or not token_ids:
        return logits.clone()

    result = logits.clone()
    indices = torch.tensor(
        sorted(set(token_ids)), dtype=torch.long, device=logits.device
    )
    if torch.any(indices < 0) or torch.any(indices >= logits.numel()):
        raise ValueError("token_ids contains an ID outside the vocabulary")

    selected = result[indices]
    adjusted = torch.where(selected < 0, selected * penalty, selected / penalty)
    result[indices] = adjusted
    return result


def apply_allowed_token_mask(
    logits: Tensor, allowed_token_mask: Tensor | None
) -> Tensor:
    """Mask tokens that the current decoding constraint does not allow."""
    if allowed_token_mask is None:
        return logits.clone()
    if allowed_token_mask.dtype != torch.bool:
        raise TypeError("allowed_token_mask must be a boolean tensor")
    if allowed_token_mask.shape != logits.shape:
        raise ValueError("allowed_token_mask must have the same shape as logits")
    if not bool(allowed_token_mask.any().item()):
        raise ValueError("allowed_token_mask must allow at least one token")
    return logits.masked_fill(
        ~allowed_token_mask.to(device=logits.device), float("-inf")
    )


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Scale logits before truncation and sampling."""
    if temperature <= 0:
        raise ValueError("temperature scaling requires temperature greater than 0")
    return logits / temperature


def top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Keep the ``top_k`` highest-scoring tokens.

    ``SamplingParams`` can validate that ``top_k`` is positive, but only this
    function knows the vocabulary size. Values larger than the vocabulary are
    therefore clamped here.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    k = min(top_k, logits.numel())
    top_logits, top_indices = torch.topk(logits, k)
    return torch.full_like(logits, float("-inf")).scatter(
        0, top_indices, top_logits
    )


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """Keep the smallest high-probability set with mass at least ``top_p``."""
    if top_p == 1.0:
        return logits.clone()

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )
    remove = cumulative_probabilities > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(
        0, sorted_indices, sorted_logits
    )


def min_p_filter(logits: Tensor, min_p: float) -> Tensor:
    """Keep tokens whose probability is plausible relative to the best token."""
    if min_p == 0:
        return logits.clone()

    probabilities = torch.softmax(logits, dim=-1)
    threshold = min_p * probabilities.max()
    return logits.masked_fill(probabilities < threshold, float("-inf"))


def typical_filter(logits: Tensor, typical_p: float) -> Tensor:
    """Keep tokens whose information content is closest to the entropy."""
    if typical_p == 1.0:
        return logits.clone()

    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = torch.exp(log_probabilities)
    entropy_terms = torch.where(
        probabilities > 0,
        probabilities * log_probabilities,
        torch.zeros_like(probabilities),
    )
    entropy = -entropy_terms.sum()
    surprise_distance = torch.abs(-log_probabilities - entropy)
    _, sorted_indices = torch.sort(surprise_distance)
    sorted_probabilities = probabilities[sorted_indices]
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative_probabilities > typical_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    remove_in_vocab_order = torch.zeros_like(remove).scatter(0, sorted_indices, remove)
    return logits.masked_fill(remove_in_vocab_order, float("-inf"))


class Sampler:
    """Turn one request's vocabulary logits into its next token ID."""

    def prepare_logits(
        self,
        logits: Tensor,
        params: SamplingParams,
        generated_token_ids: Sequence[int] = (),
        allowed_token_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply penalties, constraints, temperature, and truncation in order."""
        self._validate_logits(logits)
        prepared = apply_repetition_penalty(
            logits, generated_token_ids, params.repetition_penalty
        )
        prepared = apply_allowed_token_mask(prepared, allowed_token_mask)

        # Greedy decoding still respects penalties and structural constraints.
        if params.temperature == 0:
            return prepared

        prepared = apply_temperature(prepared, params.temperature)
        if params.top_k is not None:
            prepared = top_k_filter(prepared, params.top_k)
        if params.top_p is not None:
            prepared = top_p_filter(prepared, params.top_p)
        if params.min_p is not None:
            prepared = min_p_filter(prepared, params.min_p)
        if params.typical_p is not None:
            prepared = typical_filter(prepared, params.typical_p)
        return prepared

    def select(
        self,
        logits: Tensor,
        params: SamplingParams,
        generated_token_ids: Sequence[int] = (),
        allowed_token_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> int:
        """Select one token from a one-dimensional vocabulary distribution."""
        prepared = self.prepare_logits(
            logits,
            params,
            generated_token_ids=generated_token_ids,
            allowed_token_mask=allowed_token_mask,
        )
        if params.temperature == 0:
            return int(torch.argmax(prepared).item())

        probabilities = torch.softmax(prepared, dim=-1)
        if not bool(torch.isfinite(probabilities).all().item()):
            raise ValueError("sampling produced a non-finite probability distribution")

        # PyTorch generators are device-specific. If a backend cannot create a
        # local generator, the engine supplies a CPU generator and we sample
        # the small categorical draw on CPU to preserve per-request RNG state.
        generator_device = getattr(generator, "device", probabilities.device)
        if (
            generator is not None
            and torch.device(generator_device).type != probabilities.device.type
        ):
            sampled = torch.multinomial(
                probabilities.to("cpu"), num_samples=1, generator=generator
            )
        else:
            sampled = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            )
        return int(sampled.item())

    @staticmethod
    def _validate_logits(logits: Tensor) -> None:
        if logits.ndim != 1:
            raise ValueError("logits must have shape [vocab_size]")
        if logits.numel() == 0:
            raise ValueError("logits must contain at least one vocabulary entry")
        if bool(torch.isnan(logits).any().item()) or bool(
            torch.isposinf(logits).any().item()
        ):
            raise ValueError("logits must not contain NaN or positive infinity")
        if not bool(torch.isfinite(logits).any().item()):
            raise ValueError("logits must contain at least one finite value")
