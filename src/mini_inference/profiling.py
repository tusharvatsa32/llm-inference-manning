"""Small inference measurements and estimates used by the Chapter 1 profiler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceProfile:
    """Timings for one autoregressive generation request.

    ``prefill_to_first_token_seconds`` measures model execution from the start
    of prefill through production of the first output token. It is not
    end-to-end TTFT because it excludes queueing, scheduling, and serving time.
    Decode time covers only the remaining output tokens.
    """

    prompt_tokens: int
    output_tokens: int
    prefill_to_first_token_seconds: float
    decode_seconds: float

    @property
    def tpot_seconds(self) -> float | None:
        decode_tokens = max(self.output_tokens - 1, 0)
        return self.decode_seconds / decode_tokens if decode_tokens else None

    @property
    def total_seconds(self) -> float:
        return self.prefill_to_first_token_seconds + self.decode_seconds

    @property
    def output_tokens_per_second(self) -> float:
        return self.output_tokens / self.total_seconds if self.total_seconds else 0.0


@dataclass(frozen=True)
class TransformerFlops:
    """Approximate forward-pass FLOPs for a dense decoder-only transformer."""

    parameter_flops: int
    attention_flops: int

    @property
    def total(self) -> int:
        return self.parameter_flops + self.attention_flops


@dataclass(frozen=True)
class ProfileResult:
    """Measurements and estimates produced for one generation request."""

    profile: InferenceProfile
    prefill_flops: TransformerFlops
    decode_flops: TransformerFlops
    kv_cache_bytes: float
    generated_text: str


def estimate_prefill_flops(
    *, num_parameters: int, num_layers: int, hidden_size: int, prompt_tokens: int
) -> TransformerFlops:
    """Estimate prefill FLOPs, including the quadratic attention operations."""
    return TransformerFlops(
        parameter_flops=2 * num_parameters * prompt_tokens,
        attention_flops=4 * num_layers * hidden_size * prompt_tokens**2,
    )


def estimate_decode_flops(
    *,
    num_parameters: int,
    num_layers: int,
    hidden_size: int,
    prompt_tokens: int,
    output_tokens: int,
) -> TransformerFlops:
    """Estimate cached decode FLOPs for all output tokens after the first."""
    decode_tokens = max(output_tokens - 1, 0)
    context_sum = decode_tokens * prompt_tokens + decode_tokens * (decode_tokens + 1) // 2
    return TransformerFlops(
        parameter_flops=2 * num_parameters * decode_tokens,
        attention_flops=4 * num_layers * hidden_size * context_sum,
    )