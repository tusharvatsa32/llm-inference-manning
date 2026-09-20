"""Shared loading and display helpers for the Chapter 2 examples."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_inference.engine import Decoder, InferenceRequest, TokenOutput
from mini_inference.runner import HuggingFaceModelRunner
from mini_inference.sampling import SamplingParams

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_stack(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    runner = HuggingFaceModelRunner(model, device=device)
    return tokenizer, runner, Decoder(runner)


def encode_prompt(tokenizer, prompt: str) -> tuple[int, ...]:
    token_ids = tuple(tokenizer.encode(prompt, add_special_tokens=True))
    if not token_ids:
        raise ValueError("the prompt must produce at least one token")
    return token_ids


def generate_token_ids(
    decoder: Decoder,
    prompt_token_ids: tuple[int, ...],
    sampling: SamplingParams,
    max_new_tokens: int,
) -> list[int]:
    request = InferenceRequest(
        request_id="example",
        prompt_token_ids=prompt_token_ids,
        sampling=sampling,
        max_new_tokens=max_new_tokens,
    )
    return [
        event.token_id
        for event in decoder.generate(request)
        if isinstance(event, TokenOutput)
    ]
