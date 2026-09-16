"""Shared Hugging Face runner for the Chapter 1 experiments."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_inference.memory import ModelShape, kv_cache_bytes_for_request
from mini_inference.profiling import (
    InferenceProfile,
    ProfileResult,
    estimate_decode_flops,
    estimate_prefill_flops,
)


def best_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


def make_input(tokenizer, prompt: str, prompt_tokens: int, device: str):
    seed = tokenizer(prompt, return_tensors="pt").input_ids[0]
    if len(seed) == 0:
        raise ValueError("The prompt must produce at least one token")
    repeats = (prompt_tokens + len(seed) - 1) // len(seed)
    input_ids = seed.repeat(repeats)[:prompt_tokens].unsqueeze(0).to(device)
    return input_ids


def warm_up(model, tokenizer, device: str) -> None:
    """Run one unreported request to exclude one-time runtime initialization."""
    input_ids = make_input(tokenizer, "Warm up the model.", 8, device)
    profile_request(model, tokenizer, input_ids, output_tokens=2, device=device)


@torch.inference_mode()
def profile_request(model, tokenizer, input_ids, output_tokens: int, device: str) -> ProfileResult:
    if output_tokens < 1:
        raise ValueError("output_tokens must be at least 1")

    attention_mask = torch.ones_like(input_ids)
    synchronize(device)
    prefill_start = time.perf_counter()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    next_token = outputs.logits[:, -1:].argmax(dim=-1)
    synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_start

    generated = [next_token]
    past_key_values = outputs.past_key_values
    synchronize(device)
    decode_start = time.perf_counter()
    for _ in range(output_tokens - 1):
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated.append(next_token)
    synchronize(device)
    decode_seconds = time.perf_counter() - decode_start

    prompt_tokens = input_ids.shape[1]
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    shape = ModelShape(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
        num_kv_heads=num_kv_heads,
    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    profile = InferenceProfile(prompt_tokens, output_tokens, prefill_seconds, decode_seconds)
    return ProfileResult(
        profile=profile,
        prefill_flops=estimate_prefill_flops(
            num_parameters=num_parameters,
            num_layers=num_layers,
            hidden_size=hidden_size,
            prompt_tokens=prompt_tokens,
        ),
        decode_flops=estimate_decode_flops(
            num_parameters=num_parameters,
            num_layers=num_layers,
            hidden_size=hidden_size,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        ),
        kv_cache_bytes=kv_cache_bytes_for_request(
            shape,
            num_tokens=prompt_tokens + output_tokens,
            dtype_bytes=next(model.parameters()).element_size(),
        ),
        generated_text=tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True),
    )