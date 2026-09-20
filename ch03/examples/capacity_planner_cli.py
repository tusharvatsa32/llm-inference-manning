#!/usr/bin/env python3
"""CLI capacity planner for KV cache sizing (Chapter 3, Section 3.7).

Usage:
    python3 ch03/examples/capacity_planner_cli.py --model llama-3-8b \\
        --concurrency 32 --context-tokens 4096 --quantization fp16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mini_inference.memory import (  # noqa: E402
    ModelProfile,
    ServingConfig,
    WorkloadProfile,
    bytes_per_token,
    estimate_capacity,
)

# layers, kv_heads, head_dim, params_b -- read from each model's config.json
# (num_hidden_layers, num_key_value_heads, hidden_size / num_attention_heads).
MODEL_PRESETS = {
    "llama-3-8b": ModelProfile(layers=32, kv_heads=8, head_dim=128, params_b=8.03),
    "llama-3-70b": ModelProfile(layers=80, kv_heads=8, head_dim=128, params_b=70.6),
    "smollm2-135m": ModelProfile(layers=30, kv_heads=3, head_dim=64, params_b=0.135),
}

QUANTIZATION_BYTES = {"fp16": 2, "fp8": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate KV cache capacity for a model + workload.")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="llama-3-8b")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--quantization", choices=sorted(QUANTIZATION_BYTES), default="fp16")
    # Named -gib, not -gb: this value feeds binary (1024**3) GiB math, so an
    # "80 GB" spec-sheet number is really ~74.5 GiB -- convert before passing
    # it in if you want the two to match exactly.
    parser.add_argument("--gpu-memory-gib", type=float, default=80.0)
    parser.add_argument("--gpu-utilization", type=float, default=0.90)
    return parser.parse_args()


def format_report(model_name: str, args: argparse.Namespace) -> str:
    model = MODEL_PRESETS[model_name]
    workload = WorkloadProfile(
        current_tokens_per_request=args.context_tokens,
        max_tokens_per_request=args.context_tokens,
        concurrency=args.concurrency,
    )
    config = ServingConfig(
        kv_dtype_bytes=QUANTIZATION_BYTES[args.quantization],
        gpu_memory_gib=args.gpu_memory_gib,
        gpu_utilization=args.gpu_utilization,
    )

    report = estimate_capacity(model, workload, config)

    if report.token_pool == 0:
        usable_gib = args.gpu_memory_gib * args.gpu_utilization
        return (
            f"KV Cache Capacity Report ({model_name})\n"
            f"{'=' * 60}\n"
            f"Model weights ({report.weight_gb:.1f} GiB) exceed the usable GPU "
            f"budget ({usable_gib:.1f} GiB) -- this model needs tensor "
            f"parallelism across multiple GPUs, a lower-precision weight "
            f"dtype, or a bigger GPU. There is no KV cache budget left to size."
        )

    per_token_bytes = bytes_per_token(model, config)
    requested_kv_gib = (per_token_bytes * args.context_tokens * args.concurrency) / (1024 ** 3)
    fits = requested_kv_gib <= report.kv_budget_gb

    rows = [
        ("Model", model_name),
        ("Precision", args.quantization.upper()),
        ("Weight footprint", f"{report.weight_gb:.1f} GiB"),
        ("Per-token KV cache", f"{per_token_bytes / 1024:.1f} KB"),
        ("KV budget (after weights)", f"{report.kv_budget_gb:.1f} GiB"),
        (
            f"Requested KV ({args.concurrency} req x {args.context_tokens} tok)",
            f"{requested_kv_gib:.1f} GiB",
        ),
        ("Token pool", f"{report.token_pool:,} tokens"),
        ("Max concurrency @ this context", f"{report.max_concurrency}"),
        ("Fits requested workload?", "YES" if fits else "NO - reduce concurrency/context or apply a lever"),
    ]

    label_width = max(len(label) for label, _ in rows)
    lines = [f"KV Cache Capacity Report ({model_name})", "=" * 60]
    for label, value in rows:
        lines.append(f"{label.ljust(label_width)} : {value}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    print(format_report(args.model, args))


if __name__ == "__main__":
    main()
