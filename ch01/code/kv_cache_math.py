"""Back-of-the-envelope memory and bandwidth math for LLM serving.

Reproduces the numbers from sections 1.2.3, 1.2.4, and 1.4.2:

    KV cache per token  = 2 (K and V) x layers x KV heads x head dim x bytes
    KV cache per request = KV cache per token x context length
    Decode step time    >= bytes read per step / memory bandwidth

For Llama 3 8B in 16-bit precision on an H100 SXM this gives 128 KiB per
token, about 500 MiB for a 4,000-token context, and a ceiling of roughly
200 tokens per second for a single request.

Examples:
    python kv_cache_math.py
    python kv_cache_math.py --model llama-3-70b --gpu h100-sxm
    python kv_cache_math.py --context 8000 --batch 1 16 64
"""

import argparse
from dataclasses import dataclass

KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3
GB = 1e9


@dataclass
class ModelConfig:
    name: str
    params: float      # nominal parameter count
    layers: int
    kv_heads: int      # with grouped-query attention, fewer than query heads
    head_dim: int


@dataclass
class GPUConfig:
    name: str
    memory_gb: float
    bandwidth_tb_s: float  # peak memory bandwidth


MODELS = {
    "llama-3-8b": ModelConfig("Llama 3 8B", 8e9, layers=32, kv_heads=8, head_dim=128),
    "llama-3-70b": ModelConfig("Llama 3 70B", 70e9, layers=80, kv_heads=8, head_dim=128),
}

GPUS = {
    "h100-sxm": GPUConfig("H100 SXM 80GB", memory_gb=80, bandwidth_tb_s=3.35),
    "a100-sxm": GPUConfig("A100 SXM 80GB", memory_gb=80, bandwidth_tb_s=2.04),
    "l40s": GPUConfig("L40S 48GB", memory_gb=48, bandwidth_tb_s=0.864),
}


def kv_bytes_per_token(model, bytes_per_value):
    """Bytes of KV cache one token occupies across all layers."""
    return 2 * model.layers * model.kv_heads * model.head_dim * bytes_per_value


def weight_bytes(model, bytes_per_value):
    return model.params * bytes_per_value


def decode_step_seconds(model, gpu, bytes_per_value, batch, context):
    """Lower bound on one decode step, counting only memory reads.

    Each step reads every weight once (shared by the whole batch) plus the
    KV cache of every request in the batch. Compute and overheads are
    ignored, so real steps are slower than this.
    """
    weights = weight_bytes(model, bytes_per_value)
    kv = batch * context * kv_bytes_per_token(model, bytes_per_value)
    return (weights + kv) / (gpu.bandwidth_tb_s * 1e12)


def format_bytes(n):
    """Binary units for KV cache sizes, matching the chapter (KiB, MiB)."""
    if n >= GIB:
        return f"{n / GIB:,.2f} GiB"
    if n >= MIB:
        return f"{n / MIB:,.1f} MiB"
    return f"{n / KIB:,.1f} KiB"


def main(args):
    model = MODELS[args.model]
    gpu = GPUS[args.gpu]
    bpv = args.bytes_per_value

    print(f"Model: {model.name}, {bpv * 8}-bit values")
    print(f"GPU:   {gpu.name}, {gpu.bandwidth_tb_s} TB/s peak bandwidth\n")

    # Section 1.2.3: KV cache size.
    per_token = kv_bytes_per_token(model, bpv)
    print("KV cache per token")
    print(f"  2 x {model.layers} layers x {model.kv_heads} KV heads x "
          f"{model.head_dim} dims x {bpv} bytes = {format_bytes(per_token)}")
    print("KV cache per request")
    for context in sorted(set([6, *args.context])):
        print(f"  {context:>7,} tokens: {format_bytes(context * per_token)}")
    print()

    # Section 1.4.2: do the weights fit, and how many requests fit beside them?
    weights = weight_bytes(model, bpv)
    gpu_bytes = gpu.memory_gb * GB
    print("Memory capacity")
    print(f"  Weights: {weights / GB:,.1f} GB of {gpu.memory_gb:.0f} GB")
    if weights >= gpu_bytes:
        print("  The weights alone do not fit on one GPU. Quantize them or "
              "split the model across GPUs (tensor parallelism).")
    else:
        free = gpu_bytes - weights
        context = max(args.context)
        fits = int(free // (context * per_token))
        print(f"  Left for KV cache (ignoring activations and overhead): "
              f"{free / GB:,.1f} GB")
        print(f"  Requests at {context:,} tokens that fit at once: {fits:,}")
    print()

    # Section 1.2.4: decode is bounded by memory bandwidth.
    print("Decode ceiling from memory bandwidth (lower bound on step time)")
    if weights >= gpu_bytes:
        print("  Skipped: the model does not fit on one GPU.")
        return
    context = max(args.context)
    print(f"  Context length per request: {context:,} tokens")
    print("    Batch | Step (ms) | Tokens/s per request | Tokens/s total")
    for batch in args.batch:
        step = decode_step_seconds(model, gpu, bpv, batch, context)
        print(f"  {batch:>7} | {step * 1000:>9.2f} | {1 / step:>20,.0f} | "
              f"{batch / step:>14,.0f}")
    weights_only = weights / (gpu.bandwidth_tb_s * 1e12)
    print(f"\n  Reading the weights once takes {weights_only * 1000:.1f} ms, "
          f"so one request can never exceed ~{1 / weights_only:,.0f} tokens/s.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, default="llama-3-8b")
    parser.add_argument("--gpu", choices=GPUS, default="h100-sxm")
    parser.add_argument("--bytes-per-value", type=int, default=2,
                        help="2 for FP16/BF16, 1 for FP8/INT8 (default: 2)")
    parser.add_argument("--context", type=int, nargs="+", default=[4000],
                        help="context lengths to size, in tokens (default: 4000)")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32],
                        help="decode batch sizes to compare (default: 1 8 32)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
