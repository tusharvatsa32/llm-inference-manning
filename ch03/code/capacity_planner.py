"""The KV cache capacity planner built through Chapter 3.

Brings together Listings 3.1 to 3.7 and 3.10:

    estimate_kv_cache_bytes   ideal KV cache size (Listing 3.2, Equation 3.2)
    reserved_token_slots      slots reserved by static or paged allocation (3.5)
    bytes_per_token           per-token KV cache cost (3.6)
    reserved_kv_gb            reserved slots converted to memory (3.6)
    active_tokens             sliding-window cap on the active length (3.7)
    estimate_capacity         token pool and concurrency for one GPU (3.10)

It also handles multi-head latent attention (section 3.4.3) and reads a
model's architecture straight from its Hugging Face config.json (sections
3.4.1 and 3.7.1).

Memory sizes are binary: 1 GB here means 1024**3 bytes and 1 KB means 1024
bytes, as in the chapter.

Examples:
    python capacity_planner.py                    # the chapter's worked examples
    python capacity_planner.py --preset llama-3-8b --context 32768
    python capacity_planner.py --preset llama-3-8b --kv-heads 32      # as if MHA
    python capacity_planner.py --preset llama-3-8b --kv-dtype-bytes 1 # FP8 cache
    python capacity_planner.py --config path/to/config.json --params-b 8.03
"""

import argparse
import json
import math
from dataclasses import dataclass, replace

GB = 1024 ** 3


@dataclass
class ModelProfile:
    layers: int
    kv_heads: int
    head_dim: int
    params_b: float                      # parameter count, in billions
    sliding_window: int | None = None
    attention_type: str = "gqa"          # "mha", "mqa", "gqa", or "mla"
    kv_latent_dim: int | None = None     # MLA only: cached values per token per layer


@dataclass
class WorkloadProfile:
    current_tokens_per_request: int
    max_tokens_per_request: int
    concurrency: int


@dataclass
class ServingConfig:
    allocation: str = "paged"
    block_size: int = 16
    kv_dtype_bytes: int = 2
    weight_dtype_bytes: int = 2          # bytes per weight (2 = FP16)
    gpu_memory_gb: float = 80.0          # HBM on one GPU
    gpu_utilization: float = 0.90        # fraction the server may use


@dataclass
class CapacityReport:
    weight_gb: float
    kv_budget_gb: float
    per_token_kb: float
    token_pool: int
    max_concurrency: int


# Listing 3.2: the ideal KV cache size, Equation 3.2.
def estimate_kv_cache_bytes(
    model: ModelProfile,
    workload: WorkloadProfile,
    dtype_bytes: int = 2,
):
    return (workload.concurrency
            * workload.current_tokens_per_request
            * model.layers
            * model.kv_heads
            * model.head_dim
            * 2
            * dtype_bytes
            )


# Listing 3.5: token slots the allocator reserves.
def reserved_token_slots(workload: WorkloadProfile, config: ServingConfig) -> int:
    if config.allocation == "static":
        return (
            workload.concurrency
            * workload.max_tokens_per_request
        )

    if config.allocation == "paged":
        blocks_per_request = math.ceil(
            workload.current_tokens_per_request / config.block_size
        )
        return (
            workload.concurrency
            * blocks_per_request
            * config.block_size
        )

    raise ValueError(f"unknown allocation: {config.allocation}")


# Section 3.4.3: MHA, MQA, and GQA cache full keys and values. MLA caches
# one compressed latent vector per token per layer instead.
def cache_values_per_token_per_layer(model: ModelProfile) -> int:
    if model.attention_type in {"mha", "mqa", "gqa"}:
        return 2 * model.kv_heads * model.head_dim

    if model.attention_type == "mla":
        return model.kv_latent_dim

    raise ValueError(f"unknown attention type: {model.attention_type}")


# Listing 3.6: per-token cost and reserved memory. For MHA, MQA, and GQA,
# this is layers x kv_heads x head_dim x 2 x kv_dtype_bytes, as in the book.
def bytes_per_token(model: ModelProfile, config: ServingConfig) -> int:
    return (
        model.layers
        * cache_values_per_token_per_layer(model)
        * config.kv_dtype_bytes
    )


def reserved_kv_gb(model: ModelProfile, workload: WorkloadProfile, config: ServingConfig) -> float:
    slots = reserved_token_slots(workload, config)
    total_bytes = slots * bytes_per_token(model, config)
    return total_bytes / (1024 ** 3)


# Listing 3.7: sliding-window attention caps the active cache length.
def active_tokens(model: ModelProfile, workload: WorkloadProfile) -> int:
    if model.sliding_window is None:
        return workload.current_tokens_per_request
    return min(workload.current_tokens_per_request, model.sliding_window)


# Listing 3.10: the token pool and concurrency for one GPU.
def estimate_capacity(
    model: ModelProfile,
    workload: WorkloadProfile,
    config: ServingConfig,
    weight_gb: float | None = None,        # override the estimate...
    available_kv_gb: float | None = None,  # ...or pass a measured KV budget
) -> CapacityReport:
    # First-pass weight estimate: params x bytes. Ignores activations,
    # tensor parallelism, LoRA, and framework overhead, so the real KV
    # budget is somewhat smaller. Pass the measured numbers when you have them.
    if weight_gb is None:
        weight_gb = (model.params_b * 1e9 * config.weight_dtype_bytes) / (1024 ** 3)
    if available_kv_gb is None:
        usable_gb = config.gpu_memory_gb * config.gpu_utilization
        available_kv_gb = usable_gb - weight_gb

    per_token_bytes = bytes_per_token(model, config)
    token_pool = max(0, int((available_kv_gb * 1024 ** 3) / per_token_bytes))

    return CapacityReport(
        weight_gb=round(weight_gb, 1),
        kv_budget_gb=round(available_kv_gb, 1),
        per_token_kb=round(per_token_bytes / 1024, 1),
        token_pool=token_pool,
        max_concurrency=token_pool // active_tokens(model, workload),
    )


# Sections 3.4.1 and 3.7.1: read the architecture from config.json.
def model_profile_from_config(path, params_b=0.0):
    with open(path, encoding="utf-8") as f:
        return model_profile_from_dict(json.load(f), params_b)


def model_profile_from_dict(cfg, params_b=0.0):
    cfg = cfg.get("text_config") or cfg  # multimodal models nest the text model

    query_heads = cfg["num_attention_heads"]
    # If num_key_value_heads is absent, the model uses full MHA.
    kv_heads = cfg.get("num_key_value_heads") or query_heads
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // query_heads

    if kv_heads == query_heads:
        attention_type = "mha"
    elif kv_heads == 1:
        attention_type = "mqa"
    else:
        attention_type = "gqa"

    # DeepSeek-style MLA caches a compressed latent plus a small
    # rotary-position part instead of full keys and values.
    kv_latent_dim = None
    if cfg.get("kv_lora_rank"):
        attention_type = "mla"
        kv_latent_dim = cfg["kv_lora_rank"] + cfg.get("qk_rope_head_dim", 0)

    # Some configs list a window but turn it off with use_sliding_window.
    # Models that mix windowed and full-attention layers (layer_types) keep
    # the full sequence in some layers; treating every layer as full
    # attention overestimates memory instead of underestimating it.
    sliding_window = cfg.get("sliding_window")
    if cfg.get("use_sliding_window") is False:
        sliding_window = None
    if "full_attention" in cfg.get("layer_types", []):
        sliding_window = None

    return ModelProfile(
        layers=cfg["num_hidden_layers"],
        kv_heads=kv_heads,
        head_dim=head_dim,
        params_b=params_b,
        sliding_window=sliding_window,
        attention_type=attention_type,
        kv_latent_dim=kv_latent_dim,
    )


PRESETS = {
    "llama-3-8b": ModelProfile(layers=32, kv_heads=8, head_dim=128, params_b=8.03),
    "llama-3-70b": ModelProfile(layers=80, kv_heads=8, head_dim=128, params_b=70.6),
    "mistral-7b-v0.1": ModelProfile(layers=32, kv_heads=8, head_dim=128, params_b=7.24,
                                    sliding_window=4096),
    "deepseek-v3": ModelProfile(layers=61, kv_heads=128, head_dim=128, params_b=671,
                                attention_type="mla", kv_latent_dim=512 + 64),
}


def gb(num_bytes):
    return num_bytes / GB


def print_report(model, workload, config):
    report = estimate_capacity(model, workload, config)
    ideal = workload.concurrency * active_tokens(model, workload) * bytes_per_token(model, config)
    print(f"Per-token KV cache:   {report.per_token_kb:,} KB")
    print(f"Weights (estimate):   {report.weight_gb:,} GB")
    print(f"KV cache budget:      {report.kv_budget_gb:,} GB "
          f"({config.gpu_memory_gb:g} GB x {config.gpu_utilization:.0%} - weights)")
    if report.token_pool == 0:
        print("Token pool:           0 (the weights do not fit; quantize them or "
              "split the model across GPUs)")
        return
    print(f"Token pool:           {report.token_pool:,} tokens")
    print(f"Max concurrency at {active_tokens(model, workload):,} active tokens: "
          f"{report.max_concurrency:,} requests")
    fits = "fits" if ideal <= report.kv_budget_gb * GB else "does NOT fit"
    print(f"Workload: {workload.concurrency} requests x "
          f"{workload.current_tokens_per_request:,} tokens needs "
          f"{gb(ideal):,.1f} GB of KV cache -> {fits}")


def chapter_examples():
    fp16 = ServingConfig()
    llama8, llama70 = PRESETS["llama-3-8b"], PRESETS["llama-3-70b"]

    print("== Section 3.2: one token of Llama 3 8B at FP16")
    per_layer = 2 * llama8.kv_heads * llama8.head_dim * 2
    print(f"  per layer: 2 x 8 KV heads x 128 dims x 2 bytes = {per_layer:,} bytes")
    print(f"  32 layers: {bytes_per_token(llama8, fp16) / 1024:,.0f} KB per token\n")

    print("== Table 3.1: KV cache size at FP16")
    print("  Model | Requests | Context | KV cache")
    for name, model in (("8B", llama8), ("70B", llama70)):
        for concurrency, context in ((1, 8192), (32, 8192), (1, 131072)):
            workload = WorkloadProfile(context, context, concurrency)
            size = gb(estimate_kv_cache_bytes(model, workload))
            print(f"  {name:>5} | {concurrency:>8} | {context // 1024:>5}K "
                  f"| {size:,.1f} GB")
    print()

    print("== Figure 3.6: one Llama 3 8B request, 4,000-token prompt, 200-token reply")
    prompt = gb(estimate_kv_cache_bytes(llama8, WorkloadProfile(4000, 4200, 1))) * 1024
    reply = gb(estimate_kv_cache_bytes(llama8, WorkloadProfile(200, 200, 1))) * 1024
    print(f"  after prefill: {prompt:,.0f} MB; decode adds {reply:,.0f} MB\n")

    print("== Section 3.3.4: 32 requests at 201 tokens, max 2,048, block size 16")
    workload = WorkloadProfile(201, 2048, 32)
    ideal = workload.concurrency * workload.current_tokens_per_request
    for allocation in ("static", "paged"):
        config = ServingConfig(allocation=allocation)
        slots = reserved_token_slots(workload, config)
        print(f"  {allocation:>6}: {slots:>6,} slots reserved for {ideal:,} tokens "
              f"({reserved_kv_gb(llama8, workload, config):.2f} GB on Llama 3 8B)")
    print()

    print("== Section 3.4.1: Llama 3 70B, GQA (8 KV heads) vs MHA (64 KV heads)")
    for label, model in (("GQA", llama70), ("MHA", replace(llama70, kv_heads=64))):
        per_token = bytes_per_token(model, fp16)
        at_32k = gb(per_token * 32_000)
        print(f"  {label}: {per_token / 1024 ** 2:.2f} MB per token, "
              f"{at_32k:,.1f} GB for one 32,000-token request")
    print()

    print("== Section 3.4.2: sliding window (Mistral 7B v0.1, W = 4,096)")
    mistral = PRESETS["mistral-7b-v0.1"]
    workload = WorkloadProfile(32_000, 32_000, 1)
    print(f"  32,000-token request: {active_tokens(mistral, workload):,} active tokens, "
          f"{gb(active_tokens(mistral, workload) * bytes_per_token(mistral, fp16)):.2f} GB "
          f"instead of {gb(32_000 * bytes_per_token(mistral, fp16)):.2f} GB\n")

    print("== Section 3.4.3: MLA (DeepSeek-V3 caches a 576-value latent per layer)")
    deepseek = PRESETS["deepseek-v3"]
    # Without MLA, each of its 128 heads would cache a 192-value key and a
    # 128-value value: the same as 2 x 128 heads x 160 values.
    as_mha = replace(deepseek, attention_type="mha", head_dim=(192 + 128) // 2)
    print(f"  MLA: {bytes_per_token(deepseek, fp16) / 1024:,.1f} KB per token; "
          f"full K/V for its 128 heads would be about "
          f"{bytes_per_token(as_mha, fp16) / 1024:,.0f} KB\n")

    print("== Section 3.7.1: token pool for Llama 3 8B on one 80 GB GPU")
    for label, model, config in (
        ("FP16, GQA (8 KV heads)", llama8, fp16),
        ("FP16, as if MHA (32 KV heads)", replace(llama8, kv_heads=32), fp16),
        ("FP8 cache, GQA", llama8, ServingConfig(kv_dtype_bytes=1)),
    ):
        report = estimate_capacity(model, WorkloadProfile(8192, 8192, 1), config)
        print(f"  {label:<30} pool {report.token_pool:>9,} tokens")
    for context in (8192, 32768, 131072):
        report = estimate_capacity(llama8, WorkloadProfile(context, context, 1), fp16)
        print(f"  at {context // 1024:>3}K tokens per request: "
              f"~{report.max_concurrency} concurrent requests")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preset", choices=PRESETS)
    source.add_argument("--config", help="path to a Hugging Face config.json")
    parser.add_argument("--params-b", type=float, default=None,
                        help="parameter count in billions (needed with --config)")
    parser.add_argument("--kv-heads", type=int, default=None,
                        help="override the KV head count, e.g. to compare with MHA")
    parser.add_argument("--context", type=int, default=8192,
                        help="active tokens per request (default: 8192)")
    parser.add_argument("--max-context", type=int, default=None,
                        help="maximum tokens per request (default: --context)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--allocation", choices=("paged", "static"), default="paged")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2,
                        help="2 for FP16/BF16, 1 for FP8 (default: 2)")
    parser.add_argument("--weight-dtype-bytes", type=float, default=2)
    parser.add_argument("--gpu-memory-gb", type=float, default=80.0)
    parser.add_argument("--gpu-utilization", type=float, default=0.90)
    args = parser.parse_args()

    if not args.preset and not args.config:
        chapter_examples()
        return

    if args.config:
        model = model_profile_from_config(args.config, args.params_b or 0.0)
    else:
        model = PRESETS[args.preset]
        if args.params_b is not None:
            model = replace(model, params_b=args.params_b)
    if args.kv_heads is not None:
        model = replace(model, kv_heads=args.kv_heads)

    workload = WorkloadProfile(
        current_tokens_per_request=args.context,
        max_tokens_per_request=args.max_context or args.context,
        concurrency=args.concurrency,
    )
    config = ServingConfig(
        allocation=args.allocation,
        block_size=args.block_size,
        kv_dtype_bytes=args.kv_dtype_bytes,
        weight_dtype_bytes=args.weight_dtype_bytes,
        gpu_memory_gb=args.gpu_memory_gb,
        gpu_utilization=args.gpu_utilization,
    )

    print(f"Model: {args.preset or args.config}")
    print(f"  {model.layers} layers, {model.kv_heads} KV heads, head_dim {model.head_dim}, "
          f"{model.attention_type.upper()}"
          + (f", sliding window {model.sliding_window:,}" if model.sliding_window else "")
          + (f", latent {model.kv_latent_dim}" if model.kv_latent_dim else ""))
    if not model.params_b:
        print("  Pass --params-b to estimate the weight footprint and token pool.")
        print(f"  Per-token KV cache: {bytes_per_token(model, config) / 1024:,.1f} KB")
        return
    reserved = reserved_kv_gb(model, workload, config)
    print(f"  {config.allocation} allocation reserves {reserved:,.1f} GB for this workload\n")
    print_report(model, workload, config)


if __name__ == "__main__":
    main()
