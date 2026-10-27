# Chapter 3: KV Cache and Memory

Code for Chapter 3. `capacity_planner.py` is the planner the chapter builds one
listing at a time. The other scripts show the mechanisms behind each memory
lever: the block allocator, prefix caching, and the vLLM settings for
quantization, offloading, and chunked prefill.

| File | Section | What it contains |
|------|---------|------------------|
| `kv_cache_basics.py` | 3.1 | Equation 3.1 recompute cost, plus an optional real-model check that measures the KV cache and times decoding with and without it |
| `capacity_planner.py` | 3.2, 3.3.4, 3.4, 3.7.1 | Listings 3.1, 3.2, 3.5, 3.6, 3.7, and 3.10, the MLA cache width, and a `config.json` reader |
| `block_allocator.py` | 3.3 | `SequenceBlocks` (Listing 3.3) and `block_stats` (Listing 3.4), with the chapter's allocation examples |
| `prefix_cache.py` | 3.5 | A toy prefix cache with chained block hashes and LRU eviction |
| `vllm_memory_settings.py` | 3.5, 3.6 | vLLM settings for FP8 cache, CPU offloading (Listing 3.8), chunked prefill (Listing 3.9), and prefix caching |
| `deployment_health.py` | 3.7.2 | Reads vLLM's `/metrics` and classifies a deployment as STARVED, HEALTHY, or SLACK |

Listings 3.1 and 3.2 grow into Listing 3.10, so all of them live in
`capacity_planner.py` in their final form. Memory sizes are binary, as in the
chapter: 1 GB means 1024³ bytes.

## Setup

Python 3.10 or later. `capacity_planner.py`, `block_allocator.py`,
`prefix_cache.py`, and `deployment_health.py` use only the standard library.

```bash
cd ch03/code
pip install -r requirements.txt        # for kv_cache_basics.py --model
pip install vllm                       # only for vllm_memory_settings.py (needs a GPU)
```

## Running the code

### Why the KV cache exists (3.1)

```bash
python kv_cache_basics.py
python kv_cache_basics.py --model Qwen/Qwen2.5-0.5B-Instruct
```

Without `--model`, the script prints Equation 3.1: generating 200 tokens after
a 10,000-token prompt recomputes about 2 million token positions without a
cache. With `--model`, it measures the KV cache the model allocates after
prefill, compares it with the planner's estimate from the model's
`config.json` (they match to the byte), and times greedy decoding with and
without the cache.

### The capacity planner (3.2, 3.4, 3.7)

```bash
python capacity_planner.py                                       # the chapter's worked examples
python capacity_planner.py --preset llama-3-8b --context 32768 --concurrency 16
python capacity_planner.py --preset llama-3-8b --kv-heads 32     # as if the model used MHA
python capacity_planner.py --preset llama-3-8b --kv-dtype-bytes 1   # FP8 cache
python capacity_planner.py --preset llama-3-8b --allocation static --max-context 2048 --context 201 --concurrency 32
python capacity_planner.py --config path/to/config.json --params-b 8.03
```

With no arguments the planner prints the chapter's examples: the 128 KB
per-token cost of Llama 3 8B, Table 3.1, the prefill and decode growth from
figure 3.6, static versus paged reservation, GQA versus MHA for Llama 3 70B,
a sliding window, MLA, and the 467,000-token pool from section 3.7.1.

Presets: `llama-3-8b`, `llama-3-70b`, `mistral-7b-v0.1` (4,096-token sliding
window), and `deepseek-v3` (MLA). To plan for any other model, download its
`config.json` from Hugging Face and pass it with `--config`. The reader takes
`num_hidden_layers`, `num_key_value_heads` (falling back to
`num_attention_heads` for MHA), `head_dim` (or `hidden_size /
num_attention_heads`), `sliding_window`, and the MLA fields `kv_lora_rank` and
`qk_rope_head_dim`. For models that mix sliding-window and full-attention
layers, it treats every layer as full attention, which overestimates memory
rather than underestimating it.

The weight estimate is parameters × bytes. Pass the measured numbers through
`estimate_capacity(..., weight_gb=...)` or `available_kv_gb=...` when your
serving stack reports them.

### The block allocator (3.3)

```bash
python block_allocator.py
python block_allocator.py --block-size 8 --tokens 33
```

Shows one request claiming a new block every 16 tokens, two requests whose
blocks interleave in physical memory (figure 3.10), waste and block-table size
for different block sizes, and static versus paged reservation (figure 3.8 and
section 3.3.4).

### Prefix caching (3.5)

```bash
python prefix_cache.py
python prefix_cache.py --block-size 4
```

Shows three requests sharing a 2,000-token system prompt (figure 3.15), a good
and a bad prompt layout (figure 3.16), and eviction. Only full blocks are
shared, and blocks are released last-first so that eviction removes the end
of a cached prefix before its beginning, as vLLM does.

### vLLM memory settings (3.5, 3.6)

```bash
python vllm_memory_settings.py --lever fp8
python vllm_memory_settings.py --lever offload
python vllm_memory_settings.py --lever chunked
python vllm_memory_settings.py --lever prefix
python vllm_memory_settings.py --lever chunked --model Qwen/Qwen2.5-0.5B-Instruct
```

These need vLLM and a supported GPU. The arguments were checked against vLLM
0.32. Older vLLM releases also accepted `swap_space` and
`preemption_mode="swap"`. Current releases removed both and always preempt by
recomputing, so the offloading example uses `kv_offloading_size` and
`kv_offloading_backend` instead.

### Reading a live deployment (3.7.2)

```bash
python deployment_health.py --url http://localhost:8000/metrics
python deployment_health.py --url http://localhost:8000/metrics --interval 60 --full 0.9
```

Samples `vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`, and
`vllm:num_requests_waiting` twice and prints STARVED, HEALTHY, or SLACK. When
the signals disagree, for example a nearly full cache with no preemptions, it
prints WATCH. You can also pass two saved snapshots with
`--files before.txt after.txt`.
