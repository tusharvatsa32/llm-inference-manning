# Chapter 1: Inference Is the New Bottleneck

Chapter 1 builds a small inference profiler. The measurements are the main
artifact; memory and FLOPs estimates help explain what the measurements show.

## Start without a model

Install the reusable package from the repository root. The KV-cache calculator
has no third-party dependencies and does not download a model:

```bash
python3 -m pip install -e .
python3 ch01/supplementary/kv_cache_memory_calculator.py
```

## Profile a request

Install the Chapter 1 extras from the repository root. The default 135M-parameter
model is small enough to run on a MacBook and uses MPS automatically when it is
available.

```bash
python3 -m pip install -e '.[chapter1]'
python3 ch01/examples/profile_one_request.py
```

The script runs greedy generation with an explicit cached prefill/decode loop and
reports prefill-to-first-token latency, TPOT, output throughput, estimated
KV-cache size, and approximate FLOPs. The first-token measurement covers only
model execution; it excludes queueing, scheduling, tokenization, and serving
overhead, so it is not end-to-end TTFT. When only one token is generated, TPOT
is reported as `N/A` because there is no decode interval to measure.

Then vary one constraint at a time:

```bash
python3 ch01/examples/profile_one_request.py --prompt-tokens 256 --output-tokens 32
python3 ch01/examples/profile_one_request.py --prompt-tokens 64 --output-tokens 128
python3 ch01/examples/compare_prefill_decode.py
```

Look for three effects: longer prompts primarily increase prefill time, every
generated token pays decode cost, and the KV cache grows with total sequence
length. Results vary by hardware and software version; the relationships are
more important than a benchmark table.

Memory values use decimal units throughout the repository: 1 MB is `10**6`
bytes and 1 GB is `10**9` bytes.

The examples import measurements, FLOPs estimates, and KV-cache calculations
from `src/mini_inference`. Chapters 2–4 extend that same package with generation,
caching, scheduling, and optimization code instead of copying Chapter 1.