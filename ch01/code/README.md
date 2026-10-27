# Chapter 1: Inference Is the New Bottleneck

Companion scripts for Chapter 1. They reproduce the experiment from the opening
of the chapter and the numbers worked out in the text.

| Script | Section | What it shows |
|--------|---------|---------------|
| `concurrency_latency.py` | Introduction, 1.1 | How latency grows when the same request is sent many times at once |
| `tokenize_prompt.py` | 1.2.1 | How a prompt becomes token IDs, and how many tokens your text really costs |
| `kv_cache_math.py` | 1.2.3, 1.2.4, 1.4.2 | KV cache size per token and per request, the decode ceiling set by memory bandwidth, and whether a model fits on one GPU |
| `streaming_metrics.py` | 1.3.1 | TTFT, ITL, TPOT (Equation 1.1), and end-to-end latency for streaming requests |
| `cost_breakdown.py` | 1.3.3 | How much of an inference bill comes from prompt tokens versus completion tokens |

`kv_cache_math.py` and `cost_breakdown.py` only do arithmetic and need nothing
beyond Python. The other scripts need the packages in `requirements.txt`.

## Setup

Python 3.10 or later.

```bash
cd ch01/code
pip install -r requirements.txt
```

## Starting a local model server

`concurrency_latency.py` and `streaming_metrics.py` talk to any server with an
OpenAI-compatible `/v1/completions` endpoint. The chapter uses MLX LM on an
Apple Silicon Mac:

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

The server listens on `http://localhost:8080`. In another terminal, check that
it answers:

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 8}'
```

On a machine with an NVIDIA GPU you can use vLLM or SGLang instead. Pass the
server's URL and the model name, because both require it:

```bash
python concurrency_latency.py \
  --url http://localhost:8000/v1/completions \
  --model meta-llama/Llama-3.1-8B-Instruct
```

## Running the scripts

### Latency under concurrency

```bash
python concurrency_latency.py                        # 1 request, then 20 at once
python concurrency_latency.py --sweep 1 2 4 8 16 20  # watch latency grow
```

The script sends warm-up requests first, measures one request on its own, and
then sends each burst. For every concurrency level it prints the median (P50),
P95, and slowest latency, and how long the whole burst took. On a 24 GiB Apple
Silicon Mac the chapter measured about 176 ms for a single request and a median
of about 1.2 s with 20 concurrent requests. Your numbers depend on your
hardware.

### Tokenization

```bash
python tokenize_prompt.py
python tokenize_prompt.py "SELECT name FROM users WHERE id = 42;"
python tokenize_prompt.py --file my_prompt.txt
```

With the default prompt you should see the five IDs from the chapter:
`[791, 5765, 315, 9822, 374]`. The tokenizer is downloaded from the Hugging Face
Hub on the first run.

### KV cache and bandwidth math

```bash
python kv_cache_math.py                              # Llama 3 8B on an H100 SXM
python kv_cache_math.py --context 8000 --batch 1 16 64
python kv_cache_math.py --model llama-3-70b          # does not fit on one GPU
```

These are lower bounds. They count only memory reads and leave out compute,
activations, and framework overhead, so a real engine is slower and fits fewer
requests.

### Streaming latency metrics

```bash
python streaming_metrics.py                          # one request
python streaming_metrics.py --concurrency 20         # distributions under load
```

ITL is measured at the client. Network buffering can deliver several tokens in
one chunk, so the script counts output tokens from the server's usage report
when the server sends one.

### Cost breakdown

```bash
python cost_breakdown.py                             # the example from figure 1.6
python cost_breakdown.py --cached-fraction 0.5       # effect of prefix caching
python cost_breakdown.py --requests 10000 --prompt-tokens 2e6 --completion-tokens 5e6
```
