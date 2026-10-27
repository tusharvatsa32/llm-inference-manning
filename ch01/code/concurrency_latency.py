"""Measure how request latency changes as concurrency increases.

Reproduces the experiment from the opening of Chapter 1: send one warmed-up
request to a local model server, then send the same request N times at once
and compare the latencies.

Works with any server that exposes an OpenAI-compatible /v1/completions
endpoint, such as MLX LM, vLLM, or SGLang.

Example (MLX LM on Apple Silicon):
    mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
    python concurrency_latency.py --concurrency 20
    python concurrency_latency.py --sweep 1 2 4 8 16 20
"""

import argparse
import asyncio
import statistics
import time

import httpx

DEFAULT_URL = "http://localhost:8080/v1/completions"
DEFAULT_PROMPT = "The capital of France is"


def percentile(values, pct):
    """Return the pct-th percentile using linear interpolation."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


async def send_request(client, args):
    """Send one completion request and return (latency_seconds, text)."""
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    if args.model:
        payload["model"] = args.model

    start = time.perf_counter()
    response = await client.post(args.url, json=payload)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    text = response.json()["choices"][0]["text"]
    return elapsed, text


async def run_burst(client, args, concurrency):
    """Send `concurrency` identical requests at the same time."""
    start = time.perf_counter()
    results = await asyncio.gather(
        *(send_request(client, args) for _ in range(concurrency))
    )
    wall_time = time.perf_counter() - start
    latencies = [latency for latency, _ in results]
    return latencies, wall_time


def print_burst(concurrency, latencies, wall_time):
    print(
        f"{concurrency:>11} | "
        f"{statistics.median(latencies) * 1000:>9.0f} | "
        f"{percentile(latencies, 95) * 1000:>9.0f} | "
        f"{max(latencies) * 1000:>9.0f} | "
        f"{wall_time * 1000:>10.0f}"
    )


async def main(args):
    levels = args.sweep or [args.concurrency]
    limits = httpx.Limits(max_connections=max(levels))
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        # Warm-up: the first request pays one-time costs such as loading
        # weights and compiling kernels, so we leave it out of the results.
        for _ in range(args.warmup):
            await send_request(client, args)

        latency, text = await send_request(client, args)
        print(f"Prompt:     {args.prompt!r}")
        print(f"Completion: {text!r}")
        print(f"Single warmed-up request: {latency * 1000:.0f} ms\n")

        print("Concurrency | P50 (ms)  | P95 (ms)  | Max (ms)  | Burst (ms)")
        print("------------+-----------+-----------+-----------+-----------")
        for concurrency in levels:
            latencies, wall_time = await run_burst(client, args, concurrency)
            print_burst(concurrency, latencies, wall_time)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"completions endpoint (default: {DEFAULT_URL})")
    parser.add_argument("--model", default=None,
                        help="model name to send; required by vLLM and SGLang")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=8,
                        help="tokens to generate per request (default: 8)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="requests to send at once (default: 20)")
    parser.add_argument("--sweep", type=int, nargs="+", default=None,
                        help="run several concurrency levels, e.g. 1 2 4 8 16")
    parser.add_argument("--warmup", type=int, default=2,
                        help="warm-up requests before measuring (default: 2)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-request timeout in seconds (default: 300)")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
