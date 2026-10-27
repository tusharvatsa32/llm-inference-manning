"""Measure TTFT, ITL, TPOT, and end-to-end latency for streaming requests.

Implements the latency metrics from section 1.3.1. The script streams a
completion, records when each chunk arrives at the client, and computes:

    TTFT  time from sending the request to receiving the first token
    ITL   gaps between consecutive chunks, as the client sees them
    TPOT  (E2E latency - TTFT) / (output tokens - 1)       (Equation 1.1)
    E2E   time from sending the request to receiving the final token

Run it with --concurrency to stream several requests at once. Under load,
compare how TTFT and ITL move: if TTFT rises while ITL stays flat, the time
is going into queueing or prefill; if ITL rises, decode itself is slowing.

Example (MLX LM on Apple Silicon):
    mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
    python streaming_metrics.py
    python streaming_metrics.py --concurrency 20
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

DEFAULT_URL = "http://localhost:8080/v1/completions"
DEFAULT_PROMPT = "Explain in a few sentences why the sky is blue."


def percentile(values, pct):
    """Return the pct-th percentile using linear interpolation."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def tpot(e2e, ttft, output_tokens):
    """Time per output token, Equation 1.1."""
    if output_tokens < 2:
        return None
    return (e2e - ttft) / (output_tokens - 1)


async def stream_request(client, args):
    """Stream one completion and return its timing measurements."""
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.model:
        payload["model"] = args.model

    start = time.perf_counter()
    arrivals = []
    reported_tokens = None

    async with client.stream("POST", args.url, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            # Servers that honour include_usage send token counts at the end.
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                reported_tokens = usage["completion_tokens"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                arrivals.append(time.perf_counter() - start)

    if not arrivals:
        raise RuntimeError("the server returned no tokens")

    # One chunk usually carries one token, but network buffering can merge
    # several, so prefer the server's own count when it reports one.
    output_tokens = reported_tokens or len(arrivals)
    ttft = arrivals[0]
    e2e = arrivals[-1]
    gaps = [later - earlier for earlier, later in zip(arrivals, arrivals[1:])]
    return {
        "ttft": ttft,
        "e2e": e2e,
        "tpot": tpot(e2e, ttft, output_tokens),
        "itl": gaps,
        "output_tokens": output_tokens,
    }


def ms(seconds):
    return f"{seconds * 1000:.1f} ms" if seconds is not None else "n/a"


def print_single(result):
    print(f"Output tokens: {result['output_tokens']}")
    print(f"TTFT:          {ms(result['ttft'])}")
    print(f"TPOT:          {ms(result['tpot'])}  (Equation 1.1)")
    print(f"E2E latency:   {ms(result['e2e'])}")
    if result["itl"]:
        print(f"ITL P50:       {ms(percentile(result['itl'], 50))}")
        print(f"ITL P99:       {ms(percentile(result['itl'], 99))}")
        print(f"ITL max:       {ms(max(result['itl']))}  (longest stall)")


def print_distribution(results):
    def row(name, values):
        values = [v for v in values if v is not None]
        if not values:
            print(f"{name:<6} | n/a")
            return
        print(
            f"{name:<6} | {statistics.mean(values) * 1000:>9.1f} | "
            f"{percentile(values, 50) * 1000:>9.1f} | "
            f"{percentile(values, 95) * 1000:>9.1f} | "
            f"{percentile(values, 99) * 1000:>9.1f}"
        )

    print(f"{len(results)} concurrent requests\n")
    print("Metric | Mean (ms) | P50 (ms)  | P95 (ms)  | P99 (ms)")
    print("-------+-----------+-----------+-----------+----------")
    row("TTFT", [r["ttft"] for r in results])
    row("ITL", [gap for r in results for gap in r["itl"]])
    row("TPOT", [r["tpot"] for r in results])
    row("E2E", [r["e2e"] for r in results])


async def main(args):
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        for _ in range(args.warmup):
            await stream_request(client, args)

        results = await asyncio.gather(
            *(stream_request(client, args) for _ in range(args.concurrency))
        )

    if args.concurrency == 1:
        print_single(results[0])
    else:
        print_distribution(results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"completions endpoint (default: {DEFAULT_URL})")
    parser.add_argument("--model", default=None,
                        help="model name to send; required by vLLM and SGLang")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="tokens to generate per request (default: 200)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="streaming requests to run at once (default: 1)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warm-up requests before measuring (default: 1)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-request timeout in seconds (default: 300)")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
