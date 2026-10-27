"""Read a live vLLM deployment as STARVED, HEALTHY, or SLACK.

Section 3.7.2. vLLM serves Prometheus metrics at /metrics. This script reads
three of them twice, some seconds apart:

    vllm:kv_cache_usage_perc    fraction of the token pool in use (0 to 1)
    vllm:num_preemptions_total  running count of preempted requests
    vllm:num_requests_waiting   requests queued for admission

and classifies the deployment:

    STARVED  cache above 95%, with preemptions or the queue rising
    HEALTHY  cache 30-95%, no new preemptions
    SLACK    cache below 30%

The thresholds are starting points. Calibrate them against your own baseline.
Uses only the Python standard library.

Examples:
    python deployment_health.py --url http://localhost:8000/metrics
    python deployment_health.py --url http://localhost:8000/metrics --interval 60
    python deployment_health.py --files before.txt after.txt   # saved snapshots
"""

import argparse
import time
import urllib.request

USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")  # new, old name
PREEMPTIONS = ("vllm:num_preemptions_total",)
WAITING = ("vllm:num_requests_waiting",)


def parse_metrics(text):
    """Parse Prometheus text format into {metric name: [values]}."""
    metrics = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        metrics.setdefault(name, []).append(value)
    return metrics


def read(metrics, names, combine):
    """Combine one metric across its label sets, trying each name in turn."""
    for name in names:
        if name in metrics:
            return combine(metrics[name])
    return None


def snapshot(text):
    metrics = parse_metrics(text)
    usage = read(metrics, USAGE, max)  # the fullest engine
    if usage is None:
        raise ValueError("no KV cache usage metric found; is this a vLLM /metrics page?")
    return {
        "usage": usage,
        "preemptions": read(metrics, PREEMPTIONS, sum) or 0.0,
        "waiting": read(metrics, WAITING, sum) or 0.0,
    }


def classify(before, after, full=0.95, slack=0.30):
    usage = after["usage"]
    new_preemptions = after["preemptions"] - before["preemptions"]
    queue_growth = after["waiting"] - before["waiting"]

    if usage < slack:
        return "SLACK", ("The cache is not your bottleneck. You could run longer "
                         "contexts, admit more users, or use a smaller GPU.")
    if usage > full and (new_preemptions > 0 or queue_growth > 0):
        return "STARVED", ("The token pool is too small for the offered load. Running "
                           "requests are preempted and must rebuild their cache. "
                           "Apply a lever before adding hardware.")
    if new_preemptions == 0 and usage <= full:
        return "HEALTHY", "The GPU is in use without living on the edge. Nothing to change."
    return "WATCH", ("Mixed signals: the cache is nearly full or requests are being "
                     "preempted, but not both. Keep sampling.")


def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="vLLM metrics URL, e.g. http://localhost:8000/metrics")
    source.add_argument("--files", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="two saved /metrics snapshots instead of a live server")
    parser.add_argument("--interval", type=float, default=30,
                        help="seconds between the two samples (default: 30)")
    parser.add_argument("--full", type=float, default=0.95,
                        help="cache usage treated as full (default: 0.95)")
    parser.add_argument("--slack", type=float, default=0.30,
                        help="cache usage below which the cache is idle (default: 0.30)")
    args = parser.parse_args()

    if args.files:
        texts = [open(path, encoding="utf-8").read() for path in args.files]
    else:
        texts = [fetch(args.url)]
        print(f"Sampled {args.url}; sampling again in {args.interval:g} s...")
        time.sleep(args.interval)
        texts.append(fetch(args.url))

    before, after = snapshot(texts[0]), snapshot(texts[1])
    print(f"KV cache usage:    {after['usage']:.1%}")
    print(f"New preemptions:   {after['preemptions'] - before['preemptions']:+.0f}")
    print(f"Requests waiting:  {after['waiting']:.0f} "
          f"({after['waiting'] - before['waiting']:+.0f})")
    state, advice = classify(before, after, args.full, args.slack)
    print(f"\n{state}: {advice}")


if __name__ == "__main__":
    main()
