"""Why decoding needs a KV cache, measured on a real model.

Section 3.1. The script first works through Equation 3.1: without a cache,
every decode step recomputes keys and values for all earlier tokens, so the
work grows quadratically.

With --model, it then loads a Hugging Face model and:
    1. runs prefill on a prompt and measures the KV cache the model actually
       allocates, next to the planner's estimate from config.json
    2. generates the same tokens with and without the cache (figure 3.3)
       and times both

Examples:
    python kv_cache_basics.py
    python kv_cache_basics.py --model Qwen/Qwen2.5-0.5B-Instruct
    python kv_cache_basics.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt-tokens 1000 --new-tokens 50
"""

import argparse
import time

from capacity_planner import ServingConfig, bytes_per_token, model_profile_from_dict


def recompute_cost(prompt_tokens, new_tokens):
    """K/V computations to generate new_tokens after a prompt (Equation 3.1).

    Without a cache, step i recomputes K and V for every token so far.
    With a cache, each token's K and V are computed once.
    """
    without_cache = sum(prompt_tokens + i for i in range(new_tokens))
    with_cache = prompt_tokens + new_tokens
    return without_cache, with_cache


def print_recompute_cost():
    print("== Equation 3.1: K/V computations with and without a cache")
    print("  Prompt  | New tokens | Without cache | With cache")
    for prompt, new in ((1_000, 200), (10_000, 200), (100_000, 200)):
        without, with_ = recompute_cost(prompt, new)
        print(f"  {prompt:>7,} | {new:>10,} | {without:>13,} | {with_:>10,}")
    print("  Generating 200 tokens after a 10,000-token prompt recomputes about"
          " 2 million token positions without a cache.\n")


def kv_tensors(cache):
    """Yield the key and value tensors of every layer in a Hugging Face cache."""
    if hasattr(cache, "layers"):              # transformers 4.56 and later
        for layer in cache.layers:
            yield layer.keys
            yield layer.values
    elif hasattr(cache, "key_cache"):         # older DynamicCache
        yield from cache.key_cache
        yield from cache.value_cache
    else:                                     # legacy tuple of (key, value)
        for key, value in cache:
            yield key
            yield value


def measure(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()

    # A prompt of exactly --prompt-tokens tokens.
    text = "The quick brown fox jumps over the lazy dog. " * args.prompt_tokens
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :args.prompt_tokens]
    ids = ids.to(model.device)

    print(f"== {args.model}: KV cache after prefilling {ids.shape[1]:,} tokens")
    with torch.no_grad():
        cache = model(ids, use_cache=True).past_key_values
    tensors = list(kv_tensors(cache))
    measured = sum(t.numel() * t.element_size() for t in tensors)
    dtype_bytes = tensors[0].element_size()

    profile = model_profile_from_dict(model.config.to_dict())
    estimate = ids.shape[1] * bytes_per_token(profile, ServingConfig(kv_dtype_bytes=dtype_bytes))
    print(f"  config.json: {profile.layers} layers, {profile.kv_heads} KV heads, "
          f"head_dim {profile.head_dim}, {dtype_bytes}-byte values "
          f"({tensors[0].dtype})")
    print(f"  shape of one layer's keys: {tuple(tensors[0].shape)} "
          f"(batch, KV heads, tokens, head_dim)")
    print(f"  measured:         {measured:>13,} bytes ({measured / 1024 ** 2:,.2f} MB)")
    print(f"  planner estimate: {estimate:>13,} bytes ({estimate / 1024 ** 2:,.2f} MB)\n")

    print(f"== Generating {args.new_tokens} tokens with and without the cache (figure 3.3)")
    results = {}
    for use_cache in (False, True):
        out, seconds = greedy(model, ids, args.new_tokens, use_cache)
        results[use_cache] = out
        label = "with cache   " if use_cache else "without cache"
        print(f"  {label}: {seconds:6.2f} s ({seconds / args.new_tokens * 1000:6.1f} ms per token)")
    same = torch.equal(results[False], results[True])
    print(f"  Both produce the same tokens: {same}")


def greedy(model, ids, new_tokens, use_cache):
    """Greedy decoding, either reusing the cache or recomputing everything."""
    import torch

    start = time.perf_counter()
    generated = []
    with torch.no_grad():
        if use_cache:
            out = model(ids, use_cache=True)
            cache = out.past_key_values
            next_id = out.logits[0, -1].argmax().view(1, 1)
            for _ in range(new_tokens):
                generated.append(next_id)
                # Only the new token goes through the model; earlier keys
                # and values come from the cache.
                out = model(next_id, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                next_id = out.logits[0, -1].argmax().view(1, 1)
        else:
            seq = ids
            for _ in range(new_tokens):
                # The whole sequence goes through the model at every step.
                next_id = model(seq, use_cache=False).logits[0, -1].argmax().view(1, 1)
                generated.append(next_id)
                seq = torch.cat([seq, next_id], dim=1)
    return torch.cat(generated, dim=1), time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None,
                        help="Hugging Face model ID to measure, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps (default: cpu)")
    parser.add_argument("--prompt-tokens", type=int, default=500)
    parser.add_argument("--new-tokens", type=int, default=30)
    args = parser.parse_args()

    print_recompute_cost()
    if args.model:
        measure(args)
    else:
        print("Pass --model to measure a real model's KV cache, for example:")
        print("  python kv_cache_basics.py --model Qwen/Qwen2.5-0.5B-Instruct")


if __name__ == "__main__":
    main()
