"""vLLM settings for the memory levers in section 3.6.

Each lever is one engine configuration. Choose one with --lever, and the
script starts vLLM with it and generates a short completion. It needs vLLM and
a supported GPU, and was checked against the engine arguments in vLLM 0.32.

    fp8           KV cache quantization (section 3.6.1)
    offload       KV cache offloading to CPU RAM (section 3.6.2, Listing 3.8)
    chunked       chunked prefill (section 3.6.3, Listing 3.9)
    prefix        prefix caching (section 3.5)

Examples:
    python vllm_memory_settings.py --lever chunked
    python vllm_memory_settings.py --lever fp8 --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# Section 3.6.1: store the cache in FP8, one byte per value instead of two.
def fp8_cache(model):
    from vllm import LLM

    return LLM(
        model=model,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.9,
    )


# Listing 3.8, section 3.6.2: reserve a 32 GiB CPU buffer for KV cache blocks.
# Older vLLM releases also took swap_space and preemption_mode="swap". Current
# releases removed both and always preempt by recomputing, so passing them
# raises a TypeError.
def offloading(model):
    from vllm import LLM

    return LLM(
        model=model,
        gpu_memory_utilization=0.9,
        kv_offloading_size=32,          # GiB of CPU RAM for KV cache blocks
        kv_offloading_backend="native", # vLLM's built-in CPU offloading
    )


# Listing 3.9, section 3.6.3: bounding the prefill spike.
def chunked_prefill(model):
    from vllm import LLM

    return LLM(
        model=model,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,  # no single forward pass exceeds this many tokens
        gpu_memory_utilization=0.9,
    )


# Section 3.5: reuse the KV cache of shared prompt prefixes.
def prefix_caching(model):
    from vllm import LLM

    return LLM(
        model=model,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    )


LEVERS = {
    "fp8": fp8_cache,
    "offload": offloading,
    "chunked": chunked_prefill,
    "prefix": prefix_caching,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lever", choices=LEVERS, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    from vllm import SamplingParams

    llm = LEVERS[args.lever](args.model)
    output = llm.generate([args.prompt], SamplingParams(temperature=0, max_tokens=16))
    print(args.prompt + output[0].outputs[0].text)


if __name__ == "__main__":
    main()
