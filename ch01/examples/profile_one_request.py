"""Measure one small language-model request on CPU or Apple Silicon."""

import argparse

from model_runner import best_device, load_model, make_input, profile_request, warm_up


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--prompt", default="Explain why language model inference has two phases.")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "mps"), default=best_device())
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.device)
    warm_up(model, tokenizer, args.device)
    input_ids = make_input(tokenizer, args.prompt, args.prompt_tokens, args.device)
    result = profile_request(model, tokenizer, input_ids, args.output_tokens, args.device)
    profile = result.profile

    print(f"device:                 {args.device}")
    print(f"prompt / output tokens: {profile.prompt_tokens} / {profile.output_tokens}")
    print(f"TTFT (prefill):          {profile.ttft_seconds * 1_000:.1f} ms")
    print(f"TPOT (decode):           {profile.tpot_seconds * 1_000:.1f} ms/token")
    print(f"output throughput:       {profile.output_tokens_per_second:.1f} tokens/s")
    print(f"estimated KV cache:      {result.kv_cache_bytes / 1e6:.2f} MB")
    print(f"estimated prefill FLOPs: {result.prefill_flops.total / 1e9:.2f} GFLOPs")
    print(f"estimated decode FLOPs:  {result.decode_flops.total / 1e9:.2f} GFLOPs")
    print(f"generated text:          {result.generated_text!r}")


if __name__ == "__main__":
    main()