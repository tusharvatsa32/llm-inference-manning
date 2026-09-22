"""Observe how prompt and output lengths affect prefill and decode."""

import argparse

from mini_inference import bytes_to_mb

from model_runner import best_device, load_model, make_input, profile_request, warm_up


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--device", choices=("cpu", "mps"), default=best_device())
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.device)
    warm_up(model, tokenizer, args.device)
    prompt = "Inference systems trade latency, throughput, memory, and cost. "
    experiments = ((32, 16), (128, 16), (512, 16), (128, 64))

    print("prompt  output  prefill-to-1st(ms)  TPOT(ms)  tok/s  KV cache(MB)")
    for prompt_tokens, output_tokens in experiments:
        input_ids = make_input(tokenizer, prompt, prompt_tokens, args.device)
        result = profile_request(model, tokenizer, input_ids, output_tokens, args.device)
        profile = result.profile
        tpot = "N/A" if profile.tpot_seconds is None else f"{profile.tpot_seconds * 1_000:.1f}"
        print(
            f"{prompt_tokens:>6}  {output_tokens:>6}  "
            f"{profile.prefill_to_first_token_seconds * 1_000:>18.1f}  "
            f"{tpot:>8}  "
            f"{profile.output_tokens_per_second:>5.1f}  "
            f"{bytes_to_mb(result.kv_cache_bytes):>12.2f}"
        )


if __name__ == "__main__":
    main()