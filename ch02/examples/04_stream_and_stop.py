"""Print the token event stream and the terminal finish reason."""

import argparse

from mini_inference.engine import InferenceRequest, RequestFinished, TokenOutput
from mini_inference.sampling import SamplingParams

from _common import DEFAULT_MODEL, best_device, encode_prompt, load_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Three benefits of model caching are")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--stop-text",
        help="stop after this text if it maps to exactly one token",
    )
    parser.add_argument("--stop-token-id", type=int, action="append", default=[])
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default=best_device()
    )
    args = parser.parse_args()

    tokenizer, _, decoder = load_stack(args.model, args.device)
    stop_token_ids = list(args.stop_token_id)
    if args.stop_text is not None:
        encoded_stop = tokenizer.encode(args.stop_text, add_special_tokens=False)
        if len(encoded_stop) != 1:
            parser.error(
                f"--stop-text must map to one token; {args.stop_text!r} mapped to "
                f"{encoded_stop}. Use --stop-token-id for token-level stopping."
            )
        stop_token_ids.append(encoded_stop[0])

    request = InferenceRequest(
        request_id="stream-demo",
        prompt_token_ids=encode_prompt(tokenizer, args.prompt),
        sampling=SamplingParams(temperature=0),
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=tuple(stop_token_ids),
    )

    generated_token_ids = []
    for event in decoder.generate(request):
        if isinstance(event, TokenOutput):
            generated_token_ids.append(event.token_id)
            piece = tokenizer.decode(
                [event.token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print(f"TokenOutput(token_id={event.token_id}, text_piece={piece!r})")
        elif isinstance(event, RequestFinished):
            print(f"RequestFinished(reason={event.reason!r})")

    continuation = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    print(f"final text: {args.prompt + continuation!r}")


if __name__ == "__main__":
    main()
