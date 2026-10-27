"""A simplified sampling decoder (Listing 2.10).

The Decoder runs the autoregressive loop from section 2.1 and selects each
token with the controls from sections 2.2 and 2.3: greedy decoding,
temperature, top-k, top-p, min-p, locally typical sampling, and a repetition
penalty.

Examples:
    python decoder.py --temperature 0
    python decoder.py --temperature 0.6 --top-p 0.9
    python decoder.py --temperature 1.5 --min-p 0.1
    python decoder.py --temperature 1.0 --typical-p 0.9
"""

import argparse
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class DecoderConfig:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    typical_p: float | None = None
    repetition_penalty: float = 1.0
    max_new_tokens: int = 256


class Decoder:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def top_k_filter(self, logits):
        kth = torch.topk(
            logits, self.config.top_k
        ).values[-1]
        return logits.masked_fill(
            logits < kth, float("-inf")
        )

    def top_p_filter(self, logits):
        sorted_logits, sorted_idx = torch.sort(
            logits, descending=True
        )
        cum = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )

        remove = cum > self.config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove, float("-inf")
        )
        return torch.full_like(
            logits, float("-inf")
        ).scatter(0, sorted_idx, sorted_logits)

    def min_p_filter(self, logits):
        probs = torch.softmax(logits, dim=-1)
        threshold = self.config.min_p * probs.max()
        return logits.masked_fill(
            probs < threshold, float("-inf")
        )

    def typical_filter(self, logits):
        # Locally typical sampling (Meister et al., 2022). A token's
        # surprise is -log p. The entropy is the average surprise of the
        # distribution. Keep the tokens whose surprise is closest to the
        # entropy until their probabilities add up to typical_p.
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).nansum()
        distance = (-log_probs - entropy).abs()

        _, sorted_idx = torch.sort(distance)
        cum = torch.cumsum(probs[sorted_idx], dim=-1)

        remove = cum > self.config.typical_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False

        remove = torch.zeros_like(remove).scatter(
            0, sorted_idx, remove
        )
        return logits.masked_fill(remove, float("-inf"))

    def apply_repetition_penalty(self, logits, generated):
        logits = logits.clone()
        for tid in set(generated.tolist()):
            if logits[tid] > 0:
                logits[tid] /= self.config.repetition_penalty
            else:
                logits[tid] *= self.config.repetition_penalty
        return logits

    def select_next_token(self, logits, generated):
        c = self.config

        if c.repetition_penalty != 1.0:
            logits = self.apply_repetition_penalty(
                logits, generated
            )

        if c.temperature == 0:
            return logits.argmax()

        logits = logits / c.temperature

        if c.top_k is not None:
            logits = self.top_k_filter(logits)
        if c.top_p is not None:
            logits = self.top_p_filter(logits)
        if c.min_p is not None:
            logits = self.min_p_filter(logits)
        if c.typical_p is not None:
            logits = self.typical_filter(logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)[0]

    @torch.no_grad()
    def generate(self, prompt):
        ids = self.tokenizer.encode(
            prompt, return_tensors="pt"
        ).to(self.model.device)
        start = ids.shape[1]

        for _ in range(self.config.max_new_tokens):
            logits = self.model(ids).logits[0, -1, :]
            generated = ids[0, start:]

            next_id = self.select_next_token(
                logits, generated
            )
            ids = torch.cat(
                [ids, next_id.view(1, 1)], dim=1
            )

            if next_id.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(ids[0, start:])


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name, device=None):
    """Load a Hugging Face causal language model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return model.to(device or default_device()).eval(), tokenizer


def add_model_args(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Hugging Face model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", default=None,
                        help="cpu, cuda, or mps (default: best available)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for sampling (default: 0)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--typical-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.device)
    config = DecoderConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        typical_p=args.typical_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    decoder = Decoder(model, tokenizer, config)
    print(args.prompt + decoder.generate(args.prompt))


if __name__ == "__main__":
    main()
