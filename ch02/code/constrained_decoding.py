"""Constrained decoding with a finite-state machine (section 2.5).

Completes Listing 2.12. A small state machine accepts exactly the restricted
JSON format from section 2.5.1:

    {"city": "<string>"}

START -> "{" -> quoted key "city" -> ":" -> quoted string value -> "}" -> ACCEPT,
with optional whitespace before each part. At every decode step the grammar
masks every token whose text would leave that path, the decoder samples from
the tokens that remain, and the grammar advances by the chosen token's text.

The script asks the model for the same JSON several times, with and without
the constraint, and counts how many outputs parse.

Examples:
    python constrained_decoding.py
    python constrained_decoding.py --runs 20 --temperature 1.2
"""

import argparse
import json

import torch

from decoder import Decoder, DecoderConfig, add_model_args, load_model

STRING = object()  # placeholder for a quoted string value
FORMAT = ["{", '"city"', ":", STRING, "}"]
WHITESPACE = " \t\n\r"


def step(state, ch):
    """Advance the state machine by one character, or return None."""
    part, pos = state
    if part == len(FORMAT):
        return None  # ACCEPT: nothing may follow

    expected = FORMAT[part]
    if expected is STRING:
        if pos == 0:  # before the opening quote
            if ch in WHITESPACE:
                return state
            return (part, 1) if ch == '"' else None
        if ch == '"':  # closing quote
            return (part + 1, 0)
        if ch == "\\" or ord(ch) < 0x20:
            return None  # keep the example simple: no escapes
        return state

    if pos == 0 and ch in WHITESPACE:
        return state
    if ch != expected[pos]:
        return None
    pos += 1
    return (part + 1, 0) if pos == len(expected) else (part, pos)


class CityGrammar:
    """Tracks the state machine and turns each state into a token mask."""

    def __init__(self, tokenizer, vocab_size):
        self.eos_token_id = tokenizer.eos_token_id
        # Text of every token. Special tokens and partial UTF-8 byte
        # sequences are never allowed, so they get no text.
        special = set(tokenizer.added_tokens_decoder)
        self.token_text = [None] * vocab_size
        for tid in range(min(len(tokenizer), vocab_size)):
            text = tokenizer.decode([tid])
            if tid not in special and text and "�" not in text:
                self.token_text[tid] = text
        self.masks = {}  # state -> mask, built on first use
        self.reset()

    def reset(self):
        self.state = (0, 0)

    def is_complete(self):
        return self.state[0] == len(FORMAT)

    def consume(self, state, text):
        for ch in text:
            state = step(state, ch)
            if state is None:
                return None
        return state

    def valid_next_tokens(self):
        if self.state not in self.masks:
            mask = torch.zeros(len(self.token_text), dtype=torch.bool)
            if self.is_complete():
                mask[self.eos_token_id] = True  # only end-of-sequence
            else:
                for tid, text in enumerate(self.token_text):
                    if text and self.consume(self.state, text) is not None:
                        mask[tid] = True
            self.masks[self.state] = mask
        return self.masks[self.state]

    def advance(self, token_id):
        if token_id != self.eos_token_id:
            self.state = self.consume(self.state, self.token_text[token_id])


class ConstrainedDecoder(Decoder):
    def __init__(self, model, tokenizer, config, grammar):
        super().__init__(model, tokenizer, config)
        self.grammar = grammar

    def select_next_token(self, logits, generated):
        mask = self.grammar.valid_next_tokens().to(logits.device)  # tokens that keep the output valid
        logits = logits.masked_fill(~mask, float("-inf"))
        next_id = super().select_next_token(logits, generated)
        self.grammar.advance(next_id.item())
        return next_id

    def generate(self, prompt):
        self.grammar.reset()
        return super().generate(prompt)


def check(text):
    """Return 'valid' or the reason the output fails the format."""
    # Parse the first JSON value and ignore any text the model adds after it.
    try:
        value, _ = json.JSONDecoder().raw_decode(text.strip())
    except json.JSONDecodeError:
        return "invalid JSON"
    if not isinstance(value, dict) or set(value) != {"city"}:
        return "wrong fields"
    if not isinstance(value["city"], str):
        return "city is not a string"
    return "valid"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--prompt", default=(
        'Reply with only JSON in the form {"city": "..."}.\n'
        "Question: What is the weather in Paris?\n"
        "JSON:"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.device)
    config = DecoderConfig(temperature=args.temperature,
                           max_new_tokens=args.max_new_tokens)
    vocab_size = model.get_output_embeddings().weight.shape[0]

    decoders = {
        "unconstrained": Decoder(model, tokenizer, config),
        "constrained": ConstrainedDecoder(
            model, tokenizer, config, CityGrammar(tokenizer, vocab_size)),
    }
    for name, decoder in decoders.items():
        print(f"== {name}, temperature={args.temperature}")
        valid = 0
        for _ in range(args.runs):
            text = decoder.generate(args.prompt).replace(tokenizer.eos_token, "")
            result = check(text)
            valid += result == "valid"
            print(f"  {result:<20} {text.strip()!r}")
        print(f"  {valid}/{args.runs} valid\n")


if __name__ == "__main__":
    main()
