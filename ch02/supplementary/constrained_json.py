"""Show how a grammar supplies an allowed-token mask to the shared sampler.

This deliberately uses a character vocabulary and fixed logits so the state
machine stays visible. Production constrained decoders apply the same mask at
subword-token boundaries and support a much larger grammar.
"""

import torch

from mini_inference.sampling import Sampler, SamplingParams

TARGET = '{"answer":true}'
VOCABULARY = tuple(dict.fromkeys(TARGET + "fals"))
TOKEN_ID = {character: index for index, character in enumerate(VOCABULARY)}


def allowed_mask(expected_character: str) -> torch.Tensor:
    mask = torch.zeros(len(VOCABULARY), dtype=torch.bool)
    mask[TOKEN_ID[expected_character]] = True
    return mask


def main() -> None:
    # Token 0 always has the largest raw score. The grammar mask, rather than
    # the model preference, determines which token is legal at each state.
    logits = torch.linspace(2.0, -2.0, len(VOCABULARY))
    sampler = Sampler()
    output = []
    for expected_character in TARGET:
        token_id = sampler.select(
            logits,
            SamplingParams(temperature=0),
            allowed_token_mask=allowed_mask(expected_character),
        )
        output.append(VOCABULARY[token_id])
    print("".join(output))


if __name__ == "__main__":
    main()
