# Chapter 2: Autoregressive Decoding and Generation Control

Code for Chapter 2. `decoder.py` is the complete decoder from Listing 2.10. The
other scripts use it to compare sampling strategies, run beam search, and
constrain the output to a fixed JSON format.

| File | Section | What it contains |
|------|---------|------------------|
| `decoder.py` | 2.1–2.3 | `DecoderConfig` and `Decoder` from Listing 2.10: the autoregressive loop, greedy decoding, temperature, top-k, top-p, min-p, locally typical sampling, and the repetition penalty |
| `compare_strategies.py` | 2.3 | Runs each strategy on the same prompt and prints how many candidate tokens survive at chosen decode steps |
| `beam_search.py` | 2.4 | A compact beam search, compared with greedy decoding |
| `constrained_decoding.py` | 2.5 | `ConstrainedDecoder` from Listing 2.12 with a finite-state machine for `{"city": "<string>"}` |

Listings 2.1 to 2.9 build `Decoder` one step at a time, and Listing 2.10 is
the result, so they all live in `decoder.py`.

## Two strategies that are not in the chapter text

**Min-p** keeps every token whose probability is at least `min_p` times the
probability of the most likely token. When the model is confident, the cutoff
is high and few tokens survive. When it is unsure, the cutoff drops and more
tokens survive. This keeps sampling stable at high temperatures, where top-k
and top-p let in too many unlikely tokens.

**Locally typical sampling** (Meister et al., 2022) measures each token's
surprise, `-log p`. The average surprise of the distribution is its entropy.
The filter keeps the tokens whose surprise is closest to the entropy, adding
them until their probabilities sum to `typical_p`. It drops tokens that are
far more predictable than usual as well as the unlikely tail, which can reduce
repetitive or bland text.

## Setup

Python 3.10 or later.

```bash
cd ch02/code
pip install -r requirements.txt
```

Every script defaults to `Qwen/Qwen2.5-0.5B-Instruct`, which is small enough
to run on a laptop CPU and needs no Hugging Face account. The first run
downloads it (about 1 GB). To use the model from the chapter, pass
`--model meta-llama/Llama-3.1-8B-Instruct`. That model needs you to accept its
license on Hugging Face, log in with `huggingface-cli login`, and have about
16 GB of free GPU or unified memory.

Scripts pick CUDA, then Apple Silicon (`mps`), then CPU. Use `--device` to
choose, and `--seed` to change the random seed.

## Running the code

### The decoder

```bash
python decoder.py --temperature 0                     # greedy (2.2)
python decoder.py --temperature 0.6 --top-p 0.9       # Llama 3 defaults (2.3.4)
python decoder.py --temperature 0.6 --top-k 20 --top-p 0.95   # Qwen3 defaults (2.3.5)
python decoder.py --temperature 1.5 --min-p 0.1       # min-p at high temperature
python decoder.py --temperature 1.0 --typical-p 0.9   # locally typical sampling
python decoder.py --prompt "Write a poem about the ocean." --repetition-penalty 1.1 --max-new-tokens 200
```

Use the `Decoder` from your own code:

```python
from decoder import Decoder, DecoderConfig, load_model

model, tokenizer = load_model("Qwen/Qwen2.5-0.5B-Instruct")
config = DecoderConfig(temperature=0.6, top_p=0.9, max_new_tokens=100)
print(Decoder(model, tokenizer, config).generate("The capital of France is"))
```

### Comparing strategies

```bash
python compare_strategies.py
python compare_strategies.py --steps 1 5 10 --max-new-tokens 10
python compare_strategies.py --strategies top-k top-p --repetition-penalty 1.1
```

The truncation strategies all use temperature 1.0, so the filter is the only
thing that changes. Notice that top-k keeps exactly 50 candidates at every
step, while top-p, min-p, and locally typical sampling keep more candidates
when the model is unsure and fewer when it is confident.

### Beam search

```bash
python beam_search.py
python beam_search.py --num-beams 5 --max-new-tokens 30
```

Greedy decoding is beam search with one beam, so the script runs both and
prints each output with its cumulative log probability. Beam search usually
finds a sequence with a higher (less negative) log probability than greedy,
though it is not guaranteed to. For clarity, each step reruns the model on
every full beam instead of reusing a KV cache.

### Constrained decoding

```bash
python constrained_decoding.py
python constrained_decoding.py --runs 20 --temperature 1.2
```

The script asks for `{"city": "..."}` several times with and without the
constraint and reports how many outputs parse. With the constraint every
completed output is valid JSON with exactly one string field, `city`. An
output can still fail if it hits `--max-new-tokens` before the closing brace.
The constraint guarantees the structure, not the content: the model can still
pick the wrong city.

The first constrained run takes a few seconds longer because the grammar
checks every vocabulary token once for each new state and caches the result.
