# Chapter 2: Autoregressive Decoding and Generation Control

Chapter 2 builds the single-request engine that later chapters extend. The
engine accepts token IDs, performs a cached prefill/decode loop, applies a
per-request sampling policy, emits one event per generated token, and ends with
an explicit finish reason.

## Setup

From the repository root, create an environment with Python 3.10 or newer and
install the chapter dependencies:

```bash
python3 -m pip install -e '.[chapter2]'
```

The required examples default to `HuggingFaceTB/SmolLM2-135M`. The first run
downloads the model. It runs on CPU, Apple Silicon (`mps`), or CUDA; use
`--device` to override automatic selection. Larger instruct-model experiments
are optional because they require substantially more memory.

## Run the examples in order

1. Inspect the logits and probabilities produced by the prompt prefill:

   ```bash
   python3 ch02/examples/01_inspect_next_token.py
   ```

   Observe that the model returns one score for every vocabulary token. A
   sampler chooses among those scores; the model does not return text.

2. Compare deterministic greedy decoding with reproducible seeded sampling:

   ```bash
   python3 ch02/examples/02_greedy_vs_sampling.py
   ```

   The two greedy runs should match. Sampling runs with different seeds can
   diverge even though they use the same prompt and policy.

3. Compare temperature, top-k, top-p, min-p, locally typical sampling, and a
   combined policy:

   ```bash
   python3 ch02/examples/03_compare_sampling_strategies.py
   ```

   Candidate counts explain what each truncation strategy removes. Output
   quality varies with the model, so compare mechanisms rather than treating
   one short continuation as a quality benchmark.

4. Inspect the engine event stream and its finish reason:

   ```bash
   # Ends because it reaches the request's token budget.
   python3 ch02/examples/04_stream_and_stop.py --max-new-tokens 4

   # Ends if the model emits this single-token stop marker first.
   python3 ch02/examples/04_stream_and_stop.py --stop-text .
   ```

   A terminal EOS or stop token is emitted as a raw `TokenOutput` before the
   final `RequestFinished`. Text rendering decides whether special tokens are
   visible. Text stop strings that span multiple tokens belong to the Chapter 5
   service adapter, not the token engine.

The `cancelled` finish reason is reserved in the shared vocabulary. Chapter 4
implements cancellation when it introduces request scheduling and lifecycle
states.

## The two code tracks

Reusable contracts and behavior live in `src/mini_inference/`:

- `sampling.py` owns every logits-to-token transformation.
- `runner.py` owns Hugging Face model execution and `past_key_values`.
- `engine.py` owns the request lifecycle and event stream.

The scripts in `examples/` are chapter labs that import this cumulative
package. `supplementary/` holds beam-search and constrained-decoding material
that later engine chapters do not need to extend.

Run the model-free test suite with:

```bash
python3 -m pip install -e '.[chapter2,test]'
python3 -m pytest
```

Chapter 3 keeps the request, sampling, and event contracts and replaces opaque
framework-owned KV state with explicit memory ownership and accounting.
