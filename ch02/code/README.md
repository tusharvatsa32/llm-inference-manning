# Chapter 2: A Minimal Inference Stack — Code

Code examples for Chapter 2. These accompany the chapter's hands-on walkthrough of building an inference pipeline from scratch.

## Files

| File | Section | What It Covers |
|------|---------|----------------|
| `nanogpt.py` | 2.2 | GPT-2 model implementation (tokenizer, causal self-attention, full transformer) |
| `probability.py` | 2.1 | Probability fundamentals: bigram models, conditional probability, joint distributions, sampling |
| `generation.py` | 2.2–2.4 | Temperature sampling, autoregressive generation loop, diversity/fluency evaluation |
| `meta_generation.py` | 2.5 | Cross-model evaluation: generate with GPT-2 small, score with GPT-2 medium |
| `plotting_utils.py` | — | Visualization helpers for probability distributions and evaluation metrics |

## Notebooks

Interactive Jupyter notebook versions of the Python files above:
- `probability.ipynb` — Probability foundations (Section 2.1)
- `generation.ipynb` — Temperature sampling and evaluation (Sections 2.2–2.4)
- `meta_generation.ipynb` — Cross-model evaluation (Section 2.5)

## Quick Start

```bash
# Install dependencies
pip install torch tiktoken litellm python-dotenv matplotlib numpy

# Run the generation demo (CPU, no GPU needed)
python -c "from generation import demonstrate_temperature_sampling; demonstrate_temperature_sampling()"

# Or open the notebooks
jupyter notebook generation.ipynb
```

## Hardware Requirements

- **Sections 2.1–2.4**: CPU is sufficient (GPT-2 base is 117M parameters)
- **Section 2.5**: Loads two models (~460M total). 4GB+ RAM recommended. GPU optional but faster.

