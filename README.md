# Building Fast and Reliable LLM Inference Systems

**A Manning Publications book by Tushar Vatsa and Karthik Suresh**

A practitioner book that treats LLM inference as a systems engineering discipline — covering decoding, memory, batching, speculative execution, and inference-time reasoning for teams building production services under latency and cost constraints.

---

## Who This Book Is For

- Machine Learning Engineers optimizing deployed models
- Backend Engineers building AI-powered services
- Platform and SRE Engineers managing inference infrastructure
- Applied Scientists deploying models under real-world constraints

**Prerequisites**: Working knowledge of Python, basic familiarity with ML concepts (neural networks, transformers, embeddings), and some experience deploying or consuming APIs.

---

## Repository Structure

Each chapter has its own directory with code examples and diagrams referenced in the text.

```
llm-inference-manning/
├── ch01/                    # Inference Is the New Bottleneck
│   ├── code/
│   └── diagrams/
├── ch02/                    # A Minimal Inference Stack
│   ├── code/
│   └── diagrams/
```

More chapters will be added as the book progresses.

---

## Chapters

### Part I — Inference as a Systems Discipline

**Chapter 1: Inference Is the New Bottleneck**
Why inference has replaced training as the primary systems bottleneck. Introduces the prefill/decode split, KV cache, and the four interacting constraints (compute, memory capacity, memory bandwidth, scheduling). Defines latency, throughput, and cost as first-class metrics.

**Chapter 2: A Minimal Inference Stack**
Hands-on chapter. Build a working inference pipeline from scratch using GPT-2: probability foundations, the autoregressive generation loop, temperature sampling, and generation evaluation. Every section includes runnable code and Try It Now exercises.

---

## Running the Code

### Requirements
- Python 3.10+
- PyTorch 2.0+
- See individual chapter directories for specific dependencies

### Quick Start
```bash
# Clone the repo
git clone https://github.com/tusharvatsa32/llm-inference-manning.git
cd llm-inference-manning

# Chapter 2 example (CPU, no GPU needed)
cd ch02/code
pip install torch tiktoken litellm python-dotenv matplotlib numpy
python -c "from generation import demonstrate_temperature_sampling; demonstrate_temperature_sampling()"
```

---

## Authors

**Tushar Vatsa** — Machine Learning Engineer at Adobe. Master's from Carnegie Mellon University. Published at AAAI, NAACL, ICDM. Builds agentic AI systems and optimizes LLMs for production inference.

**Karthik Suresh** — Machine Learning Engineer at Adobe (6+ years). Published at ICCV, NeurIPS. Builds and deploys LLMs and VLMs end-to-end for large-scale design applications serving millions of users.

---

## License

Code in this repository is provided under the [MIT License](LICENSE). The book content itself is copyright Manning Publications.
