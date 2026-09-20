# Building Fast and Reliable LLM Inference Systems

**A Manning Publications book by Tushar Vatsa, Karthik Suresh and Ishita Verma**

A practitioner book that treats LLM inference as a systems engineering discipline — covering decoding, memory, batching, serving, distributed execution, and speculative decoding for teams building production services under latency and cost constraints.

---

## Who This Book Is For

- Machine Learning Engineers optimizing deployed models
- Backend Engineers building AI-powered services
- Platform and SRE Engineers managing inference infrastructure
- Applied Scientists deploying models under real-world constraints

**Prerequisites**: Working knowledge of Python, basic familiarity with ML concepts (neural networks, transformers, embeddings), and some experience deploying or consuming APIs.

---

## Repository Structure

The repository has two complementary tracks. Every chapter is an independently
runnable lab, while `src/mini_inference` is the tested cumulative engine that
later chapters extend.

```
llm-inference-manning/
├── src/mini_inference/       # The educational inference engine, built progressively across chapters
├── ch01/                     # Inference as the New Bottleneck
│   ├── examples/             # Runnable latency, TPOT, and prefill/decode experiments
│   ├── supplementary/        # KV-cache memory calculator (Section 1.3.2)
│   └── diagrams/
├── ch02/                     # Autoregressive Decoding and Generation Control
│   ├── examples/             # Logits, sampling, and streaming demonstrations
│   ├── supplementary/        # Beam search and constrained decoding
│   └── diagrams/
└── tests/                    # Tests for the living src/mini_inference package
```

Chapter examples import the installed package; they never import code from an
earlier chapter directory. Each chapter README gives the exact setup and run
order. Reusable behavior enters `src` only when a later chapter needs to call,
test, or extend it.

More chapters will be added as the book progresses.

---

## Chapters

### Part I — Inference as a Systems Discipline

**Chapter 1: Inference as the New Bottleneck**
Why inference has replaced training as the primary systems bottleneck. Introduces the prefill/decode split, KV cache, and the four interacting constraints (compute, memory capacity, memory bandwidth, scheduling). Defines latency, throughput, and cost as first-class metrics. Readers build a small profiler that measures prefill-to-first-token latency, TPOT, and throughput, then use memory and FLOPs estimates to explain the results. A separate KV-cache calculator requires no model download or GPU.

**Chapter 2: Autoregressive Decoding and Generation Control**
Build the first working token engine: a cached prefill/decode loop, greedy and
stochastic sampling, token-level stopping, streamed output events, and explicit
finish reasons. The required examples use a laptop-friendly 135M-parameter
model; larger quality experiments are optional.

---

## The End-to-End Project

The book culminates in a progressive inference service built across chapters. Starting from the minimal loop in Chapter 2, each chapter adds a layer:

| Chapter | What Gets Added |
|---------|----------------|
| Ch 1 | Inference profiling and KV-cache memory sizing |
| Ch 2 | Autoregressive Decoding and Generation Control |


The final project is a production-oriented teaching system. It makes the core
mechanisms of modern inference stacks concrete while remaining small enough to
read and modify.


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

# Chapter 1 calculator (no GPU, model download, or dependencies)
python3 -m pip install -e .
python3 ch01/supplementary/kv_cache_memory_calculator.py

# Chapter 1 profiler (downloads and caches a 135M-parameter model on first run)
python3 -m pip install -e '.[chapter1]'
python3 ch01/examples/profile_one_request.py

# Chapter 2 token engine (downloads the same small model on first run)
python3 -m pip install -e '.[chapter2]'
python3 ch02/examples/01_inspect_next_token.py
```

---

## Authors

**Tushar Vatsa** — Machine Learning Engineer at Adobe. Master's from Carnegie Mellon University. Published at AAAI, NAACL, ICDM. Builds agentic AI systems and optimizes LLMs for production inference.

**Karthik Suresh** — Machine Learning Engineer at Adobe (6+ years). Published at ICCV, NeurIPS. Builds and deploys LLMs and VLMs end-to-end for large-scale design applications serving millions of users.

**Ishita Verma** - Senior Machine Learning Engineer at Netflix. Master's from University of Maryland. Published at NAACL, and other top-tier conferences, and authored papers on multimodal and multilingual GenAI systems.

---

## License

Code in this repository is provided under the [MIT License](LICENSE). The book content itself is copyright Manning Publications.
