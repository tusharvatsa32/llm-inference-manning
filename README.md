# Building Fast and Reliable LLM Inference Systems

**A Manning Publications book by Tushar Vatsa, Karthik Suresh and Ishita Verma**

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

Each chapter has its own directory with code examples and diagrams referenced in the text. The `end-to-end-project/` directory contains a progressive inference service that integrates techniques from across the book.

```
llm-inference-manning/
├── ch01/                    # Inference Is the New Bottleneck : No Code for this
│   ├── code/
├── ch02/                    # A Minimal Inference Stack
│   ├── code/
│   └── diagrams/
├── ch03/                    # Decoding and Generation Policies
│   ├── code/
│   └── diagrams/
├── ch04/                    # Scaling Reasoning Without Retraining
│   ├── code/
│   └── diagrams/
├── ch05/                    # GPU Execution Model for Transformer Inference
│   ├── code/
│   └── diagrams/
├── ch06/                    # KV-Cache Architecture and Memory Management
│   ├── code/
│   └── diagrams/
├── ch07/                    # Batching and Cache-Aware Scheduling
│   ├── code/
│   └── diagrams/
├── ch08/                    # Quantization and Memory Compression
│   ├── code/
│   └── diagrams/
├── ch09/                    # Speculative Decoding
│   ├── code/
│   └── diagrams/
├── ch10/                    # Framework Internals: vLLM, SGLang, TensorRT-LLM
│   ├── code/
│   └── diagrams/
├── ch11/                    # Distributed Inference Systems
│   ├── code/
│   └── diagrams/
├── ch12/                    # Performance and Cost Engineering
│   ├── code/
│   └── diagrams/
├── ch13/                    # Building a Production Inference Service
│   ├── code/
│   └── diagrams/
└── end-to-end-project/      # Progressive inference service built across chapters
```

---

## Table of Contents

### Part I — Inference as a Systems Discipline

**Chapter 1: Inference Is the New Bottleneck**
Why inference has replaced training as the primary systems bottleneck. Introduces the prefill/decode split, KV cache, and the four interacting constraints (compute, memory capacity, memory bandwidth, scheduling). Defines latency, throughput, and cost as first-class metrics.

**Chapter 2: A Minimal Inference Stack**
Hands-on chapter. Build a working inference pipeline from scratch using GPT-2: probability foundations, the autoregressive generation loop, temperature sampling, and generation evaluation. Every section includes runnable code and Try It Now exercises.
---

## The End-to-End Project

The book culminates in a progressive inference service built across chapters. Starting from the minimal loop in Chapter 2, each chapter adds a layer:

| Chapter | What Gets Added |
|---------|----------------|
| Ch 1 | Inference is the new Bottleneck |
| Ch 2 | Minimal autoregressive generation loop |


The final system is representative of modern production inference stacks (vLLM, SGLang) — not a toy, but a real system the reader can extend.

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
pip install -r requirements.txt
python autoregressive_loop.py
```

---

## Authors

**Tushar Vatsa** — Machine Learning Engineer at Adobe. Master's from Carnegie Mellon University. Published at AAAI, NAACL, ICDM. Builds agentic AI systems and optimizes LLMs for production inference.

**Karthik Suresh** — Machine Learning Engineer at Adobe (6+ years). Published at ICCV, NeurIPS. Builds and deploys LLMs and VLMs end-to-end for large-scale design applications serving millions of users.

**Ishita Verma** - Senior Machine Learning Engineer at Netflix. Master's from University of Maryland. Published at NAACL, and other top-tier conferences, and authored papers on multimodal and multilingual GenAI systems.

---

## License

Code in this repository is provided under the [MIT License](LICENSE). The book content itself is copyright Manning Publications.
