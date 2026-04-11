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

Each chapter has its own directory with code examples and diagrams referenced in the text. The `end-to-end-project/` directory contains a progressive inference service that integrates techniques from across the book.

```
llm-inference-manning/
├── ch01/                    # Inference Is the New Bottleneck
│   ├── code/
│   └── diagrams/
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

### Part II — How Generation Actually Works

**Chapter 3: Decoding and Generation Policies**
How token generation works and why decoding strategy choices dominate inference performance. Covers top-k, top-p (nucleus sampling), beam search, diversity vs. determinism, constrained/structured decoding, JSON mode, grammar constraints, and tool invocation.

### Part III — Inference-Time Scaling and Reasoning

**Chapter 4: Scaling Reasoning Without Retraining**
Inference-time techniques that improve reasoning without modifying model weights: chain-of-thought, best-of-N decoding, reranking, self-refinement, adaptive compute, and multi-token prediction. Focuses on quality-latency-cost tradeoffs.

### Part IV — Memory and GPU Systems

**Chapter 5: GPU Execution Model for Transformer Inference**
How transformer inference executes on GPUs: memory hierarchy, attention kernels, prefill vs. decode workload profiles. Builds hardware-level intuition for performance bottlenecks. Covers GQA/MQA, sparse attention, and profiling.

**Chapter 6: KV-Cache Architecture and Memory Management**
The KV-cache as the dominant memory bottleneck. Paged memory layouts, fragmentation, PagedAttention, cache eviction, prefix reuse, and why long context degrades latency.

### Part V — Batching, Scheduling, and Throughput

**Chapter 7: Batching and Cache-Aware Scheduling**
Dynamic and in-flight batching, token-level scheduling, prefill-decode interleaving, prefix reuse, radix trees, cache-aware routing, and tail latency amplification.

### Part VI — High-Performance Inference Techniques

**Chapter 8: Quantization and Memory Compression**
Post-training quantization, weight-only quantization (GPTQ), KV-cache quantization, compression techniques (KIVI, GEAR), and prefix caching. Choosing the right compression strategy for your workload.

**Chapter 9: Speculative Decoding**
Draft-model execution for throughput improvement. Draft models, acceptance rates, Medusa, multi-token speculation, and failure modes.

### Part VII — Framework Internals

**Chapter 10: Inference Framework Internals: vLLM, SGLang, TensorRT-LLM**
How modern frameworks implement memory management, scheduling, and kernel optimization. Comparative trade-offs: PagedAttention vs. RadixAttention vs. compiled inference.

### Part VIII — Distributed and Production Inference

**Chapter 11: Distributed Inference Systems**
Multi-GPU and multi-node: tensor/pipeline parallelism, MoE expert parallelism, disaggregated prefill/decode, LLM+SLM routing, failure handling, canary deployments.

### Part IX — Performance, Cost, and Reliability Engineering

**Chapter 12: Performance and Cost Engineering**
SLOs, P50 vs. P99 latency, cost per million tokens, autoscaling, backpressure, observability, benchmarking methodology, and regression testing for inference optimizations.

### Part X — End-to-End Synthesis

**Chapter 13: Building and Evolving a Production Inference Service**
Synthesizes all techniques into a cohesive production system. Sub-100ms latency targets, cost/performance tuning, and inference-time scaling trends.

---

## The End-to-End Project

The book culminates in a progressive inference service built across chapters. Starting from the minimal loop in Chapter 2, each chapter adds a layer:

| Chapter | What Gets Added |
|---------|----------------|
| Ch 2 | Minimal autoregressive generation loop |
| Ch 3 | Configurable decoding strategies |
| Ch 5 | GPU-aware execution |
| Ch 6 | Paged KV-cache management |
| Ch 7 | Continuous batching and scheduling |
| Ch 8 | Quantized model and cache |
| Ch 9 | Speculative decoding |
| Ch 10 | Framework integration (vLLM/SGLang) |
| Ch 11 | Multi-GPU serving and routing |
| Ch 12 | SLO monitoring and autoscaling |
| Ch 13 | Full production deployment |

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

---

## License

Code in this repository is provided under the [MIT License](LICENSE). The book content itself is copyright Manning Publications.
