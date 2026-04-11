# Chapter 2 — A Minimal Inference Stack

> **Pre-meeting notes — Tushar Vatsa**
> Putting this down to anchor our discussion. Everything here is open for debate — the goal is to have a concrete starting point so we spend meeting time on decisions, not on generating options from scratch.

---

## Where We Are After Chapter 1

Chapter 1 gave the reader the **conceptual vocabulary**: prefill vs. decode, the KV cache, the four constraints (compute, memory capacity, memory bandwidth, scheduling), and the metrics that govern production systems (latency, throughput, cost). It was deliberately system-level and diagram-driven — no code.

Chapter 2 needs to **make it real**. The reader should finish this chapter having built a working inference pipeline from scratch, run it, measured it, and understood what every piece does. By the end, they should look at the generation loop and think: *"I see exactly where the bottlenecks Ch1 described would show up."*

---

## Guiding Principles

1. **Code-first, theory-as-needed.** Every concept earns its place by explaining what the code does. No standalone theory sections.
2. **Ch2 owns the system; Ch3 owns decoding strategies.** We introduce only greedy and temperature sampling here — enough to run the loop. Top-k, top-p, beam search, constrained/structured decoding, and the full diversity-vs-determinism treatment belong in Ch3.
3. **Try It Now in every section.** Short, runnable exercises with a clear "what you'll see" outcome. Tiered: Starter → Intermediate → Challenge.
4. **Diagrams show what code can't.** Architecture flows, tensor shapes through the model, and the autoregressive loop unrolled over time.

---

## Proposed Outline

### 2.1 — What a Language Model Actually Computes
*~4 pages | 1 diagram | 2 code listings*

The minimal probability foundation needed to understand generation. Not a textbook chapter on probability — a practitioner's answer to *"what comes out of the model and why?"*

**Content:**
- A language model outputs P(next token | all previous tokens) — a conditional distribution over the vocabulary
- The chain rule: how a full sequence probability decomposes into a product of per-token conditionals
- Why we work with log probabilities (numerical stability, additive scoring)
- Demonstrating this with a toy bigram model before jumping to neural LMs

**Diagram — Conditional Probability Heatmap:**
A color-coded grid showing P(next | current) for a small vocabulary. Annotated to show: *"after 'the', nouns are likely; after verbs, prepositions dominate."* Makes the abstract idea of a conditional distribution visually concrete.

**Code Listing 2.1:** A hand-crafted bigram probability table — setting up, sampling from it, computing sequence probabilities.

**Code Listing 2.2:** Computing log P("the cat sat on the mat") step by step, showing the chain rule in action.

**Try It Now: Sampling from a Bigram Model**

| Level | Exercise |
|-------|----------|
| Starter | Examine the conditional probability heatmap. Which word is most predictable? Which is least? |
| Intermediate | Modify the probability table so that "the cat sat on the mat" becomes the single most likely 6-word sentence. What did you change? |
| Challenge | Generate 20 sentences by sampling from the bigram model. How many are grammatical? What does this tell you about the gap between bigram and full-context models? |

> **Key Takeaway:** Every language model — from a bigram table to GPT-4 — is a machine that outputs a probability distribution over the next token. The rest of this chapter is about what happens when you sample from that distribution in a loop.

---

### 2.2 — Inside the Autoregressive Loop
*~6 pages | 2 diagrams | 2 code listings*

The mechanical heart of inference. We open the black box and trace what happens from input tokens to generated output, step by step.

**Content:**
- The forward pass pipeline: token IDs → embeddings → transformer blocks → logits
- What logits are (raw scores, not probabilities) and why softmax converts them
- The autoregressive generation loop: forward pass → sample token → append → repeat
- Sequential dependence: each token depends on every previous token — this is why decode can't be parallelized
- Stopping conditions: EOS token, max length, stop sequences
- Light walkthrough of one transformer block (attention + FFN) — enough to connect "logits" to "learned representations," not a full architecture tutorial

**Diagram — Anatomy of a Forward Pass:**
Vertical flow: Token IDs → Embedding → [Transformer Block x 12] → LayerNorm → Linear → Logits [50,257]. Tensor shapes annotated at each stage for GPT-2 (e.g., `[1, seq_len, 768]`). One transformer block expanded to show attention + FFN.

**Diagram — The Autoregressive Loop Unrolled:**
Horizontal timeline showing 5 decode steps. Each step: the growing sequence, the forward pass, the sampled token appended. Color distinguishes prompt tokens (blue) from generated tokens (green). Side panel shows KV cache growing at each step — connecting back to Ch1's memory discussion.

**Code Listing 2.3:** Key components from `nanogpt.py` — model config, the forward pass method, and CausalSelfAttention — with inline annotations focused on *what matters for inference* (not training).

**Code Listing 2.4:** A minimal 15-line generation loop: load GPT-2, encode prompt, loop (forward → argmax → append), decode. No sampling yet — pure greedy.

**Try It Now: Your First Generation Loop**

| Level | Exercise |
|-------|----------|
| Starter | Run greedy generation with "The future of artificial intelligence is" for 25 tokens. Is the output coherent? Repetitive? |
| Intermediate | Add a print inside the loop showing the top-5 tokens and their probabilities at each step. When is the model confident vs. uncertain? |
| Challenge | Time the loop. How many ms/token on your hardware? Extrapolate to 500 tokens. Now you see why decode latency is the bottleneck Ch1 described. |

> **Key Takeaway:** Autoregressive generation is a sequential loop — each token requires a full forward pass. This is the fundamental reason decode is slow, and why every optimization in this book exists.

---

### 2.3 — Temperature Sampling
*~5 pages | 2 diagrams | 3 code listings*

Greedy decoding is deterministic and repetitive. Temperature is the simplest fix — and the first sampling concept every practitioner needs.

**Content:**
- The problem with greedy: always picks argmax → repetition loops, no diversity
- Temperature scaling: `P(token) = softmax(logits / T)`
  - T → 0: collapses to greedy
  - T = 1.0: unmodified model distribution
  - T → ∞: approaches uniform random
- Intuition: temperature controls the entropy (uncertainty) of the distribution
- Practical guidance: T = 0.0–0.3 for factual tasks, T = 0.7–1.0 for general use, T > 1.0 for creative tasks
- Why this is a *blunt* instrument — it scales the entire distribution uniformly. (Foreshadow: Ch3 introduces surgical tools like top-k and top-p.)

**Diagram — Temperature's Effect on the Distribution:**
Four-panel bar chart showing token probabilities for the same logit vector at T = 0.0, 0.5, 1.0, 1.5. The distribution goes from a single spike → peaked → spread → nearly flat. Entropy value labeled on each panel.

**Diagram — The Diversity-Coherence Spectrum:**
Horizontal slider from "Deterministic / Repetitive (T=0)" to "Random / Incoherent (T→∞)". Sweet-spot region highlighted. Real generated text examples placed along the spectrum.

**Code Listing 2.5:** `temperature_sample()` — the 8-line core function.

**Code Listing 2.6:** `generate_with_temperature()` — the full generation function using temperature sampling.

**Code Listing 2.7:** Generating 3 completions at each of T = [0.0, 0.5, 1.0, 1.5] for the same prompt, printing results side by side.

**Try It Now: Temperature Experiments**

| Level | Exercise |
|-------|----------|
| Starter | Generate at T=0.0 three times. Are all outputs identical? Why? |
| Intermediate | Find the "breaking point": increase T from 1.0 in 0.1 increments. At what value does output become incoherent? |
| Challenge | Implement *dynamic temperature*: T=0.3 for the first 10 tokens, then T=1.0 for the rest. Compare against fixed T=0.3 and T=1.0. Does starting coherent and then going creative produce better text? |

> **Key Takeaway:** Temperature is simple and effective, but it's a uniform scaling of the entire distribution. Chapter 3 will introduce more precise tools: top-k, top-p, beam search, and structured decoding.

---

### 2.4 — Measuring Generation Quality
*~5 pages | 1 diagram | 3 code listings*

You can't improve what you can't measure. This section establishes the evaluation toolkit used throughout the book.

**Content:**
- Why evaluation matters: in production, you can't read every response
- Diversity metrics:
  - Within-text: unique word ratio (simple but limited)
  - Cross-generation: bigram uniqueness across N samples from the same prompt (reveals how much the model "explores")
- Fluency evaluation: the LLM-as-judge pattern
  - Ask an external LLM to rate coherence on a 0–10 scale
  - Limitations: cost, judge bias, non-determinism
- Log probability as a model-intrinsic quality signal (bridge to 2.5)
- The diversity-fluency tradeoff: plotting metrics across temperatures

**Diagram — Evaluation Framework:**
Flow: Generated Texts → three parallel evaluators [Diversity (bigram uniqueness), Fluency (LLM judge), Log Probability (model-intrinsic)] → Metrics Table. Each box annotated with inputs, outputs, and cost (free / API call / forward pass).

**Code Listing 2.8:** `calculate_cross_generation_diversity()` — bigram uniqueness across multiple generations.

**Code Listing 2.9:** `evaluate_fluency()` — calling an external LLM to score text quality.

**Code Listing 2.10:** Full evaluation pipeline: `run_evaluation()` that combines all metrics and prints a summary table.

**Try It Now: Evaluating Your Outputs**

| Level | Exercise |
|-------|----------|
| Starter | Run `run_evaluation_demo()` on your temperature outputs. Which temperature wins on fluency? On diversity? |
| Intermediate | Plot diversity (y) vs. fluency (x) across temperatures. Is there a Pareto-optimal temperature? |
| Challenge | Modify the fluency scorer to make 3 independent LLM calls and average them. Does variance decrease? What does this tell you about LLM-as-judge reliability? |

> **Key Takeaway:** Generation quality is multi-dimensional — no single number captures "good." The tension between diversity and fluency is controlled by decoding strategy, and measuring it quantitatively is essential for production systems.

---

### 2.5 — Cross-Model Evaluation: When Models Judge Models
*~5 pages | 2 diagrams | 2 code listings*

> **Discussion point:** This section could alternatively live in Ch4 (Scaling Reasoning) alongside best-of-N and reward models. Keeping it here gives Ch2 a strong finish and plants seeds for later chapters. Open to debate.

**Content:**
- The pattern: generate with a small/fast model, score with a larger/better model
- Log probability scoring: computing log P(text | model) from per-token log probs
- Experiment: GPT-2 small (117M) generates, GPT-2 medium (345M) evaluates
- Results and analysis:
  - When do the two models agree vs. disagree?
  - Temperature's effect on cross-model agreement
  - What "false confidence" looks like (small model loves it, large model doesn't)
- Why this matters for production — three patterns that build on this idea:
  - *Best-of-N sampling* — generate N, pick the best (→ Ch4)
  - *Speculative decoding* — draft with small, verify with large (→ Ch9)
  - *Reward model ranking* — score candidates with a trained evaluator (→ Ch4)

**Diagram — Cross-Model Evaluation Pipeline:**
Left: Small model generates N candidates. Middle: Both models score each candidate. Right: Scatter plot of small_score vs. large_score, with quadrants labeled (agree-good, agree-bad, small-likes/large-doesn't, large-likes/small-doesn't).

**Diagram — The Generate-Then-Verify Pattern:**
Abstract flow: Fast Model → Candidates → Scoring Model → Best Output. Three concrete instantiations shown below with chapter references: Best-of-N (Ch4), Speculative Decoding (Ch9), RLHF Reward Ranking (Ch4).

**Code Listing 2.11:** `calculate_log_probability()` — computing sequence log probability under a model.

**Code Listing 2.12:** The full cross-model experiment: generate, score with both models, display comparison.

**Try It Now: Cross-Model Analysis**

| Level | Exercise |
|-------|----------|
| Starter | Generate 10 completions at T=0.8 and score with both models. Do they rank the completions the same way? |
| Intermediate | Find a completion where the small model gives a high score but the medium model gives a low score. Read it — what went wrong? |
| Challenge | Build a best-of-N sampler: generate N=10 candidates with the small model, select the one the medium model scores highest. Compare against a single generation. Is the quality difference visible? |

> **Key Takeaway:** Smaller models can generate candidates cheaply; larger models can judge them. This generate-then-verify pattern underpins speculative decoding, reward-model ranking, and best-of-N — all covered later in the book.

---

### 2.6 — From Notebook to Production: What's Missing
*~2 pages | 1 diagram (annotated recap)*

A short, no-code bridge section. The reader has a working pipeline — now we show them the gap between that and a production system.

**Content:**
- What we built: single-request, single-user, CPU, unbounded memory, one sampling strategy
- What production demands (each bullet foreshadows a later chapter):

| Gap | What's Missing | Where It's Covered |
|-----|----------------|-------------------|
| Decoding | Only greedy + temperature; no top-k, top-p, beam search, structured output | Ch 3 |
| Reasoning | No multi-pass, no chain-of-thought, no best-of-N at scale | Ch 4 |
| Hardware | Running on CPU; GPU memory hierarchy changes everything | Ch 5 |
| Memory | KV cache grows unbounded; no paging, no eviction | Ch 6 |
| Batching | One request at a time; no concurrent serving | Ch 7 |
| Compression | Full precision weights and cache; no quantization | Ch 8 |
| Speed | No speculative decoding, no draft models | Ch 9 |
| Frameworks | Hand-rolled loop; no vLLM, SGLang, TensorRT-LLM | Ch 10 |
| Scale | Single GPU; no distribution, no routing | Ch 11 |
| Operations | No SLOs, no monitoring, no cost tracking | Ch 12 |

**Diagram — The Canonical Inference Architecture (annotated):**
Reuse the system diagram from Ch1 (Figure 1.3), now with chapter numbers overlaid on each component. This becomes the reader's roadmap for the rest of the book.

---

### 2.7 — Summary
*~0.5 pages*

- A language model outputs a conditional probability distribution over the next token. Generation means sampling from this distribution in a loop.
- The autoregressive loop is sequential: one forward pass per token, making decode latency proportional to output length.
- Temperature is the simplest diversity-coherence control, but it's a blunt instrument — Ch3 introduces finer tools.
- Generation quality is multi-dimensional. Measuring diversity, fluency, and log probability systematically is essential.
- Smaller models can generate cheaply; larger models can verify. This pattern shows up throughout the book.
- The gap between a working notebook and a production service spans memory, batching, hardware, and scheduling — the subject of everything that follows.

---

## Chapter 2 at a Glance

| Section | Pages | Diagrams | Code Listings | Try It Now |
|---------|-------|----------|---------------|------------|
| 2.1 What a Language Model Computes | ~4 | 1 | 2 | Yes |
| 2.2 Inside the Autoregressive Loop | ~6 | 2 | 2 | Yes |
| 2.3 Temperature Sampling | ~5 | 2 | 3 | Yes |
| 2.4 Measuring Generation Quality | ~5 | 1 | 3 | Yes |
| 2.5 Cross-Model Evaluation | ~5 | 2 | 2 | Yes |
| 2.6 From Notebook to Production | ~2 | 1 | 0 | No |
| 2.7 Summary | ~0.5 | 0 | 0 | No |
| **Total** | **~28–32** | **9** | **12** | **5** |

---

## Relationship to Chapter 3

To avoid overlap, here's the explicit split:

| Chapter 2 Owns | Chapter 3 Owns |
|-----------------|----------------|
| Conditional probability basics | Logits, softmax, numerical stability (in depth) |
| Greedy decoding + temperature | Top-k, top-p (nucleus), beam search |
| The autoregressive loop mechanics | Why decoding strategy dominates latency |
| Evaluation basics (diversity, fluency, log prob) | Diversity vs. determinism (formal treatment) |
| Cross-model evaluation intro | Controlled generation, constraints |
| | Constrained and structured decoding |
| | JSON mode, grammar constraints, tool invocation |

Ch2 = *"build the loop, run it, measure it"*
Ch3 = *"control how tokens are chosen"*

---

## Open Questions for Discussion

1. **Section 2.5 placement:** Cross-model evaluation is interesting but makes the chapter longer. Should it stay in Ch2 or move to Ch4 (where best-of-N and reward models are covered in depth)?

2. **Probability foundations depth (2.1):** The reference code covers joint distributions, marginalization, latent variables, MLE. Our audience is ML engineers. Do we trim to just conditional probability + log probs, or include a fuller refresher?

3. **nanogpt walkthrough depth (2.2):** Three options:
   - (a) Black-box: *"here's the model, call forward()"*
   - (b) One-block walkthrough: trace through attention + FFN once
   - (c) Full architecture walkthrough
   Leaning toward (b).

4. **Try It Now tiering:** Current proposal uses Starter / Intermediate / Challenge. Do we like this, or keep exercises flat?
