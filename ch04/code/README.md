# Chapter 4: Schedulers and Batching

Code for Chapter 4. `scheduler.py` holds the scheduler built from the
chapter's listings. The other scripts run it on the chapter's scenarios.

| File | Section | What it contains |
|------|---------|------------------|
| `scheduler.py` | 4.1–4.4 | `RequestStatus` (Listing 4.2), `SchedulerState` (4.1), the step-loop methods (4.3), admission under the KV budget (4.6), preemption (4.7), the token budget with chunked prefill (Equation 4.1), `run_static` (4.4), `run_continuous` (4.5), and TTFT, TPOT, throughput, and goodput (Table 4.1). Run it to trace Requests A, B, C, and D pass by pass. |
| `static_vs_continuous.py` | 4.2.4, 4.3 | Figure 4.7, figure 4.6, and the section 4.2.4 Try It Yourself in simulation |
| `memory_pressure.py` | 4.4.1, 4.4.2 | Admission stopping on KV memory, preemption, and recomputation under different cache sizes and batch limits |
| `chunked_prefill.py` | 4.1.3, 4.4.3 | The Pause and Think in section 4.1.4: a 4,000-token prompt arriving while eight requests stream, under the three choices in figure 4.4 |

Everything uses only the Python standard library and runs in seconds on a
laptop. No GPU or model is needed.

## A simulated model executor

The scheduler is real. The forward pass is simulated, so that every scenario
runs anywhere and prints the same numbers each time. A pass takes a fixed time
to read the weights plus a cost per token it processes, by default 5 ms plus
0.04 ms per token, roughly Llama 3 8B on one H100. A decode step for 32
requests then takes about 6 ms, and prefilling a 4,000-token prompt about
165 ms. Change the cost model with `CostModel(base_ms=..., per_token_ms=...)`.

Each request carries the step at which the simulated model emits its stop
token. Only the executor reads it. The scheduler never does, just as a real
scheduler never knows how long a response will be.

## Running the code

```bash
cd ch04/code
python scheduler.py                       # Requests A-D, pass by pass
python scheduler.py --token-budget 4096   # the same with a larger budget
python static_vs_continuous.py
python memory_pressure.py
python chunked_prefill.py
python chunked_prefill.py --prompt 8000 --budgets 4096 1024 256
```

### What to look for

- **`scheduler.py`**: A arrives at 12 ms, joins at the next pass, and
  contributes its six prompt tokens in one pass (figure 4.2). While B's
  2,000-token prompt shares the budget, C and D's next tokens slow from about
  5 ms to 25 ms.
- **`static_vs_continuous.py`**: under static batching the slots of finished
  requests sit empty (figure 4.7), only about 40% of slots do useful work in
  figure 4.6, and short requests wait for the longest one in their batch.
  Continuous batching refills slots after every pass.
- **`memory_pressure.py`**: with a small KV cache, admission stops before the
  batch is full, and growing requests force preemptions that recompute their
  prompts. Lowering the batch limit (vLLM's `max_num_seqs`) reduces them.
- **`chunked_prefill.py`**: prefilling the prompt in one pass stalls every
  stream for about 165 ms. Holding it back keeps the streams smooth but delays
  its first token. Smaller chunks move between the two.

## Where the code goes beyond the listings

The listings show the shape of each method. To run end to end, a few need
details the book leaves out:

- `run_static` (Listing 4.4) calls `update_requests` and `record_metrics`
  inside its inner loop, as `run_continuous` (Listing 4.5) does. Without
  `update_requests`, no request in a static batch would ever finish.
- `admit_from_waiting` (Listing 4.6) allocates the blocks it checked for, so
  two requests cannot both be admitted against the same free blocks.
- `make_room` (Listing 4.7) preempts the newest request that is still
  running. A static batch keeps finished requests in the running set until
  the batch ends, and those must not be preempted. It also resets the
  victim's progress so that its cache is recomputed when it resumes.
