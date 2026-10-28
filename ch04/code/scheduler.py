"""A scheduler for LLM serving, with a simulated model executor.

Chapter 4. The scheduler is built from the chapter's listings:

    RequestStatus          the request lifecycle (Listing 4.2)
    SchedulerState         waiting queue, running set, ledger (Listing 4.1)
    Scheduler              the step-loop methods (Listing 4.3), admission
                           under the KV budget (Listing 4.6), preemption
                           (Listing 4.7), and the per-step token budget with
                           chunked prefill (Equation 4.1, section 4.4.3)
    run_static             static batching (Listing 4.4)
    run_continuous         continuous batching (Listing 4.5)

The model executor is simulated so that every experiment runs on a laptop and
gives the same numbers each time. A forward pass takes a fixed time to read
the weights plus a cost per token it processes. The defaults are rough
figures for Llama 3 8B on one H100: about 5 ms per pass plus 0.04 ms per
token, so a decode step for 32 requests takes about 6 ms and prefilling a
4,000-token prompt takes about 165 ms.

Each request knows the step at which its model would emit the stop token
(output_len). Only the simulated model reads it. The scheduler never does,
just as a real scheduler never knows how long a response will be.

Run this file to trace Requests A, B, C, and D from section 4.1 step by step:
    python scheduler.py
    python scheduler.py --token-budget 2048
"""

import argparse
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

BLOCK_SIZE = 16  # tokens per KV cache block, as in Chapter 3


# Listing 4.2: the request lifecycle.
class RequestStatus(Enum):
    WAITING = auto()    # arrived and tokenized, not yet on the GPU
    RUNNING = auto()    # admitted: prefilling, then decoding one token per step
    FINISHED = auto()   # stop token or max_tokens reached; KV blocks freed
    PREEMPTED = auto()  # bumped off under memory pressure, back to WAITING (Section 4.4)


@dataclass(eq=False)
class Request:
    name: str
    prompt_len: int
    output_len: int                 # the simulated model's stop point; the scheduler never reads it
    arrival: float = 0.0            # ms
    status: RequestStatus = RequestStatus.WAITING
    computed: int = 0               # positions whose keys and values are in the KV cache
    generated: int = 0              # output tokens produced so far
    blocks: list = field(default_factory=list)
    token_times: list = field(default_factory=list)
    finish_time: float | None = None
    preemptions: int = 0
    recomputed: int = 0             # tokens prefilled again after preemption

    @property
    def target(self):
        """Positions to process before the next token: prompt plus output so far."""
        return self.prompt_len + self.generated

    @property
    def remaining(self):
        return self.target - self.computed

    @property
    def is_decoding(self):
        return self.generated > 0 and self.remaining == 1

    # Latency metrics from section 1.3.1, measured inside the server.
    @property
    def ttft(self):
        return self.token_times[0] - self.arrival

    @property
    def tpot(self):
        if len(self.token_times) < 2:
            return 0.0
        return (self.token_times[-1] - self.token_times[0]) / (len(self.token_times) - 1)

    @property
    def max_itl(self):
        gaps = [b - a for a, b in zip(self.token_times, self.token_times[1:])]
        return max(gaps, default=0.0)

    @property
    def e2e(self):
        return self.finish_time - self.arrival


def blocks_to_start(req):
    """KV cache blocks needed to hold the request's prompt (and any output
    it must recompute after preemption)."""
    return math.ceil(req.target / BLOCK_SIZE)


class ResourceLedger:
    """Free KV cache blocks and the per-step token budget."""

    def __init__(self, total_blocks, token_budget):
        self.total_blocks = total_blocks
        self.token_budget = token_budget        # vLLM: max_num_batched_tokens
        self.free_list = list(range(total_blocks))

    @property
    def free_blocks(self):
        return len(self.free_list)

    def allocate(self, n):
        taken = [self.free_list.pop() for _ in range(n)]
        return taken

    def free(self, blocks):
        self.free_list.extend(blocks)


# Listing 4.1: what the scheduler sees.
@dataclass
class SchedulerState:
    waiting: deque          # arrived, not yet running  (A, B)
    running: list           # on the GPU now            (C, D)
    ledger: ResourceLedger  # free KV blocks, per-step token budget


@dataclass
class CostModel:
    """Simulated forward-pass time: read the weights once, then pay per token."""
    base_ms: float = 5.0
    per_token_ms: float = 0.04

    def step_ms(self, tokens):
        return self.base_ms + self.per_token_ms * tokens


class Batch:
    """The requests in one batch and the tokens each processes this step."""

    def __init__(self, members):
        self.members = list(members)
        self.tokens = None  # request -> tokens this step; planned before each pass

    def all_finished(self):
        return all(r.status != RequestStatus.RUNNING for r in self.members)


@dataclass
class StepOutput:
    time: float
    duration: float
    tokens: dict              # request -> tokens processed this step
    produced: list            # requests that emitted a token this step
    batch_size: int           # requests in the batch, finished or not


class Scheduler(SchedulerState):
    def __init__(self, requests, total_blocks=10_000, token_budget=2048,
                 chunked_prefill=True, cost=None):
        super().__init__(deque(), [], ResourceLedger(total_blocks, token_budget))
        self.pending = sorted(requests, key=lambda r: r.arrival)  # not arrived yet
        self.finished = []
        self.chunked_prefill = chunked_prefill
        self.cost = cost or CostModel()
        self.clock = 0.0
        self.steps = []
        self.num_preemptions = 0

    # --- Listing 4.3: the methods of the scheduling loop ------------------

    def has_work(self):
        return bool(self.pending or self.waiting or self.running)

    # Listing 4.6: admission under the KV budget.
    def admit_from_waiting(self, up_to):
        self.receive_arrivals()
        while self.waiting and len(self.running) < up_to:
            req = self.waiting[0]
            if self.ledger.free_blocks < blocks_to_start(req):
                break                       # batch has room, but KV-cache memory does not
            self.running.append(self.waiting.popleft())
            req.blocks = self.ledger.allocate(blocks_to_start(req))
            req.status = RequestStatus.RUNNING

    def form_batch(self):
        batch = Batch(self.running)
        batch.tokens = self.plan(batch)
        return batch

    def step(self, batch):
        """Run one forward pass over the batch."""
        if batch.tokens is None:                  # a frozen batch plans each pass
            batch.tokens = self.plan(batch)
        tokens = batch.tokens
        batch.tokens = None

        if not tokens:
            if self.pending:
                self.clock = max(self.clock, self.pending[0].arrival)  # idle until the next arrival
            elif any(r.status == RequestStatus.RUNNING for r in self.running):
                raise RuntimeError("running requests cannot be scheduled; is a prompt "
                                   "larger than the token budget without chunked prefill?")
            elif not self.waiting:
                raise RuntimeError("no request can run and none are arriving")
            # Otherwise every request in the batch was preempted; the loop
            # goes back to admission and resumes them.
            return StepOutput(self.clock, 0.0, {}, [], len(batch.members))

        duration = self.cost.step_ms(sum(tokens.values()))
        self.clock += duration
        produced = []
        for req, n in tokens.items():
            req.computed += n
            if req.remaining == 0:  # its whole context is processed: it emits a token
                produced.append(req)
        return StepOutput(self.clock, duration, tokens, produced, len(batch.members))

    def update_requests(self, outputs):
        for req in outputs.produced:
            req.generated += 1
            req.token_times.append(outputs.time)
            if req.generated >= req.output_len:  # the model emitted its stop token
                req.status = RequestStatus.FINISHED
                req.finish_time = outputs.time

    def retire_finished(self, outputs):
        for req in [r for r in self.running if r.status == RequestStatus.FINISHED]:
            self.running.remove(req)
            self.ledger.free(req.blocks)
            req.blocks = []
            self.finished.append(req)

    def record_metrics(self, outputs):
        if outputs.tokens:
            self.steps.append(outputs)

    # Listing 4.7: preemption.
    def make_room(self):
        # a running request needs a block and none are free
        while self.ledger.free_blocks == 0 and self.running:
            # Newest request admitted that is still running. A static batch keeps
            # finished requests in the running set until the batch ends; those
            # cannot be preempted.
            still_running = [r for r in self.running if r.status == RequestStatus.RUNNING]
            if not still_running:
                break
            victim = still_running[-1]
            self.running.remove(victim)
            self.ledger.free(victim.blocks)        # reclaim its KV blocks
            victim.status = RequestStatus.PREEMPTED
            self.waiting.appendleft(victim)        # resume it ahead of new arrivals
            # Recompute: its keys and values are gone and must be rebuilt.
            victim.blocks = []
            victim.recomputed += victim.computed
            victim.computed = 0
            victim.preemptions += 1
            self.num_preemptions += 1

    # --- Equation 4.1: fill the per-step token budget ---------------------

    def plan(self, batch):
        """Choose how many tokens each batch member processes this pass.

        Decodes go first, one token each. Prefills share what is left of the
        budget in admission order. With chunked prefill a prompt can take
        any slice of it; without, a prompt must fit whole or wait.
        """
        budget = self.ledger.token_budget
        tokens = {}

        for req in list(batch.members):
            if req.status == RequestStatus.RUNNING and req.is_decoding and budget > 0:
                if self.reserve_block(req, tokens):
                    tokens[req] = 1
                    budget -= 1

        for req in batch.members:
            if req.status != RequestStatus.RUNNING or req.is_decoding or budget == 0:
                continue
            chunk = min(req.remaining, budget) if self.chunked_prefill else req.remaining
            if chunk > budget:
                continue
            tokens[req] = chunk
            budget -= chunk

        return tokens

    def reserve_block(self, req, tokens):
        """Make sure a decoding request has room for its next token's KV entry."""
        if req.computed < len(req.blocks) * BLOCK_SIZE:
            return True
        if self.ledger.free_blocks == 0:
            self.make_room()
            for victim in [r for r in tokens if r.status != RequestStatus.RUNNING]:
                del tokens[victim]  # already planned, but preempted
        if req.status != RequestStatus.RUNNING:
            return False            # it was the newest request, so it preempted itself
        req.blocks += self.ledger.allocate(1)
        return True

    # --- helpers ------------------------------------------------------------

    def receive_arrivals(self):
        while self.pending and self.pending[0].arrival <= self.clock:
            self.waiting.append(self.pending.pop(0))


# Listing 4.4: static batching. The inner loop also updates requests and
# records metrics; without those calls no request would ever finish.
def run_static(scheduler, max_batch_size):
    while scheduler.has_work():
        # steps 1 and 2, run once: fill the batch, then freeze it
        scheduler.admit_from_waiting(up_to=max_batch_size)
        batch = scheduler.form_batch()
        while not batch.all_finished():    # frozen: no admits, no re-forming
            outputs = scheduler.step(batch)    # one forward pass over the whole batch
            scheduler.update_requests(outputs)
            scheduler.record_metrics(outputs)
        scheduler.retire_finished(batch)   # everyone leaves together
        if not batch.members:              # nothing had arrived yet
            scheduler.step(batch)          # idle until the next arrival


# Listing 4.5: continuous batching.
def run_continuous(scheduler, max_batch_size):
    while scheduler.has_work():
        # Fill available positions with waiting requests.
        scheduler.admit_from_waiting(up_to=max_batch_size)
        # Form the batch from the current running requests.
        batch = scheduler.form_batch()
        # Run one forward pass.
        outputs = scheduler.step(batch)
        # Update each request after the forward pass.
        scheduler.update_requests(outputs)
        # Remove completed requests and release their KV-cache blocks.
        scheduler.retire_finished(outputs)
        # Record the latency and throughput produced by this step.
        scheduler.record_metrics(outputs)


# --- Table 4.1: TTFT, TPOT, throughput, and goodput ------------------------

@dataclass
class Summary:
    requests: int
    makespan_ms: float
    throughput_tok_s: float
    mean_ttft_ms: float
    p95_ttft_ms: float
    mean_tpot_ms: float
    max_itl_ms: float
    goodput_tok_s: float
    met_slo: int
    preemptions: int


def percentile(values, pct):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(scheduler, ttft_slo_ms=float("inf"), tpot_slo_ms=float("inf")):
    """Throughput counts every token. Goodput counts only the tokens of
    requests that met both latency targets."""
    done = scheduler.finished
    start = min(r.arrival for r in done)
    makespan = max(r.finish_time for r in done) - start
    tokens = sum(r.generated for r in done)
    good = [r for r in done if r.ttft <= ttft_slo_ms and r.tpot <= tpot_slo_ms]
    return Summary(
        requests=len(done),
        makespan_ms=makespan,
        throughput_tok_s=tokens / makespan * 1000,
        mean_ttft_ms=statistics.mean(r.ttft for r in done),
        p95_ttft_ms=percentile([r.ttft for r in done], 95),
        mean_tpot_ms=statistics.mean(r.tpot for r in done),
        max_itl_ms=max(r.max_itl for r in done),
        goodput_tok_s=sum(r.generated for r in good) / makespan * 1000,
        met_slo=len(good),
        preemptions=scheduler.num_preemptions,
    )


def print_summary(label, s):
    print(f"  {label}")
    print(f"    {s.requests} requests in {s.makespan_ms:,.0f} ms, "
          f"throughput {s.throughput_tok_s:,.0f} tokens/s")
    print(f"    TTFT mean {s.mean_ttft_ms:,.1f} ms, p95 {s.p95_ttft_ms:,.1f} ms | "
          f"TPOT mean {s.mean_tpot_ms:,.1f} ms | longest gap between tokens "
          f"{s.max_itl_ms:,.1f} ms")
    if s.met_slo < s.requests:
        print(f"    goodput {s.goodput_tok_s:,.0f} tokens/s "
              f"({s.met_slo} of {s.requests} requests met the latency targets)")
    if s.preemptions:
        print(f"    preemptions: {s.preemptions}")


def print_trace(scheduler):
    """Print each forward pass, collapsing runs of identical passes."""
    def describe(step):
        return " ".join(f"{r.name}:{n}" for r, n in step.tokens.items())

    steps, i = scheduler.steps, 0
    while i < len(steps):
        step = steps[i]
        done = " ".join(r.name for r in step.produced if r.finish_time == step.time)
        print(f"  {step.time:8.1f} ms | {sum(step.tokens.values()):>5} tokens | {describe(step)}"
              + (f"  -> {done} finished" if done else ""))
        j = i + 1
        while (j < len(steps) and describe(steps[j]) == describe(step)
               and not any(r.finish_time == steps[j].time for r in steps[j].produced)):
            j += 1
        if j - i > 1:
            print(f"  {'':>8}    | {'':>5}        | ... {j - i - 1} more passes like this")
        i = j


def requests_abcd():
    """Requests A, B, C, and D from section 4.1.1."""
    return [
        Request("C", prompt_len=300, output_len=40, arrival=0.0),   # already streaming
        Request("D", prompt_len=200, output_len=4, arrival=0.0),    # almost finished
        Request("A", prompt_len=6, output_len=3, arrival=12.0),     # "The capital of France is"
        Request("B", prompt_len=2000, output_len=20, arrival=12.0), # summarize a document
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--token-budget", type=int, default=512,
                        help="per-step token budget, max_num_batched_tokens (default: 512)")
    args = parser.parse_args()

    print("Requests A, B, C, and D (section 4.1.1), continuous batching, "
          f"token budget {args.token_budget}")
    print("Each line is one forward pass: request:tokens it processes (Equation 4.1)\n")
    scheduler = Scheduler(requests_abcd(), token_budget=args.token_budget)
    run_continuous(scheduler, max_batch_size=8)
    print_trace(scheduler)
    print()
    for req in sorted(scheduler.finished, key=lambda r: r.name):
        print(f"  {req.name}: prompt {req.prompt_len:>5}, {req.generated:>3} tokens, "
              f"TTFT {req.ttft:6.1f} ms, TPOT {req.tpot:5.1f} ms")


if __name__ == "__main__":
    main()
