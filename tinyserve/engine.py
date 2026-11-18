"""The engine loop: schedule, run the model, sample, repeat.

One call to :meth:`Engine.step` is one "iteration" of the serving system:

1. the scheduler picks which requests run (and which need a prefill),
2. the model runner executes the prefill batch and/or the decode batch,
3. each request receives its new token, and finished requests leave the batch.

The engine is synchronous and single-threaded on purpose. The API server wraps
it in a background thread; the benchmarks drive it directly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from tinyserve.kv_cache import (
    ContiguousKVCache,
    KVCacheManager,
    KVPool,
    KVUsage,
    PagedKVCache,
)
from tinyserve.model import ModelConfig, TinyGPT
from tinyserve.policy import POLICIES, make_policy
from tinyserve.request import Request, SamplingParams
from tinyserve.runner import ModelRunner
from tinyserve.scheduler import (
    ContinuousScheduler,
    NaiveScheduler,
    Scheduler,
    StaticScheduler,
)

STRATEGIES = ("naive", "static", "continuous")
KV_LAYOUTS = ("contiguous", "paged")


@dataclass(frozen=True)
class StepOutput:
    """One token produced for one request during a step."""

    request: Request
    token_id: int
    finished: bool


@dataclass(frozen=True)
class StepTrace:
    """What happened during one engine step (recorded for benchmarks and plots).

    ``slots`` maps batch seat -> (request id, phase) where phase is
    ``"prefill"`` or ``"decode"``.
    """

    start: float
    end: float
    num_prefill: int
    num_prefill_tokens: int
    num_decode: int
    num_waiting: int
    num_running: int
    kv: KVUsage
    slots: dict[int, tuple[int, str]] = field(default_factory=dict)


class Engine:
    """Drives a scheduler and a model runner one step at a time.

    Args:
        model: the transformer.
        scheduler: decides who runs each step; owns the KV cache manager.
        clock: time source, in seconds. Injectable for tests.
        record_trace: keep a :class:`StepTrace` for every step in ``self.trace``.
    """

    def __init__(
        self,
        model: TinyGPT,
        scheduler: Scheduler,
        clock: Callable[[], float] = time.perf_counter,
        record_trace: bool = False,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.runner = ModelRunner(model, scheduler.kv)
        self.clock = clock
        self.record_trace = record_trace
        self.trace: list[StepTrace] = []
        self.num_steps = 0
        self.num_generated_tokens = 0

    @property
    def kv(self) -> KVCacheManager:
        """The KV cache manager shared by the scheduler and the runner."""
        return self.scheduler.kv

    def validate(self, request: Request) -> None:
        """Raise ``ValueError`` if the engine could never serve ``request``.

        Pure check with no side effects, so it is safe to call from another thread.
        """
        if request.max_length > self.model.cfg.max_seq_len:
            raise ValueError(
                f"prompt ({len(request.prompt_ids)}) + max_new_tokens "
                f"({request.params.max_new_tokens}) exceeds max_seq_len "
                f"({self.model.cfg.max_seq_len})"
            )
        self.scheduler.validate(request)

    def add_request(self, request: Request) -> None:
        """Queue a request. Its ``arrival_time`` defaults to now."""
        self.validate(request)
        if request.arrival_time is None:
            request.arrival_time = self.clock()
        self.scheduler.add(request)

    def has_unfinished(self) -> bool:
        """True while any request is waiting or running."""
        return self.scheduler.has_unfinished()

    def abort(self, request_id: int) -> Request | None:
        """Cancel a request and free its memory."""
        return self.scheduler.abort(request_id, self.clock())

    def step(self) -> list[StepOutput]:
        """Run one iteration and return the tokens it produced."""
        start = self.clock()
        plan = self.scheduler.schedule(start)
        if plan.is_empty:
            return []

        # Snapshot the batch layout before tokens are appended (for the trace).
        slots: dict[int, tuple[int, str]] = {}
        if self.record_trace:
            for phase, reqs in (("prefill", plan.prefill), ("decode", plan.decode)):
                for r in reqs:
                    slots[self.scheduler.slot_of[r.request_id]] = (r.request_id, phase)
        kv_usage = self.kv.usage()
        chunk_sizes = [plan.prefill_len(r) for r in plan.prefill]
        prefill_tokens = sum(chunk_sizes)

        results: list[tuple[Request, int | None]] = []
        if plan.prefill:
            tokens = self.runner.prefill(plan.prefill, chunk_sizes)
            for req, n in zip(plan.prefill, chunk_sizes, strict=True):
                req.num_computed_tokens += n
            results += zip(plan.prefill, tokens, strict=True)
        if plan.decode:
            tokens = self.runner.decode(plan.decode)
            for req in plan.decode:
                req.num_computed_tokens = req.length
            results += zip(plan.decode, tokens, strict=True)

        now = self.clock()
        outputs = []
        for req, token in results:
            if token is None:  # a prefill chunk that has not reached the end yet
                continue
            req.append_token(token, now)
            outputs.append(StepOutput(req, token, req.is_finished))
        self.scheduler.finish_step()

        self.num_steps += 1
        self.num_generated_tokens += len(outputs)
        if self.record_trace:
            self.trace.append(
                StepTrace(
                    start=start,
                    end=now,
                    num_prefill=len(plan.prefill),
                    num_prefill_tokens=prefill_tokens,
                    num_decode=len(plan.decode),
                    num_waiting=len(self.scheduler.waiting),
                    num_running=len(self.scheduler.running),
                    kv=kv_usage,
                    slots=slots,
                )
            )
        return outputs

    def generate(self, prompts: list[list[int]], params: SamplingParams) -> list[Request]:
        """Offline helper: run all prompts to completion and return the finished requests."""
        requests = [Request(p, params) for p in prompts]
        for r in requests:
            self.add_request(r)
        while self.has_unfinished():
            self.step()
        return requests


@dataclass(frozen=True)
class EngineConfig:
    """Everything needed to build an engine for one serving strategy.

    Attributes:
        strategy: one of :data:`STRATEGIES`.
        kv_capacity_tokens: size of the KV pool, in token slots. Every strategy
            gets the same pool so comparisons are apples to apples.
        max_batch_size: most requests running at once (ignored by ``naive``).
        max_tokens_per_step: per-step token budget of the continuous scheduler.
        kv_layout: ``"contiguous"`` (one max_seq_len slab per request) or
            ``"paged"`` (fixed-size blocks on demand). Paged needs preemption to
            handle running out of memory, so only ``continuous`` supports it.
        block_size: tokens per block for the paged layout.
        queue_policy: admission order of the continuous scheduler, one of
            :data:`~tinyserve.policy.POLICIES`.
        ttft_slo_s: TTFT target used by the deadline-aware policies.
        sjf_aging: tokens of priority a waiting request gains per second (``sjf``).
        num_threads: PyTorch intra-op threads (None leaves the default).
    """

    strategy: str = "naive"
    model: ModelConfig = field(default_factory=ModelConfig)
    kv_capacity_tokens: int = 8192
    max_batch_size: int = 32
    max_tokens_per_step: int = 2048
    kv_layout: str = "contiguous"
    block_size: int = 16
    queue_policy: str = "fcfs"
    ttft_slo_s: float = 1.0
    sjf_aging: float = 100.0
    num_threads: int | None = None


def build_engine(cfg: EngineConfig, record_trace: bool = False) -> Engine:
    """Construct the model, KV cache, scheduler and engine described by ``cfg``."""
    if cfg.strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {cfg.strategy!r}; choose from {STRATEGIES}")
    if cfg.kv_layout not in KV_LAYOUTS:
        raise ValueError(f"unknown kv_layout {cfg.kv_layout!r}; choose from {KV_LAYOUTS}")
    if cfg.queue_policy not in POLICIES:
        raise ValueError(f"unknown queue_policy {cfg.queue_policy!r}; choose from {POLICIES}")
    if cfg.queue_policy != "fcfs" and cfg.strategy != "continuous":
        raise ValueError("queue policies other than 'fcfs' need the 'continuous' strategy")
    if cfg.kv_layout == "paged" and cfg.strategy != "continuous":
        raise ValueError("the paged KV cache needs preemption, which only 'continuous' has")
    if cfg.kv_capacity_tokens < cfg.model.max_seq_len:
        raise ValueError("kv_capacity_tokens must hold at least one max_seq_len sequence")
    if cfg.num_threads is not None:
        torch.set_num_threads(cfg.num_threads)
    m = cfg.model
    model = TinyGPT(m)
    pool = KVPool(cfg.kv_capacity_tokens, m.n_layers, m.n_kv_heads, m.head_dim)
    kv: KVCacheManager
    if cfg.kv_layout == "paged":
        kv = PagedKVCache(pool, cfg.block_size)
    else:
        kv = ContiguousKVCache(pool, m.max_seq_len)
    scheduler: Scheduler
    if cfg.strategy == "naive":
        scheduler = NaiveScheduler(kv)
    elif cfg.strategy == "static":
        scheduler = StaticScheduler(kv, cfg.max_batch_size)
    else:
        policy = make_policy(cfg.queue_policy, cfg.ttft_slo_s, cfg.sjf_aging)
        scheduler = ContinuousScheduler(kv, cfg.max_batch_size, cfg.max_tokens_per_step, policy)
    return Engine(model, scheduler, record_trace=record_trace)
