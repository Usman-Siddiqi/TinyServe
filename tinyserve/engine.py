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

from tinyserve.draft import early_exit_draft, load_distilled_draft
from tinyserve.kv_cache import (
    ContiguousKVCache,
    KVCacheManager,
    KVPool,
    KVUsage,
    PagedKVCache,
    PrefixCachingKVCache,
)
from tinyserve.kv_quant import KV_DTYPES, calibrate_key_smoothing
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
from tinyserve.speculative import (
    DraftLengthController,
    DraftModelProposer,
    NgramProposer,
    Proposer,
    SpeculativeDecoder,
)

STRATEGIES = ("naive", "static", "continuous")
KV_LAYOUTS = ("contiguous", "paged")
SPEC_METHODS = ("ngram", "early-exit", "distilled")


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
        attention: decode attention backend (see :class:`ModelRunner`).
        speculator: if given, decode steps draft and verify several tokens at once.
    """

    def __init__(
        self,
        model: TinyGPT,
        scheduler: Scheduler,
        clock: Callable[[], float] = time.perf_counter,
        record_trace: bool = False,
        attention: str | None = None,
        speculator: SpeculativeDecoder | None = None,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.runner = ModelRunner(model, scheduler.kv, attention)
        self.speculator = speculator
        if speculator is not None:
            scheduler.num_lookahead_tokens = speculator.num_speculative_tokens
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
        request = self.scheduler.abort(request_id, self.clock())
        if request is not None and self.speculator is not None:
            self.speculator.release(request_id)
        return request

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

        if self.speculator is not None:
            for req in plan.preempted:
                self.speculator.release(req.request_id)

        results: list[tuple[Request, list[int]]] = []
        if plan.prefill:
            tokens = self.runner.prefill(plan.prefill, chunk_sizes)
            for req, n in zip(plan.prefill, chunk_sizes, strict=True):
                req.num_computed_tokens += n
            results += [
                (r, [] if t is None else [t]) for r, t in zip(plan.prefill, tokens, strict=True)
            ]
        if plan.decode and self.speculator is None:
            tokens = self.runner.decode(plan.decode)
            for req in plan.decode:
                req.num_computed_tokens = req.length
            results += [(r, [t]) for r, t in zip(plan.decode, tokens, strict=True)]
        elif plan.decode:
            lookahead = [self.scheduler.num_lookahead(r) for r in plan.decode]
            accepted = self.speculator.decode(self.runner, plan.decode, lookahead)
            for req, new in zip(plan.decode, accepted, strict=True):
                req.num_computed_tokens = req.length + len(new) - 1
            results += zip(plan.decode, accepted, strict=True)

        now = self.clock()
        outputs = []
        for req, tokens in results:
            for token in tokens:
                req.append_token(token, now)
                outputs.append(StepOutput(req, token, req.is_finished))
                if req.is_finished:  # e.g. EOS in the middle of accepted drafts
                    break
        if self.speculator is not None:
            # Give back the KV slots of rejected drafts.
            for req in plan.decode:
                if not req.is_finished:
                    self.kv.truncate(req.request_id, req.length - 1)
        for req in self.scheduler.finish_step():
            if self.speculator is not None:
                self.speculator.release(req.request_id)

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
        chunked_prefill: let the continuous scheduler split long prefills across
            steps so they do not stall running requests.
        kv_layout: ``"contiguous"`` (one max_seq_len slab per request) or
            ``"paged"`` (fixed-size blocks on demand). Paged needs preemption to
            handle running out of memory, so only ``continuous`` supports it.
        block_size: tokens per block for the paged layout.
        enable_prefix_caching: share the KV of common prompt prefixes between
            requests (paged layout only).
        queue_policy: admission order of the continuous scheduler, one of
            :data:`~tinyserve.policy.POLICIES`.
        ttft_slo_s: TTFT target used by the deadline-aware policies.
        sjf_aging: tokens of priority a waiting request gains per second (``sjf``).
        num_threads: PyTorch intra-op threads (None leaves the default).
        kv_dtype: KV storage format, one of :data:`~tinyserve.kv_quant.KV_DTYPES`.
        kv_memory_bytes: if set, size the pool by bytes instead of
            ``kv_capacity_tokens``, so a smaller ``kv_dtype`` gets more slots.
        kv_smooth_keys: for ``int8``, calibrate per-channel key smoothing at startup.
        attention: decode attention backend, ``"torch"`` or ``"cpp"``. None reads
            ``$TINYSERVE_ATTENTION`` and defaults to ``"torch"``.
        speculative: speculative decoding proposer, one of :data:`SPEC_METHODS`,
            or None (off). ``early-exit`` drafts with the target's first layer,
            ``distilled`` with a small model trained by ``bench/distill_draft.py``.
        num_speculative_tokens: most draft tokens verified per request per step.
        adaptive_speculation: pick each request's draft length from its recent
            acceptance rate instead of always drafting ``num_speculative_tokens``.
    """

    strategy: str = "naive"
    model: ModelConfig = field(default_factory=ModelConfig)
    kv_capacity_tokens: int = 8192
    max_batch_size: int = 32
    max_tokens_per_step: int = 2048
    chunked_prefill: bool = False
    kv_layout: str = "contiguous"
    block_size: int = 16
    enable_prefix_caching: bool = False
    queue_policy: str = "fcfs"
    ttft_slo_s: float = 1.0
    sjf_aging: float = 100.0
    num_threads: int | None = None
    kv_dtype: str = "float32"
    kv_memory_bytes: int | None = None
    kv_smooth_keys: bool = True
    attention: str | None = None
    speculative: str | None = None
    num_speculative_tokens: int = 4
    adaptive_speculation: bool = True

    @property
    def num_kv_slots(self) -> int:
        """Token slots in the pool: ``kv_capacity_tokens``, or as many as fit the byte budget."""
        if self.kv_memory_bytes is None:
            return self.kv_capacity_tokens
        return self.kv_memory_bytes // self.model.kv_bytes_per_token(self.kv_dtype)


def calibrate_kv(model: TinyGPT, pool: KVPool, num_seqs: int = 4) -> None:
    """Set ``pool.key_smoothing`` from the key magnitudes on a few calibration sequences.

    The model has random weights, so random tokens are as good as text; with a
    real checkpoint these would be sample prompts.
    """
    m = model.cfg
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(4, m.vocab_size, (num_seqs, m.max_seq_len), generator=g)
    pool.key_smoothing = calibrate_key_smoothing(model, ids)


# Rough cost of drafting one token relative to a target decode step (bench/spec_costs.py).
DRAFT_COSTS = {"ngram": 0.01, "early-exit": 0.35, "distilled": 0.2}


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
    if cfg.speculative is not None and cfg.speculative not in SPEC_METHODS:
        raise ValueError(f"unknown speculative {cfg.speculative!r}; choose from {SPEC_METHODS}")
    if cfg.kv_dtype not in KV_DTYPES:
        raise ValueError(f"unknown kv_dtype {cfg.kv_dtype!r}; choose from {KV_DTYPES}")
    if cfg.enable_prefix_caching and cfg.kv_layout != "paged":
        raise ValueError("prefix caching needs the paged KV cache")
    if cfg.num_kv_slots < cfg.model.max_seq_len:
        raise ValueError("the KV pool must hold at least one max_seq_len sequence")
    if cfg.num_threads is not None:
        torch.set_num_threads(cfg.num_threads)
    m = cfg.model
    model = TinyGPT(m)
    pool = KVPool(cfg.num_kv_slots, m.n_layers, m.n_kv_heads, m.head_dim, cfg.kv_dtype)
    if pool.format.quantized and cfg.kv_smooth_keys:
        calibrate_kv(model, pool)
    kv: KVCacheManager
    if cfg.enable_prefix_caching:
        kv = PrefixCachingKVCache(pool, cfg.block_size)
    elif cfg.kv_layout == "paged":
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
        scheduler = ContinuousScheduler(
            kv,
            cfg.max_batch_size,
            cfg.max_tokens_per_step,
            chunked_prefill=cfg.chunked_prefill,
            policy=policy,
        )
    speculator = None
    if cfg.speculative is not None:
        controller = None
        if cfg.adaptive_speculation:
            controller = DraftLengthController(DRAFT_COSTS[cfg.speculative])
        speculator = SpeculativeDecoder(
            _build_proposer(cfg, model), cfg.num_speculative_tokens, controller
        )
    return Engine(
        model,
        scheduler,
        record_trace=record_trace,
        attention=cfg.attention,
        speculator=speculator,
    )


def _build_proposer(cfg: EngineConfig, model: TinyGPT) -> Proposer:
    """The speculative proposer named by ``cfg.speculative``."""
    if cfg.speculative == "ngram":
        return NgramProposer()
    if cfg.speculative == "early-exit":
        draft = early_exit_draft(model, n_layers=1)
    else:
        draft = load_distilled_draft(model.cfg)
    d = draft.cfg
    # Same token capacity as the target: the draft caches the same sequences.
    pool = KVPool(cfg.num_kv_slots, d.n_layers, d.n_kv_heads, d.head_dim)
    return DraftModelProposer(draft, PagedKVCache(pool, cfg.block_size))
