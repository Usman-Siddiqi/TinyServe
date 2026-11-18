"""Schedulers decide, at every engine step, which requests run and in which phase.

All schedulers share the same bookkeeping (a ``waiting`` queue and a
``running`` list) and differ only in their admission policy:

* :class:`NaiveScheduler` serves one request at a time, start to finish. Think
  of a **taxi**: one passenger, door to door, everybody else waits at the curb.
* :class:`StaticScheduler` fills a batch, then runs it until *every* request in
  it has finished. A **bus that waits for everyone**: nobody new can board until
  the last passenger gets off, so seats sit empty while the long trips finish.
* :class:`ContinuousScheduler` re-plans the batch at *every* decode step:
  finished requests leave immediately and waiting ones take their seats. A
  **shuttle** that drops people off and picks people up at every stop. Who
  boards first is up to its :mod:`~tinyserve.policy` (FCFS by default).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

from tinyserve.kv_cache import KVCacheManager
from tinyserve.policy import FCFSPolicy, QueuePolicy
from tinyserve.request import FinishReason, Request, RequestStatus


@dataclass
class SchedulerOutput:
    """What to run in one engine step.

    Attributes:
        prefill: requests whose sequence (or the next chunk of it) is processed
            this step. A request samples its first token once its prefill completes.
        decode: requests that generate one token this step.
        preempted: requests that were kicked out of memory to make room.
        prefill_chunks: optional ``request_id -> number of tokens`` to process for
            a prefill request this step. Missing entries mean "everything that is
            not computed yet".
    """

    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    prefill_chunks: dict[int, int] = field(default_factory=dict)

    def prefill_len(self, request: Request) -> int:
        """Tokens of ``request`` this step's prefill will process."""
        remaining = request.length - request.num_computed_tokens
        return self.prefill_chunks.get(request.request_id, remaining)

    @property
    def is_empty(self) -> bool:
        """True if there is nothing to run."""
        return not self.prefill and not self.decode

    @property
    def num_tokens(self) -> int:
        """Tokens processed by the model this step."""
        return sum(self.prefill_len(r) for r in self.prefill) + len(self.decode)


class Scheduler(ABC):
    """Shared queue/slot bookkeeping. Subclasses implement :meth:`schedule`.

    Args:
        kv: the KV cache manager whose memory this scheduler hands out.
        max_batch_size: most requests that may be running at the same time.
    """

    name = "base"

    def __init__(self, kv: KVCacheManager, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.kv = kv
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        # Each running request occupies a numbered "seat" in the batch. Seats are
        # purely for visualizing who sits where; the model never sees them.
        self.slot_of: dict[int, int] = {}
        self._free_slots = list(range(max_batch_size - 1, -1, -1))
        self.num_preemptions = 0

    def validate(self, request: Request) -> None:  # noqa: B027 - optional hook
        """Raise ``ValueError`` if this scheduler could never run ``request``."""

    def add(self, request: Request) -> None:
        """Queue a new request."""
        self.validate(request)
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        """True while any request is waiting or running."""
        return bool(self.waiting or self.running)

    @abstractmethod
    def schedule(self, now: float) -> SchedulerOutput:
        """Pick the work for the next engine step, at time ``now``."""

    def finish_step(self) -> list[Request]:
        """Release memory and seats of requests that finished this step. Returns them."""
        if self.kv.prefix_caching:
            for req in self.running:
                self.kv.cache_blocks(req.request_id, req.all_ids[: req.num_computed_tokens])
        done = [r for r in self.running if r.is_finished]
        for req in done:
            self._release(req)
        return done

    def abort(self, request_id: int, now: float) -> Request | None:
        """Cancel a request wherever it is (e.g. the client disconnected)."""
        for req in list(self.waiting):
            if req.request_id == request_id:
                self.waiting.remove(req)
                req.finish(FinishReason.ABORTED, now)
                return req
        for req in self.running:
            if req.request_id == request_id:
                req.finish(FinishReason.ABORTED, now)
                self._release(req)
                return req
        return None

    # --- helpers for subclasses ----------------------------------------------------

    def _admit(self, request: Request) -> bool:
        """Move a waiting request into the batch, if its KV memory fits.

        With prefix caching, the request's cached prefix is skipped by prefill.
        """
        token_ids = request.all_ids
        num_cached = self.kv.match_prefix(token_ids)
        if not self.kv.allocate(request.request_id, request.length, token_ids):
            return False
        self.waiting.remove(request)
        request.num_computed_tokens = request.num_cached_tokens = num_cached
        request.status = RequestStatus.RUNNING
        self.running.append(request)
        self.slot_of[request.request_id] = self._free_slots.pop()
        return True

    def _release(self, request: Request) -> None:
        """Take a request out of the batch and free its KV memory."""
        self.running.remove(request)
        self.kv.free(request.request_id)
        self._free_slots.append(self.slot_of.pop(request.request_id))
        self._free_slots.sort(reverse=True)  # keep handing out the lowest seat first

    def _grow_or_fail(self, request: Request) -> None:
        """Reserve KV room for the token a decode step is about to write."""
        if not self.kv.grow(request.request_id, request.length):
            raise RuntimeError(
                f"{self.name} scheduler ran out of KV memory; it does not support preemption"
            )


class NaiveScheduler(Scheduler):
    """One request at a time, from first token to last. Baseline #1 (the taxi)."""

    name = "naive"

    def __init__(self, kv: KVCacheManager) -> None:
        super().__init__(kv, max_batch_size=1)

    def schedule(self, now: float) -> SchedulerOutput:
        """Decode the current request, or start the next one if the taxi is empty."""
        if not self.running:
            if self.waiting and self._admit(self.waiting[0]):
                return SchedulerOutput(prefill=[self.running[0]])
            return SchedulerOutput()
        for req in self.running:
            self._grow_or_fail(req)
        return SchedulerOutput(decode=list(self.running))


class StaticScheduler(Scheduler):
    """Classic static batching. Baseline #2 (the bus that waits for everyone).

    When the batch is empty, admit up to ``max_batch_size`` waiting requests (as
    many as fit in KV memory) and prefill them together. Then decode the batch
    until all of them are done. Requests that finish early leave a dead seat
    behind; it is only refilled once the whole batch has drained.

    To be generous to this baseline, finished requests are dropped from the
    compute instead of being decoded as padding (which is what a plain
    ``model.generate`` on a padded batch would do). The seat stays empty either way.
    """

    name = "static"

    def schedule(self, now: float) -> SchedulerOutput:
        """Decode the current batch, or board a new one if the bus is empty."""
        if self.running:
            for req in self.running:
                self._grow_or_fail(req)
            return SchedulerOutput(decode=list(self.running))
        while self.waiting and len(self.running) < self.max_batch_size:
            if not self._admit(self.waiting[0]):
                break
        return SchedulerOutput(prefill=list(self.running))


class ContinuousScheduler(Scheduler):
    """Continuous (a.k.a. iteration-level or in-flight) batching, as in Orca and vLLM.

    Every step:

    1. **Decode** every running request (one new token each). Before that, make
       sure each has KV room for the token it is about to write. If memory runs
       out, *preempt* the lowest-ranked running request: free its KV and put it
       back in the queue. When it is readmitted its whole sequence is prefilled
       again ("recompute" preemption; nothing it generated is lost).
    2. **Prefill** with whatever is left of the step's token budget: first
       requests whose prefill is already under way, then newly admitted ones
       (best-ranked first, while there is a free seat and their KV memory fits).

    Without chunking, a prompt must fit in the leftover budget in one piece, so
    the step that prefills it takes as long as the whole prompt and every running
    request waits that long for its next token. With ``chunked_prefill`` a prompt
    is split across steps (Sarathi-Serve), which keeps every step roughly the
    same size and smooths out inter-token latency.

    The ranking comes from ``policy``. The default, FCFS, admits oldest first
    and preempts the most recently admitted request.

    Args:
        kv: the KV cache manager.
        max_batch_size: most requests running at once.
        max_tokens_per_step: token budget per step (prefill tokens + one per
            decode). Must be at least ``max_batch_size``.
        chunked_prefill: allow a prefill to be split across several steps.
        policy: admission and preemption order (default :class:`FCFSPolicy`).
    """

    name = "continuous"

    def __init__(
        self,
        kv: KVCacheManager,
        max_batch_size: int,
        max_tokens_per_step: int = 2048,
        *,
        chunked_prefill: bool = False,
        policy: QueuePolicy | None = None,
    ) -> None:
        super().__init__(kv, max_batch_size)
        if max_tokens_per_step < max_batch_size:
            raise ValueError("max_tokens_per_step must be >= max_batch_size")
        self.max_tokens_per_step = max_tokens_per_step
        self.chunked_prefill = chunked_prefill
        self.policy = policy or FCFSPolicy()

    def validate(self, request: Request) -> None:
        """Without chunking, reject requests that could never fit in one step."""
        if not self.chunked_prefill and request.max_length > self.max_tokens_per_step:
            raise ValueError("request is longer than max_tokens_per_step")

    def schedule(self, now: float) -> SchedulerOutput:
        """Plan the decode batch (preempting if needed), then spend the rest on prefill."""
        out = SchedulerOutput()

        # 1. Decode everyone past prefill, best-ranked first. Preempt from the back.
        self.running = self.policy.order(self.running, now)
        i = 0
        while i < len(self.running):
            req = self.running[i]
            if req.needs_prefill:
                i += 1
            elif self.kv.grow(req.request_id, req.length):
                out.decode.append(req)
                i += 1
            else:
                victim = self.running[-1]
                self._preempt(victim)
                out.preempted.append(victim)
                # If the victim was `req` itself the loop ends; otherwise retry `req`.

        # 2. Continue prefills that are under way (only happens with chunking).
        budget = self.max_tokens_per_step - len(out.decode)
        for req in self.running:
            if budget == 0:
                break
            if req.needs_prefill:
                budget -= self._add_prefill(out, req, budget)

        # 3. Admit new requests in policy order while seats, budget and memory allow.
        # Stopping at the first one that does not fit (rather than skipping it)
        # keeps big requests from being overtaken forever.
        for req in self.policy.order(self.waiting, now):
            if len(self.running) >= self.max_batch_size or budget == 0:
                break
            cost = req.length - self.kv.match_prefix(req.all_ids)  # cached tokens are free
            if not self.chunked_prefill and cost > budget:
                break
            if not self._admit(req):
                break
            budget -= self._add_prefill(out, req, budget)
        return out

    def _add_prefill(self, out: SchedulerOutput, request: Request, budget: int) -> int:
        """Schedule the next prefill chunk of ``request`` within ``budget``. Returns its size."""
        n = min(request.length - request.num_computed_tokens, budget)
        out.prefill.append(request)
        out.prefill_chunks[request.request_id] = n
        return n

    def _preempt(self, request: Request) -> None:
        """Evict a running request back to the head of the waiting queue."""
        self._release(request)
        request.status = RequestStatus.WAITING
        request.num_computed_tokens = 0  # its KV is gone; recompute from scratch
        request.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(request)
