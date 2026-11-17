"""Schedulers decide, at every engine step, which requests run and in which phase.

All schedulers share the same bookkeeping (a FIFO ``waiting`` queue and a
``running`` list) and differ only in their admission policy:

* :class:`NaiveScheduler` serves one request at a time, start to finish. Think
  of a **taxi**: one passenger, door to door, everybody else waits at the curb.
* :class:`StaticScheduler` fills a batch, then runs it until *every* request in
  it has finished. A **bus that waits for everyone**: nobody new can board until
  the last passenger gets off, so seats sit empty while the long trips finish.
* :class:`ContinuousScheduler` re-plans the batch at *every* decode step:
  finished requests leave immediately and waiting ones take their seats. A
  **shuttle** that drops people off and picks people up at every stop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

from tinyserve.kv_cache import KVCacheManager
from tinyserve.request import FinishReason, Request, RequestStatus


@dataclass
class SchedulerOutput:
    """What to run in one engine step.

    Attributes:
        prefill: requests that need their whole sequence processed this step.
        decode: requests that generate one token this step.
        preempted: requests that were kicked out of memory to make room.
    """

    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if there is nothing to run."""
        return not self.prefill and not self.decode

    @property
    def num_tokens(self) -> int:
        """Tokens processed by the model this step."""
        return sum(r.length for r in self.prefill) + len(self.decode)


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

    def add(self, request: Request) -> None:
        """Queue a new request."""
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        """True while any request is waiting or running."""
        return bool(self.waiting or self.running)

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        """Pick the work for the next engine step."""

    def finish_step(self) -> list[Request]:
        """Release memory and seats of requests that finished this step. Returns them."""
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
        """Move the head of the waiting queue into the batch, if its KV memory fits."""
        if not self.kv.allocate(request.request_id, request.length):
            return False
        assert self.waiting[0] is request, "requests are admitted in FIFO order"
        self.waiting.popleft()
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

    def schedule(self) -> SchedulerOutput:
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

    def schedule(self) -> SchedulerOutput:
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
       out, *preempt* the most recently admitted request: free its KV and put it
       back at the front of the queue. When it is readmitted its whole sequence is
       prefilled again ("recompute" preemption; nothing it generated is lost).
    2. **Admit** waiting requests, oldest first, while there is a free seat, the
       step's token budget allows it and their KV memory fits. Newly admitted
       requests are prefilled in this same step.

    Args:
        kv: the KV cache manager.
        max_batch_size: most requests running at once.
        max_tokens_per_step: token budget per step (prefill tokens + one per
            decode). Caps how long a step can take, so a burst of long prompts
            cannot stall everybody's decoding for too long.
    """

    name = "continuous"

    def __init__(
        self, kv: KVCacheManager, max_batch_size: int, max_tokens_per_step: int = 2048
    ) -> None:
        super().__init__(kv, max_batch_size)
        self.max_tokens_per_step = max_tokens_per_step

    def add(self, request: Request) -> None:
        """Queue a request, rejecting any that could never fit in one step's budget."""
        if request.max_length > self.max_tokens_per_step:
            raise ValueError("request is longer than max_tokens_per_step")
        super().add(request)

    def schedule(self) -> SchedulerOutput:
        """Plan the decode batch, preempting if needed, then fill free seats."""
        out = SchedulerOutput()

        # 1. Decode everyone already running, oldest first. Preempt from the back.
        i = 0
        while i < len(self.running):
            req = self.running[i]
            if self.kv.grow(req.request_id, req.length):
                out.decode.append(req)
                i += 1
                continue
            victim = self.running[-1]
            self._preempt(victim)
            out.preempted.append(victim)
            # If the victim was `req` itself the loop ends; otherwise retry `req`.

        # 2. Admit new requests (FCFS) into free seats while budget and memory allow.
        budget = self.max_tokens_per_step - len(out.decode)
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting[0]
            if req.length > budget or not self._admit(req):
                break
            out.prefill.append(req)
            budget -= req.length
        return out

    def _preempt(self, request: Request) -> None:
        """Evict a running request back to the head of the waiting queue."""
        self._release(request)
        request.status = RequestStatus.WAITING
        request.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(request)
