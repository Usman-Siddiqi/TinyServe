"""Bridge between the synchronous engine loop and asyncio request handlers.

The model math is blocking PyTorch code, so the engine runs in one dedicated
background thread that loops ``engine.step()`` forever. HTTP handlers live on
the asyncio event loop. They talk to the engine thread through two channels:

* an **inbox** (a thread-safe queue) carrying "add this request" and "abort
  that request" commands into the engine thread, and
* one **asyncio.Queue per request** carrying generated tokens back out, filled
  with ``loop.call_soon_threadsafe`` so the event loop is never touched from
  the wrong thread.

Only the engine thread ever touches the engine, so the engine itself needs no locks.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from tinyserve.engine import Engine, StepOutput
from tinyserve.request import Request, SamplingParams


@dataclass
class _Stream:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[StepOutput]


class AsyncEngine:
    """Runs an :class:`Engine` in a background thread and streams tokens to coroutines.

    Args:
        engine: the engine to drive.
        stats_window: seconds over which ``tokens_per_sec`` is averaged.
    """

    def __init__(self, engine: Engine, stats_window: float = 5.0) -> None:
        self.engine = engine
        self.stats_window = stats_window
        self._inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._streams: dict[int, _Stream] = {}
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._token_log: deque[tuple[float, int]] = deque()
        self._snapshot: dict[str, Any] = {}
        self._started_at = time.monotonic()
        self._num_finished = 0

    # --- lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        """Start the background engine thread."""
        if self._thread is not None:
            return
        self._running.set()
        self._started_at = time.monotonic()
        self._update_snapshot()
        self._thread = threading.Thread(target=self._loop, name="tinyserve-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the engine thread and wait for it to exit."""
        self._running.clear()
        self._inbox.put(("wake", None))
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # --- public API (called from the event loop) -----------------------------------

    async def generate(
        self, prompt_ids: list[int], params: SamplingParams
    ) -> AsyncGenerator[StepOutput, None]:
        """Submit a request and yield its tokens as the engine produces them.

        Raises ``ValueError`` before yielding anything if the request can never be
        served. If the consumer stops early (client disconnected), the request is
        aborted and its KV memory freed.
        """
        request = Request(prompt_ids, params, arrival_time=self.engine.clock())
        self.engine.validate(request)
        stream = _Stream(asyncio.get_running_loop(), asyncio.Queue())
        self._streams[request.request_id] = stream
        self._inbox.put(("add", request))
        finished = False
        try:
            while not finished:
                out = await stream.queue.get()
                finished = out.finished
                yield out
        finally:
            self._streams.pop(request.request_id, None)
            if not finished:
                self._inbox.put(("abort", request.request_id))

    def stats(self) -> dict[str, Any]:
        """A JSON-friendly snapshot of what the engine is doing right now."""
        with self._lock:
            snap = dict(self._snapshot)
            now = time.monotonic()
            while self._token_log and self._token_log[0][0] < now - self.stats_window:
                self._token_log.popleft()
            window = min(self.stats_window, max(now - self._started_at, 1e-9))
            snap["tokens_per_sec"] = round(sum(n for _, n in self._token_log) / window, 1)
        snap["uptime_sec"] = round(time.monotonic() - self._started_at, 1)
        return snap

    # --- engine thread -------------------------------------------------------------

    def _loop(self) -> None:
        while self._running.is_set():
            # Block for new work only when idle; otherwise just drain what is there.
            self._drain_inbox(block=not self.engine.has_unfinished())
            if not self.engine.has_unfinished():
                continue
            outputs = self.engine.step()
            with self._lock:
                self._token_log.append((time.monotonic(), len(outputs)))
            self._num_finished += sum(o.finished for o in outputs)
            # Update stats before streaming, so a client that has its last token
            # never sees a snapshot from before its request finished.
            self._update_snapshot()
            for out in outputs:
                stream = self._streams.get(out.request.request_id)
                if stream is not None:
                    stream.loop.call_soon_threadsafe(stream.queue.put_nowait, out)

    def _drain_inbox(self, block: bool) -> None:
        try:
            cmd, arg = self._inbox.get(timeout=0.1) if block else self._inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            if cmd == "add":
                self.engine.add_request(arg)
            elif cmd == "abort":
                self.engine.abort(arg)
            try:
                cmd, arg = self._inbox.get_nowait()
            except queue.Empty:
                break
        self._update_snapshot()

    def _update_snapshot(self) -> None:
        sched = self.engine.scheduler
        kv = self.engine.kv.usage()
        snap = {
            "strategy": sched.name,
            "kv_layout": self.engine.kv.layout,
            "queue_depth": len(sched.waiting),
            "batch_size": len(sched.running),
            "max_batch_size": sched.max_batch_size,
            "kv_used_tokens": kv.used_tokens,
            "kv_reserved_tokens": kv.reserved_tokens,
            "kv_capacity_tokens": kv.capacity_tokens,
            "kv_utilization": round(kv.utilization, 4),
            "kv_occupancy": round(kv.occupancy, 4),
            "num_steps": self.engine.num_steps,
            "num_generated_tokens": self.engine.num_generated_tokens,
            "num_finished_requests": self._num_finished,
            "num_preemptions": sched.num_preemptions,
        }
        with self._lock:
            self._snapshot = snap
