"""Queue policies: the order in which the continuous scheduler admits and preempts requests.

A policy ranks requests; the scheduler admits waiting requests best-first and
preempts running ones worst-first. Requests that were preempted after
streaming some tokens are always readmitted first, whatever the policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from tinyserve.request import Request

POLICIES = ("fcfs", "sjf", "edf", "slo")

Priority = tuple[float, ...]


def _arrival(request: Request, now: float) -> float:
    """Arrival time, treating a request with none as arriving ``now``."""
    return now if request.arrival_time is None else request.arrival_time


class QueuePolicy(ABC):
    """Ranks requests. Lower :meth:`priority` runs first."""

    name = "base"

    @abstractmethod
    def priority(self, request: Request, now: float) -> Priority:
        """Sort key of ``request`` at time ``now``."""

    def order(self, requests: Iterable[Request], now: float) -> list[Request]:
        """``requests`` best-first. The sort is stable, so ties keep queue order."""
        return sorted(requests, key=lambda r: (not r.output_ids, *self.priority(r, now)))


class FCFSPolicy(QueuePolicy):
    """First come, first served: keep the queue's own order."""

    name = "fcfs"

    def priority(self, request: Request, now: float) -> Priority:
        """Arrival time (only used by callers that sort explicitly)."""
        return (_arrival(request, now),)

    def order(self, requests: Iterable[Request], now: float) -> list[Request]:
        """The queue as is: arrival order, preempted requests already at the front."""
        return list(requests)


class SJFPolicy(QueuePolicy):
    """Shortest job first, with aging so long jobs cannot starve.

    A job's size is its prompt length plus ``max_new_tokens``. Every second of
    waiting subtracts ``aging`` tokens from it, so a job that has waited
    ``(largest - smallest) / aging`` seconds outranks any newcomer.

    Args:
        aging: tokens of priority gained per second of waiting. 0 is pure SJF.
    """

    name = "sjf"

    def __init__(self, aging: float = 100.0) -> None:
        if aging < 0:
            raise ValueError("aging must be >= 0")
        self.aging = aging

    def priority(self, request: Request, now: float) -> Priority:
        """Size hint minus the aging credit earned while waiting."""
        waited = now - _arrival(request, now)
        return (request.max_length - self.aging * waited,)


class EDFPolicy(QueuePolicy):
    """Earliest TTFT deadline first, where deadline = arrival + ``ttft_slo``.

    With ``demote_missed``, requests whose deadline has already passed go behind
    every request that can still make it, so under overload one late request
    does not make the next one late too. Without it, and with one SLO for every
    request, EDF orders exactly like FCFS.

    Args:
        ttft_slo: time-to-first-token target, in seconds.
        demote_missed: rank requests that already missed their deadline last.
    """

    name = "edf"

    def __init__(self, ttft_slo: float = 1.0, demote_missed: bool = True) -> None:
        if ttft_slo <= 0:
            raise ValueError("ttft_slo must be > 0")
        self.ttft_slo = ttft_slo
        self.demote_missed = demote_missed

    def missed(self, request: Request, now: float) -> bool:
        """True if ``request`` is still waiting for its first token past its deadline."""
        deadline = _arrival(request, now) + self.ttft_slo
        return self.demote_missed and not request.output_ids and now > deadline

    def priority(self, request: Request, now: float) -> Priority:
        """``(missed, deadline)``."""
        return (float(self.missed(request, now)), _arrival(request, now) + self.ttft_slo)


class SLOPolicy(EDFPolicy):
    """Shortest feasible job first: SJF among requests that can still meet their TTFT target.

    Requests past their deadline are demoted behind all others and served oldest
    first. Short jobs free their seats quickly, so more of the requests that are
    still on time get in before their deadline.

    Args:
        ttft_slo: time-to-first-token target, in seconds.
        aging: as in :class:`SJFPolicy`.
    """

    name = "slo"

    def __init__(self, ttft_slo: float = 1.0, aging: float = 100.0) -> None:
        super().__init__(ttft_slo, demote_missed=True)
        self.sjf = SJFPolicy(aging)

    def priority(self, request: Request, now: float) -> Priority:
        """``(1, deadline)`` if late, else ``(0, aged size)``."""
        if self.missed(request, now):
            return (1.0, _arrival(request, now) + self.ttft_slo)
        return (0.0, *self.sjf.priority(request, now))


def make_policy(name: str, ttft_slo: float = 1.0, sjf_aging: float = 100.0) -> QueuePolicy:
    """Build a policy by name (one of :data:`POLICIES`)."""
    if name == "fcfs":
        return FCFSPolicy()
    if name == "sjf":
        return SJFPolicy(sjf_aging)
    if name == "edf":
        return EDFPolicy(ttft_slo)
    if name == "slo":
        return SLOPolicy(ttft_slo, sjf_aging)
    raise ValueError(f"unknown queue policy {name!r}; choose from {POLICIES}")
