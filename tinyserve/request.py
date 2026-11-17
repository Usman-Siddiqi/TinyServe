"""Requests, sampling parameters and the per-request metrics the engine records."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

import torch

from tinyserve.tokenizer import EOS_ID

_request_ids = itertools.count()


@dataclass(frozen=True)
class SamplingParams:
    """How to generate tokens for one request.

    Attributes:
        max_new_tokens: stop after this many generated tokens.
        temperature: 0 means greedy (argmax). Higher values sample more randomly.
        ignore_eos: keep going past the end-of-sequence token. Benchmarks set this
            so that output lengths are exactly ``max_new_tokens``.
        seed: seed for this request's private RNG, so sampled outputs do not depend
            on what else happens to be in the batch.
    """

    max_new_tokens: int = 64
    temperature: float = 0.0
    ignore_eos: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")


class RequestStatus(enum.Enum):
    """Lifecycle of a request inside the engine."""

    WAITING = "waiting"  # queued, holds no KV memory
    RUNNING = "running"  # has KV memory and is generating
    FINISHED = "finished"


class FinishReason(enum.Enum):
    """Why a request stopped generating."""

    LENGTH = "length"
    EOS = "eos"
    ABORTED = "aborted"


@dataclass
class Request:
    """One generation request and everything the engine tracks about it.

    The KV-cache invariant the engine maintains: after every step, the cache holds
    KV for every token of ``all_ids`` except the most recently sampled one. The
    next decode step feeds exactly that token.
    """

    prompt_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float | None = None  # set by the engine if not given
    request_id: int = field(default_factory=lambda: next(_request_ids))

    output_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: FinishReason | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    num_preemptions: int = 0
    _generator: torch.Generator | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("prompt must contain at least one token")

    @property
    def all_ids(self) -> list[int]:
        """Prompt followed by everything generated so far."""
        return self.prompt_ids + self.output_ids

    @property
    def length(self) -> int:
        """Total tokens in the sequence (prompt + generated)."""
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def max_length(self) -> int:
        """The longest this sequence can ever get."""
        return len(self.prompt_ids) + self.params.max_new_tokens

    @property
    def is_finished(self) -> bool:
        """True once the request has stopped generating."""
        return self.status is RequestStatus.FINISHED

    @property
    def generator(self) -> torch.Generator:
        """The request's private RNG for sampling (created lazily)."""
        if self._generator is None:
            seed = self.params.seed if self.params.seed is not None else self.request_id
            self._generator = torch.Generator().manual_seed(seed)
        return self._generator

    def append_token(self, token_id: int, now: float) -> None:
        """Record a newly generated token and check the stopping conditions."""
        self.output_ids.append(token_id)
        self.token_times.append(now)
        if self.first_token_time is None:
            self.first_token_time = now
        if token_id == EOS_ID and not self.params.ignore_eos:
            self.finish(FinishReason.EOS, now)
        elif len(self.output_ids) >= self.params.max_new_tokens:
            self.finish(FinishReason.LENGTH, now)

    def finish(self, reason: FinishReason, now: float) -> None:
        """Mark the request as done."""
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.finish_time = now

    # --- metrics ------------------------------------------------------------------

    @property
    def ttft(self) -> float | None:
        """Time to first token: arrival until the first generated token."""
        if self.first_token_time is None or self.arrival_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def latency(self) -> float | None:
        """End-to-end latency: arrival until the last token."""
        if self.finish_time is None or self.arrival_time is None:
            return None
        return self.finish_time - self.arrival_time
