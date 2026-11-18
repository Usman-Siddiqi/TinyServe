"""Speculative decoding: a cheap proposer drafts tokens, the target model verifies them.

Each decode step, a proposer guesses the next ``k`` tokens of every request. The
target then scores the latest token plus all ``k`` guesses in *one* forward pass
and keeps the longest prefix it agrees with, plus one token of its own. A step
therefore yields between 1 and ``k + 1`` tokens for the price of one (wider)
target pass plus the drafting.

Acceptance follows Leviathan et al. (2023) and Chen et al. (2023):

* greedy requests accept a draft token iff it equals the target's argmax, so the
  output is the same as ordinary greedy decoding;
* sampled requests use rejection sampling: accept draft token ``x`` with
  probability ``min(1, p(x) / q(x))``, otherwise resample from
  ``max(0, p - q)``. The output then follows the target distribution exactly.

Two proposers are provided: :class:`NgramProposer` (prompt lookup, no model) and
:class:`DraftModelProposer` (a small transformer with its own paged KV cache).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor

from tinyserve.kv_cache import KVCacheManager
from tinyserve.model import TinyGPT
from tinyserve.request import Request
from tinyserve.runner import ModelRunner, forward_chunks


@dataclass
class Draft:
    """Proposed continuation of one request.

    Attributes:
        tokens: the drafted token ids.
        probs: ``(len(tokens), vocab)`` draft distribution each token was sampled
            from, or ``None`` if the tokens were chosen deterministically (then
            ``q`` is treated as one-hot on the chosen token).
    """

    tokens: list[int]
    probs: Tensor | None = None


class Proposer(ABC):
    """Drafts candidate tokens for a batch of requests."""

    @abstractmethod
    def propose(self, requests: list[Request], num_tokens: list[int]) -> list[Draft]:
        """Draft up to ``num_tokens[i]`` tokens continuing ``requests[i]``."""

    def release(self, request_id: int) -> None:  # noqa: B027 - optional hook
        """Forget any state held for a request that finished or was preempted."""


class NgramProposer(Proposer):
    """Prompt-lookup decoding: copy what followed an earlier occurrence of the tail.

    Finds the most recent earlier match of the last ``n`` tokens (trying
    ``max_ngram`` down to 1) and proposes the tokens that came after it,
    extending the lookup over its own guesses until ``k`` tokens are drafted.
    Costs no model calls; works when text repeats itself.
    """

    def __init__(self, max_ngram: int = 3) -> None:
        self.max_ngram = max_ngram

    def propose(self, requests: list[Request], num_tokens: list[int]) -> list[Draft]:
        """Draft by lookup in each request's own history."""
        return [
            Draft(self._lookup(r.all_ids, k)) for r, k in zip(requests, num_tokens, strict=True)
        ]

    def _lookup(self, ids: list[int], k: int) -> list[int]:
        ids = list(ids)
        draft: list[int] = []
        while len(draft) < k:
            continuation = self._match(ids)
            if not continuation:
                break
            continuation = continuation[: k - len(draft)]
            draft += continuation
            ids += continuation
        return draft

    def _match(self, ids: list[int]) -> list[int]:
        for n in range(min(self.max_ngram, len(ids) - 1), 0, -1):
            tail = ids[-n:]
            for start in range(len(ids) - n - 1, -1, -1):
                if ids[start : start + n] == tail:
                    return ids[start + n : len(ids)]
        return []


class DraftModelProposer(Proposer):
    """Drafts with a small transformer that keeps its own KV cache.

    The draft cache holds KV for a prefix of each request. Because the engine only
    ever appends tokens, the first ``request.length - 1`` cached positions stay
    valid after a verification, except where a draft token was rejected: that
    position (and everything after it) is simply recomputed on the next call.

    Args:
        model: the draft model. It must share the target's tokenizer and context size.
        kv: a KV cache manager over a pool shaped for ``model``.
    """

    def __init__(self, model: TinyGPT, kv: KVCacheManager) -> None:
        self.model = model
        self.kv = kv
        self._computed: dict[int, int] = {}  # request_id -> tokens with valid draft KV

    @torch.inference_mode()
    def propose(self, requests: list[Request], num_tokens: list[int]) -> list[Draft]:
        """Catch the draft cache up on each sequence, then draft ``k`` tokens autoregressively."""
        drafts = [Draft([]) for _ in requests]
        active = [i for i, k in enumerate(num_tokens) if k > 0 and self._reserve(requests[i], k)]
        if not active:
            return drafts
        reqs = [requests[i] for i in active]
        ks = [num_tokens[i] for i in active]
        seq_ids = [r.request_id for r in reqs]
        tokens: list[list[int]] = [[] for _ in reqs]
        probs: list[list[Tensor]] = [[] for _ in reqs]

        # Feed every token the draft has not seen yet (at least the latest one).
        starts = [min(self._computed[r.request_id], r.length - 1) for r in reqs]
        chunks = [r.all_ids[s:] for r, s in zip(reqs, starts, strict=True)]
        hidden, lengths = forward_chunks(self.model, self.kv, seq_ids, chunks, starts)
        last = self.model.logits(hidden[torch.arange(len(reqs)), lengths - 1])
        rows = list(range(len(reqs)))
        while True:
            for row, logits in zip(rows, last, strict=True):
                token, q = _draft_sample(logits, reqs[row])
                tokens[row].append(token)
                if q is not None:
                    probs[row].append(q)
            rows = [row for row in rows if len(tokens[row]) < ks[row]]
            if not rows:
                break
            hidden, _ = forward_chunks(
                self.model,
                self.kv,
                [seq_ids[row] for row in rows],
                [[tokens[row][-1]] for row in rows],
                [reqs[row].length + len(tokens[row]) - 1 for row in rows],
            )
            last = self.model.logits(hidden[:, 0])

        for row, i in enumerate(active):
            # The last draft token is never fed, so its KV is not cached.
            self._computed[seq_ids[row]] = reqs[row].length + ks[row] - 1
            drafts[i] = Draft(tokens[row], torch.stack(probs[row]) if probs[row] else None)
        return drafts

    def release(self, request_id: int) -> None:
        """Free the request's draft KV."""
        if request_id in self.kv:
            self.kv.free(request_id)
        self._computed.pop(request_id, None)

    def _reserve(self, request: Request, k: int) -> bool:
        """Make room for the KV this call writes. False (no drafting) if the pool is full."""
        rid = request.request_id
        needed = request.length + k - 1
        if rid not in self.kv:
            self._computed[rid] = 0
            return self.kv.allocate(rid, needed)
        return self.kv.grow(rid, needed)


def _draft_sample(logits: Tensor, request: Request) -> tuple[int, Tensor | None]:
    """Pick a draft token like the target would: argmax, or a sample with its distribution."""
    temperature = request.params.temperature
    if temperature == 0:
        return int(logits.argmax()), None
    q = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(q, 1, generator=request.generator)), q


def accept_greedy(draft: list[int], target_logits: Tensor) -> list[int]:
    """Keep the drafts matching the target's argmax, then append the target's own token."""
    target = target_logits.argmax(dim=-1).tolist()
    n = 0
    while n < len(draft) and draft[n] == target[n]:
        n += 1
    return [*draft[:n], target[n]]


def accept_sampled(
    draft: Draft, target_logits: Tensor, temperature: float, generator: torch.Generator
) -> list[int]:
    """Rejection sampling: the returned tokens are distributed exactly as the target's samples."""
    p = torch.softmax(target_logits.float() / temperature, dim=-1)
    for i, token in enumerate(draft.tokens):
        p_x = float(p[i, token])
        q_x = 1.0 if draft.probs is None else float(draft.probs[i, token])
        if float(torch.rand((), generator=generator)) * q_x < p_x:
            continue
        if draft.probs is None:
            residual = p[i].clone()
            residual[token] = 0.0
        else:
            residual = (p[i] - draft.probs[i]).clamp_min(0.0)
        if float(residual.sum()) <= 0.0:  # p == q up to rounding
            residual = p[i]
        return [*draft.tokens[:i], int(torch.multinomial(residual, 1, generator=generator))]
    bonus = int(torch.multinomial(p[len(draft.tokens)], 1, generator=generator))
    return [*draft.tokens, bonus]


@dataclass
class SpecStats:
    """Counters for how well speculation is working.

    Attributes:
        num_verifications: request-steps that ran a verification.
        num_drafted: draft tokens proposed.
        num_accepted: draft tokens accepted.
        drafted_at: ``drafted_at[i]`` counts drafts at draft position ``i``.
        accepted_at: ``accepted_at[i]`` counts accepted drafts at position ``i``.
    """

    num_verifications: int = 0
    num_drafted: int = 0
    num_accepted: int = 0
    drafted_at: list[int] = field(default_factory=list)
    accepted_at: list[int] = field(default_factory=list)

    def record(self, num_drafted: int, num_accepted: int) -> None:
        """Account for one request's verification."""
        self.num_verifications += 1
        self.num_drafted += num_drafted
        self.num_accepted += num_accepted
        for counts in (self.drafted_at, self.accepted_at):
            counts.extend([0] * (num_drafted - len(counts)))
        for i in range(num_drafted):
            self.drafted_at[i] += 1
            self.accepted_at[i] += i < num_accepted

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that were accepted."""
        return self.num_accepted / self.num_drafted if self.num_drafted else 0.0

    @property
    def tokens_per_verification(self) -> float:
        """Tokens produced per request per target pass (1.0 = no gain)."""
        if not self.num_verifications:
            return 0.0
        return (self.num_accepted + self.num_verifications) / self.num_verifications

    def acceptance_by_position(self) -> list[float]:
        """Probability that draft position ``i`` is accepted, given it was proposed."""
        return [a / d for a, d in zip(self.accepted_at, self.drafted_at, strict=False) if d]


class DraftLengthController:
    """Picks the draft length for each decode batch from recent acceptance rates.

    With per-token acceptance rate ``a`` and ``k`` drafts, a request gains
    ``(1 - a^(k+1)) / (1 - a)`` tokens per step on average (Leviathan et al., 2023),
    while the step costs about ``1 + c*k`` plain decode steps. Verification pads
    every request to the longest draft, so one ``k`` is chosen per batch: the one
    with the most expected tokens per unit cost. When drafts keep failing it
    picks 0 and the step is an ordinary decode.

    ``c = draft_cost + verify_cost * (1 + batch_size / 4)``: verifying extra
    tokens gets pricier as the batch grows, because CPU decoding is close to
    compute-bound. The defaults are rough fits from ``bench/spec_costs.py``.

    Args:
        draft_cost: cost of drafting one token, relative to a plain decode step.
        verify_cost: extra cost of verifying one more token per request.
        decay: weight of older verifications in each acceptance estimate.
        probe_every: after this many steps without drafting, draft one token to
            re-check acceptance.
    """

    def __init__(
        self,
        draft_cost: float,
        verify_cost: float = 0.12,
        decay: float = 0.8,
        probe_every: int = 8,
    ) -> None:
        self.draft_cost = draft_cost
        self.verify_cost = verify_cost
        self.decay = decay
        self.probe_every = probe_every
        self._counts: dict[int, tuple[float, float]] = {}  # request_id -> (accepted, rejected)
        self._idle_steps = 0

    def acceptance(self, request_id: int) -> float:
        """Estimated per-token acceptance rate (a prior of 2/3 for new requests)."""
        accepted, rejected = self._counts.get(request_id, (2.0, 1.0))
        return accepted / (accepted + rejected)

    def choose(self, requests: list[Request], limits: list[int]) -> list[int]:
        """Draft lengths for a decode batch, each capped by ``limits[i]``."""
        rates = [self.acceptance(r.request_id) for r in requests]
        c = self.draft_cost + self.verify_cost * (1 + len(requests) / 4)
        best_k, best = 0, float(len(requests))
        for k in range(1, max(limits, default=0) + 1):
            expected = sum(
                sum(a**i for i in range(min(k, n) + 1)) for a, n in zip(rates, limits, strict=True)
            )
            if expected / (1 + c * k) > best:
                best_k, best = k, expected / (1 + c * k)
        if best_k == 0:
            self._idle_steps += 1
            if self._idle_steps >= self.probe_every:
                self._idle_steps, best_k = 0, 1
        return [min(best_k, n) for n in limits]

    def update(self, request_id: int, drafted: int, accepted: int) -> None:
        """Fold one verification into the request's acceptance estimate."""
        if drafted:
            acc, rej = self._counts.get(request_id, (2.0, 1.0))
            self._counts[request_id] = (
                self.decay * acc + accepted,
                self.decay * rej + (accepted < drafted),
            )

    def release(self, request_id: int) -> None:
        """Forget a finished request."""
        self._counts.pop(request_id, None)


class SpeculativeDecoder:
    """Replaces the plain decode step with draft, verify, accept.

    Args:
        proposer: where the drafts come from.
        num_speculative_tokens: most drafts per request per step (``k``).
        controller: adapts each request's draft length; ``None`` always drafts ``k``.
    """

    def __init__(
        self,
        proposer: Proposer,
        num_speculative_tokens: int = 4,
        controller: DraftLengthController | None = None,
    ) -> None:
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        self.proposer = proposer
        self.num_speculative_tokens = num_speculative_tokens
        self.controller = controller
        self.stats = SpecStats()

    def decode(
        self, runner: ModelRunner, requests: list[Request], num_tokens: list[int]
    ) -> list[list[int]]:
        """Draft up to ``num_tokens[i]`` tokens per request, verify, return the new tokens."""
        if self.controller is not None:
            num_tokens = self.controller.choose(requests, num_tokens)
        drafts = self.proposer.propose(requests, num_tokens)
        logits = runner.verify(requests, [d.tokens for d in drafts])
        results = []
        for req, draft, row in zip(requests, drafts, logits, strict=True):
            t = req.params.temperature
            if t == 0:
                tokens = accept_greedy(draft.tokens, row)
            else:
                tokens = accept_sampled(draft, row, t, req.generator)
            self.stats.record(len(draft.tokens), len(tokens) - 1)
            if self.controller is not None:
                self.controller.update(req.request_id, len(draft.tokens), len(tokens) - 1)
            results.append(tokens)
        return results

    def release(self, request_id: int) -> None:
        """Forget all state held for a request."""
        self.proposer.release(request_id)
        if self.controller is not None:
            self.controller.release(request_id)
