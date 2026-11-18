"""Replay a workload through a real scheduler on a virtual clock, without running the model.

Scheduling decisions do not depend on token values (benchmarks ignore EOS), so a
scheduler can be studied by charging each step a predicted duration instead of
computing it. :class:`StepCostModel` is fitted to a real engine trace, which
makes policy sweeps fast, deterministic and immune to a noisy machine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tinyserve.engine import StepTrace
from tinyserve.request import Request, SamplingParams
from tinyserve.scheduler import Scheduler, SchedulerOutput
from tinyserve.tokenizer import CharTokenizer
from tinyserve.workload import WorkloadItem

DUMMY_TOKEN = 5  # any non-EOS id


@dataclass(frozen=True)
class StepCostModel:
    """Linear model of one step's duration, in seconds.

    ``base + prefill_token * prefill tokens + decode_seq * decoded sequences
    + context_token * KV tokens the decoders attend to``.
    """

    base: float
    prefill_token: float
    decode_seq: float
    context_token: float

    def predict(self, prefill_tokens: int, num_decode: int, context_tokens: int) -> float:
        """Predicted duration of a step."""
        return (
            self.base
            + self.prefill_token * prefill_tokens
            + self.decode_seq * num_decode
            + self.context_token * context_tokens
        )

    def predict_plan(self, plan: SchedulerOutput) -> float:
        """Predicted duration of a scheduler plan."""
        prefill = sum(plan.prefill_len(r) for r in plan.prefill)
        context = sum(r.length for r in plan.decode)
        return self.predict(prefill, len(plan.decode), context)

    @classmethod
    def fit(cls, trace: Sequence[StepTrace]) -> StepCostModel:
        """Least-squares fit to a recorded trace (coefficients clipped at zero)."""
        rows, durations = [], []
        for s in trace:
            # The KV snapshot is taken after admission, so it includes the new prompts.
            context = max(0, s.kv.used_tokens - s.num_prefill_tokens)
            rows.append([1.0, s.num_prefill_tokens, s.num_decode, context if s.num_decode else 0])
            durations.append([s.end - s.start])
        x = torch.tensor(rows, dtype=torch.float64)
        y = torch.tensor(durations, dtype=torch.float64)
        coef = torch.linalg.lstsq(x, y).solution.clamp(min=0).flatten().tolist()
        return cls(*coef)


def simulate(
    scheduler: Scheduler, items: Sequence[WorkloadItem], cost: StepCostModel
) -> tuple[list[Request], float]:
    """Serve ``items`` with ``scheduler`` on a virtual clock that starts at 0.

    Returns the finished requests (in workload order) and the time the last one finished.
    """
    tok = CharTokenizer()
    requests = [
        Request(
            tok.encode(it.prompt),
            SamplingParams(max_new_tokens=it.max_tokens, ignore_eos=True),
            arrival_time=it.arrival,
        )
        for it in items
    ]
    now, nxt = 0.0, 0
    while nxt < len(requests) or scheduler.has_unfinished():
        while nxt < len(requests) and requests[nxt].arrival_time <= now:
            scheduler.add(requests[nxt])
            nxt += 1
        if not scheduler.has_unfinished():
            now = requests[nxt].arrival_time
            continue
        plan = scheduler.schedule(now)
        if plan.is_empty:
            raise RuntimeError("scheduler made no progress")
        now += cost.predict_plan(plan)
        for req in plan.prefill:
            req.num_computed_tokens += plan.prefill_len(req)
            if req.num_computed_tokens == req.length:
                req.append_token(DUMMY_TOKEN, now)
        for req in plan.decode:
            req.num_computed_tokens = req.length
            req.append_token(DUMMY_TOKEN, now)
        scheduler.finish_step()
    return requests, now
