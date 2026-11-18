"""The virtual-clock simulator must make the same scheduling decisions as the real engine."""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import SMALL_MODEL

from tinyserve.engine import Engine, StepTrace
from tinyserve.kv_cache import KVPool, KVUsage, PagedKVCache
from tinyserve.model import TinyGPT
from tinyserve.request import Request, SamplingParams
from tinyserve.scheduler import ContinuousScheduler
from tinyserve.simulator import StepCostModel, simulate
from tinyserve.tokenizer import CharTokenizer
from tinyserve.workload import WorkloadSpec, make_workload

M = SMALL_MODEL
SPEC = WorkloadSpec(num_requests=10, max_prompt_len=40, max_output_len=40, seed=3)


def scheduler() -> ContinuousScheduler:
    pool = KVPool(256, M.n_layers, M.n_kv_heads, M.head_dim)
    return ContinuousScheduler(PagedKVCache(pool, block_size=8), max_batch_size=4)


def test_simulator_takes_the_same_steps_as_the_engine(model: TinyGPT) -> None:
    items = [replace(it, arrival=0.0) for it in make_workload(SPEC)]
    engine = Engine(model, scheduler())
    tok = CharTokenizer()
    for it in items:
        params = SamplingParams(max_new_tokens=it.max_tokens, ignore_eos=True)
        engine.add_request(Request(tok.encode(it.prompt), params, arrival_time=it.arrival))
    while engine.has_unfinished():
        engine.step()

    one_per_step = StepCostModel(base=1.0, prefill_token=0, decode_seq=0, context_token=0)
    requests, end = simulate(scheduler(), items, one_per_step)
    assert end == engine.num_steps
    assert [len(r.output_ids) for r in requests] == [it.max_tokens for it in items]


def test_simulator_waits_for_arrivals() -> None:
    items = make_workload(WorkloadSpec(num_requests=3, rate=0.01, max_output_len=8, seed=1))
    cost = StepCostModel(base=0.01, prefill_token=0, decode_seq=0, context_token=0)
    requests, end = simulate(scheduler(), items, cost)
    for r in requests:
        assert r.ttft == pytest.approx(0.01)
    assert end == pytest.approx(items[-1].arrival + 0.01 * items[-1].max_tokens)


def test_cost_model_fit_recovers_coefficients() -> None:
    truth = StepCostModel(base=0.002, prefill_token=1e-5, decode_seq=3e-4, context_token=2e-6)
    trace, t = [], 0.0
    for i in range(50):
        prefill, decode, context = (i * 37) % 200, (i * 7) % 16 + 1, (i * 113) % 900
        dt = truth.predict(prefill, decode, context)
        usage = KVUsage(context + prefill, context + prefill, 4096)
        trace.append(StepTrace(t, t + dt, int(prefill > 0), prefill, decode, 0, decode, usage))
        t += dt
    fitted = StepCostModel.fit(trace)
    for name in ("base", "prefill_token", "decode_seq", "context_token"):
        assert getattr(fitted, name) == pytest.approx(getattr(truth, name), rel=1e-6)
