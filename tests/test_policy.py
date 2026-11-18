"""Queue policies: ordering, starvation freedom, preemption order and unchanged outputs."""

from __future__ import annotations

import itertools

import pytest
from conftest import SMALL_MODEL, make_prompts, reference_generate

from tinyserve.engine import Engine, EngineConfig, build_engine
from tinyserve.kv_cache import KVPool, PagedKVCache
from tinyserve.model import TinyGPT
from tinyserve.policy import POLICIES, EDFPolicy, FCFSPolicy, SJFPolicy, SLOPolicy, make_policy
from tinyserve.request import Request, SamplingParams
from tinyserve.scheduler import ContinuousScheduler

M = SMALL_MODEL


def paged_kv(num_slots: int, block_size: int = 8) -> PagedKVCache:
    pool = KVPool(num_slots, M.n_layers, M.n_kv_heads, M.head_dim)
    return PagedKVCache(pool, block_size=block_size, watermark=0)


def req(prompt_len: int, new_tokens: int, arrival: float) -> Request:
    params = SamplingParams(max_new_tokens=new_tokens, ignore_eos=True)
    return Request([7] * prompt_len, params, arrival_time=arrival)


def fake_step(sched: ContinuousScheduler, now: float) -> list[Request]:
    """One scheduler step without the model: every scheduled request gets a dummy token."""
    plan = sched.schedule(now)
    for r in plan.prefill + plan.decode:
        r.num_computed_tokens = r.length
        r.append_token(5, now)
    sched.finish_step()
    return plan.prefill


def test_fcfs_keeps_queue_order() -> None:
    reqs = [req(10, 50, 2.0), req(10, 5, 0.0), req(10, 20, 1.0)]
    assert FCFSPolicy().order(reqs, now=3.0) == reqs


def test_sjf_orders_by_size() -> None:
    big, small, mid = req(10, 50, 0.0), req(10, 5, 0.0), req(10, 20, 0.0)
    assert SJFPolicy(aging=0).order([big, small, mid], now=0.0) == [small, mid, big]


def test_sjf_aging_lets_an_old_long_job_overtake() -> None:
    old_big, new_small = req(10, 100, 0.0), req(10, 5, 9.0)
    assert SJFPolicy(aging=0).order([old_big, new_small], now=10.0)[0] is new_small
    # 95 tokens of size difference vs. 9 s of extra waiting: 9 * 20 = 180 tokens of credit.
    assert SJFPolicy(aging=20).order([old_big, new_small], now=10.0)[0] is old_big


def test_edf_orders_by_deadline_and_demotes_missed() -> None:
    late, early, fresh = req(10, 5, 1.0), req(10, 5, 0.0), req(10, 5, 4.5)
    plain = EDFPolicy(ttft_slo=1.0, demote_missed=False)
    assert plain.order([late, early, fresh], now=5.0) == [early, late, fresh]
    # At t=5 only `fresh` (deadline 5.5) can still meet its TTFT target.
    assert EDFPolicy(ttft_slo=1.0).order([late, early, fresh], now=5.0) == [fresh, early, late]


def test_slo_runs_short_feasible_jobs_first_then_late_ones_oldest_first() -> None:
    late_small, late_old = req(10, 5, 1.0), req(10, 90, 0.0)
    fresh_big, fresh_small = req(10, 80, 4.5), req(10, 20, 4.6)
    policy = SLOPolicy(ttft_slo=1.0, aging=0)
    order = policy.order([late_small, fresh_big, late_old, fresh_small], now=5.0)
    assert order == [fresh_small, fresh_big, late_old, late_small]


@pytest.mark.parametrize("name", ["sjf", "edf", "slo"])  # FCFS: the queue's appendleft
def test_preempted_requests_are_readmitted_first(name: str) -> None:
    resumed, newcomer = req(10, 50, 3.0), req(5, 5, 0.0)
    resumed.output_ids = [5, 5]  # it streamed tokens before being preempted
    assert make_policy(name).order([newcomer, resumed], now=3.0)[0] is resumed


def test_sjf_without_aging_starves_and_with_aging_does_not() -> None:
    """One seat, a long job, and a one-step job arriving every step."""

    def admission_step(aging: float, steps: int = 400) -> int | None:
        sched = ContinuousScheduler(
            paged_kv(4096), max_batch_size=1, max_tokens_per_step=4096, policy=SJFPolicy(aging)
        )
        long_job = req(10, 200, 0.0)
        sched.add(long_job)
        for step in range(steps):
            now = step * 0.01
            sched.add(req(10, 1, now))
            if long_job in fake_step(sched, now):
                return step
        return None

    assert admission_step(aging=0) is None
    # 199 tokens of size difference at 100 tokens/s: admitted after about 2 s (200 steps).
    step = admission_step(aging=100)
    assert step is not None and 190 <= step <= 210


def test_sjf_preempts_the_largest_job() -> None:
    sched = ContinuousScheduler(
        paged_kv(64, block_size=8), max_batch_size=4, max_tokens_per_step=64, policy=SJFPolicy(0)
    )
    small, big = req(8, 40, 0.0), req(8, 50, 0.0)
    for r in (big, small):
        sched.add(r)
    preempted: list[Request] = []
    for step in range(60):
        now = float(step)
        plan = sched.schedule(now)
        preempted += plan.preempted
        for r in plan.prefill + plan.decode:
            r.num_computed_tokens = r.length
            r.append_token(5, now)
        sched.finish_step()
        if not sched.has_unfinished():
            break
    assert preempted and all(r is big for r in preempted)


@pytest.mark.parametrize("name", POLICIES)
def test_outputs_do_not_depend_on_policy(model: TinyGPT, name: str) -> None:
    """Greedy outputs must be identical under every policy, even with preemptions."""
    prompts = make_prompts(8, seed=5, min_len=10, max_len=30)
    lengths = [24, 3, 17, 9, 30, 5, 12, 20]
    counter = itertools.count()
    sched = ContinuousScheduler(
        paged_kv(96), max_batch_size=6, max_tokens_per_step=96, policy=make_policy(name, 2.0)
    )
    engine = Engine(model, sched, clock=lambda: float(next(counter)))
    reqs = [
        Request(p, SamplingParams(max_new_tokens=n, ignore_eos=True))
        for p, n in zip(prompts, lengths, strict=True)
    ]
    for r in reqs:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()
    for r in reqs:
        assert r.output_ids == reference_generate(model, r.prompt_ids, r.params)
    assert sched.num_preemptions > 0
    assert sched.kv.allocator.num_free == sched.kv.allocator.num_blocks


def test_build_engine_validates_policy() -> None:
    with pytest.raises(ValueError, match="unknown queue_policy"):
        build_engine(EngineConfig(strategy="continuous", queue_policy="lifo"))
    with pytest.raises(ValueError, match="continuous"):
        build_engine(EngineConfig(strategy="static", queue_policy="sjf"))
