"""Phases 2-4: every scheduling strategy must generate exactly the reference tokens."""

from __future__ import annotations

import itertools

from conftest import SMALL_MODEL, make_prompts, reference_generate

from tinyserve.engine import Engine
from tinyserve.kv_cache import ContiguousKVCache, KVPool
from tinyserve.model import TinyGPT
from tinyserve.request import FinishReason, Request, SamplingParams
from tinyserve.scheduler import ContinuousScheduler, NaiveScheduler, StaticScheduler

GREEDY = SamplingParams(max_new_tokens=12, ignore_eos=True)


def contiguous_kv(rows: int) -> ContiguousKVCache:
    m = SMALL_MODEL
    pool = KVPool(rows * m.max_seq_len, m.n_layers, m.n_kv_heads, m.head_dim)
    return ContiguousKVCache(pool, m.max_seq_len)


def fake_clock():
    counter = itertools.count()
    return lambda: float(next(counter))


def test_naive_matches_reference(model: TinyGPT) -> None:
    prompts = make_prompts(4)
    engine = Engine(model, NaiveScheduler(contiguous_kv(1)))
    done = engine.generate(prompts, GREEDY)
    for req, prompt in zip(done, prompts, strict=True):
        assert req.output_ids == reference_generate(model, prompt, GREEDY)
        assert req.finish_reason is FinishReason.LENGTH


def test_naive_runs_one_request_at_a_time(model: TinyGPT) -> None:
    engine = Engine(model, NaiveScheduler(contiguous_kv(4)), record_trace=True)
    engine.generate(make_prompts(3), GREEDY)
    assert all(t.num_running <= 1 and t.num_prefill + t.num_decode == 1 for t in engine.trace)
    # Each request needs 1 prefill step + (max_new_tokens - 1) decode steps.
    assert engine.num_steps == 3 * GREEDY.max_new_tokens


def test_metrics_are_recorded(model: TinyGPT) -> None:
    engine = Engine(model, NaiveScheduler(contiguous_kv(1)), clock=fake_clock())
    req = Request(make_prompts(1)[0], GREEDY)
    engine.add_request(req)
    while engine.has_unfinished():
        engine.step()
    assert req.ttft is not None and req.ttft > 0
    assert req.latency is not None and req.latency > req.ttft
    assert len(req.token_times) == GREEDY.max_new_tokens


def test_eos_stops_generation(model: TinyGPT) -> None:
    engine = Engine(model, NaiveScheduler(contiguous_kv(1)))
    req = Request([5, 6, 7], SamplingParams(max_new_tokens=10))
    engine.add_request(req)
    req.append_token(2, 0.0)  # pretend the model emitted EOS
    assert req.finish_reason is FinishReason.EOS


def test_rejects_requests_longer_than_context(model: TinyGPT) -> None:
    engine = Engine(model, NaiveScheduler(contiguous_kv(1)))
    too_long = Request([5] * 100, SamplingParams(max_new_tokens=SMALL_MODEL.max_seq_len))
    try:
        engine.add_request(too_long)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_abort_frees_memory(model: TinyGPT) -> None:
    kv = contiguous_kv(1)
    engine = Engine(model, NaiveScheduler(kv))
    req = Request(make_prompts(1)[0], GREEDY)
    engine.add_request(req)
    engine.step()
    assert kv.num_sequences == 1
    engine.abort(req.request_id)
    assert kv.num_sequences == 0
    assert req.finish_reason is FinishReason.ABORTED
    assert not engine.has_unfinished()


def test_static_matches_reference(model: TinyGPT) -> None:
    prompts = make_prompts(7, seed=1)
    engine = Engine(model, StaticScheduler(contiguous_kv(4), max_batch_size=4))
    done = engine.generate(prompts, GREEDY)
    for req, prompt in zip(done, prompts, strict=True):
        assert req.output_ids == reference_generate(model, prompt, GREEDY)


def test_static_waits_for_whole_batch(model: TinyGPT) -> None:
    """A short request finishing early must not let a waiting request board."""
    engine = Engine(model, StaticScheduler(contiguous_kv(2), max_batch_size=2), record_trace=True)
    lengths = [2, 10, 3]
    reqs = [
        Request(p, SamplingParams(max_new_tokens=n, ignore_eos=True))
        for p, n in zip(make_prompts(3), lengths, strict=True)
    ]
    for r in reqs:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()
    third_prefill = next(
        i
        for i, t in enumerate(engine.trace)
        if any(rid == reqs[2].request_id for rid, _ in t.slots.values())
    )
    # The third request boards only after the 10-token request is done.
    assert third_prefill == lengths[1]
    assert max(t.num_running for t in engine.trace) == 2


def test_static_respects_kv_capacity(model: TinyGPT) -> None:
    """With room for only 2 slabs, a batch of 4 is capped at 2 by memory."""
    engine = Engine(model, StaticScheduler(contiguous_kv(2), max_batch_size=4), record_trace=True)
    engine.generate(make_prompts(5), GREEDY)
    assert max(t.num_prefill + t.num_decode for t in engine.trace) == 2


def test_continuous_matches_reference(model: TinyGPT) -> None:
    prompts = make_prompts(9, seed=2)
    params = [
        SamplingParams(max_new_tokens=n, ignore_eos=True) for n in (3, 15, 7, 1, 12, 9, 4, 20, 6)
    ]
    engine = Engine(model, ContinuousScheduler(contiguous_kv(4), max_batch_size=4))
    reqs = [Request(p, sp) for p, sp in zip(prompts, params, strict=True)]
    for r in reqs:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()
    for r in reqs:
        assert r.output_ids == reference_generate(model, r.prompt_ids, r.params)


def test_continuous_refills_seats_immediately(model: TinyGPT) -> None:
    """Same workload as the static test: the third request boards right after the first ends."""
    engine = Engine(
        model, ContinuousScheduler(contiguous_kv(2), max_batch_size=2), record_trace=True
    )
    lengths = [2, 10, 3]
    reqs = [
        Request(p, SamplingParams(max_new_tokens=n, ignore_eos=True))
        for p, n in zip(make_prompts(3), lengths, strict=True)
    ]
    for r in reqs:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()
    third_prefill = next(
        i
        for i, t in enumerate(engine.trace)
        if any(rid == reqs[2].request_id for rid, _ in t.slots.values())
    )
    assert third_prefill == lengths[0]
    # Prefill of the newcomer and decode of the others happen in the same step.
    assert engine.trace[third_prefill].num_prefill == 1
    assert engine.trace[third_prefill].num_decode == 1
    # The whole workload takes as long as the longest request, not the sum of batches.
    assert engine.num_steps == lengths[1]


def test_continuous_respects_max_batch_size(model: TinyGPT) -> None:
    engine = Engine(
        model, ContinuousScheduler(contiguous_kv(8), max_batch_size=3), record_trace=True
    )
    engine.generate(make_prompts(10), GREEDY)
    assert max(t.num_running for t in engine.trace) == 3
    assert max(len(t.slots) for t in engine.trace) == 3


def test_continuous_respects_token_budget(model: TinyGPT) -> None:
    """With a budget of 40 tokens per step, two 30-token prompts cannot prefill together."""
    sched = ContinuousScheduler(contiguous_kv(4), max_batch_size=4, max_tokens_per_step=40)
    engine = Engine(model, sched, record_trace=True)
    params = SamplingParams(max_new_tokens=5, ignore_eos=True)
    for _ in range(2):
        engine.add_request(Request([7] * 30, params))
    while engine.has_unfinished():
        engine.step()
    assert [t.num_prefill for t in engine.trace[:2]] == [1, 1]
    assert all(t.num_prefill_tokens + t.num_decode <= 40 for t in engine.trace)


def test_prefill_can_resume_in_chunks(model: TinyGPT) -> None:
    """Prefilling a prompt in three chunks samples the same first token as one pass."""
    from tinyserve.runner import ModelRunner

    prompt = make_prompts(1, seed=9, min_len=25, max_len=25)[0]
    kv = contiguous_kv(1)
    runner = ModelRunner(model, kv)
    req = Request(prompt, GREEDY)
    kv.allocate(req.request_id, req.length)
    outs = []
    for n in (10, 9, 6):
        outs += runner.prefill([req], [n])
        req.num_computed_tokens += n
    assert outs[:2] == [None, None]
    assert outs[2] == reference_generate(model, prompt, SamplingParams(max_new_tokens=1))[0]
