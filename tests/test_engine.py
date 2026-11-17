"""Phases 2-4: every scheduling strategy must generate exactly the reference tokens."""

from __future__ import annotations

import itertools

from conftest import SMALL_MODEL, make_prompts, reference_generate

from tinyserve.engine import Engine
from tinyserve.kv_cache import ContiguousKVCache, KVPool
from tinyserve.model import TinyGPT
from tinyserve.request import FinishReason, Request, SamplingParams
from tinyserve.scheduler import NaiveScheduler

GREEDY = SamplingParams(max_new_tokens=12, ignore_eos=True)


def contiguous_kv(rows: int) -> ContiguousKVCache:
    m = SMALL_MODEL
    pool = KVPool(rows * m.max_seq_len, m.n_layers, m.n_heads, m.head_dim)
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
