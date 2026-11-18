"""Chunked prefill: long prompts are split across steps without changing any output."""

from __future__ import annotations

import pytest
from conftest import SMALL_MODEL, make_prompts, reference_generate

from tinyserve.engine import Engine
from tinyserve.kv_cache import ContiguousKVCache, KVCacheManager, KVPool, PagedKVCache
from tinyserve.model import TinyGPT
from tinyserve.request import FinishReason, Request, SamplingParams
from tinyserve.scheduler import ContinuousScheduler

M = SMALL_MODEL


def pool(num_slots: int) -> KVPool:
    return KVPool(num_slots, M.n_layers, M.n_kv_heads, M.head_dim)


def run(engine: Engine, requests: list[Request]) -> None:
    for r in requests:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()


@pytest.mark.parametrize("layout", ["contiguous", "paged"])
@pytest.mark.parametrize("budget", [4, 16, 64])
def test_chunked_matches_reference(model: TinyGPT, layout: str, budget: int) -> None:
    kv: KVCacheManager
    if layout == "paged":
        kv = PagedKVCache(pool(4 * M.max_seq_len), block_size=8)
    else:
        kv = ContiguousKVCache(pool(4 * M.max_seq_len), M.max_seq_len)
    sched = ContinuousScheduler(
        kv, max_batch_size=4, max_tokens_per_step=budget, chunked_prefill=True
    )
    engine = Engine(model, sched, record_trace=True)
    prompts = make_prompts(7, seed=11, min_len=5, max_len=70)
    reqs = [
        Request(p, SamplingParams(max_new_tokens=n, ignore_eos=True))
        for p, n in zip(prompts, (6, 1, 12, 3, 9, 5, 8), strict=True)
    ]
    run(engine, reqs)
    for r in reqs:
        assert r.output_ids == reference_generate(model, r.prompt_ids, r.params)
    assert all(t.num_prefill_tokens + t.num_decode <= budget for t in engine.trace)
    assert kv.num_sequences == 0


def test_budget_is_filled_decode_first(model: TinyGPT) -> None:
    """A long prompt is fed in chunks that use exactly what decodes leave over."""
    kv = ContiguousKVCache(pool(4 * M.max_seq_len), M.max_seq_len)
    sched = ContinuousScheduler(kv, max_batch_size=4, max_tokens_per_step=16, chunked_prefill=True)
    engine = Engine(model, sched, record_trace=True)
    short = Request([5] * 4, SamplingParams(max_new_tokens=30, ignore_eos=True))
    prompt = make_prompts(1, seed=14, min_len=100, max_len=100)[0]
    long = Request(prompt, SamplingParams(max_new_tokens=3, ignore_eos=True))
    run(engine, [short, long])

    assert long.output_ids == reference_generate(model, long.prompt_ids, long.params)
    # Step 0 prefills `short` (4) and the first 12 tokens of `long`. After that
    # `short` decodes every step and `long` gets the other 15 tokens.
    chunks = [t.num_prefill_tokens for t in engine.trace[:8]]
    assert chunks == [16, 15, 15, 15, 15, 15, 13, 0]
    assert all(t.num_decode >= 1 for t in engine.trace[1:8])


def test_long_prompt_is_rejected_without_chunking(model: TinyGPT) -> None:
    kv = ContiguousKVCache(pool(M.max_seq_len), M.max_seq_len)
    prompt = [5] * 50
    with pytest.raises(ValueError):
        ContinuousScheduler(kv, 4, max_tokens_per_step=16).add(Request(prompt))
    ContinuousScheduler(kv, 4, max_tokens_per_step=16, chunked_prefill=True).add(Request(prompt))


def test_budget_must_cover_a_full_decode_batch() -> None:
    kv = ContiguousKVCache(pool(M.max_seq_len), M.max_seq_len)
    with pytest.raises(ValueError):
        ContinuousScheduler(kv, max_batch_size=8, max_tokens_per_step=4)


class RecordingScheduler(ContinuousScheduler):
    """Remembers how far each victim's prefill had got when it was preempted."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.victims: list[tuple[int, int]] = []

    def _preempt(self, request: Request) -> None:
        self.victims.append((request.num_computed_tokens, request.length))
        super()._preempt(request)


def test_preemption_mid_prefill_keeps_outputs_correct(model: TinyGPT) -> None:
    """A decode that needs a block evicts a request whose prefill is only partly done."""
    kv = PagedKVCache(pool(64), block_size=8, watermark=0)  # 8 blocks
    sched = RecordingScheduler(kv, max_batch_size=2, max_tokens_per_step=2, chunked_prefill=True)
    engine = Engine(model, sched)
    a = Request(
        make_prompts(1, seed=12, min_len=7, max_len=7)[0], SamplingParams(30, ignore_eos=True)
    )
    # 50 prompt tokens take 7 blocks: they fit next to `a` until `a` needs its second block.
    b = Request(
        make_prompts(1, seed=13, min_len=50, max_len=50)[0], SamplingParams(5, ignore_eos=True)
    )
    run(engine, [a, b])

    assert sched.victims and all(0 < done < length for done, length in sched.victims)
    assert b.num_preemptions == 1
    for r in (a, b):
        assert r.output_ids == reference_generate(model, r.prompt_ids, r.params)
    assert kv.allocator.num_free == kv.allocator.num_blocks


def test_abort_mid_prefill_frees_memory(model: TinyGPT) -> None:
    kv = PagedKVCache(pool(4 * M.max_seq_len), block_size=8)
    sched = ContinuousScheduler(kv, max_batch_size=2, max_tokens_per_step=8, chunked_prefill=True)
    engine = Engine(model, sched)
    req = Request([5] * 40, SamplingParams(max_new_tokens=4))
    engine.add_request(req)
    assert engine.step() == []  # first chunk only, nothing sampled yet
    assert req.num_computed_tokens == 8
    engine.abort(req.request_id)
    assert req.finish_reason is FinishReason.ABORTED
    assert kv.allocator.num_free == kv.allocator.num_blocks
    assert not engine.has_unfinished()


def test_preempted_decodes_recompute_in_chunks(model: TinyGPT) -> None:
    """Requests preempted after generating tokens re-prefill prompt + outputs in chunks."""
    prompts = make_prompts(6, seed=4, min_len=20, max_len=30)
    params = SamplingParams(max_new_tokens=24, ignore_eos=True)
    kv = PagedKVCache(pool(128), block_size=8, watermark=0)
    sched = RecordingScheduler(kv, max_batch_size=6, max_tokens_per_step=8, chunked_prefill=True)
    engine = Engine(model, sched)
    for req, prompt in zip(engine.generate(prompts, params), prompts, strict=True):
        assert req.output_ids == reference_generate(model, prompt, params)
    assert any(length > 30 for _, length in sched.victims)  # victims that had outputs
