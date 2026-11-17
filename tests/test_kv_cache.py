"""Phase 5: the block allocator and the paged KV cache."""

from __future__ import annotations

import pytest
import torch
from conftest import SMALL_MODEL, make_prompts, reference_generate

from tinyserve.engine import Engine, EngineConfig, build_engine
from tinyserve.kv_cache import (
    BlockAllocator,
    ContiguousKVCache,
    KVPool,
    OutOfBlocksError,
    PagedKVCache,
)
from tinyserve.model import TinyGPT
from tinyserve.request import Request, SamplingParams
from tinyserve.scheduler import ContinuousScheduler

M = SMALL_MODEL


def pool(num_slots: int) -> KVPool:
    return KVPool(num_slots, M.n_layers, M.n_kv_heads, M.head_dim)


# --- BlockAllocator ------------------------------------------------------------------


def test_allocator_hands_out_distinct_blocks() -> None:
    alloc = BlockAllocator(4)
    blocks = [alloc.allocate() for _ in range(4)]
    assert sorted(blocks) == [0, 1, 2, 3]
    assert alloc.num_free == 0
    with pytest.raises(OutOfBlocksError):
        alloc.allocate()


def test_allocator_reuses_freed_blocks() -> None:
    alloc = BlockAllocator(3)
    a, b, _ = alloc.allocate(), alloc.allocate(), alloc.allocate()
    alloc.free(b)
    alloc.free(a)
    assert alloc.num_free == 2
    assert alloc.allocate() == a  # last freed, first reused
    assert alloc.allocate() == b


def test_allocator_rejects_double_free() -> None:
    alloc = BlockAllocator(2)
    blk = alloc.allocate()
    alloc.free(blk)
    with pytest.raises(ValueError):
        alloc.free(blk)


# --- PagedKVCache --------------------------------------------------------------------


def test_paged_allocates_blocks_on_demand() -> None:
    kv = PagedKVCache(pool(64), block_size=4, watermark=0)
    assert kv.allocate(seq_id=0, num_tokens=5)  # needs 2 blocks
    assert len(kv.block_tables[0]) == 2
    assert kv.grow(0, 8)  # still fits in 2 blocks
    assert len(kv.block_tables[0]) == 2
    assert kv.grow(0, 9)  # crosses into a third block
    assert len(kv.block_tables[0]) == 3
    usage = kv.usage()
    assert (usage.used_tokens, usage.reserved_tokens, usage.capacity_tokens) == (9, 12, 64)


def test_paged_free_returns_blocks_for_reuse() -> None:
    kv = PagedKVCache(pool(16), block_size=4, watermark=0)
    assert kv.allocate(0, 16)
    assert not kv.can_allocate(1)
    first_blocks = set(kv.block_tables[0])
    kv.free(0)
    assert kv.allocator.num_free == 4
    assert kv.allocate(1, 16)
    assert set(kv.block_tables[1]) == first_blocks


def test_paged_grow_fails_cleanly_when_full() -> None:
    kv = PagedKVCache(pool(8), block_size=4, watermark=0)
    assert kv.allocate(0, 4)
    assert kv.allocate(1, 4)
    assert not kv.grow(0, 5)
    assert kv.usage().used_tokens == 8  # failed grow changed nothing


def test_paged_addresses_follow_block_table() -> None:
    kv = PagedKVCache(pool(32), block_size=4, watermark=0)
    kv.allocate(0, 4)
    kv.allocate(1, 4)
    kv.grow(0, 8)  # seq 0 now owns blocks [b0, b2]: not contiguous
    b0, b2 = kv.block_tables[0]
    addrs = kv.addresses([0], torch.arange(8)[None])[0].tolist()
    assert addrs == [b0 * 4 + i for i in range(4)] + [b2 * 4 + i for i in range(4)]


def test_paged_wastes_far_less_than_contiguous() -> None:
    """Same sequences: contiguous pins max_seq_len each; paged only rounds up to a block."""
    lengths = [10, 37, 64, 5]
    contiguous = ContiguousKVCache(pool(4 * M.max_seq_len), M.max_seq_len)
    paged = PagedKVCache(pool(4 * M.max_seq_len), block_size=16, watermark=0)
    for i, n in enumerate(lengths):
        assert contiguous.allocate(i, n)
        assert paged.allocate(i, n)
    c, p = contiguous.usage(), paged.usage()
    assert c.used_tokens == p.used_tokens == sum(lengths)
    assert c.reserved_tokens == 4 * M.max_seq_len
    assert p.reserved_tokens == 16 * sum(-(-n // 16) for n in lengths)
    assert p.utilization > 0.8 > 0.4 > c.utilization


@torch.inference_mode()
def test_paged_and_contiguous_give_identical_logits(model: TinyGPT) -> None:
    ids = torch.randint(4, M.vocab_size, (1, 40), generator=torch.Generator().manual_seed(0))
    outs = []
    for kv in (
        ContiguousKVCache(pool(2 * M.max_seq_len), M.max_seq_len),
        PagedKVCache(pool(2 * M.max_seq_len), block_size=8, watermark=0),
    ):
        kv.allocate(99, 1)  # a neighbour, so seq 0's blocks are not at the start
        kv.allocate(0, 25)
        pos = torch.arange(25)[None]
        logits = [model(ids[:, :25], pos, kv.build_batch([0], pos, torch.ones_like(pos) > 0))]
        for j in range(25, 40):
            kv.grow(0, j + 1)
            p = torch.tensor([[j]])
            logits.append(model(ids[:, j : j + 1], p, kv.build_batch([0], p, p >= 0)))
        outs.append(torch.cat(logits, dim=1))
    torch.testing.assert_close(outs[0], outs[1], atol=1e-5, rtol=0)
    torch.testing.assert_close(outs[0], model(ids), atol=1e-5, rtol=0)


# --- engine integration --------------------------------------------------------------


def test_continuous_paged_matches_reference(model: TinyGPT) -> None:
    prompts = make_prompts(8, seed=3)
    params = SamplingParams(max_new_tokens=10, ignore_eos=True)
    kv = PagedKVCache(pool(4 * M.max_seq_len), block_size=8)
    engine = Engine(model, ContinuousScheduler(kv, max_batch_size=8))
    for req, prompt in zip(engine.generate(prompts, params), prompts, strict=True):
        assert req.output_ids == reference_generate(model, prompt, params)
    assert kv.allocator.num_free == kv.allocator.num_blocks  # everything returned


def test_preemption_keeps_outputs_correct(model: TinyGPT) -> None:
    """A pool too small for everyone forces preemptions; outputs must not change."""
    prompts = make_prompts(6, seed=4, min_len=20, max_len=30)
    params = SamplingParams(max_new_tokens=24, ignore_eos=True)
    kv = PagedKVCache(pool(128), block_size=8, watermark=0)  # 16 blocks for ~6 x 6 needed
    sched = ContinuousScheduler(kv, max_batch_size=6, max_tokens_per_step=128)
    engine = Engine(model, sched)
    for req, prompt in zip(engine.generate(prompts, params), prompts, strict=True):
        assert req.output_ids == reference_generate(model, prompt, params)
    assert sched.num_preemptions > 0


def test_paged_fits_more_requests_in_same_memory() -> None:
    """With an identical pool, paged runs a bigger batch than contiguous."""
    batch = {}
    for layout in ("contiguous", "paged"):
        cfg = EngineConfig(
            strategy="continuous",
            model=M,
            kv_capacity_tokens=4 * M.max_seq_len,
            max_batch_size=16,
            kv_layout=layout,
        )
        engine = build_engine(cfg, record_trace=True)
        for p in make_prompts(16, seed=5):
            engine.add_request(Request(p, SamplingParams(max_new_tokens=8, ignore_eos=True)))
        while engine.has_unfinished():
            engine.step()
        batch[layout] = max(t.num_running for t in engine.trace)
    assert batch["contiguous"] == 4
    assert batch["paged"] == 16


def test_paged_requires_continuous() -> None:
    with pytest.raises(ValueError):
        build_engine(EngineConfig(strategy="static", model=M, kv_layout="paged"))
