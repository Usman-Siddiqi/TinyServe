"""Automatic prefix caching: hashing, reference counts, LRU eviction and the engine."""

from __future__ import annotations

import pytest
import torch
from conftest import SMALL_MODEL, make_prompts, reference_generate

from tinyserve.engine import Engine, EngineConfig, build_engine
from tinyserve.kv_cache import (
    CachingBlockAllocator,
    KVPool,
    OutOfBlocksError,
    PagedKVCache,
    PrefixCachingKVCache,
    block_hash,
)
from tinyserve.model import TinyGPT
from tinyserve.request import Request, SamplingParams
from tinyserve.scheduler import ContinuousScheduler
from tinyserve.workload import WorkloadSpec, make_workload

M = SMALL_MODEL
GREEDY = SamplingParams(max_new_tokens=10, ignore_eos=True)


def pool(num_slots: int) -> KVPool:
    return KVPool(num_slots, M.n_layers, M.n_kv_heads, M.head_dim)


def shared_prefix_prompts(n: int, prefix_len: int = 40, seed: int = 0) -> list[list[int]]:
    """``n`` prompts that all start with the same ``prefix_len`` tokens."""
    g = torch.Generator().manual_seed(seed)
    prefix = torch.randint(4, M.vocab_size, (prefix_len,), generator=g).tolist()
    return [prefix + p for p in make_prompts(n, seed=seed + 1, min_len=3, max_len=10)]


def assert_all_memory_released(kv: PrefixCachingKVCache) -> None:
    assert kv.num_sequences == 0
    assert kv.allocator.num_free == kv.allocator.num_blocks
    assert set(kv.allocator.ref_counts) == {0}
    assert kv.allocator.num_shared_refs == 0
    assert kv.usage().reserved_tokens == 0


# --- hashing -------------------------------------------------------------------------


def test_block_hash_is_chained_over_the_prefix() -> None:
    a = block_hash(None, [1, 2, 3, 4])
    assert a == block_hash(None, (1, 2, 3, 4))  # deterministic, list or tuple
    assert block_hash(a, [5, 6, 7, 8]) != block_hash(None, [5, 6, 7, 8])
    assert block_hash(a, [5, 6, 7, 8]) != block_hash(block_hash(None, [9, 2, 3, 4]), [5, 6, 7, 8])


# --- CachingBlockAllocator ------------------------------------------------------------


def test_shared_block_is_freed_only_by_its_last_owner() -> None:
    alloc = CachingBlockAllocator(4)
    blk = alloc.allocate()
    alloc.register(blk, 123)
    alloc.acquire(blk)
    assert alloc.ref_counts[blk] == 2 and alloc.num_shared_refs == 1
    alloc.free(blk)
    assert alloc.num_free == 3  # still owned once
    alloc.free(blk)
    assert alloc.num_free == 4 and alloc.num_evictable == 1
    with pytest.raises(ValueError):
        alloc.free(blk)


def test_freed_cached_block_can_be_revived() -> None:
    alloc = CachingBlockAllocator(2)
    blk = alloc.allocate()
    alloc.register(blk, 7)
    alloc.free(blk)
    assert alloc.lookup(7) == blk
    alloc.acquire(blk)
    assert alloc.ref_counts[blk] == 1 and alloc.num_evictable == 0
    assert alloc.num_free == 1


def test_unregistered_blocks_go_back_to_the_clean_list() -> None:
    alloc = CachingBlockAllocator(2)
    blk = alloc.allocate()
    alloc.free(blk)
    assert alloc.num_evictable == 0 and alloc.num_free == 2


def test_first_registration_of_a_hash_wins() -> None:
    alloc = CachingBlockAllocator(2)
    a, b = alloc.allocate(), alloc.allocate()
    alloc.register(a, 5)
    alloc.register(b, 5)  # same content computed twice: b stays private
    assert alloc.lookup(5) == a
    alloc.free(b)
    assert alloc.num_evictable == 0


def test_eviction_prefers_clean_blocks_then_least_recently_freed() -> None:
    alloc = CachingBlockAllocator(4)
    blocks = [alloc.allocate() for _ in range(3)]
    for i, blk in enumerate(blocks):
        alloc.register(blk, 100 + i)
    for blk in blocks:  # freed in order: blocks[0] is the least recently used
        alloc.free(blk)
    clean = alloc.allocate()
    assert clean not in blocks and alloc.num_evictions == 0
    assert alloc.allocate() == blocks[0]
    assert alloc.lookup(100) is None  # its cache entry is gone
    assert alloc.allocate() == blocks[1]
    assert alloc.lookup(102) == blocks[2]
    assert alloc.num_evictions == 2
    alloc.allocate()
    with pytest.raises(OutOfBlocksError):
        alloc.allocate()


def test_reviving_a_block_refreshes_its_lru_position() -> None:
    alloc = CachingBlockAllocator(2)
    a, b = alloc.allocate(), alloc.allocate()
    alloc.register(a, 1)
    alloc.register(b, 2)
    alloc.free(a)
    alloc.free(b)
    alloc.acquire(a)  # a is used again, then freed last
    alloc.free(a)
    assert alloc.allocate() == b


# --- PrefixCachingKVCache ------------------------------------------------------------


def test_sequences_with_a_common_prefix_share_full_blocks() -> None:
    kv = PrefixCachingKVCache(pool(64), block_size=4, watermark=0)
    first = list(range(10, 20))  # 2 full blocks + 2 tokens
    assert kv.allocate(0, len(first), first)
    kv.cache_blocks(0, first)
    second = [*first[:8], 99, 98, 97]
    assert kv.match_prefix(second) == 8
    assert kv.allocate(1, len(second), second)
    assert kv.block_tables[1][:2] == kv.block_tables[0][:2]
    assert kv.block_tables[1][2] != kv.block_tables[0][2]  # partial blocks stay private
    usage = kv.usage()
    assert usage.used_tokens == 10 + 11 - 8  # shared slots count once
    assert usage.reserved_tokens == 4 * 4
    stats = kv.prefix_cache_stats()
    assert (stats.query_tokens, stats.hit_tokens) == (21, 8)


def test_match_leaves_at_least_one_token_to_compute() -> None:
    kv = PrefixCachingKVCache(pool(64), block_size=4, watermark=0)
    ids = list(range(10, 18))  # exactly two full blocks
    kv.allocate(0, len(ids), ids)
    kv.cache_blocks(0, ids)
    assert kv.match_prefix(ids) == 4
    assert kv.match_prefix([*ids, 1]) == 8


def test_match_stops_at_the_first_miss() -> None:
    kv = PrefixCachingKVCache(pool(64), block_size=4, watermark=0)
    ids = list(range(10, 22))
    kv.allocate(0, len(ids), ids)
    kv.cache_blocks(0, ids)
    changed = [*ids[:4], 0, *ids[5:], 1]  # block 1 differs, so block 2 cannot match
    assert kv.match_prefix(changed) == 4


def test_only_computed_blocks_are_cached() -> None:
    kv = PrefixCachingKVCache(pool(64), block_size=4, watermark=0)
    ids = list(range(10, 22))
    kv.allocate(0, len(ids), ids)
    kv.cache_blocks(0, ids[:5])  # only the first block's KV exists yet
    assert kv.match_prefix([*ids, 1]) == 4


def test_freed_blocks_stay_cached_until_evicted() -> None:
    kv = PrefixCachingKVCache(pool(32), block_size=4, watermark=0)  # 8 blocks
    ids = list(range(10, 22))
    kv.allocate(0, len(ids), ids)
    kv.cache_blocks(0, ids)
    kv.free(0)
    assert kv.usage().cached_tokens == 12 and kv.allocator.num_free == 8
    assert kv.match_prefix([*ids, 1]) == 12
    assert kv.allocate(1, 24)  # 5 clean blocks are not enough, so evict the tail
    assert kv.match_prefix([*ids, 1]) == 8
    assert kv.prefix_cache_stats().evicted_blocks == 1


def test_can_allocate_counts_live_hits_as_free() -> None:
    kv = PrefixCachingKVCache(pool(16), block_size=4, watermark=0)  # 4 blocks
    ids = list(range(10, 22))
    kv.allocate(0, 12, ids)
    kv.cache_blocks(0, ids)
    other = [*ids[:8], 1, 2]  # shares 2 live blocks, needs 1 more: fits
    assert kv.can_allocate(len(other), other)
    assert not kv.can_allocate(len(other))
    kv.free(0)
    # Now the hits are evictable, so acquiring them uses free space: 3 of 4 blocks.
    assert kv.can_allocate(len(other), other)
    assert kv.allocate(1, len(other), other)
    assert kv.allocator.num_free == 1
    assert kv.can_allocate(9, [*ids[:8], 5])
    assert not kv.can_allocate(9)


def test_reallocated_sequences_release_everything() -> None:
    kv = PrefixCachingKVCache(pool(64), block_size=4, watermark=0)
    ids = list(range(10, 30))
    for seq in range(3):
        assert kv.allocate(seq, len(ids), ids)
        kv.cache_blocks(seq, ids)
    for seq in range(3):
        kv.free(seq)
    assert_all_memory_released(kv)


# --- engine integration --------------------------------------------------------------


def run_engine(
    model: TinyGPT, kv: PagedKVCache, prompts: list[list[int]], max_batch_size: int = 2
) -> tuple[Engine, list[Request]]:
    sched = ContinuousScheduler(kv, max_batch_size=max_batch_size, max_tokens_per_step=256)
    engine = Engine(model, sched, record_trace=True)
    return engine, engine.generate(prompts, GREEDY)


def test_cached_outputs_match_reference_and_skip_prefill(model: TinyGPT) -> None:
    prompts = shared_prefix_prompts(6)
    plain_kv = PagedKVCache(pool(4 * M.max_seq_len), block_size=8)
    cached_kv = PrefixCachingKVCache(pool(4 * M.max_seq_len), block_size=8)
    plain, plain_reqs = run_engine(model, plain_kv, prompts)
    cached, cached_reqs = run_engine(model, cached_kv, prompts)

    for a, b, prompt in zip(plain_reqs, cached_reqs, prompts, strict=True):
        expected = reference_generate(model, prompt, GREEDY)
        assert a.output_ids == b.output_ids == expected
    # The first batch of two computes the 40-token prefix; the rest reuse 5 blocks of it.
    assert [r.num_cached_tokens for r in cached_reqs] == [0, 0, 40, 40, 40, 40]
    stats = cached_kv.prefix_cache_stats()
    assert stats.hit_tokens == 160
    assert stats.hit_rate == pytest.approx(160 / sum(len(p) for p in prompts))

    def prefilled(e: Engine) -> int:
        return sum(t.num_prefill_tokens for t in e.trace)

    assert prefilled(plain) - prefilled(cached) == 160
    assert_all_memory_released(cached_kv)


def test_multi_turn_conversation_reuses_generated_tokens(model: TinyGPT) -> None:
    """A follow-up turn's prompt contains the previous answer, which is cached too."""
    kv = PrefixCachingKVCache(pool(4 * M.max_seq_len), block_size=8)
    engine = Engine(model, ContinuousScheduler(kv, max_batch_size=4))
    first_prompt = shared_prefix_prompts(1, prefix_len=30)[0]
    (turn1,) = engine.generate([first_prompt], GREEDY)
    follow_up = [*turn1.all_ids, 5, 6, 7]
    (turn2,) = engine.generate([follow_up], GREEDY)
    computed = len(turn1.all_ids) - 1  # the last answer token never had its KV computed
    assert turn2.num_cached_tokens == computed // 8 * 8
    assert turn2.output_ids == reference_generate(model, follow_up, GREEDY)


def test_preemption_with_shared_blocks_keeps_outputs_correct(model: TinyGPT) -> None:
    prompts = shared_prefix_prompts(6, prefix_len=24)
    params = SamplingParams(max_new_tokens=24, ignore_eos=True)
    kv = PrefixCachingKVCache(pool(128), block_size=8, watermark=0)  # 16 blocks
    sched = ContinuousScheduler(kv, max_batch_size=6, max_tokens_per_step=128)
    engine = Engine(model, sched)
    for req, prompt in zip(engine.generate(prompts, params), prompts, strict=True):
        assert req.output_ids == reference_generate(model, prompt, params)
    assert sched.num_preemptions > 0
    assert kv.prefix_cache_stats().hit_tokens > 0
    assert_all_memory_released(kv)


def test_abort_releases_shared_blocks(model: TinyGPT) -> None:
    kv = PrefixCachingKVCache(pool(4 * M.max_seq_len), block_size=8)
    engine = Engine(model, ContinuousScheduler(kv, max_batch_size=4))
    prompts = shared_prefix_prompts(3)
    engine.generate(prompts[:1], GREEDY)  # warm the cache
    reqs = [Request(p, SamplingParams(max_new_tokens=50, ignore_eos=True)) for p in prompts[1:]]
    for r in reqs:
        engine.add_request(r)
    engine.step()
    engine.step()
    assert kv.allocator.num_shared_refs > 0
    for r in reqs:
        engine.abort(r.request_id)
    assert_all_memory_released(kv)


def test_build_engine_prefix_caching_option() -> None:
    cfg = EngineConfig(
        strategy="continuous", model=M, kv_layout="paged", enable_prefix_caching=True
    )
    assert isinstance(build_engine(cfg).kv, PrefixCachingKVCache)
    with pytest.raises(ValueError):
        build_engine(EngineConfig(strategy="continuous", model=M, enable_prefix_caching=True))


def test_shared_prefix_workload() -> None:
    items = make_workload(WorkloadSpec.shared_prefix(num_system_prompts=3, num_requests=40))
    prefixes = {it.prompt[:200] for it in items}
    assert len(prefixes) == 3
    assert len({it.prompt for it in items}) == 40
    assert all(len(it.prompt) + 1 + it.max_tokens <= 512 for it in items)


def test_prefix_caching_with_chunked_prefill(model: TinyGPT) -> None:
    """Cached prefixes and 16-token prefill chunks together, under memory pressure."""
    prompts = shared_prefix_prompts(6, prefix_len=24)
    params = SamplingParams(max_new_tokens=24, ignore_eos=True)
    kv = PrefixCachingKVCache(pool(160), block_size=8, watermark=0)
    sched = ContinuousScheduler(kv, max_batch_size=6, max_tokens_per_step=16, chunked_prefill=True)
    engine = Engine(model, sched)
    for req, prompt in zip(engine.generate(prompts, params), prompts, strict=True):
        assert req.output_ids == reference_generate(model, prompt, params)
    assert kv.prefix_cache_stats().hit_tokens > 0
    assert_all_memory_released(kv)
