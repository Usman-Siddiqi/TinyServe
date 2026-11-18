"""Speculative decoding: greedy output is unchanged, sampling keeps the target distribution."""

from __future__ import annotations

import pytest
import torch
from conftest import SMALL_MODEL, make_prompts, reference_generate

from tinyserve.draft import early_exit_draft, load_distilled_draft
from tinyserve.engine import Engine, EngineConfig, build_engine
from tinyserve.kv_cache import KVPool, PagedKVCache
from tinyserve.model import ModelConfig, TinyGPT
from tinyserve.request import Request, SamplingParams
from tinyserve.scheduler import ContinuousScheduler
from tinyserve.speculative import (
    Draft,
    DraftLengthController,
    DraftModelProposer,
    NgramProposer,
    SpecStats,
    SpeculativeDecoder,
    accept_greedy,
    accept_sampled,
)


def paged_kv(model: TinyGPT, num_slots: int = 2048) -> PagedKVCache:
    m = model.cfg
    return PagedKVCache(KVPool(num_slots, m.n_layers, m.n_kv_heads, m.head_dim), block_size=8)


def draft_proposer(model: TinyGPT) -> DraftModelProposer:
    draft = early_exit_draft(model, n_layers=1)
    return DraftModelProposer(draft, paged_kv(draft))


def run(engine: Engine, prompts: list[list[int]], params: list[SamplingParams]) -> list[Request]:
    reqs = [Request(p, sp) for p, sp in zip(prompts, params, strict=True)]
    for r in reqs:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()
    return reqs


def test_truncate_releases_trailing_blocks(model: TinyGPT) -> None:
    kv = paged_kv(model)
    kv.allocate(0, 20)
    assert len(kv.block_tables[0]) == 3
    kv.truncate(0, 9)
    assert len(kv.block_tables[0]) == 2 and kv.usage().used_tokens == 9
    with pytest.raises(ValueError):
        kv.truncate(0, 10)


def test_accept_greedy_keeps_matching_prefix() -> None:
    logits = torch.zeros(4, 5)
    logits[torch.arange(4), torch.tensor([1, 2, 3, 4])] = 1.0
    assert accept_greedy([1, 2, 0], logits) == [1, 2, 3]
    assert accept_greedy([1, 2, 3], logits) == [1, 2, 3, 4]
    assert accept_greedy([0, 2, 3], logits) == [1]


@pytest.mark.parametrize("deterministic_draft", [True, False])
def test_rejection_sampling_preserves_target_distribution(deterministic_draft: bool) -> None:
    """The first emitted token must follow p regardless of the draft distribution q."""
    p_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]])
    p = torch.softmax(p_logits[0], -1)
    q = torch.tensor([0.1, 0.2, 0.3, 0.4])
    gen = torch.Generator().manual_seed(0)
    counts = torch.zeros(4)
    trials = 4000
    for _ in range(trials):
        if deterministic_draft:
            draft = Draft([2])
        else:
            draft = Draft([int(torch.multinomial(q, 1, generator=gen))], q[None])
        counts[accept_sampled(draft, p_logits, 1.0, gen)[0]] += 1
    assert torch.allclose(counts / trials, p, atol=0.03)


@pytest.mark.parametrize("proposer", ["ngram", "draft"])
def test_greedy_speculation_matches_plain_decoding(model: TinyGPT, proposer: str) -> None:
    prompts = make_prompts(6, seed=3)
    # Repetitive prompts give the n-gram proposer something to find.
    prompts[0] = [5, 6, 7, 8] * 6
    params = [SamplingParams(max_new_tokens=n, ignore_eos=True) for n in (1, 2, 9, 17, 6, 12)]
    spec = SpeculativeDecoder(NgramProposer() if proposer == "ngram" else draft_proposer(model), 3)
    kv = paged_kv(model)
    engine = Engine(model, ContinuousScheduler(kv, max_batch_size=4), speculator=spec)
    for r in run(engine, prompts, params):
        assert r.output_ids == reference_generate(model, r.prompt_ids, r.params)
    assert spec.stats.num_accepted > 0
    assert kv.num_sequences == 0
    if isinstance(spec.proposer, DraftModelProposer):
        assert spec.proposer.kv.num_sequences == 0


def test_greedy_speculation_survives_preemption(model: TinyGPT) -> None:
    """Room for only ~2 sequences: requests get preempted and recomputed mid-speculation."""
    prompts = make_prompts(5, seed=4, min_len=10, max_len=20)
    params = [SamplingParams(max_new_tokens=14, ignore_eos=True)] * 5
    sched = ContinuousScheduler(paged_kv(model, num_slots=72), max_batch_size=5)
    engine = Engine(model, sched, speculator=SpeculativeDecoder(draft_proposer(model), 4))
    for r in run(engine, prompts, params):
        assert r.output_ids == reference_generate(model, r.prompt_ids, r.params)
    assert sched.num_preemptions > 0


def test_sampled_speculation_runs_to_length(model: TinyGPT) -> None:
    prompts = make_prompts(4, seed=5)
    params = [
        SamplingParams(max_new_tokens=15, temperature=0.5, ignore_eos=True, seed=i)
        for i in range(4)
    ]
    spec = SpeculativeDecoder(draft_proposer(model), 4)
    kv = paged_kv(model)
    engine = Engine(model, ContinuousScheduler(kv, max_batch_size=4), speculator=spec)
    reqs = run(engine, prompts, params)
    assert all(len(r.output_ids) == 15 for r in reqs)
    assert all(0 <= t < SMALL_MODEL.vocab_size for r in reqs for t in r.output_ids)
    assert kv.num_sequences == 0 and spec.stats.num_verifications > 0


def test_spec_stats() -> None:
    stats = SpecStats()
    stats.record(4, 2)
    stats.record(2, 2)
    assert stats.acceptance_rate == pytest.approx(4 / 6)
    assert stats.tokens_per_verification == pytest.approx(3.0)
    assert stats.acceptance_by_position() == [1.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize("method", ["ngram", "early-exit"])
def test_build_engine_with_speculation(model: TinyGPT, method: str) -> None:
    cfg = EngineConfig(
        strategy="continuous",
        kv_layout="paged",
        model=SMALL_MODEL,
        kv_capacity_tokens=1024,
        speculative=method,
    )
    engine = build_engine(cfg)
    assert engine.speculator is not None
    prompts = make_prompts(3, seed=6)
    for r in engine.generate(prompts, SamplingParams(max_new_tokens=10, ignore_eos=True)):
        assert r.output_ids == reference_generate(model, r.prompt_ids, r.params)


def test_distilled_draft_loads() -> None:
    draft = load_distilled_draft(ModelConfig())
    assert draft.cfg.n_layers == 1 and draft.cfg.vocab_size == ModelConfig().vocab_size


def test_draft_length_controller_backs_off_and_probes() -> None:
    ctrl = DraftLengthController(draft_cost=0.2, probe_every=3)
    req = Request([5, 6], SamplingParams(max_new_tokens=50))
    for _ in range(20):
        ctrl.update(req.request_id, drafted=4, accepted=4)
    assert ctrl.choose([req], [4]) == [4]
    for _ in range(20):
        ctrl.update(req.request_id, drafted=4, accepted=0)
    assert [ctrl.choose([req], [4]) for _ in range(3)] == [[0], [0], [1]]
