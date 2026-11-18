"""The fused C++ paged-attention kernel must match the PyTorch attention path."""

from __future__ import annotations

import pytest
import torch
from conftest import SMALL_MODEL, make_prompts, reference_generate

from tinyserve import attention
from tinyserve.attention import paged_attention_decode, reference_paged_attention
from tinyserve.engine import Engine, EngineConfig, build_engine
from tinyserve.kv_cache import ContiguousKVCache, KVCacheManager, KVPool, PagedKVCache
from tinyserve.model import TinyGPT
from tinyserve.request import SamplingParams
from tinyserve.runner import ModelRunner
from tinyserve.scheduler import ContinuousScheduler

requires_cpp = pytest.mark.skipif(
    not attention.cpp_available(), reason="C++ attention extension could not be built"
)
ATOL = 1e-5


def random_paged_inputs(
    lengths: list[int], n_heads: int, n_kv_heads: int, head_dim: int, block_size: int, seed: int
) -> tuple[torch.Tensor, ...]:
    """Random queries and a pool whose blocks are scattered in random order."""
    g = torch.Generator().manual_seed(seed)
    width = -(-max(lengths) // block_size)
    num_blocks = len(lengths) * width + 3
    k = torch.randn(num_blocks * block_size, n_kv_heads, head_dim, generator=g)
    v = torch.randn(num_blocks * block_size, n_kv_heads, head_dim, generator=g)
    tables = torch.randperm(num_blocks, generator=g)[: len(lengths) * width].view(-1, width)
    q = torch.randn(len(lengths), n_heads, head_dim, generator=g)
    return q, k, v, tables, torch.tensor(lengths)


@requires_cpp
@pytest.mark.parametrize(
    ("n_heads", "n_kv_heads", "head_dim"), [(8, 2, 64), (8, 8, 64), (4, 1, 32), (6, 3, 40)]
)
@pytest.mark.parametrize("block_size", [1, 16, 512])
def test_kernel_matches_reference(
    n_heads: int, n_kv_heads: int, head_dim: int, block_size: int
) -> None:
    lengths = [1, 2, 31, 32, 33, 150, 400, 7]  # ragged, around the kernel's 32-token tile
    q, k, v, tables, lens = random_paged_inputs(
        lengths, n_heads, n_kv_heads, head_dim, block_size, seed=block_size
    )
    out = paged_attention_decode(q, k, v, tables, lens, block_size)
    expected = reference_paged_attention(q, k, v, tables, lens, block_size)
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=0)


@requires_cpp
def test_kernel_is_stable_for_large_scores() -> None:
    q, k, v, tables, lens = random_paged_inputs([300, 5], 8, 2, 64, 16, seed=1)
    q *= 50  # softmax over scores in the hundreds must not overflow
    out = paged_attention_decode(q, k, v, tables, lens, 16)
    expected = reference_paged_attention(q, k, v, tables, lens, 16)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-5)


@requires_cpp
def test_kernel_rejects_bad_inputs() -> None:
    q, k, v, tables, lens = random_paged_inputs([10], 8, 2, 64, 16, seed=0)
    with pytest.raises(RuntimeError, match="block table"):
        paged_attention_decode(q, k, v, tables, torch.tensor([10_000]), 16)
    with pytest.raises(RuntimeError, match="float32"):
        paged_attention_decode(q.double(), k, v, tables, lens, 16)


def make_kv(layout: str, rows: int = 8) -> KVCacheManager:
    m = SMALL_MODEL
    pool = KVPool(rows * m.max_seq_len, m.n_layers, m.n_kv_heads, m.head_dim)
    if layout == "paged":
        return PagedKVCache(pool, block_size=8, watermark=0)
    return ContiguousKVCache(pool, m.max_seq_len)


@requires_cpp
@pytest.mark.parametrize("layout", ["contiguous", "paged"])
@torch.inference_mode()
def test_decode_step_matches_torch_path(model: TinyGPT, layout: str) -> None:
    """One decode step over a ragged batch gives the same hidden states either way."""
    kv = make_kv(layout)
    lengths = [3, 60, 17, 1, 100, 33]
    g = torch.Generator().manual_seed(0)
    for seq_id, n in enumerate(lengths):
        kv.allocate(seq_id, n + 1)
        ids = torch.randint(4, SMALL_MODEL.vocab_size, (1, n), generator=g)
        positions = torch.arange(n)[None]
        batch = kv.build_batch([seq_id], positions, torch.ones_like(ids, dtype=torch.bool))
        model.hidden_states(ids, positions, batch)

    seq_ids = list(range(len(lengths)))
    ids = torch.randint(4, SMALL_MODEL.vocab_size, (len(lengths), 1), generator=g)
    positions = torch.tensor(lengths)[:, None]
    valid = torch.ones_like(ids, dtype=torch.bool)
    hidden = {}
    for fused in (False, True):
        batch = kv.build_batch(seq_ids, positions, valid, paged_attention=fused)
        hidden[fused] = model.hidden_states(ids, positions, batch)
    torch.testing.assert_close(hidden[True], hidden[False], atol=ATOL, rtol=0)


@requires_cpp
@pytest.mark.parametrize("layout", ["contiguous", "paged"])
def test_greedy_outputs_identical(layout: str) -> None:
    prompts = make_prompts(10, seed=3)
    params = SamplingParams(max_new_tokens=20, ignore_eos=True)
    outputs = {}
    for backend in ("torch", "cpp"):
        cfg = EngineConfig(
            strategy="continuous",
            model=SMALL_MODEL,
            kv_capacity_tokens=8 * SMALL_MODEL.max_seq_len,
            kv_layout=layout,
            attention=backend,
        )
        engine = build_engine(cfg)
        assert engine.runner.attention == backend
        outputs[backend] = [r.output_ids for r in engine.generate(prompts, params)]
    assert outputs["cpp"] == outputs["torch"]


@requires_cpp
def test_cpp_backend_survives_preemption(model: TinyGPT) -> None:
    prompts = make_prompts(6, seed=4, min_len=20, max_len=30)
    params = SamplingParams(max_new_tokens=24, ignore_eos=True)
    pool = KVPool(128, SMALL_MODEL.n_layers, SMALL_MODEL.n_kv_heads, SMALL_MODEL.head_dim)
    sched = ContinuousScheduler(PagedKVCache(pool, block_size=8, watermark=0), 6, 128)
    engine = Engine(model, sched, attention="cpp")
    for req, prompt in zip(engine.generate(prompts, params), prompts, strict=True):
        assert req.output_ids == reference_generate(model, prompt, params)
    assert sched.num_preemptions > 0


@requires_cpp
@torch.inference_mode()
def test_fused_batch_rejects_prefill(model: TinyGPT) -> None:
    kv = make_kv("paged")
    kv.allocate(0, 4)
    positions = torch.arange(4)[None]
    batch = kv.build_batch([0], positions, torch.ones(1, 4, dtype=torch.bool), paged_attention=True)
    with pytest.raises(ValueError, match="one token"):
        model.hidden_states(torch.zeros(1, 4, dtype=torch.long), positions, batch)


def test_backend_selection(monkeypatch: pytest.MonkeyPatch, model: TinyGPT) -> None:
    monkeypatch.delenv(attention.ENV_VAR, raising=False)
    assert attention.resolve_backend() == "torch"
    with pytest.raises(ValueError):
        attention.resolve_backend("cuda")
    # Without a working compiler, asking for cpp quietly falls back to torch.
    monkeypatch.setattr(attention, "cpp_available", lambda: False)
    monkeypatch.setenv(attention.ENV_VAR, "cpp")
    assert attention.resolve_backend() == "torch"
    assert ModelRunner(model, make_kv("paged")).attention == "torch"


@requires_cpp
def test_env_var_selects_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(attention.ENV_VAR, "cpp")
    assert attention.resolve_backend() == "cpp"
    assert attention.resolve_backend("torch") == "torch"  # explicit choice wins
