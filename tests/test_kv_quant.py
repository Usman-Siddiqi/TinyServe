"""Quantized KV cache: round-trip error, pool storage, capacity math, engine integration."""

from __future__ import annotations

import pytest
import torch
from conftest import SMALL_MODEL, make_prompts, reference_generate

from tinyserve.engine import EngineConfig, build_engine
from tinyserve.kv_cache import ContiguousKVCache, KVPool, PagedKVCache
from tinyserve.kv_quant import (
    KV_DTYPES,
    calibrate_key_smoothing,
    dequantize_int8,
    kv_format,
    quantize_int8,
)
from tinyserve.model import TinyGPT
from tinyserve.request import SamplingParams

M = SMALL_MODEL


def pool(num_slots: int, kv_dtype: str) -> KVPool:
    return KVPool(num_slots, M.n_layers, M.n_kv_heads, M.head_dim, kv_dtype)


def test_int8_round_trip_error_is_within_half_a_step() -> None:
    x = torch.randn(200, 2, 64, generator=torch.Generator().manual_seed(0)) * 3
    q, scale = quantize_int8(x)
    assert q.dtype == torch.int8 and scale.shape == (200, 2)
    err = (dequantize_int8(q, scale) - x).abs()
    assert (err <= scale[..., None] / 2 + 1e-6).all()
    # The absmax element of every vector is represented exactly (as +-127).
    assert (q.abs().amax(-1) == 127).all()


def test_int8_handles_all_zero_vectors() -> None:
    q, scale = quantize_int8(torch.zeros(3, 64))
    assert torch.equal(dequantize_int8(q, scale), torch.zeros(3, 64))


@pytest.mark.parametrize(("kv_dtype", "rel_tol"), [("float32", 0), ("bfloat16", 2**-8)])
def test_float_pool_round_trip(kv_dtype: str, rel_tol: float) -> None:
    p = pool(32, kv_dtype)
    k, v = torch.randn(5, M.n_kv_heads, M.head_dim), torch.randn(5, M.n_kv_heads, M.head_dim)
    slots = torch.tensor([7, 3, 30, 0, 12])
    p.write(1, slots, k, v)
    got_k, got_v = p.read(1, slots[None])
    assert got_k.shape == (1, 5, M.n_kv_heads, M.head_dim)
    assert got_k.dtype == kv_format(kv_dtype).compute
    torch.testing.assert_close(got_k[0].float(), k, atol=0, rtol=rel_tol)
    torch.testing.assert_close(got_v[0].float(), v, atol=0, rtol=rel_tol)


def test_int8_pool_round_trip() -> None:
    p = pool(32, "int8")
    k = torch.randn(4, M.n_kv_heads, M.head_dim)
    slots = torch.tensor([9, 2, 31, 5])
    p.write(0, slots, k, -k)
    got_k, got_v = p.read(0, slots.view(2, 2))
    step = k.abs().amax(-1, keepdim=True) / 127
    # Half a quantization step, plus bf16 rounding of the scale and of the result.
    bound = step / 2 + step * 127 * 2**-8 + k.abs() * 2**-8 + 1e-6
    assert ((got_k.reshape(4, *k.shape[1:]).float() - k).abs() <= bound).all()
    assert ((got_v.reshape(4, *k.shape[1:]).float() + k).abs() <= bound).all()


def test_bytes_per_token_and_capacity_from_a_byte_budget() -> None:
    per_head = {"float32": 64 * 4, "bfloat16": 64 * 2, "int8": 64 + 4}
    for kv_dtype, nbytes in per_head.items():
        expected = 2 * M.n_layers * M.n_kv_heads * nbytes
        assert M.kv_bytes_per_token(kv_dtype) == expected
        assert pool(10, kv_dtype).nbytes == 10 * expected
    budget = 1024 * M.kv_bytes_per_token("float32")
    slots = {
        d: EngineConfig(model=M, kv_dtype=d, kv_memory_bytes=budget).num_kv_slots for d in KV_DTYPES
    }
    assert slots["float32"] == 1024
    assert slots["bfloat16"] == 2048
    assert slots["int8"] == 1024 * 256 // 68  # 3.76x: scales cost 4 bytes per head
    # Without a byte budget the token count is used as before.
    assert EngineConfig(model=M, kv_dtype="int8", kv_capacity_tokens=777).num_kv_slots == 777


def test_unknown_kv_dtype_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_engine(EngineConfig(model=M, kv_dtype="float8"))
    with pytest.raises(ValueError):
        KVPool(8, 1, 1, 4, "fp4")


@torch.inference_mode()
@pytest.mark.parametrize(("kv_dtype", "atol"), [("bfloat16", 1e-2), ("int8", 3e-2)])
def test_quantized_cache_logits_stay_close(model: TinyGPT, kv_dtype: str, atol: float) -> None:
    """Incremental decoding through a low-precision cache tracks the float32 logits.

    Logits have a std of about 0.47 here; measured max errors are ~3.5e-3 (bf16), ~1.1e-2 (int8).
    """
    ids = torch.randint(4, M.vocab_size, (1, 60), generator=torch.Generator().manual_seed(1))
    kv = PagedKVCache(pool(2 * M.max_seq_len, kv_dtype), block_size=8, watermark=0)
    kv.allocate(0, 60)
    pos = torch.arange(30)[None]
    logits = [model(ids[:, :30], pos, kv.build_batch([0], pos, pos >= 0))]
    for j in range(30, 60):
        p = torch.tensor([[j]])
        logits.append(model(ids[:, j : j + 1], p, kv.build_batch([0], p, p >= 0)))
    torch.testing.assert_close(torch.cat(logits, dim=1), model(ids), atol=atol, rtol=0)


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
def test_engine_runs_with_each_kv_dtype(model: TinyGPT, kv_dtype: str) -> None:
    cfg = EngineConfig(
        strategy="continuous",
        model=M,
        kv_layout="paged",
        kv_dtype=kv_dtype,
        kv_memory_bytes=4 * M.max_seq_len * M.kv_bytes_per_token("float32"),
        max_batch_size=8,
    )
    engine = build_engine(cfg)
    assert engine.kv.pool.kv_dtype == kv_dtype
    prompts = make_prompts(8, seed=6)
    params = SamplingParams(max_new_tokens=12, ignore_eos=True)
    requests = engine.generate(prompts, params)
    assert all(len(r.output_ids) == 12 for r in requests)
    assert engine.kv.usage().used_tokens == 0
    if kv_dtype == "float32":
        for req, prompt in zip(requests, prompts, strict=True):
            assert req.output_ids == reference_generate(model, prompt, params)


@torch.inference_mode()
def test_key_smoothing_is_exact_in_float32(model: TinyGPT) -> None:
    """Dividing keys by the smoothing and multiplying queries by it leaves attention unchanged."""
    ids = torch.randint(4, M.vocab_size, (2, 40), generator=torch.Generator().manual_seed(2))
    smoothing = calibrate_key_smoothing(model, ids)
    assert smoothing.shape == (M.n_layers, M.n_kv_heads, M.head_dim)
    assert (smoothing >= 0.1).all()
    outs = []
    for s in (None, smoothing):
        p = pool(2 * M.max_seq_len, "float32")
        p.key_smoothing = s
        kv = ContiguousKVCache(p, M.max_seq_len)
        kv.allocate(0, 40)
        pos = torch.arange(40)[None]
        outs.append(model(ids[:1], pos, kv.build_batch([0], pos, pos >= 0)))
    torch.testing.assert_close(outs[0], outs[1], atol=1e-5, rtol=1e-5)


def test_key_smoothing_helps_int8_with_outlier_channels() -> None:
    g = torch.Generator().manual_seed(3)
    keys = torch.randn(512, 2, 64, generator=g)
    keys[..., :2] *= 20  # a couple of huge channels, as real LLM keys have
    smoothing = keys.abs().amax(0) / keys.abs().amax(0).mean(-1, keepdim=True)

    def error(s: torch.Tensor) -> float:
        q, scale = quantize_int8(keys / s)
        return float(((dequantize_int8(q, scale) * s - keys)[..., 2:]).abs().mean())

    assert error(smoothing) < error(torch.ones(2, 64)) / 5


def test_build_engine_calibrates_int8_smoothing() -> None:
    engine = build_engine(EngineConfig(model=M, kv_dtype="int8"))
    assert engine.kv.pool.key_smoothing is not None
    off = build_engine(EngineConfig(model=M, kv_dtype="int8", kv_smooth_keys=False))
    assert off.kv.pool.key_smoothing is None
    assert build_engine(EngineConfig(model=M, kv_dtype="bfloat16")).kv.pool.key_smoothing is None
