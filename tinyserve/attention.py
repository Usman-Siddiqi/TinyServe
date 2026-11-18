"""Attention backends for decode: the PyTorch gather path or a fused C++ paged kernel.

``torch`` gathers each sequence's K/V into a dense padded tensor and calls
``scaled_dot_product_attention``. ``cpp`` reads K/V straight from the pool
through block tables (see ``csrc/paged_attention.cpp``). The C++ extension is
compiled on first use; if that fails, TinyServe warns and falls back to ``torch``.
"""

from __future__ import annotations

import functools
import logging
import os
import warnings
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor

ATTENTION_BACKENDS = ("torch", "cpp")
ENV_VAR = "TINYSERVE_ATTENTION"

_SOURCE = Path(__file__).parent / "csrc" / "paged_attention.cpp"
logger = logging.getLogger(__name__)


@functools.cache
def load_extension() -> ModuleType | None:
    """JIT-compile and import the C++ kernel, or return None if it cannot be built."""
    from torch.utils.cpp_extension import load_inline

    try:
        return load_inline(
            name="tinyserve_paged_attention",
            cpp_sources=_SOURCE.read_text(),
            # FMA contraction and errno-free math, without -ffast-math's global effects.
            extra_cflags=[
                "-O3",
                "-march=native",
                "-fopenmp",
                "-std=c++17",
                "-fno-math-errno",
                "-fno-trapping-math",
                "-ffp-contract=fast",
            ],
            extra_ldflags=["-fopenmp"],
        )
    except Exception as exc:  # compiler missing, build error, unsupported platform...
        logger.debug("building the C++ attention kernel failed", exc_info=True)
        warnings.warn(
            f"C++ attention kernel unavailable ({type(exc).__name__}); using the torch backend",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def cpp_available() -> bool:
    """Whether the C++ kernel builds and loads on this machine."""
    return load_extension() is not None


def resolve_backend(requested: str | None = None) -> str:
    """Pick the attention backend: ``requested``, else ``$TINYSERVE_ATTENTION``, else torch.

    Asking for ``cpp`` when the extension cannot be built falls back to ``torch``.
    """
    backend = requested or os.environ.get(ENV_VAR) or "torch"
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"unknown attention backend {backend!r}; choose from {ATTENTION_BACKENDS}")
    if backend == "cpp" and not cpp_available():
        return "torch"
    return backend


def paged_attention_decode(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    block_size: int,
) -> Tensor:
    """Single-token GQA attention over K/V stored in a paged pool.

    Args:
        q: ``(B, H, D)`` one query token per sequence.
        k_cache, v_cache: ``(num_slots, H_kv, D)`` one layer of the pool.
        block_tables: ``(B, W)`` int64 physical block of each logical block.
        context_lens: ``(B,)`` int64 tokens each sequence attends to.
        block_size: tokens per block.

    Returns:
        ``(B, H, D)`` attention output.
    """
    ext = load_extension()
    if ext is None:
        raise RuntimeError("the C++ attention kernel is not available")
    scale = q.shape[-1] ** -0.5
    return ext.paged_attention_decode(
        q, k_cache, v_cache, block_tables, context_lens, block_size, scale
    )


def reference_paged_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    block_size: int,
) -> Tensor:
    """Slow, obviously-correct PyTorch version of :func:`paged_attention_decode`."""
    out = torch.empty_like(q)
    for b, length in enumerate(context_lens.tolist()):
        j = torch.arange(length)
        slots = block_tables[b, j // block_size] * block_size + j % block_size
        out[b] = torch.nn.functional.scaled_dot_product_attention(
            q[b, :, None],  # (H, 1, D)
            k_cache[slots].transpose(0, 1),  # (H_kv, L, D)
            v_cache[slots].transpose(0, 1),
            enable_gqa=True,
        )[:, 0]
    return out
