"""Storage formats for the KV cache: float32, bfloat16 and int8 with per-token scales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from tinyserve.model import TinyGPT

INT8_MAX = 127


@dataclass(frozen=True)
class KVFormat:
    """How one KV dtype is stored and attended.

    Attributes:
        name: the config name (``"float32"``, ``"bfloat16"`` or ``"int8"``).
        storage: dtype of the stored keys/values.
        compute: dtype attention runs in after the gather.
        scale_bytes: bytes of scale stored per (token, head), 0 if unscaled.
    """

    name: str
    storage: torch.dtype
    compute: torch.dtype
    scale_bytes: int = 0

    @property
    def quantized(self) -> bool:
        """Whether values are stored as integers with a scale."""
        return self.scale_bytes > 0

    def bytes_per_head(self, head_dim: int) -> int:
        """Bytes for one head's key (or value) of one token."""
        return head_dim * torch.empty((), dtype=self.storage).element_size() + self.scale_bytes


KV_FORMATS: dict[str, KVFormat] = {
    "float32": KVFormat("float32", torch.float32, torch.float32),
    "bfloat16": KVFormat("bfloat16", torch.bfloat16, torch.bfloat16),
    "int8": KVFormat("int8", torch.int8, torch.bfloat16, scale_bytes=4),
}
KV_DTYPES = tuple(KV_FORMATS)


def kv_format(name: str) -> KVFormat:
    """Look up a :class:`KVFormat` by name."""
    try:
        return KV_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown kv_dtype {name!r}; choose from {KV_DTYPES}") from None


def quantize_int8(x: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric absmax quantization over the last dim.

    Returns int8 values of ``x``'s shape and float32 scales of shape ``x.shape[:-1]``,
    with ``|x - q * scale| <= scale / 2`` elementwise.
    """
    scale = x.abs().amax(dim=-1).float().clamp(min=1e-12) / INT8_MAX
    q = torch.round(x / scale[..., None]).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, scale


@torch.inference_mode()
def calibrate_key_smoothing(model: TinyGPT, input_ids: Tensor) -> Tensor:
    """Per-channel key magnitudes ``(n_layers, n_kv_heads, head_dim)`` measured on ``input_ids``.

    A few key channels are often much larger than the rest, and with one absmax
    scale per token they leave the other channels only a few bits. Dividing keys
    by these factors before quantizing (and multiplying queries by them) evens the
    channels out without changing ``q . k``.
    """
    c = model.cfg
    kv_dim = c.n_kv_heads * c.head_dim
    absmax: list[Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], out: Tensor) -> None:
        keys = out[..., c.d_model : c.d_model + kv_dim].reshape(-1, c.n_kv_heads, c.head_dim)
        absmax.append(keys.abs().amax(dim=0))

    handles = [block.attn.qkv.register_forward_hook(hook) for block in model.blocks]
    try:
        model(input_ids)
    finally:
        for h in handles:
            h.remove()
    scales = torch.stack(absmax)
    # Normalize per head and keep near-dead channels from being blown up.
    scales = scales / scales.mean(dim=-1, keepdim=True)
    return scales.clamp(min=0.1)


def dequantize_int8(q: Tensor, scale: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Inverse of :func:`quantize_int8`, returning ``dtype``."""
    # Casting the scale first makes this a single pass that writes ``dtype`` directly.
    return q * scale.to(dtype)[..., None]
