"""A tiny GPT-style decoder-only transformer whose attention reads from an external KV cache.

The weights are random (fixed seed), so the model produces gibberish. That is on
purpose: TinyServe is about the serving system around the model, and the serving
system only needs a model with realistic *shapes* and *costs*.

The one thing that makes this model "servable" is that attention does not own its
key/value history. Every forward pass receives a :class:`~tinyserve.kv_cache.KVBatch`
that tells each attention layer where in a shared memory pool to write the new
keys/values and where to read the full history from. That indirection is what lets
the engine batch together sequences of different lengths and manage KV memory
however it likes (contiguous slabs or paged blocks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyserve.kv_cache import KVBatch
from tinyserve.tokenizer import CharTokenizer


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters of the tiny transformer."""

    vocab_size: int = CharTokenizer().vocab_size
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 512
    seed: int = 0

    @property
    def head_dim(self) -> int:
        """Size of each attention head."""
        return self.d_model // self.n_heads

    def kv_bytes_per_token(self, dtype: torch.dtype = torch.float32) -> int:
        """Bytes of KV cache needed for one token across all layers (keys + values)."""
        elem = torch.empty((), dtype=dtype).element_size()
        return 2 * self.n_layers * self.d_model * elem


class Attention(nn.Module):
    """Multi-head causal self-attention backed by an optional external KV cache."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor, positions: Tensor, kv: KVBatch | None) -> Tensor:
        """Attend from the ``S`` new tokens in ``x`` to every earlier token of their sequence.

        Args:
            x: ``(B, S, d_model)`` hidden states of the new tokens.
            positions: ``(B, S)`` absolute position of each new token in its sequence.
            kv: where to store and fetch keys/values. ``None`` means "no cache": the
                tokens in ``x`` are the whole sequence and attend only to each other.
        """
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.n_heads, self.head_dim).unbind(dim=2)

        if kv is None:
            keys, values = k, v
            key_positions = positions  # (B, S)
        else:
            # Write the new keys/values into the cache, then read the full history
            # (old tokens + the ones we just wrote) back as a dense (B, L, H, D) tensor.
            keys, values = kv.write_and_read(self.layer_idx, k, v)
            key_positions = torch.arange(keys.shape[1], device=x.device).expand(b, -1)

        # Causal mask from absolute positions: a query at position p may see keys at
        # positions <= p. This one rule also hides padding and stale cache entries,
        # because those always sit at positions beyond the sequence's current length.
        mask = key_positions[:, None, :] <= positions[:, :, None]  # (B, S, L)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # (B, H, S, D)
            keys.transpose(1, 2),  # (B, H, L, D)
            values.transpose(1, 2),
            attn_mask=mask[:, None, :, :],
        )
        return self.out(out.transpose(1, 2).reshape(b, s, -1))


class Block(nn.Module):
    """A pre-norm transformer block: attention then an MLP, each with a residual."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: Tensor, positions: Tensor, kv: KVBatch | None) -> Tensor:
        """Apply the block to ``(B, S, d_model)`` hidden states."""
        x = x + self.attn(self.ln1(x), positions, kv)
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Decoder-only transformer with learned positional embeddings and tied output weights."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        # Build the weights under a private RNG so the model is identical on every
        # machine and does not disturb (or depend on) the global torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seed)
            self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
            self.pos_emb = nn.Embedding(c.max_seq_len, c.d_model)
            self.blocks = nn.ModuleList(Block(c, i) for i in range(c.n_layers))
            self.ln_f = nn.LayerNorm(c.d_model)
            self._init_weights()
        self.eval()

    def _init_weights(self) -> None:
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)
        # GPT-2 style scaled init for the residual projections.
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=std / math.sqrt(2 * self.cfg.n_layers))
            nn.init.normal_(block.mlp[2].weight, std=std / math.sqrt(2 * self.cfg.n_layers))

    def hidden_states(self, input_ids: Tensor, positions: Tensor, kv: KVBatch | None) -> Tensor:
        """Run the transformer and return final hidden states ``(B, S, d_model)``."""
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x, positions, kv)
        return self.ln_f(x)

    def logits(self, hidden: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits (weights tied to the embedding)."""
        return hidden @ self.tok_emb.weight.T

    def forward(
        self, input_ids: Tensor, positions: Tensor | None = None, kv: KVBatch | None = None
    ) -> Tensor:
        """Compute logits ``(B, S, vocab)`` for ``input_ids`` of shape ``(B, S)``.

        Without ``kv`` this is an ordinary full forward pass over a batch of
        equal-length sequences. With ``kv`` the input is only the *new* tokens
        and attention also sees whatever the cache already holds.
        """
        if positions is None:
            positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        return self.logits(self.hidden_states(input_ids, positions, kv))
