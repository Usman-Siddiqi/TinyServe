"""Draft models for speculative decoding.

The target has random weights, so an independently trained small model would
almost never agree with it. Two drafts that do:

* :func:`early_exit_draft`: the target's own first layers plus its embeddings and
  LM head (self-speculation, as in LayerSkip). No training needed.
* :func:`load_distilled_draft`: a 1-layer, 128-wide model trained to imitate the
  target's next-token distribution (see ``bench/distill_draft.py``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from tinyserve.model import ModelConfig, TinyGPT

DISTILLED_DRAFT_PATH = Path(__file__).parent / "assets" / "draft_distilled.pt"


def distilled_draft_config(target: ModelConfig) -> ModelConfig:
    """Architecture of the distilled draft for ``target``."""
    return ModelConfig(
        vocab_size=target.vocab_size,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        d_model=128,
        d_ff=512,
        max_seq_len=target.max_seq_len,
    )


def early_exit_draft(target: TinyGPT, n_layers: int = 1) -> TinyGPT:
    """A draft made of the target's first ``n_layers`` blocks, embeddings and output head."""
    if not 0 < n_layers < target.cfg.n_layers:
        raise ValueError("n_layers must be between 1 and the target's depth - 1")
    draft = TinyGPT(replace(target.cfg, n_layers=n_layers))
    draft.tok_emb = target.tok_emb
    draft.pos_emb = target.pos_emb
    draft.ln_f = target.ln_f
    draft.blocks = torch.nn.ModuleList(target.blocks[:n_layers])
    return draft


def load_distilled_draft(target: ModelConfig, path: Path = DISTILLED_DRAFT_PATH) -> TinyGPT:
    """Load the distilled draft weights saved by ``bench/distill_draft.py``."""
    draft = TinyGPT(distilled_draft_config(target))
    state = torch.load(path, map_location="cpu", weights_only=True)
    draft.load_state_dict({k: v.float() for k, v in state.items()})
    return draft.eval()
