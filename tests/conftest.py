"""Shared fixtures and a slow-but-obviously-correct reference generator."""

from __future__ import annotations

import pytest
import torch

from tinyserve.model import ModelConfig, TinyGPT
from tinyserve.request import SamplingParams

SMALL_MODEL = ModelConfig(max_seq_len=128)


@pytest.fixture(scope="session")
def model() -> TinyGPT:
    return TinyGPT(SMALL_MODEL)


@torch.inference_mode()
def reference_generate(model: TinyGPT, prompt: list[int], params: SamplingParams) -> list[int]:
    """Greedy generation with no KV cache: recompute the whole sequence every token."""
    ids = list(prompt)
    out: list[int] = []
    for _ in range(params.max_new_tokens):
        logits = model(torch.tensor([ids]))[0, -1]
        tok = int(logits.argmax())
        out.append(tok)
        ids.append(tok)
    return out


def make_prompts(n: int, seed: int = 0, min_len: int = 3, max_len: int = 30) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(min_len, max_len + 1, (n,), generator=g).tolist()
    return [torch.randint(4, SMALL_MODEL.vocab_size, (k,), generator=g).tolist() for k in lengths]
