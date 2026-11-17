"""Turning logits into the next token for each request in a batch."""

from __future__ import annotations

import torch
from torch import Tensor

from tinyserve.request import Request


def sample(logits: Tensor, requests: list[Request]) -> list[int]:
    """Pick one token per row of ``logits`` (shape ``(B, vocab)``).

    Greedy rows are handled with one batched ``argmax``. Rows with a temperature
    sample from the softmax using the request's own RNG, so a request generates
    the same tokens no matter which other requests share its batch.
    """
    tokens = logits.argmax(dim=-1).tolist()
    for i, req in enumerate(requests):
        t = req.params.temperature
        if t > 0:
            probs = torch.softmax(logits[i].float() / t, dim=-1)
            tokens[i] = int(torch.multinomial(probs, 1, generator=req.generator))
    return tokens
