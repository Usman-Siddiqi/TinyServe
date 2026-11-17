"""The model runner: turns a list of requests into one batched forward pass.

Every scheduling strategy in TinyServe (naive, static, continuous) uses this
same runner. They only differ in *which* requests they hand it each step, which
keeps the benchmark comparison fair.

There are two kinds of forward passes:

* **Prefill** processes a request's whole sequence at once (its prompt, or prompt
  plus earlier outputs after a preemption), fills the KV cache and samples the
  first new token. It is compute-heavy: many tokens per request.
* **Decode** processes one token per request, the one sampled last step. It is
  cheap per request, which is exactly why batching many requests together pays off.
"""

from __future__ import annotations

import torch

from tinyserve.kv_cache import KVCacheManager
from tinyserve.model import TinyGPT
from tinyserve.request import Request
from tinyserve.sampler import sample
from tinyserve.tokenizer import PAD_ID


class ModelRunner:
    """Builds padded input tensors, runs the model, and samples next tokens."""

    def __init__(self, model: TinyGPT, kv: KVCacheManager) -> None:
        self.model = model
        self.kv = kv

    @torch.inference_mode()
    def prefill(self, requests: list[Request]) -> list[int]:
        """Run every token of each request's sequence and sample one new token each.

        Sequences are right-padded to the longest one. Padding is excluded from the
        cache writes, and its outputs are simply ignored.
        """
        seqs = [r.all_ids for r in requests]
        lengths = torch.tensor([len(s) for s in seqs])
        width = int(lengths.max())
        input_ids = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s)
        arange = torch.arange(width).expand(len(seqs), -1)
        valid = arange < lengths[:, None]
        positions = arange * valid  # padding gets position 0; it is never written

        kv = self.kv.build_batch([r.request_id for r in requests], positions, valid)
        hidden = self.model.hidden_states(input_ids, positions, kv)
        last_hidden = hidden[torch.arange(len(seqs)), lengths - 1]
        return sample(self.model.logits(last_hidden), requests)

    @torch.inference_mode()
    def decode(self, requests: list[Request]) -> list[int]:
        """Feed each request's latest token at its position and sample the next one."""
        input_ids = torch.tensor([[r.output_ids[-1]] for r in requests])
        positions = torch.tensor([[r.length - 1] for r in requests])
        valid = torch.ones_like(input_ids, dtype=torch.bool)

        kv = self.kv.build_batch([r.request_id for r in requests], positions, valid)
        hidden = self.model.hidden_states(input_ids, positions, kv)
        return sample(self.model.logits(hidden[:, 0]), requests)
