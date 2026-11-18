"""The model runner: turns a list of requests into one batched forward pass.

Every scheduling strategy in TinyServe (naive, static, continuous) uses this
same runner. They only differ in *which* requests they hand it each step, which
keeps the benchmark comparison fair.

There are three kinds of forward passes:

* **Prefill** processes a request's whole sequence at once (its prompt, or prompt
  plus earlier outputs after a preemption), fills the KV cache and samples the
  first new token. It is compute-heavy: many tokens per request.
* **Decode** processes one token per request, the one sampled last step. It is
  cheap per request, which is exactly why batching many requests together pays off.
* **Verify** (speculative decoding) feeds the latest token plus a few drafted
  ones and returns the logits at every position, so all drafts are checked at once.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tinyserve.attention import resolve_backend
from tinyserve.kv_cache import KVCacheManager
from tinyserve.model import TinyGPT
from tinyserve.request import Request
from tinyserve.sampler import sample
from tinyserve.tokenizer import PAD_ID


@torch.inference_mode()
def forward_chunks(
    model: TinyGPT,
    kv: KVCacheManager,
    seq_ids: list[int],
    chunks: list[list[int]],
    starts: list[int],
) -> tuple[Tensor, Tensor]:
    """Run ``chunks[i]`` (tokens at positions ``starts[i]...``) through the model.

    Chunks are right-padded to the longest one; padding is never written to the
    cache. Returns the final hidden states ``(B, S, d_model)`` and the chunk lengths.
    """
    lengths = torch.tensor([len(c) for c in chunks])
    width = int(lengths.max())
    input_ids = torch.full((len(chunks), width), PAD_ID, dtype=torch.long)
    for i, c in enumerate(chunks):
        input_ids[i, : len(c)] = torch.tensor(c)
    arange = torch.arange(width).expand(len(chunks), -1)
    valid = arange < lengths[:, None]
    # Absolute positions; padding gets position 0 and is never written.
    positions = (arange + torch.tensor(starts)[:, None]) * valid
    batch = kv.build_batch(seq_ids, positions, valid)
    return model.hidden_states(input_ids, positions, batch), lengths


class ModelRunner:
    """Builds padded input tensors, runs the model, and samples next tokens.

    Args:
        model: the transformer.
        kv: the KV cache manager.
        attention: decode attention backend, ``"torch"`` or ``"cpp"`` (see
            :func:`tinyserve.attention.resolve_backend`). Prefill always uses torch.
    """

    def __init__(self, model: TinyGPT, kv: KVCacheManager, attention: str | None = None) -> None:
        self.model = model
        self.kv = kv
        self.attention = resolve_backend(attention)

    @torch.inference_mode()
    def prefill(
        self, requests: list[Request], num_tokens: list[int] | None = None
    ) -> list[int | None]:
        """Process the next ``num_tokens[i]`` uncomputed tokens of each request.

        Each request resumes at ``request.num_computed_tokens``, so a long prompt
        can be prefilled over several steps (chunked prefill) or skip tokens whose
        KV is already cached. A request whose chunk reaches the end of its sequence
        samples its next token; the others get ``None``.
        """
        starts = [r.num_computed_tokens for r in requests]
        if num_tokens is None:
            num_tokens = [r.length - s for r, s in zip(requests, starts, strict=True)]
        chunks = [
            r.all_ids[s : s + n] for r, s, n in zip(requests, starts, num_tokens, strict=True)
        ]
        hidden, lengths = forward_chunks(
            self.model, self.kv, [r.request_id for r in requests], chunks, starts
        )
        last_hidden = hidden[torch.arange(len(chunks)), lengths - 1]
        done = [s + n == r.length for r, s, n in zip(requests, starts, num_tokens, strict=True)]
        if not any(done):
            return [None] * len(requests)
        sampled = iter(
            sample(
                self.model.logits(last_hidden[done]),
                [r for r, d in zip(requests, done, strict=True) if d],
            )
        )
        return [next(sampled) if d else None for d in done]

    @torch.inference_mode()
    def decode(self, requests: list[Request]) -> list[int]:
        """Feed each request's latest token at its position and sample the next one."""
        input_ids = torch.tensor([[r.output_ids[-1]] for r in requests])
        positions = torch.tensor([[r.length - 1] for r in requests])
        valid = torch.ones_like(input_ids, dtype=torch.bool)

        kv = self.kv.build_batch(
            [r.request_id for r in requests],
            positions,
            valid,
            paged_attention=self.attention == "cpp" and self.kv.pool.k.dtype == torch.float32,
        )
        hidden = self.model.hidden_states(input_ids, positions, kv)
        return sample(self.model.logits(hidden[:, 0]), requests)

    @torch.inference_mode()
    def verify(self, requests: list[Request], drafts: list[list[int]]) -> list[Tensor]:
        """Score each request's latest token followed by its draft tokens in one pass.

        Returns, per request, logits ``(len(draft) + 1, vocab)``: row ``i`` is the
        target's next-token distribution after the latest token and ``draft[:i]``.
        The KV of every fed token is written; the caller truncates rejected ones.
        """
        chunks = [[r.output_ids[-1], *d] for r, d in zip(requests, drafts, strict=True)]
        starts = [r.length - 1 for r in requests]
        hidden, lengths = forward_chunks(
            self.model, self.kv, [r.request_id for r in requests], chunks, starts
        )
        logits = self.model.logits(hidden)
        return [logits[i, :n] for i, n in enumerate(lengths.tolist())]
