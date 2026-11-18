"""Distill a tiny draft model from the (random-weight) target, for speculative decoding.

1. Sample sequences from the target with the engine, from random prompts and at
   random temperatures in [0.2, 1.0], so the data is not just greedy repeats.
2. Train a 1-layer, 128-wide model to match the target's next-token distribution
   (KL divergence at several temperatures) on those sequences.
3. Report how often each draft would be accepted on held-out sequences and save
   the weights to ``tinyserve/assets/draft_distilled.pt``.

Usage: ``python bench/distill_draft.py`` (a few minutes on 2 CPU threads).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from tinyserve.draft import DISTILLED_DRAFT_PATH, distilled_draft_config, early_exit_draft
from tinyserve.engine import EngineConfig, build_engine
from tinyserve.model import TinyGPT
from tinyserve.request import Request, SamplingParams

TRAIN_TEMPERATURES = (1.0, 0.5, 0.25)
EVAL_TEMPERATURES = (0.2, 0.3, 0.5, 1.0)


def sample_sequences(n: int, seq_len: int, seed: int) -> Tensor:
    """``(n, seq_len)`` token ids: random prompts continued by the target."""
    engine = build_engine(
        EngineConfig(strategy="continuous", kv_layout="paged", kv_capacity_tokens=16384)
    )
    g = torch.Generator().manual_seed(seed)
    vocab = engine.model.cfg.vocab_size
    reqs = []
    for i in range(n):
        prompt_len = int(torch.randint(8, 128, (), generator=g))
        prompt = torch.randint(4, vocab, (prompt_len,), generator=g).tolist()
        temperature = float(torch.empty(()).uniform_(0.2, 1.0, generator=g))
        params = SamplingParams(
            max_new_tokens=seq_len - prompt_len,
            temperature=temperature,
            ignore_eos=True,
            seed=seed * 100_000 + i,
        )
        reqs.append(Request(prompt, params))
        engine.add_request(reqs[-1])
    while engine.has_unfinished():
        engine.step()
    return torch.tensor([r.all_ids for r in reqs])


@torch.inference_mode()
def teacher_logits(model: TinyGPT, seqs: Tensor, batch: int = 16) -> Tensor:
    """Target logits at every position of ``seqs``."""
    return torch.cat([model(seqs[i : i + batch]) for i in range(0, len(seqs), batch)])


def distill_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Mean KL(teacher || student) over :data:`TRAIN_TEMPERATURES`."""
    loss = student.new_zeros(())
    for t in TRAIN_TEMPERATURES:
        loss = loss + F.kl_div(
            F.log_softmax(student / t, -1),
            F.log_softmax(teacher / t, -1),
            log_target=True,
            reduction="batchmean",
        )
    return loss / len(TRAIN_TEMPERATURES)


@torch.inference_mode()
def acceptance(draft: TinyGPT, seqs: Tensor, teacher: Tensor, prompt_skip: int = 128) -> dict:
    """Expected per-token acceptance ``sum(min(p, q))`` and greedy top-1 agreement."""
    q_logits = teacher_logits(draft, seqs)[:, prompt_skip:]
    p_logits = teacher[:, prompt_skip:]
    out = {"greedy": (q_logits.argmax(-1) == p_logits.argmax(-1)).float().mean().item()}
    for t in EVAL_TEMPERATURES:
        p, q = (torch.softmax(x / t, -1) for x in (p_logits, q_logits))
        out[f"T={t}"] = torch.minimum(p, q).sum(-1).mean().item()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-seqs", type=int, default=64)
    parser.add_argument("--eval-seqs", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DISTILLED_DRAFT_PATH)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    train = sample_sequences(args.train_seqs, args.seq_len, seed=1)
    held_out = sample_sequences(args.eval_seqs, args.seq_len, seed=2)
    target = TinyGPT()
    train_logits = teacher_logits(target, train)
    eval_logits = teacher_logits(target, held_out)
    print(f"sampled and scored {train.numel():,} tokens in {time.perf_counter() - t0:.0f} s")

    student = TinyGPT(distilled_draft_config(target.cfg)).train()
    opt = torch.optim.AdamW(student.parameters(), lr=5e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-3, total_steps=args.steps)
    g = torch.Generator().manual_seed(0)
    t0 = time.perf_counter()
    for step in range(args.steps):
        idx = torch.randint(len(train), (8,), generator=g)
        loss = distill_loss(student(train[idx]), train_logits[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  {time.perf_counter() - t0:.0f} s")
    student.eval()

    for name, draft in (
        ("distilled (1 layer, d=128)", student),
        ("early exit (target layer 1)", early_exit_draft(target, 1)),
        ("early exit (target layers 1-2)", early_exit_draft(target, 2)),
    ):
        scores = acceptance(draft, held_out, eval_logits)
        print(f"{name:32s} " + "  ".join(f"{k} {v:.3f}" for k, v in scores.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), args.out)
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
