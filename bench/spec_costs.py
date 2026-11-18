"""Measure what speculative decoding costs per step, relative to a plain decode step.

Times, interleaved and repeated: a plain decode step, a verification of ``k``
drafts per request, and drafting ``k`` tokens with each draft model, for several
batch sizes at a fixed context length. Prints medians relative to plain decode.

Usage: ``python bench/spec_costs.py --batch-sizes 1 4 8 --context 96``
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch

from tinyserve.draft import early_exit_draft, load_distilled_draft
from tinyserve.kv_cache import KVPool, PagedKVCache
from tinyserve.model import TinyGPT
from tinyserve.request import Request, SamplingParams
from tinyserve.runner import ModelRunner
from tinyserve.speculative import DraftModelProposer


def paged(model: TinyGPT, slots: int = 8192) -> PagedKVCache:
    """A paged KV cache shaped for ``model``."""
    c = model.cfg
    return PagedKVCache(KVPool(slots, c.n_layers, c.n_kv_heads, c.head_dim))


def measure(
    target: TinyGPT, drafts: dict[str, TinyGPT], batch: int, args: argparse.Namespace
) -> dict[str, float]:
    """Median seconds per call of each job for one batch size."""
    g = torch.Generator().manual_seed(batch)
    kv = paged(target)
    runner = ModelRunner(target, kv)
    reqs = []
    for _ in range(batch):
        ids = torch.randint(4, target.cfg.vocab_size, (args.context,), generator=g).tolist()
        r = Request(ids[:-1], SamplingParams(max_new_tokens=64))
        r.output_ids = [ids[-1]]
        kv.allocate(r.request_id, r.length + args.k)
        reqs.append(r)
    runner.prefill(reqs, [r.length - 1 for r in reqs])
    proposers = {n: DraftModelProposer(m, paged(m)) for n, m in drafts.items()}

    def draft_step(p: DraftModelProposer) -> None:
        p.propose(reqs, [args.k] * batch)
        for r in reqs:  # so the next call does the usual one-token catch-up
            p._computed[r.request_id] = r.length - 1

    jobs: dict[str, Callable[[], object]] = {"decode": lambda: runner.decode(reqs)}
    for k in range(1, args.k + 1):
        jobs[f"verify k={k}"] = lambda k=k: runner.verify(reqs, [[5] * k] * batch)
    for name, p in proposers.items():
        jobs[f"draft {name} k={args.k}"] = lambda p=p: draft_step(p)
    times: dict[str, list[float]] = {n: [] for n in jobs}
    for _ in range(3):
        for job in jobs.values():
            job()
    for _ in range(args.repeats):
        for name, job in jobs.items():
            start = time.perf_counter()
            job()
            times[name].append(time.perf_counter() - start)
    return {name: statistics.median(t) for name, t in times.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--context", type=int, default=96)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    target = TinyGPT()
    drafts = {
        "early-exit": early_exit_draft(target, 1),
        "distilled": load_distilled_draft(target.cfg),
    }
    for batch in args.batch_sizes:
        medians = measure(target, drafts, batch, args)
        base = medians["decode"]
        print(f"batch {batch}: decode {base * 1000:.2f} ms")
        for name, med in medians.items():
            print(f"  {name:24s} {med * 1000:6.2f} ms  {med / base:5.2f}x decode")


if __name__ == "__main__":
    main()
