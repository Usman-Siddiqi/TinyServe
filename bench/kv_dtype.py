"""Compare KV cache dtypes (float32, bfloat16, int8): output quality, decode speed, end to end.

Usage::

    python bench/kv_dtype.py quality              # agreement and logit error vs float32
    python bench/kv_dtype.py decode               # decode step time by batch size
    python bench/kv_dtype.py e2e --kv-mib 32      # same KV bytes, Poisson workload

Every comparison runs the dtypes interleaved in one process so machine drift
hits them equally, and reports medians over repeats.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace

import torch
import torch.nn.functional as F
from run import drive, summarize, warm_up

from tinyserve.engine import EngineConfig, build_engine, calibrate_kv
from tinyserve.kv_cache import KVPool, PagedKVCache
from tinyserve.kv_quant import KV_DTYPES
from tinyserve.model import ModelConfig, TinyGPT
from tinyserve.request import Request, SamplingParams
from tinyserve.workload import WorkloadSpec, make_workload


class AdvancedIndexingPool(KVPool):
    """The float32 pool as it was before this change (advanced-indexing gather)."""

    def read(self, layer: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer][slots], self.v[layer][slots]


# Quality variants compared against float32: name -> (kv_dtype, smooth keys).
VARIANTS = {
    "bfloat16": ("bfloat16", False),
    "int8": ("int8", True),
    "int8-raw": ("int8", False),
}


def make_pool(name: str, num_slots: int, model: TinyGPT) -> KVPool:
    """A pool for a kv dtype, a quality variant, or ``"float32-old"`` (the pre-change gather)."""
    m = model.cfg
    if name == "float32-old":
        return AdvancedIndexingPool(num_slots, m.n_layers, m.n_kv_heads, m.head_dim)
    kv_dtype, smooth = VARIANTS.get(name, (name, name == "int8"))
    pool = KVPool(num_slots, m.n_layers, m.n_kv_heads, m.head_dim, kv_dtype)
    if smooth:
        calibrate_kv(model, pool)
    return pool


def random_prompts(n: int, seed: int, min_len: int, max_len: int, vocab: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(min_len, max_len + 1, (n,), generator=g).tolist()
    return [torch.randint(4, vocab, (k,), generator=g).tolist() for k in lengths]


# --- quality -------------------------------------------------------------------------


def generate_all(
    model: TinyGPT, cfg: EngineConfig, prompts: list[list[int]], params: list[SamplingParams]
) -> list[list[int]]:
    engine = build_engine(cfg)
    engine.model.load_state_dict(model.state_dict())
    if engine.kv.pool.key_smoothing is not None:
        calibrate_kv(engine.model, engine.kv.pool)  # redo it for the loaded weights
    requests = [Request(p, sp) for p, sp in zip(prompts, params, strict=True)]
    for r in requests:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()
    return [r.output_ids for r in requests]


@torch.inference_mode()
def teacher_forced_logits(
    model: TinyGPT, variant: str, prompts: list[list[int]], continuations: list[list[int]]
) -> torch.Tensor:
    """Prefill each prompt, then decode the given continuations in one batch, token by token.

    Returns ``(num_seqs * T, vocab)`` logits predicting each of the ``T`` continuation tokens.
    """
    m = model.cfg
    kv = PagedKVCache(make_pool(variant, len(prompts) * m.max_seq_len, model), watermark=0)
    steps = [[] for _ in range(len(continuations[0]))]
    for i, (prompt, cont) in enumerate(zip(prompts, continuations, strict=True)):
        kv.allocate(i, len(prompt) + len(cont))
        pos = torch.arange(len(prompt))[None]
        batch = kv.build_batch([i], pos, pos >= 0)
        steps[0].append(model(torch.tensor([prompt]), pos, batch)[:, -1])
    rows = list(range(len(prompts)))
    starts = torch.tensor([len(p) for p in prompts])[:, None]
    for t in range(1, len(steps)):
        ids = torch.tensor([[c[t - 1]] for c in continuations])
        pos = starts + t - 1
        steps[t].append(model(ids, pos, kv.build_batch(rows, pos, pos >= 0))[:, 0])
    per_step = [torch.cat(s) for s in steps]  # (T, num_seqs, vocab) once stacked
    return torch.stack(per_step, dim=1).reshape(-1, m.vocab_size)


@torch.no_grad()
def stress(model: TinyGPT) -> None:
    """Make the random model harder to quantize: peaked attention, key outliers, sharp logits.

    Real LLMs have a few key channels with very large magnitudes, which per-token
    absmax scaling handles worst; random weights have none, so we plant two per head.
    """
    c = model.cfg
    qk = slice(0, c.d_model + c.n_kv_heads * c.head_dim)
    for block in model.blocks:
        block.attn.qkv.weight[qk] *= 4  # attention scores scale by 16x
        keys = block.attn.qkv.weight[c.d_model : c.d_model + c.n_kv_heads * c.head_dim]
        for h in range(c.n_kv_heads):
            keys[h * c.head_dim : h * c.head_dim + 2] *= 8
    model.ln_f.weight *= 8  # logit std goes from ~0.5 to ~4


def quality(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    m = ModelConfig()
    model = TinyGPT(m)
    if args.stress:
        stress(model)
    n = args.num_requests
    prompts = random_prompts(n, seed=args.seed, min_len=16, max_len=192, vocab=m.vocab_size)
    greedy = [SamplingParams(max_new_tokens=args.max_tokens, ignore_eos=True)] * n
    sampled = [
        SamplingParams(max_new_tokens=args.max_tokens, temperature=1.0, seed=i, ignore_eos=True)
        for i in range(n)
    ]
    base = EngineConfig(strategy="continuous", kv_layout="paged", model=m, max_batch_size=32)
    base = replace(base, kv_capacity_tokens=n * m.max_seq_len)

    print(f"== end-to-end agreement with float32 ({n} requests x {args.max_tokens} tokens)")
    for mode, params in (("greedy", greedy), ("sampled", sampled)):
        ref = generate_all(model, base, prompts, params)
        for name, (dtype, smooth) in VARIANTS.items():
            cfg = replace(base, kv_dtype=dtype, kv_smooth_keys=smooth)
            outs = generate_all(model, cfg, prompts, params)
            same = sum(o == r for o, r in zip(outs, ref, strict=True))
            first = [
                next(j for j, (a, b) in enumerate(zip(o, r, strict=True)) if a != b)
                for o, r in zip(outs, ref, strict=True)
                if o != r
            ]
            tok_agree = sum(
                a == b for o, r in zip(outs, ref, strict=True) for a, b in zip(o, r, strict=True)
            ) / sum(len(r) for r in ref)
            med_first = statistics.median(first) if first else None
            print(
                f"{mode:7s} {name:9s} identical {same}/{n} ({same / n:.0%}) | "
                f"token agreement {tok_agree:.1%} | median first divergence {med_first}"
            )

    print("== teacher-forced next-token logits vs float32 (sampled float32 continuations)")
    conts = generate_all(model, base, prompts, sampled)
    ref_logits = teacher_forced_logits(model, "float32", prompts, conts)
    ref_logp = F.log_softmax(ref_logits, -1)
    logit_std = ref_logits.std().item()
    print(f"float32 logit std {logit_std:.4f} over {len(ref_logits)} positions")
    for name in VARIANTS:
        got = teacher_forced_logits(model, name, prompts, conts)
        err = (got - ref_logits).abs()
        kl = F.kl_div(F.log_softmax(got, -1), ref_logp, log_target=True, reduction="none")
        kl = kl.sum(-1)
        top1 = (got.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        print(
            f"{name:9s} max |dlogit| {err.max():.2e} | mean {err.mean():.2e} "
            f"({err.mean() / logit_std:.2%} of logit std) | KL mean {kl.mean():.2e} "
            f"max {kl.max():.2e} | top-1 agreement {top1:.2%}"
        )


# --- decode step time ----------------------------------------------------------------


@torch.inference_mode()
def decode(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    m = ModelConfig()
    model = TinyGPT(m)
    names = ["float32-old", *KV_DTYPES]
    g = torch.Generator().manual_seed(0)
    print(f"== decode step time (ms), contexts {args.min_ctx}-{args.max_ctx}, median")
    print(f"{'batch':>5} " + " ".join(f"{n:>12}" for n in names) + "   speedup vs float32-old")
    for batch in args.batches:
        ctx = torch.randint(args.min_ctx, args.max_ctx + 1, (batch,), generator=g).tolist()
        setups = {}
        for name in names:
            kv = PagedKVCache(make_pool(name, (batch + 1) * m.max_seq_len, model), watermark=0)
            kv.allocate(10_000, 1)  # a neighbour so blocks do not start at slot 0
            for i, c in enumerate(ctx):
                kv.allocate(i, c)
                pos = torch.arange(c - 1)[None]
                ids = torch.randint(4, m.vocab_size, (1, c - 1), generator=g)
                model.hidden_states(ids, pos, kv.build_batch([i], pos, pos >= 0))
            setups[name] = kv
        ids = torch.randint(4, m.vocab_size, (batch, 1), generator=g)
        pos = torch.tensor(ctx)[:, None] - 1
        valid = torch.ones_like(pos, dtype=torch.bool)
        times: dict[str, list[float]] = {n: [] for n in names}
        for rep in range(args.repeats + 1):
            for name, kv in setups.items():
                t = time.perf_counter()
                for _ in range(args.steps):
                    batch_kv = kv.build_batch(list(range(batch)), pos, valid)
                    model.logits(model.hidden_states(ids, pos, batch_kv))
                if rep:  # the first round is warm-up
                    times[name].append((time.perf_counter() - t) / args.steps * 1000)
        med = {n: statistics.median(ts) for n, ts in times.items()}
        speed = " ".join(f"{n}={med['float32-old'] / med[n]:.2f}x" for n in names[1:])
        print(f"{batch:>5} " + " ".join(f"{med[n]:>12.2f}" for n in names) + f"   {speed}")


# --- end to end ----------------------------------------------------------------------


def e2e(args: argparse.Namespace) -> None:
    spec = WorkloadSpec(num_requests=args.num_requests, rate=args.rate, seed=0)
    items = make_workload(spec)
    base = EngineConfig(
        strategy="continuous",
        kv_layout="paged",
        max_batch_size=args.max_batch_size,
        num_threads=args.threads,
        kv_memory_bytes=int(args.kv_mib * 2**20),
    )
    runs: dict[str, list[dict]] = {d: [] for d in KV_DTYPES}
    for rep in range(args.repeats):
        for dtype in KV_DTYPES:
            cfg = replace(base, kv_dtype=dtype)
            engine = build_engine(cfg, record_trace=True)
            warm_up(engine)
            requests, t0, t_end = drive(engine, items)
            s = summarize(engine, requests, t0, t_end)
            s["slots"] = cfg.num_kv_slots
            runs[dtype].append(s)
            print(
                f"[{rep + 1}] {dtype:9s} slots {s['slots']:6d} | "
                f"{s['throughput_tok_s']:6.0f} tok/s | p50 lat {s['latency_p50_s']:5.2f}s | "
                f"p95 lat {s['latency_p95_s']:5.2f}s | p50 TTFT {s['ttft_p50_s']:6.3f}s | "
                f"tpot {s['tpot_p50_ms']:5.1f}ms | batch mean {s['mean_batch_size']:4.1f} "
                f"max {s['max_batch_size']} | preempt {s['num_preemptions']}",
                flush=True,
            )
    print("== medians")
    ref = statistics.median(r["throughput_tok_s"] for r in runs["float32"])
    for dtype, reps in runs.items():

        def med(key: str, reps: list[dict] = reps) -> float:
            return statistics.median(r[key] for r in reps)

        tput = med("throughput_tok_s")
        print(
            f"{dtype:9s} {tput:6.0f} tok/s ({tput / ref:.2f}x)"
            f" | p50 lat {med('latency_p50_s'):5.2f}s | p95 lat {med('latency_p95_s'):5.2f}s"
            f" | p50 TTFT {med('ttft_p50_s'):6.3f}s | mean batch {med('mean_batch_size'):4.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality")
    q.add_argument("--num-requests", type=int, default=64)
    q.add_argument("--max-tokens", type=int, default=128)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--threads", type=int, default=1)
    q.add_argument("--stress", action="store_true", help="sharpen the model, plant key outliers")
    d = sub.add_parser("decode")
    d.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32])
    d.add_argument("--min-ctx", type=int, default=20)
    d.add_argument("--max-ctx", type=int, default=400)
    d.add_argument("--steps", type=int, default=5)
    d.add_argument("--repeats", type=int, default=7)
    d.add_argument("--threads", type=int, default=4)
    e = sub.add_parser("e2e")
    e.add_argument("--kv-mib", type=float, default=32.0, help="32 MiB = 8192 fp32 slots")
    e.add_argument("--num-requests", type=int, default=200)
    e.add_argument("--rate", type=float, default=40.0)
    e.add_argument("--max-batch-size", type=int, default=32)
    e.add_argument("--repeats", type=int, default=3)
    e.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    {"quality": quality, "decode": decode, "e2e": e2e}[args.cmd](args)


if __name__ == "__main__":
    main()
