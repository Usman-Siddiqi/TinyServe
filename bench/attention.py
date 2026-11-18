"""Compare the torch and fused C++ decode-attention backends.

Usage::

    python bench/attention.py                 # decode-step timings + end-to-end run
    python bench/attention.py --mode step     # decode-step timings only

``step`` times one full decode step (KV batch build, all layers, logits) over a
batch whose context lengths are drawn uniformly from 20 to 400. ``e2e`` serves
the same Poisson workload with each backend. A/B runs are interleaved in one
process so machine noise hits both backends alike; medians are reported.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import replace

import torch
from run import drive, summarize, warm_up

from tinyserve.attention import cpp_available
from tinyserve.engine import EngineConfig, build_engine
from tinyserve.kv_cache import KVPool, PagedKVCache
from tinyserve.model import ModelConfig, TinyGPT
from tinyserve.workload import WorkloadSpec, make_workload

BACKENDS = ("torch", "cpp")


def filled_cache(cfg: ModelConfig, lengths: list[int]) -> PagedKVCache:
    """A paged cache holding random K/V for sequences ``0..len(lengths)-1``."""
    pool = KVPool(sum(lengths) + 16 * len(lengths) + 16, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim)
    kv = PagedKVCache(pool, block_size=16, watermark=0)
    for seq_id, n in enumerate(lengths):
        kv.allocate(seq_id, n + 1)
        # Random K/V stands in for a prefill; attention cost does not depend on values.
        slots = kv.addresses([seq_id], torch.arange(n)[None])[0]
        pool.k[:, slots] = torch.randn(cfg.n_layers, n, cfg.n_kv_heads, cfg.head_dim)
        pool.v[:, slots] = torch.randn(cfg.n_layers, n, cfg.n_kv_heads, cfg.head_dim)
    return kv


def decode_step(model: TinyGPT, kv: PagedKVCache, lengths: list[int], fused: bool) -> None:
    """One decode step for every cached sequence, mirroring ``ModelRunner.decode``."""
    ids = torch.full((len(lengths), 1), 5, dtype=torch.long)
    positions = torch.tensor(lengths)[:, None]
    valid = torch.ones_like(ids, dtype=torch.bool)
    kv_batch = kv.build_batch(list(range(len(lengths))), positions, valid, paged_attention=fused)
    model.logits(model.hidden_states(ids, positions, kv_batch)[:, 0])


@torch.inference_mode()
def bench_step(batch_sizes: list[int], repeats: int, iters: int, seed: int) -> None:
    """Print median decode-step time per backend for each batch size."""
    cfg = ModelConfig()
    model = TinyGPT(cfg)
    rng = random.Random(seed)
    print("| batch | torch (ms) | cpp (ms) | speedup |\n|---:|---:|---:|---:|")
    for batch in batch_sizes:
        lengths = [rng.randint(20, 400) for _ in range(batch)]
        kv = filled_cache(cfg, lengths)
        times: dict[str, list[float]] = {b: [] for b in BACKENDS}
        for b in BACKENDS:
            decode_step(model, kv, lengths, b == "cpp")  # warm-up
        for _ in range(repeats):
            for b in BACKENDS:
                t0 = time.perf_counter()
                for _ in range(iters):
                    decode_step(model, kv, lengths, b == "cpp")
                times[b].append((time.perf_counter() - t0) / iters * 1e3)
        med = {b: statistics.median(times[b]) for b in BACKENDS}
        print(
            f"| {batch} | {med['torch']:.2f} | {med['cpp']:.2f} | "
            f"{med['torch'] / med['cpp']:.2f}x |",
            flush=True,
        )


def bench_e2e(num_requests: int, repeats: int, threads: int, layout: str) -> None:
    """Serve one workload with each backend (interleaved) and print median throughput."""
    items = make_workload(WorkloadSpec(num_requests=num_requests, rate=40.0, seed=0))
    base = EngineConfig(
        strategy="continuous", kv_layout=layout, max_batch_size=32, num_threads=threads
    )
    results: dict[str, list[dict[str, float]]] = {b: [] for b in BACKENDS}
    for rep in range(repeats):
        for b in BACKENDS:
            engine = build_engine(replace(base, attention=b), record_trace=True)
            warm_up(engine)
            summary = summarize(engine, *drive(engine, items))
            results[b].append(summary)
            print(
                f"[{rep + 1}/{repeats}] {b:>5}: {summary['throughput_tok_s']:6.0f} tok/s | "
                f"p50 TPOT {summary['tpot_p50_ms']:5.2f} ms | "
                f"p50 latency {summary['latency_p50_s']:5.2f} s",
                flush=True,
            )
    print("| backend | tok/s | p50 TPOT (ms) | p50 latency (s) |\n|---|---:|---:|---:|")
    for b in BACKENDS:
        tput = statistics.median(s["throughput_tok_s"] for s in results[b])
        tpot = statistics.median(s["tpot_p50_ms"] for s in results[b])
        lat = statistics.median(s["latency_p50_s"] for s in results[b])
        print(f"| {b} | {tput:.0f} | {tpot:.2f} | {lat:.2f} |")


def main() -> None:
    """Parse arguments and run the selected benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["step", "e2e", "both"], default="both")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20, help="decode steps per timing")
    parser.add_argument("--num-requests", type=int, default=60)
    parser.add_argument("--e2e-repeats", type=int, default=3)
    parser.add_argument("--layout", choices=["contiguous", "paged"], default="paged")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not cpp_available():
        raise SystemExit("the C++ attention kernel could not be built")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.mode in ("step", "both"):
        bench_step(args.batch_sizes, args.repeats, args.iters, args.seed)
    if args.mode in ("e2e", "both"):
        bench_e2e(args.num_requests, args.e2e_repeats, args.threads, args.layout)


if __name__ == "__main__":
    main()
