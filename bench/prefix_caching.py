"""A/B benchmark of automatic prefix caching on continuous batching with a paged KV cache.

Usage::

    python bench/prefix_caching.py                      # shared-prefix and default workloads
    python bench/prefix_caching.py --workloads shared --repeats 5

Caching off and on run interleaved (off-on, then on-off, to cancel drift in the
machine's speed) on fresh engines with the same requests. The median of each
metric over the repeats is reported with the relative change.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from run import drive, hardware_info, summarize, warm_up

from tinyserve.engine import EngineConfig, build_engine
from tinyserve.workload import WorkloadSpec, make_workload

METRICS = (
    "throughput_tok_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "latency_p50_s",
    "latency_p95_s",
    "prefill_tokens",
    "hit_rate",
    "mean_batch_size",
    "num_preemptions",
)


def run_once(cfg: EngineConfig, spec: WorkloadSpec) -> dict[str, Any]:
    """Serve ``spec`` on a fresh engine and summarize it, with prefix-cache metrics."""
    engine = build_engine(cfg, record_trace=True)
    warm_up(engine)
    stats = engine.kv.prefix_cache_stats()
    before = (stats.query_tokens, stats.hit_tokens) if stats else (0, 0)
    requests, t0, t_end = drive(engine, make_workload(spec))
    summary = summarize(engine, requests, t0, t_end)
    queried, hits = (stats.query_tokens, stats.hit_tokens) if stats else (0, 0)
    queried, hits = queried - before[0], hits - before[1]
    summary["prefill_tokens"] = sum(s.num_prefill_tokens for s in engine.trace)
    summary["hit_rate"] = hits / queried if queried else 0.0
    return summary


def main() -> None:
    """Run the A/B benchmark and print (and optionally save) the medians."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-requests", type=int, default=120)
    parser.add_argument("--rate", type=float, default=8.0, help="mean arrivals per second")
    parser.add_argument("--system-prompts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--workloads", nargs="+", choices=["shared", "default"], default=["shared", "default"]
    )
    parser.add_argument("--out", type=Path, help="write all summaries here as JSON")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    common = {"num_requests": args.num_requests, "rate": args.rate, "seed": args.seed}
    specs = {
        "shared": WorkloadSpec.shared_prefix(args.system_prompts, **common),
        "default": WorkloadSpec(**common),
    }
    base = EngineConfig(strategy="continuous", kv_layout="paged", num_threads=args.threads)
    modes = {"off": base, "on": replace(base, enable_prefix_caching=True)}

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        w: {m: [] for m in modes} for w in args.workloads
    }
    for rep in range(args.repeats):
        for workload in args.workloads:
            order = list(modes) if rep % 2 == 0 else list(reversed(modes))
            for mode in order:
                cfg = modes[mode]
                s = run_once(cfg, specs[workload])
                runs[workload][mode].append(s)
                print(
                    f"[{rep + 1}/{args.repeats}] {workload:>7} caching {mode:>3}: "
                    f"{s['throughput_tok_s']:6.0f} tok/s | "
                    f"TTFT p50 {s['ttft_p50_s']:6.3f}s p95 {s['ttft_p95_s']:6.3f}s | "
                    f"latency p50 {s['latency_p50_s']:5.2f}s | "
                    f"prefill {s['prefill_tokens']:6d} tok | hit {s['hit_rate']:4.0%}",
                    flush=True,
                )

    for workload, by_mode in runs.items():
        print(f"\n{workload} workload, median of {args.repeats}:")
        print(f"{'metric':>18} {'off':>10} {'on':>10} {'change':>8}")
        for metric in METRICS:
            off, on = (statistics.median(r[metric] for r in by_mode[m]) for m in ("off", "on"))
            change = f"{on / off - 1:+.0%}" if off else ""
            print(f"{metric:>18} {off:10.3f} {on:10.3f} {change:>8}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"hardware": hardware_info(), "args": vars(args) | {"out": str(args.out)}}
        payload["specs"] = {w: asdict(specs[w]) for w in args.workloads}
        payload["runs"] = runs
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
