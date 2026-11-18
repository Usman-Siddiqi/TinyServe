"""Compare chunked and unchunked prefill on the default and a prompt-heavy workload.

Usage::

    python bench/chunked_prefill.py                    # 3 interleaved repeats
    python bench/chunked_prefill.py --budgets 128 256 512 --repeats 5

Every configuration is continuous batching with a paged KV cache; only the
prefill policy and the per-step token budget change. Repeats are interleaved so
machine drift hits every configuration alike, and medians are reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from run import drive, summarize, warm_up

from tinyserve.engine import EngineConfig, build_engine
from tinyserve.workload import WorkloadSpec, make_workload

METRICS = (
    "throughput_tok_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "itl_p50_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "itl_max_ms",
    "tpot_p50_ms",
    "num_preemptions",
)


def main() -> None:
    """Run the comparison and print a table of medians."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--rate", type=float, default=40.0, help="default workload, req/s")
    parser.add_argument("--heavy-rate", type=float, default=16.0, help="prompt-heavy, req/s")
    parser.add_argument("--long-fraction", type=float, default=0.3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output")
    args = parser.parse_args()

    workloads = {
        "default": WorkloadSpec(num_requests=args.num_requests, rate=args.rate),
        "prompt-heavy": WorkloadSpec(
            num_requests=args.num_requests,
            rate=args.heavy_rate,
            long_prompt_fraction=args.long_fraction,
        ),
    }
    base = EngineConfig(strategy="continuous", kv_layout="paged", num_threads=args.threads)
    configs = {"unchunked": base} | {
        f"chunked-{b}": replace(base, chunked_prefill=True, max_tokens_per_step=b)
        for b in args.budgets
    }

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        w: {c: [] for c in configs} for w in workloads
    }
    for rep in range(args.repeats):
        for wname, spec in workloads.items():
            items = make_workload(spec)
            for cname, cfg in configs.items():
                engine = build_engine(cfg, record_trace=True)
                warm_up(engine)
                summary = summarize(engine, *drive(engine, items))
                runs[wname][cname].append(summary)
                print(
                    f"[{rep + 1}/{args.repeats}] {wname:>12} {cname:>12}: "
                    f"{summary['throughput_tok_s']:5.0f} tok/s | "
                    f"TTFT p50 {summary['ttft_p50_s'] * 1000:6.0f} ms | "
                    f"ITL p99 {summary['itl_p99_ms']:6.1f} ms | "
                    f"max stall {summary['itl_max_ms']:6.0f} ms",
                    flush=True,
                )

    for wname, by_config in runs.items():
        print(f"\n{wname} (median of {args.repeats}; change vs unchunked)")
        ref = {m: statistics.median(s[m] for s in by_config["unchunked"]) for m in METRICS}
        print(f"{'metric':>18} " + " ".join(f"{c:>22}" for c in by_config))
        for m in METRICS:
            cells = []
            for summaries in by_config.values():
                med = statistics.median(s[m] for s in summaries)
                rel = f"({med / ref[m] - 1:+.0%})" if ref[m] else ""
                cells.append(f"{med:>14.3f} {rel:>7}")
            print(f"{m:>18} " + " ".join(cells))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"args": vars(args) | {"out": str(args.out)}, "runs": runs}, indent=1)
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
