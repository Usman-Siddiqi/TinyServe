"""Compare chunked and unchunked prefill on Poisson workloads and on a prompt burst.

Usage::

    python bench/chunked_prefill.py                    # 3 interleaved repeats
    python bench/chunked_prefill.py --budgets 128 256 512 --repeats 5

Every configuration is continuous batching with a paged KV cache; only the
prefill policy and the per-step token budget change. Three scenarios run:

* ``default``: the workload of ``bench/run.py``.
* ``prompt-heavy``: the same, with a fraction of 300-450 character prompts.
* ``burst``: 16 requests are decoding when 6 long prompts arrive at once. It
  isolates the stall that chunking removes and is the least noisy of the three.

Repeats are interleaved, with the order rotated each time, and medians are reported.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from run import drive, summarize, warm_up

from tinyserve.engine import EngineConfig, build_engine
from tinyserve.loadgen import percentile
from tinyserve.request import Request, SamplingParams
from tinyserve.workload import WorkloadSpec, make_workload

WORKLOAD_METRICS = (
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


def run_burst(cfg: EngineConfig) -> dict[str, float]:
    """Decode 16 short requests, then drop 6 long prompts on them at once."""
    engine = build_engine(cfg)
    warm_up(engine)
    decoders = [
        Request([5 + i] * 32, SamplingParams(max_new_tokens=120, ignore_eos=True))
        for i in range(16)
    ]
    for r in decoders:
        engine.add_request(r)
    for _ in range(10):
        engine.step()
    burst_at = engine.clock()
    burst = [
        Request([6 + i] * 420, SamplingParams(max_new_tokens=20, ignore_eos=True)) for i in range(6)
    ]
    for r in burst:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()
    # Only the gaps the decoders saw after the burst arrived.
    itls = [b - a for r in decoders for a, b in itertools.pairwise(r.token_times) if b >= burst_at]
    ttfts = [r.ttft for r in burst if r.ttft is not None]
    return {
        "decode_itl_p50_ms": percentile(itls, 50) * 1000,
        "decode_itl_p99_ms": percentile(itls, 99) * 1000,
        "decode_itl_max_ms": max(itls) * 1000,
        "burst_ttft_p50_s": percentile(ttfts, 50),
        "burst_ttft_max_s": max(ttfts),
        "drain_time_s": engine.clock() - burst_at,
    }


def print_table(title: str, by_config: dict[str, list[dict[str, Any]]], metrics: list[str]) -> None:
    """Median of every metric per configuration, with the change vs. unchunked."""
    n = len(next(iter(by_config.values())))
    print(f"\n{title} (median of {n}; change vs unchunked)")
    print(f"{'metric':>18} " + " ".join(f"{c:>22}" for c in by_config))
    for m in metrics:
        ref = statistics.median(s[m] for s in by_config["unchunked"])
        cells = []
        for summaries in by_config.values():
            med = statistics.median(s[m] for s in summaries)
            rel = f"({med / ref - 1:+.0%})" if ref else ""
            cells.append(f"{med:>14.3f} {rel:>7}")
        print(f"{m:>18} " + " ".join(cells))


def main() -> None:
    """Run every scenario and print tables of medians."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--rate", type=float, default=40.0, help="default workload, req/s")
    parser.add_argument("--heavy-rate", type=float, default=16.0, help="prompt-heavy, req/s")
    parser.add_argument("--long-fraction", type=float, default=0.3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512])
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
    scenarios = [*workloads, "burst"]
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        s: {c: [] for c in configs} for s in scenarios
    }

    names = list(configs)
    for rep in range(args.repeats):
        order = names[rep % len(names) :] + names[: rep % len(names)]
        for scenario in scenarios:
            for name in order:
                if scenario == "burst":
                    summary = run_burst(configs[name])
                    keys = list(summary)
                else:
                    engine = build_engine(configs[name], record_trace=True)
                    warm_up(engine)
                    items = make_workload(workloads[scenario])
                    summary = summarize(engine, *drive(engine, items))
                    keys = ["throughput_tok_s", "ttft_p50_s", "itl_p99_ms", "itl_max_ms"]
                runs[scenario][name].append(summary)
                shown = ", ".join(f"{k} {summary[k]:.3g}" for k in keys)
                print(f"[{rep + 1}/{args.repeats}] {scenario} {name}: {shown}", flush=True)

    for scenario in workloads:
        print_table(scenario, runs[scenario], list(WORKLOAD_METRICS))
    print_table("burst", runs["burst"], list(runs["burst"]["unchunked"][0]))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: str(v) for k, v in vars(args).items()}
        args.out.write_text(json.dumps({"args": meta, "runs": runs}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
