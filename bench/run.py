"""Benchmark every serving strategy on the same Poisson workload and save raw results to JSON.

Usage::

    python bench/run.py                       # 200 requests, all strategies
    python bench/run.py --num-requests 50     # quicker smoke run

Each strategy gets a fresh engine with an identical model, an identical KV
memory pool and the identical list of requests. Requests are injected in real
time at their Poisson arrival times, so queueing delay counts toward latency
exactly as it would for real users.

The engine is driven in-process (no HTTP) so that the numbers measure the
serving system itself rather than JSON encoding and socket overhead.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tinyserve.engine import Engine, EngineConfig, build_engine
from tinyserve.loadgen import percentile
from tinyserve.request import Request, SamplingParams
from tinyserve.tokenizer import CharTokenizer
from tinyserve.workload import WorkloadItem, WorkloadSpec, make_workload

# name -> (strategy, kv layout). Same memory budget for all of them.
RUNS: dict[str, tuple[str, str]] = {
    "naive": ("naive", "contiguous"),
    "static": ("static", "contiguous"),
    "continuous": ("continuous", "contiguous"),
    "continuous+paged": ("continuous", "paged"),
}
MAX_TIMELINE_POINTS = 3000


def hardware_info() -> dict[str, Any]:
    """Describe the machine the benchmark ran on."""
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if "model name" in line)
    except (OSError, StopIteration):
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": commit,
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def warm_up(engine: Engine) -> None:
    """Run a few throwaway requests so one-time PyTorch setup is not timed."""
    tok = CharTokenizer()
    params = SamplingParams(max_new_tokens=8, ignore_eos=True)
    engine.generate([tok.encode("warm up " * k) for k in range(1, 9)], params)
    engine.trace.clear()


def drive(engine: Engine, items: list[WorkloadItem]) -> tuple[list[Request], float, float]:
    """Inject requests at their arrival times and step the engine until all finish."""
    tok = CharTokenizer()
    t0 = engine.clock()
    requests = [
        Request(
            tok.encode(it.prompt),
            SamplingParams(max_new_tokens=it.max_tokens, ignore_eos=True),
            arrival_time=t0 + it.arrival,
        )
        for it in items
    ]
    pending = list(requests)
    while pending or engine.has_unfinished():
        now = engine.clock()
        while pending and pending[0].arrival_time <= now:
            engine.add_request(pending.pop(0))
        if engine.has_unfinished():
            engine.step()
        elif pending:
            time.sleep(max(0.0, pending[0].arrival_time - engine.clock()))
    return requests, t0, engine.clock()


def summarize(engine: Engine, requests: list[Request], t0: float, t_end: float) -> dict[str, Any]:
    """Aggregate metrics for one run."""
    latencies = [r.latency for r in requests if r.latency is not None]
    ttfts = [r.ttft for r in requests if r.ttft is not None]
    tpots = [
        (r.finish_time - r.first_token_time) / (len(r.output_ids) - 1)
        for r in requests
        if r.finish_time is not None and r.first_token_time is not None and len(r.output_ids) > 1
    ]
    # Inter-token latency: every gap between consecutive tokens of one request,
    # pooled over all requests. Its tail shows decode stalls behind long prefills.
    itls = [b - a for r in requests for a, b in itertools.pairwise(r.token_times)]
    tokens = sum(len(r.output_ids) for r in requests)
    makespan = t_end - t0

    # Time-weighted averages over engine steps.
    trace = engine.trace
    durations = [s.end - s.start for s in trace]
    busy = sum(durations) or 1.0

    def tw(values: list[float]) -> float:
        return sum(v * d for v, d in zip(values, durations, strict=True)) / busy

    return {
        "num_requests": len(requests),
        "generated_tokens": tokens,
        "makespan_s": makespan,
        "throughput_tok_s": tokens / makespan,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_mean_s": sum(latencies) / len(latencies),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "tpot_p50_ms": percentile(tpots, 50) * 1000,
        "tpot_p95_ms": percentile(tpots, 95) * 1000,
        "tpot_p99_ms": percentile(tpots, 99) * 1000,
        "itl_p50_ms": percentile(itls, 50) * 1000,
        "itl_p95_ms": percentile(itls, 95) * 1000,
        "itl_p99_ms": percentile(itls, 99) * 1000,
        "itl_max_ms": max(itls, default=0.0) * 1000,
        "num_steps": len(trace),
        "mean_batch_size": tw([s.num_prefill + s.num_decode for s in trace]),
        "max_batch_size": max(s.num_prefill + s.num_decode for s in trace),
        "kv_utilization_mean": tw([s.kv.utilization for s in trace]),
        "kv_occupancy_mean": tw([s.kv.occupancy for s in trace]),
        "num_preemptions": engine.scheduler.num_preemptions,
    }


def timeline(engine: Engine, t0: float) -> list[dict[str, float]]:
    """Per-step series (downsampled) for plots: batch size, queue and KV memory over time."""
    trace = engine.trace
    stride = max(1, len(trace) // MAX_TIMELINE_POINTS)
    return [
        {
            "t": round(s.start - t0, 4),
            "running": s.num_prefill + s.num_decode,
            "waiting": s.num_waiting,
            "kv_used": s.kv.used_tokens,
            "kv_reserved": s.kv.reserved_tokens,
            "kv_capacity": s.kv.capacity_tokens,
        }
        for s in trace[::stride]
    ]


def per_request(requests: list[Request], t0: float) -> list[dict[str, float | int | None]]:
    """Raw per-request measurements."""

    def rel(x: float | None) -> float | None:
        return None if x is None else round(x - t0, 5)

    return [
        {
            "arrival": rel(r.arrival_time),
            "first_token": rel(r.first_token_time),
            "finish": rel(r.finish_time),
            "prompt_tokens": len(r.prompt_ids),
            "output_tokens": len(r.output_ids),
            "ttft": r.ttft,
            "latency": r.latency,
            "preemptions": r.num_preemptions,
        }
        for r in requests
    ]


def main() -> None:
    """Run the benchmark and write the JSON results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--rate", type=float, default=40.0, help="mean arrivals per second")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kv-capacity-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3, help="runs per strategy")
    parser.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS))
    parser.add_argument("--out", type=Path, default=Path("bench/results/results.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    spec = WorkloadSpec(num_requests=args.num_requests, rate=args.rate, seed=args.seed)
    items = make_workload(spec)
    base = EngineConfig(
        kv_capacity_tokens=args.kv_capacity_tokens,
        max_batch_size=args.max_batch_size,
        num_threads=args.threads,
    )

    # Repeats are interleaved (A B C D, A B C D, ...) so slow drift in the machine's
    # speed affects every strategy equally. The reported summary is the run with
    # the median throughput, and every repeat's summary is kept for transparency.
    runs: dict[str, list[dict[str, Any]]] = {name: [] for name in args.runs}
    for rep in range(args.repeats):
        for name in args.runs:
            strategy, layout = RUNS[name]
            cfg = replace(base, strategy=strategy, kv_layout=layout)
            engine = build_engine(cfg, record_trace=True)
            warm_up(engine)
            requests, t0, t_end = drive(engine, items)
            summary = summarize(engine, requests, t0, t_end)
            print(
                f"[{rep + 1}/{args.repeats}] {name:>16}: "
                f"{summary['throughput_tok_s']:6.0f} tok/s | "
                f"p50 latency {summary['latency_p50_s']:5.2f}s | "
                f"p50 TTFT {summary['ttft_p50_s']:6.3f}s | "
                f"KV util {summary['kv_utilization_mean']:4.0%} | "
                f"mean batch {summary['mean_batch_size']:4.1f}",
                flush=True,
            )
            runs[name].append(
                {
                    "summary": summary,
                    "timeline": timeline(engine, t0),
                    "requests": per_request(requests, t0),
                }
            )

    results: dict[str, Any] = {}
    for name, reps in runs.items():
        reps_sorted = sorted(reps, key=lambda r: r["summary"]["throughput_tok_s"])
        median = reps_sorted[len(reps_sorted) // 2]
        strategy, layout = RUNS[name]
        results[name] = {
            "strategy": strategy,
            "kv_layout": layout,
            **median,
            "repeat_summaries": [r["summary"] for r in reps],
        }

    out = {
        "hardware": hardware_info(),
        "engine_config": {
            "kv_capacity_tokens": base.kv_capacity_tokens,
            "max_batch_size": base.max_batch_size,
            "max_tokens_per_step": base.max_tokens_per_step,
            "chunked_prefill": base.chunked_prefill,
            "block_size": base.block_size,
            "model": asdict(base.model),
            "kv_bytes_per_token": base.model.kv_bytes_per_token(),
        },
        "workload": asdict(spec),
        "repeats": args.repeats,
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
