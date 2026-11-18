"""Load sweep: goodput and SLO attainment of each queue policy, from light load to overload.

Usage::

    python bench/sweep.py --policies fcfs sjf             # real runs
    python bench/sweep.py --policies edf slo             # appends to the same JSON
    python bench/sweep.py --mode sim --out bench/results/sweep_sim.json \\
        --policies fcfs sjf edf slo --num-requests 1000 --seeds 0 1 2
    python bench/plot.py                                  # docs/assets/goodput.png

Every point uses continuous batching with a paged KV cache. A request meets its
SLO if its TTFT and its time per output token (TPOT) are both within target;
goodput is the number of such requests per second.

Load is given as a fraction of capacity. The first run on a fresh JSON file
calibrates: it serves a burst of requests for real, which measures capacity
(requests/s) and fits a :class:`~tinyserve.simulator.StepCostModel` to the
step timings. Later invocations reuse that calibration, so a long sweep can be
split into short runs that append to the same file.

``--mode real`` injects requests in real time into the real engine.
``--mode sim`` replays them through the same scheduler on a virtual clock
charged by the cost model: much faster, deterministic, and not disturbed by
other processes, but only as good as the cost model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from run import drive, hardware_info, summarize, warm_up

from tinyserve.engine import EngineConfig, build_engine
from tinyserve.kv_cache import KVPool, PagedKVCache
from tinyserve.loadgen import percentile
from tinyserve.policy import POLICIES, make_policy
from tinyserve.request import Request
from tinyserve.scheduler import ContinuousScheduler
from tinyserve.simulator import StepCostModel, simulate
from tinyserve.workload import WorkloadItem, WorkloadSpec, make_workload


def tpot(r: Request) -> float | None:
    """Mean time per output token after the first, or None with fewer than two tokens."""
    if r.finish_time is None or r.first_token_time is None or len(r.output_ids) < 2:
        return None
    return (r.finish_time - r.first_token_time) / (len(r.output_ids) - 1)


def meets_slo(r: Request, ttft_slo: float, tpot_slo: float) -> bool:
    """True if ``r`` finished with TTFT and TPOT within target."""
    t = tpot(r)
    return r.ttft is not None and r.ttft <= ttft_slo and (t is None or t <= tpot_slo)


def slo_metrics(
    requests: list[Request], makespan: float, ttft_slo: float, tpot_slo: float
) -> dict[str, float]:
    """Goodput, attainment and latency percentiles, overall and for the longest 10% of jobs."""
    ttfts = [r.ttft for r in requests if r.ttft is not None]
    lats = [r.latency for r in requests if r.latency is not None]
    tpots = [t for r in requests if (t := tpot(r)) is not None]
    met = sum(meets_slo(r, ttft_slo, tpot_slo) for r in requests)
    longest = sorted(requests, key=lambda r: r.max_length)[-max(1, len(requests) // 10) :]
    long_lats = [r.latency for r in longest if r.latency is not None]
    out = {
        "goodput_rps": met / makespan,
        "slo_attainment": met / len(requests),
        "ttft_attainment": sum(t <= ttft_slo for t in ttfts) / len(requests),
        "tpot_attainment": sum(t <= tpot_slo for t in tpots) / max(1, len(tpots)),
        "throughput_tok_s": sum(len(r.output_ids) for r in requests) / makespan,
        "tpot_p50_ms": percentile(tpots, 50) * 1000,
        "tpot_p99_ms": percentile(tpots, 99) * 1000,
        "long10_latency_p50_s": percentile(long_lats, 50),
        "long10_latency_max_s": max(long_lats),
        "num_preemptions": sum(r.num_preemptions for r in requests),
    }
    for q in (50, 95, 99):
        out[f"ttft_p{q}_s"] = percentile(ttfts, q)
        out[f"latency_p{q}_s"] = percentile(lats, q)
    return out


def calibrate(base: EngineConfig, num_requests: int, seed: int) -> dict[str, Any]:
    """Serve a burst of requests for real: measure capacity and fit the step cost model."""
    engine = build_engine(base, record_trace=True)
    warm_up(engine)
    burst = [replace(it, arrival=0.0) for it in make_workload(workload(num_requests, 1.0, seed))]
    requests, t0, t_end = drive(engine, burst)
    cost = StepCostModel.fit(engine.trace)
    return {
        "capacity_rps": len(requests) / (t_end - t0),
        "capacity_tok_s": sum(len(r.output_ids) for r in requests) / (t_end - t0),
        "cost_model": asdict(cost),
        "num_steps": len(engine.trace),
    }


def workload(num_requests: int, rate: float, seed: int) -> WorkloadSpec:
    """The sweep's workload: same seed means same prompts and lengths at every rate."""
    return WorkloadSpec(num_requests=num_requests, rate=rate, seed=seed)


def scheduler_for(base: EngineConfig, policy: str) -> ContinuousScheduler:
    """A scheduler like ``build_engine`` would make, without the model (for simulation)."""
    m = base.model
    pool = KVPool(base.kv_capacity_tokens, m.n_layers, m.n_kv_heads, m.head_dim)
    return ContinuousScheduler(
        PagedKVCache(pool, base.block_size),
        base.max_batch_size,
        base.max_tokens_per_step,
        make_policy(policy, base.ttft_slo_s, base.sjf_aging),
    )


def run_point(
    mode: str,
    base: EngineConfig,
    policy: str,
    items: list[WorkloadItem],
    cost: StepCostModel,
    slos: tuple[float, float],
) -> dict[str, Any]:
    """Serve ``items`` with one policy and return its metrics (plus raw data for real runs)."""
    if mode == "sim":
        requests, makespan = simulate(scheduler_for(base, policy), items, cost)
        return {"summary": slo_metrics(requests, makespan, *slos)}
    engine = build_engine(replace(base, queue_policy=policy), record_trace=True)
    warm_up(engine)
    requests, t0, t_end = drive(engine, items)
    summary = {**summarize(engine, requests, t0, t_end), **slo_metrics(requests, t_end - t0, *slos)}
    raw = [
        {
            "prompt_tokens": len(r.prompt_ids),
            "output_tokens": len(r.output_ids),
            "ttft": r.ttft,
            "latency": r.latency,
            "tpot": tpot(r),
            "preemptions": r.num_preemptions,
        }
        for r in requests
    ]
    # Replay the run in the simulator, charged by its own step timings, to check the simulator.
    fitted = StepCostModel.fit(engine.trace)
    sim_requests, sim_makespan = simulate(scheduler_for(base, policy), items, fitted)
    return {
        "summary": summary,
        "sim_summary": slo_metrics(sim_requests, sim_makespan, *slos),
        "cost_model": asdict(fitted),
        "requests": raw,
    }


def main() -> None:
    """Run the sweep and append its points to the JSON file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["real", "sim"], default="real")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=["fcfs", "sjf"])
    # Offered load as a multiple of the calibrated capacity.
    parser.add_argument("--loads", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ttft-slo", type=float, default=1.0, help="seconds")
    parser.add_argument("--tpot-slo", type=float, default=0.05, help="seconds")
    parser.add_argument("--sjf-aging", type=float, default=100.0, help="tokens per second")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--calibration-requests", type=int, default=100)
    parser.add_argument("--calibration", type=Path, help="reuse the calibration of this JSON")
    parser.add_argument("--out", type=Path, default=Path("bench/results/sweep.json"))
    parser.add_argument("--fresh", action="store_true", help="overwrite instead of appending")
    args = parser.parse_args()

    torch.manual_seed(0)
    base = EngineConfig(
        strategy="continuous",
        kv_layout="paged",
        ttft_slo_s=args.ttft_slo,
        sjf_aging=args.sjf_aging,
        num_threads=args.threads,
    )
    config = {
        "ttft_slo_s": args.ttft_slo,
        "tpot_slo_s": args.tpot_slo,
        "sjf_aging": args.sjf_aging,
        "max_batch_size": base.max_batch_size,
        "kv_capacity_tokens": base.kv_capacity_tokens,
        "threads": args.threads,
    }
    if args.out.exists() and not args.fresh:
        data = json.loads(args.out.read_text())
        if data["config"] != config:
            sys.exit(f"{args.out} was made with another config; pass --fresh or another --out")
    else:
        data = {"config": config, "points": []}
        if args.calibration:
            data["calibration"] = json.loads(args.calibration.read_text())["calibration"]
        else:
            print("calibrating...", flush=True)
            data["calibration"] = calibrate(base, args.calibration_requests, seed=0)
    data["hardware"] = hardware_info()
    cal = data["calibration"]
    cost = StepCostModel(**cal["cost_model"])
    print(f"capacity {cal['capacity_rps']:.2f} req/s, cost model {cost}", flush=True)

    slos = (args.ttft_slo, args.tpot_slo)
    for rep in range(args.repeats):
        for seed in args.seeds:
            for i, load in enumerate(args.loads):
                rate = load * cal["capacity_rps"]
                items = make_workload(workload(args.num_requests, rate, seed))
                # Alternate which policy goes first so slow drift hits them equally.
                order = args.policies if (rep + i) % 2 == 0 else args.policies[::-1]
                for policy in order:
                    point = run_point(args.mode, base, policy, items, cost, slos)
                    s = point["summary"]
                    print(
                        f"[{args.mode} {rep + 1}/{args.repeats} seed {seed}] load {load:4.2f} "
                        f"({rate:5.2f}/s) {policy:>10}: goodput {s['goodput_rps']:5.2f}/s | "
                        f"SLO met {s['slo_attainment']:4.0%} | "
                        f"TTFT p50/p99 {s['ttft_p50_s']:6.2f}/{s['ttft_p99_s']:6.2f}s | "
                        f"latency p50/p99 {s['latency_p50_s']:6.2f}/{s['latency_p99_s']:6.2f}s"
                        f" | long10 p50 {s['long10_latency_p50_s']:6.2f}s",
                        flush=True,
                    )
                    data["points"].append(
                        {
                            "mode": args.mode,
                            "policy": policy,
                            "load": load,
                            "rate": rate,
                            "seed": seed,
                            "num_requests": args.num_requests,
                            **point,
                        }
                    )
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(data, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
