"""An async load generator that fires a Poisson workload at a running TinyServe server.

Usage::

    tinyserve-loadgen --url http://127.0.0.1:8000 --num-requests 50 --rate 5

Each request is sent at its scheduled arrival time (whether or not earlier ones
have finished, as real users would) and its SSE stream is read to measure
time-to-first-token and end-to-end latency from the client's point of view.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass

import httpx

from tinyserve.workload import WorkloadItem, WorkloadSpec, make_workload


@dataclass
class ClientResult:
    """What the client observed for one request."""

    arrival: float
    prompt_len: int
    max_tokens: int
    num_tokens: int = 0
    ttft: float | None = None
    latency: float | None = None
    error: str | None = None


async def send_one(
    client: httpx.AsyncClient, url: str, item: WorkloadItem, t0: float
) -> ClientResult:
    """Wait until ``item.arrival``, send it, and time the streamed response."""
    await asyncio.sleep(max(0.0, t0 + item.arrival - time.perf_counter()))
    result = ClientResult(item.arrival, len(item.prompt), item.max_tokens)
    body = {"prompt": item.prompt, "max_tokens": item.max_tokens, "ignore_eos": True}
    start = time.perf_counter()
    try:
        async with client.stream("POST", f"{url}/generate", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("finished"):
                    break
                if result.ttft is None:
                    result.ttft = time.perf_counter() - start
                result.num_tokens += 1
        result.latency = time.perf_counter() - start
    except httpx.HTTPError as err:
        result.error = repr(err)
    return result


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[ClientResult], wall: float) -> dict[str, float]:
    """Throughput and latency percentiles over the successful requests."""
    ok = [r for r in results if r.error is None and r.latency is not None]
    lat = [r.latency for r in ok if r.latency is not None]
    ttft = [r.ttft for r in ok if r.ttft is not None]
    tokens = sum(r.num_tokens for r in ok)
    return {
        "requests_ok": len(ok),
        "requests_failed": len(results) - len(ok),
        "wall_time_s": wall,
        "throughput_tok_s": tokens / wall if wall else 0.0,
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p95_s": percentile(ttft, 95),
        "ttft_mean_s": statistics.fmean(ttft) if ttft else float("nan"),
    }


async def run(url: str, items: list[WorkloadItem]) -> tuple[list[ClientResult], float]:
    """Fire every item at its arrival time and wait for all of them."""
    limits = httpx.Limits(max_connections=len(items) + 1)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(send_one(client, url, it, t0) for it in items))
        return list(results), time.perf_counter() - t0


def main() -> None:
    """Command-line entry point: ``tinyserve-loadgen``."""
    parser = argparse.ArgumentParser(description="Send a Poisson workload to a TinyServe server.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--rate", type=float, default=5.0, help="mean requests per second")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="json_path", help="write per-request results here")
    args = parser.parse_args()

    spec = WorkloadSpec(num_requests=args.num_requests, rate=args.rate, seed=args.seed)
    results, wall = asyncio.run(run(args.url.rstrip("/"), make_workload(spec)))
    summary = summarize(results, wall)
    for key, value in summary.items():
        print(f"{key:>18}: {value:.3f}" if isinstance(value, float) else f"{key:>18}: {value}")
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump({"summary": summary, "requests": [asdict(r) for r in results]}, f, indent=2)


if __name__ == "__main__":
    main()
