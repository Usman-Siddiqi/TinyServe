"""Benchmark speculative decoding against plain decoding, interleaved in one process.

For every (temperature, batch size) cell, each repeat runs the same batch of
requests through every engine configuration in rotating order, so slow phases
of a noisy machine hit all configurations alike. Reports median wall-clock
speedup over plain decoding, acceptance by draft position and tokens per
target pass, and checks that greedy outputs are identical to plain decoding.

Usage: ``python bench/speculative.py --temperatures 0 0.3 1.0 --batch-sizes 1 4 8``
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from tinyserve.engine import Engine, EngineConfig, build_engine
from tinyserve.request import Request, SamplingParams

METHODS = ("ngram", "early-exit", "distilled")
# "<method>" adapts the draft length per request; "<method>-fixed" always drafts k.
CONFIGS = ("plain", *METHODS, *(f"{m}-fixed" for m in METHODS))


def engine_config(base: EngineConfig, name: str) -> EngineConfig:
    """The engine configuration behind a benchmark config name."""
    if name == "plain":
        return base
    method = name.removesuffix("-fixed")
    return replace(base, speculative=method, adaptive_speculation=not name.endswith("-fixed"))


def make_batch(n: int, prompt_len: int, seed: int, vocab: int) -> list[list[int]]:
    """Random-character prompts."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(4, vocab, (prompt_len,), generator=g).tolist() for _ in range(n)]


def run_batch(
    engine: Engine, prompts: list[list[int]], params: list[SamplingParams]
) -> tuple[float, list[list[int]]]:
    """Wall time to finish all ``prompts`` together, and their outputs."""
    reqs = [Request(p, sp) for p, sp in zip(prompts, params, strict=True)]
    start = time.perf_counter()
    for r in reqs:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()
    return time.perf_counter() - start, [r.output_ids for r in reqs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS[:4]))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.3, 1.0])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--k", type=int, default=4, help="draft tokens per step")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None, help="append JSON results here")
    args = parser.parse_args()

    base = EngineConfig(
        strategy="continuous",
        kv_layout="paged",
        num_threads=args.threads,
        num_speculative_tokens=args.k,
    )
    configs = ["plain", *(c for c in args.configs if c != "plain")]
    engines = {name: build_engine(engine_config(base, name)) for name in configs}
    vocab = engines["plain"].model.cfg.vocab_size
    # Warm up every engine (first-call allocations, thread pools).
    for engine in engines.values():
        run_batch(engine, make_batch(2, 8, 0, vocab), [SamplingParams(8, ignore_eos=True)] * 2)

    rows = []
    for temperature in args.temperatures:
        for batch_size in args.batch_sizes:
            times: dict[str, list[float]] = {c: [] for c in configs}
            mismatches = 0
            for engine in engines.values():
                if engine.speculator is not None:
                    engine.speculator.stats = type(engine.speculator.stats)()
            for rep in range(args.repeats):
                prompts = make_batch(batch_size, args.prompt_len, 1000 + rep, vocab)
                params = [
                    SamplingParams(
                        args.max_new_tokens,
                        temperature=temperature,
                        ignore_eos=True,
                        seed=rep * 1000 + i,
                    )
                    for i in range(batch_size)
                ]
                order = configs[rep % len(configs) :] + configs[: rep % len(configs)]
                outputs = {}
                for name in order:
                    elapsed, outputs[name] = run_batch(engines[name], prompts, params)
                    times[name].append(elapsed)
                if temperature == 0:
                    mismatches += sum(outputs[c] != outputs["plain"] for c in configs)
            tokens = batch_size * args.max_new_tokens
            for name in configs:
                spec = engines[name].speculator
                speedups = [p / t for p, t in zip(times["plain"], times[name], strict=True)]
                row = {
                    "config": name,
                    "temperature": temperature,
                    "batch_size": batch_size,
                    "k": args.k,
                    "adaptive": bool(spec and spec.controller),
                    "median_s": statistics.median(times[name]),
                    "times_s": times[name],
                    "tok_per_s": tokens / statistics.median(times[name]),
                    "speedup": statistics.median(speedups),
                    "acceptance": spec.stats.acceptance_rate if spec else None,
                    "acceptance_by_pos": spec.stats.acceptance_by_position() if spec else None,
                    "tokens_per_verify": spec.stats.tokens_per_verification if spec else 1.0,
                    "greedy_mismatches": mismatches if temperature == 0 else None,
                }
                rows.append(row)
                by_pos = " ".join(f"{a:.2f}" for a in row["acceptance_by_pos"] or [])
                print(
                    f"T={temperature:<4} B={batch_size:<2} {name:17s} "
                    f"{row['median_s'] * 1000:7.0f} ms  {row['tok_per_s']:6.0f} tok/s  "
                    f"speedup {row['speedup']:.2f}x  tok/verify {row['tokens_per_verify']:.2f}  "
                    f"accept-by-pos [{by_pos}]",
                    flush=True,
                )
            if temperature == 0:
                print(f"  greedy outputs differing from plain: {mismatches}", flush=True)
    if args.out:
        existing = json.loads(args.out.read_text()) if args.out.exists() else []
        args.out.write_text(json.dumps(existing + rows, indent=1))


if __name__ == "__main__":
    main()
