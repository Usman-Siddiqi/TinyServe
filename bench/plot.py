"""Turn the benchmark JSON into the README charts and results table.

Usage::

    python bench/plot.py                       # reads bench/results/results.json

Every number in every chart comes from the JSON written by ``bench/run.py``.
Nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Entity -> color is fixed, so a strategy has the same color in every chart.
ORDER = ["naive", "static", "continuous", "continuous+paged"]
COLORS = {
    "continuous+paged": "#2a78d6",
    "continuous": "#eb6834",
    "static": "#1baf7a",
    "naive": "#eda100",
}
LABELS = {
    "naive": "Naive (taxi)",
    "static": "Static batching (bus)",
    "continuous": "Continuous batching",
    "continuous+paged": "Continuous + paged KV",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def style() -> None:
    """A quiet, consistent house style for every chart."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "font.size": 11,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def subtitle(fig: plt.Figure, text: str) -> None:
    fig.text(0.012, 0.005, text, color=INK_2, fontsize=9, ha="left", va="bottom")


def plot_throughput(data: dict[str, Any], out: Path) -> None:
    """Bar chart of generated tokens per second, labelled with the multiplier over naive."""
    res = data["results"]
    names = [n for n in ORDER if n in res]
    tput = [res[n]["summary"]["throughput_tok_s"] for n in names]
    base = res["naive"]["summary"]["throughput_tok_s"] if "naive" in res else None

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(
        [LABELS[n] for n in names],
        tput,
        color=[COLORS[n] for n in names],
        width=0.6,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for bar, value in zip(bars, tput, strict=True):
        label = f"{value:,.0f}"
        if base:
            label += f"\n{value / base:.1f}x"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
        )
    ax.set_ylabel("generated tokens / second")
    ax.set_ylim(0, max(tput) * 1.25)
    ax.grid(axis="x", visible=False)
    ax.set_title("Throughput: same model, same memory, same 200 requests")
    w = data["workload"]
    subtitle(
        fig,
        f"{w['num_requests']} requests, Poisson arrivals at {w['rate']:.0f}/s, "
        f"median of {data.get('repeats', 1)} runs. Multiplier is vs. naive.",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_latency_cdf(data: dict[str, Any], out: Path) -> None:
    """CDFs of end-to-end latency and time-to-first-token, one line per strategy."""
    res = data["results"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, title in (
        (axes[0], "latency", "End-to-end latency"),
        (axes[1], "ttft", "Time to first token"),
    ):
        for name in ORDER:
            if name not in res:
                continue
            values = np.sort([r[key] for r in res[name]["requests"] if r[key] is not None])
            frac = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, frac, color=COLORS[name], label=LABELS[name])
        ax.set_xscale("log")
        ax.set_xlabel("seconds (log scale)")
        ax.set_title(title)
        ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("fraction of requests")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kv_memory(data: dict[str, Any], out: Path) -> None:
    """Reserved vs actually used KV memory over time, contiguous vs paged."""
    res = data["results"]
    panels = [n for n in ("continuous", "continuous+paged") if n in res]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.2), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], panels, strict=True):
        tl = res[name]["timeline"]
        t = np.array([p["t"] for p in tl])
        used = np.array([p["kv_used"] for p in tl])
        reserved = np.array([p["kv_reserved"] for p in tl])
        capacity = tl[0]["kv_capacity"]
        ax.fill_between(
            t,
            0,
            reserved,
            color=COLORS[name],
            alpha=0.25,
            linewidth=0,
            label="reserved",
            step="post",
        )
        ax.fill_between(
            t,
            0,
            used,
            color=COLORS[name],
            alpha=0.9,
            linewidth=0,
            label="holding real tokens",
            step="post",
        )
        ax.axhline(capacity, color=INK_2, linewidth=1, linestyle="--")
        ax.annotate(
            "pool capacity",
            (t[-1], capacity),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=INK_2,
        )
        util = res[name]["summary"]["kv_utilization_mean"]
        ax.set_title(f"{LABELS[name]}: {util:.0%} of reserved memory used")
        ax.set_xlabel("seconds since first arrival")
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.93))
    axes[0][0].set_ylabel("KV cache (token slots)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_batch_size(data: dict[str, Any], out: Path) -> None:
    """Requests in the running batch at each step: the sawtooth of static batching."""
    res = data["results"]
    names = [n for n in ("static", "continuous", "continuous+paged") if n in res]
    fig, axes = plt.subplots(
        len(names), 1, figsize=(10, 1.9 * len(names) + 0.6), sharex=True, sharey=True, squeeze=False
    )
    max_bs = data["engine_config"]["max_batch_size"]
    for ax, name in zip(axes[:, 0], names, strict=True):
        tl = res[name]["timeline"]
        t = [p["t"] for p in tl]
        running = [p["running"] for p in tl]
        ax.fill_between(t, 0, running, step="post", color=COLORS[name], alpha=0.85, linewidth=0)
        ax.set_ylim(0, max_bs * 1.1)
        mean = res[name]["summary"]["mean_batch_size"]
        ax.set_title(f"{LABELS[name]}  (mean {mean:.1f} requests per step)", fontsize=11)
        ax.set_ylabel("batch")
    axes[-1, 0].set_xlabel("seconds since first arrival")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def results_table(data: dict[str, Any]) -> str:
    """Markdown table of the headline numbers."""
    res = data["results"]
    base = res.get("naive", {}).get("summary", {}).get("throughput_tok_s")
    rows = [
        "| Strategy | Throughput (tok/s) | vs. naive | p50 latency | p95 latency "
        "| p50 TTFT | p95 TTFT | Mean batch | KV utilization |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        if name not in res:
            continue
        s = res[name]["summary"]
        mult = f"{s['throughput_tok_s'] / base:.1f}x" if base else "-"
        rows.append(
            f"| {LABELS[name]} | {s['throughput_tok_s']:,.0f} | {mult} "
            f"| {s['latency_p50_s']:.2f} s | {s['latency_p95_s']:.2f} s "
            f"| {s['ttft_p50_s'] * 1000:,.0f} ms | {s['ttft_p95_s'] * 1000:,.0f} ms "
            f"| {s['mean_batch_size']:.1f} | {s['kv_utilization_mean']:.0%} |"
        )
    hw = data["hardware"]
    rows.append("")
    rows.append(
        f"Hardware: {hw['cpu']}, {hw['logical_cores']} logical cores, "
        f"{hw['torch_threads']} PyTorch threads, PyTorch {hw['torch_version']}, "
        f"Python {hw['python_version']}. Commit `{hw['git_commit']}`, {hw['date_utc']}."
    )
    if all("repeat_summaries" in res[n] for n in res):
        spread = ", ".join(
            f"{LABELS[n]} "
            + "/".join(f"{r['throughput_tok_s']:,.0f}" for r in res[n]["repeat_summaries"])
            for n in ORDER
            if n in res
        )
        rows.append("")
        rows.append(f"Throughput of each repeat (tok/s): {spread}.")
    return "\n".join(rows) + "\n"


def main() -> None:
    """Render all charts and the results table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("bench/results/results.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    plot_throughput(data, args.out_dir / "throughput.png")
    plot_latency_cdf(data, args.out_dir / "latency_cdf.png")
    plot_kv_memory(data, args.out_dir / "kv_memory.png")
    plot_batch_size(data, args.out_dir / "batch_size.png")
    table = results_table(data)
    (args.results.parent / "results.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
