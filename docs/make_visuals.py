"""Generate the animated GIFs in docs/assets from real engine runs.

Usage::

    python docs/make_visuals.py

Both animations replay traces recorded from the actual TinyServe engine (the
same schedulers and KV cache managers the server uses) on a small workload, so
what you see is what the code does, step by step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle

from tinyserve.engine import Engine
from tinyserve.kv_cache import ContiguousKVCache, KVPool, PagedKVCache
from tinyserve.model import ModelConfig, TinyGPT
from tinyserve.request import Request, SamplingParams
from tinyserve.scheduler import ContinuousScheduler, Scheduler, StaticScheduler

OUT = Path(__file__).parent / "assets"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
IDLE = "#e4e3df"
PREFILL = "#184f95"
DECODE = "#5598e7"
FREE = "#efeeea"
USED = "#2a78d6"
WASTE = "#f6c7b3"
WASTE_EDGE = "#eb6834"


def workload(n: int, seed: int, prompt: tuple[int, int], out: tuple[int, int]) -> list[Request]:
    rng = random.Random(seed)
    return [
        Request(
            [rng.randint(4, 100) for _ in range(rng.randint(*prompt))],
            SamplingParams(max_new_tokens=rng.randint(*out), ignore_eos=True),
            arrival_time=0.0,
        )
        for _ in range(n)
    ]


def run(model: TinyGPT, scheduler: Scheduler, requests: list[Request], on_step=None) -> Engine:
    engine = Engine(model, scheduler, record_trace=True)
    for r in requests:
        engine.add_request(r)
    while engine.has_unfinished():
        engine.step()
        if on_step is not None:
            on_step(engine)
    return engine


# --- batching: static vs continuous --------------------------------------------------


@dataclass
class Segment:
    seat: int
    label: int
    start: int  # first step (inclusive)
    end: int  # last step (exclusive)
    prefill_end: int  # steps < prefill_end were prefill


def segments(engine: Engine) -> list[Segment]:
    """Collapse the per-step seat map into one bar per (seat, request)."""
    segs: dict[tuple[int, int], Segment] = {}
    for step, t in enumerate(engine.trace):
        for seat, (rid, phase) in t.slots.items():
            key = (seat, rid)
            if key not in segs:
                segs[key] = Segment(seat, rid, step, step + 1, step + 1)
            segs[key].end = step + 1
            if phase == "prefill":
                segs[key].prefill_end = step + 1
    return list(segs.values())


def make_batching_gif(model: TinyGPT) -> None:
    seats, n = 6, 20
    kinds = {
        "Static batching: the bus waits for everyone": StaticScheduler,
        "Continuous batching: a shuttle, every stop": ContinuousScheduler,
    }
    panels = []
    for title, cls in kinds.items():
        reqs = workload(n, seed=7, prompt=(8, 40), out=(3, 40))
        first_id = min(r.request_id for r in reqs)
        pool = KVPool(seats * 512, 4, model.cfg.n_kv_heads, model.cfg.head_dim)
        engine = run(model, cls(ContiguousKVCache(pool, 512), seats), reqs)
        segs = segments(engine)
        for s in segs:
            s.label -= first_id
        busy = [[seat in t.slots for seat in range(seats)] for t in engine.trace]
        panels.append((title, segs, busy, engine.num_steps))

    horizon = max(p[3] for p in panels)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), facecolor=SURFACE)
    frames = list(range(0, horizon + 1, max(1, horizon // 90))) + [horizon] * 12

    def draw(frame: int) -> None:
        for ax, (title, segs, busy, total) in zip(axes, panels, strict=True):
            ax.clear()
            ax.set_facecolor(SURFACE)
            now = min(frame, total)
            # Empty seats while the engine is busy: wasted capacity.
            for step in range(now):
                for seat in range(seats):
                    if not busy[step][seat]:
                        ax.add_patch(Rectangle((step, seat + 0.12), 1, 0.76, color=IDLE, lw=0))
            for s in segs:
                if s.start >= now:
                    continue
                end = min(s.end, now)
                ax.add_patch(
                    FancyBboxPatch(
                        (s.start + 0.15, s.seat + 0.12),
                        end - s.start - 0.3,
                        0.76,
                        boxstyle="round,pad=0,rounding_size=0.25",
                        color=DECODE,
                        lw=0,
                    )
                )
                pf = min(s.prefill_end, end)
                ax.add_patch(
                    Rectangle(
                        (s.start + 0.15, s.seat + 0.12),
                        pf - s.start - 0.15,
                        0.76,
                        color=PREFILL,
                        lw=0,
                    )
                )
                if end - s.start >= 3:
                    ax.text(
                        s.start + 1.2,
                        s.seat + 0.5,
                        f"#{s.label}",
                        va="center",
                        ha="left",
                        fontsize=7,
                        color="white",
                    )
            done_steps = f"step {now}" + (f" - all {n} requests done" if now >= total else "")
            ax.set_title(
                f"{title}    {done_steps}", loc="left", fontsize=11, fontweight="bold", color=INK
            )
            ax.set_xlim(0, horizon + 1)
            ax.set_ylim(seats, 0)
            ax.set_yticks([i + 0.5 for i in range(seats)], [f"seat {i}" for i in range(seats)])
            ax.tick_params(colors=INK_2, labelsize=8, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if now < total:
                ax.axvline(now, color=INK_2, lw=1)
        axes[-1].set_xlabel("engine step", color=INK_2, fontsize=9)
        handles = [
            Rectangle((0, 0), 1, 1, color=PREFILL),
            Rectangle((0, 0), 1, 1, color=DECODE),
            Rectangle((0, 0), 1, 1, color=IDLE),
        ]
        axes[0].legend(
            handles,
            ["prefill", "generating", "empty seat"],
            loc="upper right",
            bbox_to_anchor=(1, 1.32),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(OUT / "batching.gif", writer=PillowWriter(fps=12), dpi=90)
    plt.close(fig)


# --- KV memory: contiguous vs paged --------------------------------------------------


def make_kv_gif(model: TinyGPT) -> None:
    capacity, max_len, block = 2048, 512, 16
    cols = 16
    snapshots: dict[str, list] = {"contiguous": [], "paged": []}

    def snap_contiguous(engine: Engine) -> None:
        kv: ContiguousKVCache = engine.kv  # type: ignore[assignment]
        rows = [None] * kv.num_rows
        for sid, length in kv._lengths.items():
            rows[kv.row_of(sid)] = (sid, length)
        snapshots["contiguous"].append((rows, kv.usage(), len(engine.scheduler.waiting)))

    def snap_paged(engine: Engine) -> None:
        kv: PagedKVCache = engine.kv  # type: ignore[assignment]
        owner: dict[int, tuple[int, int]] = {}
        for sid, table in kv.block_tables.items():
            length = kv._lengths[sid]
            for i, b in enumerate(table):
                owner[b] = (sid, min(block, length - i * block))
        snapshots["paged"].append((owner, kv.usage(), len(engine.scheduler.waiting)))

    ids = {}
    for layout, snap in (("contiguous", snap_contiguous), ("paged", snap_paged)):
        reqs = workload(18, seed=3, prompt=(20, 90), out=(10, 120))
        ids[layout] = min(r.request_id for r in reqs)
        pool = KVPool(capacity, 4, model.cfg.n_kv_heads, model.cfg.head_dim)
        kv = (
            ContiguousKVCache(pool, max_len)
            if layout == "contiguous"
            else PagedKVCache(pool, block, watermark=0)
        )
        run(model, ContinuousScheduler(kv, max_batch_size=16), reqs, on_step=snap)

    horizon = max(len(v) for v in snapshots.values())
    fig, (ax_c, ax_p) = plt.subplots(
        1, 2, figsize=(11, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1]}
    )
    frames = list(range(0, horizon, max(1, horizon // 80))) + [horizon - 1] * 10

    headers = {
        ax_c: fig.text(0.01, 0.885, "", fontsize=10, color=INK, va="top"),
        ax_p: fig.text(0.51, 0.885, "", fontsize=10, color=INK, va="top"),
    }

    def header(ax, title: str, usage, waiting: int, running: int, step: int, total: int) -> None:
        if step >= total - 1:
            state = f"all requests done after {total} steps"
        else:
            state = (
                f"step {step}: {running} running, {waiting} waiting | "
                f"{usage.utilization:.0%} of reserved memory holds tokens"
            )
        headers[ax].set_text(f"{title}\n{state}")

    def draw(frame: int) -> None:
        for ax in (ax_c, ax_p):
            ax.clear()
            ax.set_facecolor(SURFACE)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Contiguous: one 512-slot slab per request.
        i = min(frame, len(snapshots["contiguous"]) - 1)
        rows, usage, waiting = snapshots["contiguous"][i]
        for r, entry in enumerate(rows):
            y = r * 1.4
            ax_c.add_patch(Rectangle((0, y), max_len, 1, color=FREE, lw=0))
            if entry is not None:
                sid, length = entry
                ax_c.add_patch(
                    Rectangle(
                        (0, y), max_len, 1, facecolor=WASTE, edgecolor=WASTE_EDGE, hatch="///", lw=0
                    )
                )
                ax_c.add_patch(Rectangle((0, y), length, 1, color=USED, lw=0))
                ax_c.text(
                    6,
                    y + 0.5,
                    f"#{sid - ids['contiguous']}",
                    color="white",
                    fontsize=8,
                    va="center",
                )
        ax_c.set_xlim(-5, max_len + 5)
        ax_c.set_ylim(len(rows) * 1.4, -0.4)
        running = sum(e is not None for e in rows)
        header(
            ax_c,
            "Contiguous: a 512-token slab per request",
            usage,
            waiting,
            running,
            i,
            len(snapshots["contiguous"]),
        )

        # Paged: a grid of 16-token blocks handed out on demand.
        i = min(frame, len(snapshots["paged"]) - 1)
        owner, usage, waiting = snapshots["paged"][i]
        nblocks = capacity // block
        for b in range(nblocks):
            x, y = b % cols, b // cols
            if b in owner:
                sid, fill = owner[b]
                ax_p.add_patch(
                    Rectangle(
                        (x + 0.06, y + 0.06),
                        0.88,
                        0.88,
                        facecolor=WASTE,
                        edgecolor=WASTE_EDGE,
                        hatch="///",
                        lw=0,
                    )
                )
                ax_p.add_patch(
                    Rectangle((x + 0.06, y + 0.06), 0.88 * fill / block, 0.88, color=USED, lw=0)
                )
                ax_p.text(
                    x + 0.5,
                    y + 0.52,
                    f"{sid - ids['paged']}",
                    color="white",
                    fontsize=6.5,
                    ha="center",
                    va="center",
                )
            else:
                ax_p.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, color=FREE, lw=0))
        ax_p.set_xlim(0, cols)
        ax_p.set_ylim(nblocks // cols, 0)
        ax_p.set_aspect("equal")
        running = len({sid for sid, _ in owner.values()})
        header(
            ax_p,
            "Paged: 16-token blocks, allocated on demand",
            usage,
            waiting,
            running,
            i,
            len(snapshots["paged"]),
        )

        handles = [
            Rectangle((0, 0), 1, 1, color=USED),
            Rectangle((0, 0), 1, 1, facecolor=WASTE, edgecolor=WASTE_EDGE, hatch="///"),
            Rectangle((0, 0), 1, 1, color=FREE),
        ]
        fig.legend(
            handles,
            ["holds tokens", "reserved but empty (wasted)", "free"],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        fig.suptitle(
            f"Same {capacity}-token KV pool, same 18 requests (numbers = request id)",
            x=0.01,
            ha="left",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.8))

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(OUT / "paged_kv.gif", writer=PillowWriter(fps=10), dpi=90)
    plt.close(fig)


def main() -> None:
    """Render both GIFs into docs/assets."""
    OUT.mkdir(parents=True, exist_ok=True)
    model = TinyGPT(ModelConfig())
    make_batching_gif(model)
    make_kv_gif(model)
    print(f"wrote {OUT / 'batching.gif'} and {OUT / 'paged_kv.gif'}")


if __name__ == "__main__":
    main()
