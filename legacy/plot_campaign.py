"""Render PNG plots from braid_engine growth-curve CSVs.

The Rust engine writes `--curve` CSVs (`attempts,n`). This companion turns a
directory of them into figures. Curve files named `T{t}_{mode}_s{seed}.csv`
are grouped by (T, mode) and averaged/overlaid across seeds; any other CSV is
plotted as its own series.

Produces (where the data supports it):
  growth_curves.png   N vs attempts (log-x) — the approach-to-jamming crawl
  acceptance_rate.png dN/d(attempts) vs attempts (log-log), with fitted slope p
  resolution_scaling.png  N(1e9) and paths-per-decade vs time-steps T

Usage:
    python plot_campaign.py --indir /tmp/campaign --outdir /tmp/campaign/plots
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Curve = list[tuple[float, float]]


def load(path: str) -> Curve:
    pts: Curve = []
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.split(",")
            if len(parts) == 2:
                pts.append((float(parts[0]), float(parts[1])))
    return pts


def label_of(path: str) -> tuple[str, int | None]:
    """Return (group_label, seed) parsed from the filename, if possible."""
    m = re.match(r"T(\d+)_(\w+?)_s(\d+)\.csv", os.path.basename(path))
    if m:
        return f"T={m.group(1)} {m.group(2)}", int(m.group(3))
    return os.path.splitext(os.path.basename(path))[0], None


def t_of(group: str) -> int | None:
    m = re.search(r"T=(\d+)", group)
    return int(m.group(1)) if m else None


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return slope, (sy - slope * sx) / n


def collect(indir: str) -> dict[str, list[Curve]]:
    groups: dict[str, list[Curve]] = {}
    for path in sorted(glob.glob(os.path.join(indir, "*.csv"))):
        pts = load(path)
        if len(pts) < 4:
            continue
        group, _ = label_of(path)
        groups.setdefault(group, []).append(pts)
    return groups


def plot_growth(groups: dict[str, list[Curve]], out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, (group, curves) in enumerate(sorted(groups.items())):
        color = f"C{ci % 10}"
        for i, pts in enumerate(curves):
            xs = [a for a, _ in pts]
            ys = [n for _, n in pts]
            ax.plot(xs, ys, lw=1.4, alpha=0.85, color=color, label=group if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("attempts")
    ax.set_ylabel("N (paths packed)")
    ax.set_title("Approach to jamming: N vs attempts (fixed time-steps)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def rate_series(pts: Curve) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for (a0, n0), (a1, n1) in zip(pts, pts[1:]):
        da, dn = a1 - a0, n1 - n0
        if da > 0 and dn > 0:
            xs.append((a0 + a1) / 2)
            ys.append(dn / da)
    return xs, ys


def plot_rate(groups: dict[str, list[Curve]], out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, (group, curves) in enumerate(sorted(groups.items())):
        xs, ys = rate_series(curves[0])
        if not xs:
            continue
        # fit slope p over the tail (last ~2 decades)
        lo = xs[-1] / 100
        lx = [math.log10(x) for x, y in zip(xs, ys) if x >= lo]
        ly = [math.log10(y) for x, y in zip(xs, ys) if x >= lo]
        p = linfit(lx, ly)[0] if len(lx) > 4 else float("nan")
        ax.plot(xs, ys, lw=1.4, alpha=0.85, color=f"C{ci % 10}", label=f"{group}  (p={p:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("attempts")
    ax.set_ylabel("acceptance rate  dN/d(attempts)")
    ax.set_title("Acceptance-rate decay  (p = -1 ⇒ N ~ log attempts)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def per_decade(pts: Curve) -> float:
    final_a = pts[-1][0]
    tail = [(a, n) for a, n in pts if a >= final_a / 100]
    return linfit([math.log10(a) for a, _ in tail], [n for _, n in tail])[0]


def plot_scaling(groups: dict[str, list[Curve]], out: str) -> bool:
    rows = []
    for group, curves in groups.items():
        T = t_of(group)
        if T is None or "smart" not in group:
            continue
        finals = [c[-1][1] for c in curves]
        bs = [per_decade(c) for c in curves]
        rows.append((T, sum(finals) / len(finals), sum(bs) / len(bs)))
    if len(rows) < 2:
        return False
    rows.sort()
    Ts = [r[0] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.plot(Ts, [r[1] for r in rows], "o-")
    a1.set_xlabel("time-steps T")
    a1.set_ylabel("N at final budget")
    a1.set_title("Packing vs resolution")
    a1.grid(True, alpha=0.25)
    a2.plot(Ts, [r[2] for r in rows], "s-", color="C1")
    a2.set_xlabel("time-steps T")
    a2.set_ylabel("paths per decade (B)")
    a2.set_title("Growth rate vs resolution")
    a2.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", default="/tmp/campaign", help="directory of curve CSVs")
    ap.add_argument("--outdir", default=None, help="output dir (default <indir>/plots)")
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(args.indir, "plots")
    os.makedirs(outdir, exist_ok=True)

    groups = collect(args.indir)
    if not groups:
        print(f"no curve CSVs found in {args.indir}")
        return
    print(f"loaded {sum(len(v) for v in groups.values())} curves in {len(groups)} groups")

    plot_growth(groups, os.path.join(outdir, "growth_curves.png"))
    plot_rate(groups, os.path.join(outdir, "acceptance_rate.png"))
    scaled = plot_scaling(groups, os.path.join(outdir, "resolution_scaling.png"))

    made = ["growth_curves.png", "acceptance_rate.png"] + (["resolution_scaling.png"] if scaled else [])
    print("wrote: " + ", ".join(os.path.join(outdir, m) for m in made))


if __name__ == "__main__":
    main()
