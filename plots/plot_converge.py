"""CONVERGE chart: the same two views of the exponent, per dimension.

A symmetric 2x2 grid so the dimensions can be compared panel-for-panel
(one-off asymmetric charts made the story hard to follow):

  row = dimension (3+1 top, 2+1 bottom)
  left column  = local two-point slope between adjacent ladder rungs,
                 plotted at the geometric-mean T of the pair, one series
                 per cutoff. Shows where (whether) the exponent converges
                 in resolution, and whether the plateau moves with depth.
  right column = the fitted exponent vs cutoff depth: converged-window
                 weighted fit plus a fixed high-T two-point slope. Shows
                 the depth (in)dependence at a glance.

3+1 arms stitch the PACK (T <= 160) and CONVERGE (T = 200-360) stores;
2+1 uses the full T = 20-300 ladders at each cutoff (the 1e-9 anchor arm
has only two T values, so it appears in the right column only).

Usage::

    uv run python -m plots.plot_converge --out converge_grid.png
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

#: Per dimension: (cutoff label, stores to stitch, color).
LADDERS: dict[int, tuple[tuple[str, tuple[str, ...], str], ...]] = {
    3: (
        (
            "1e-6",
            ("data/pack/pack3d_e6.db", "data/converge/converge3d_e6.db"),
            "tab:blue",
        ),
        (
            "1e-7",
            ("data/pack/pack3d_e7.db", "data/converge/converge3d_e7.db"),
            "tab:red",
        ),
    ),
    2: (
        ("1e-6", ("data/pack/pack2d_e6.db",), "tab:blue"),
        ("1e-7", ("data/pack/pack2d_e7.db",), "tab:red"),
        ("1e-8", ("data/converge/converge2d_e8.db",), "tab:purple"),
        ("1e-9", ("data/converge/converge2d_e9.db",), "tab:orange"),
    ),
}
#: Converged/fit window start and the fixed high-T two-point pair, per dim.
WINDOW = {3: 160, 2: 100}
TWO_POINT = {3: (80, 160), 2: (140, 300)}
#: Reference lines drawn on the local-slope panel, per dim.
REFERENCES = {3: ((2.25, "D/d = 3/4"), (7 / 3, "7/3")), 2: ((1.5, "3/2"),)}


def cells(dbs: tuple[str, ...]) -> dict[int, tuple[float, float]]:
    """Per-T seed mean and SEM of the jam count (2-term rows), stitched."""
    by_t: dict[int, list[int]] = {}
    for db in dbs:
        rows = (
            sqlite3.connect(db)
            .execute("select t, n_final from runs where status='done' and terms=2")
            .fetchall()
        )
        for t, n in rows:
            by_t.setdefault(t, []).append(n)
    return {
        t: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
        for t, v in sorted(by_t.items())
        if len(v) > 1
    }


def local_slopes(
    c: dict[int, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-point slope between adjacent rungs, at the geometric-mean T."""
    ts = sorted(c)
    mid, slope, err = [], [], []
    for a, b in zip(ts, ts[1:]):
        (n1, e1), (n2, e2) = c[a], c[b]
        lr = np.log(b / a)
        mid.append(np.sqrt(a * b))
        slope.append(np.log(n2 / n1) / lr)
        err.append(np.hypot(e1 / n1, e2 / n2) / lr)
    return np.array(mid), np.array(slope), np.array(err)


def ladder_fit(c: dict[int, tuple[float, float]], tmin: int) -> tuple[float, float]:
    """Weighted log-log fit over T >= tmin."""
    ts = np.array([t for t in c if t >= tmin])
    ns = np.array([c[t][0] for t in ts])
    sig = np.array([c[t][1] / c[t][0] for t in ts])
    p, cov = np.polyfit(np.log(ts), np.log(ns), 1, w=1 / sig, cov=True)
    return float(p[0]), float(np.sqrt(cov[0][0]))


def two_point(
    c: dict[int, tuple[float, float]], pair: tuple[int, int]
) -> tuple[float, float] | None:
    """Slope between the fixed high-T pair, if both rungs exist."""
    lo, hi = pair
    if lo not in c or hi not in c:
        return None
    (n1, e1), (n2, e2) = c[lo], c[hi]
    lr = np.log(hi / lo)
    return np.log(n2 / n1) / lr, float(np.hypot(e1 / n1, e2 / n2) / lr)


def plot(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))
    summary: list[str] = []

    for row, dim in enumerate((3, 2)):
        ax_slope, ax_depth = axes[row]
        window, pair = WINDOW[dim], TWO_POINT[dim]
        depth_x: list[float] = []
        depth_fit: list[tuple[float, float] | None] = []
        depth_two: list[tuple[float, float] | None] = []

        for label, dbs, color in LADDERS[dim]:
            c = cells(dbs)
            decade = -float(label.split("-")[1])
            depth_x.append(-decade)
            depth_two.append(two_point(c, pair))
            if len(c) > 2:
                d, e = ladder_fit(c, window)
                depth_fit.append((d, e))
                mid, slope, err = local_slopes(c)
                ax_slope.errorbar(
                    mid,
                    slope,
                    yerr=err,
                    fmt="o",
                    color=color,
                    ms=4.5,
                    capsize=2,
                    alpha=0.85,
                    label=f"{label}: fit T>={window} D = {d:.4f} ± {e:.4f}",
                )
                ax_slope.axhline(d, ls="--", color=color, lw=1.0, alpha=0.55)
                summary.append(f"{dim}+1 @{label}: window fit {d:.4f} +/- {e:.4f}")
            else:
                depth_fit.append(None)

        for ref, name in REFERENCES[dim]:
            ax_slope.axhline(ref, ls=":", color="gray", lw=1.2)
            ax_slope.annotate(
                name,
                (0.02, ref),
                xycoords=("axes fraction", "data"),
                fontsize=8.5,
                color="gray",
                va="bottom",
            )
        ax_slope.set_xscale("log")
        ax_slope.set_xlabel("T (geometric mean of the rung pair)")
        ax_slope.set_ylabel("local slope d log N / d log T")
        ax_slope.set_title(f"{dim}+1: local exponent vs resolution, per cutoff")
        ax_slope.grid(True, which="both", alpha=0.3)
        ax_slope.legend(fontsize=8.5, loc="lower right")

        fit_x = [x for x, f in zip(depth_x, depth_fit) if f]
        fit_y = [f[0] for f in depth_fit if f]
        fit_e = [f[1] for f in depth_fit if f]
        ax_depth.errorbar(
            fit_x,
            fit_y,
            yerr=fit_e,
            fmt="s-",
            color="tab:blue",
            ms=6,
            capsize=2,
            label=f"window fit (T>={window})",
        )
        two_x = [x for x, f in zip(depth_x, depth_two) if f]
        two_y = [f[0] for f in depth_two if f]
        two_e = [f[1] for f in depth_two if f]
        ax_depth.errorbar(
            two_x,
            two_y,
            yerr=two_e,
            fmt="o--",
            color="tab:orange",
            ms=5,
            capsize=2,
            label=f"two-point {pair[0]}->{pair[1]}",
        )
        ax_depth.set_xticks(depth_x)
        ax_depth.set_xticklabels([f"1e-{int(x)}" for x in depth_x])
        ax_depth.set_xlabel("acceptance-rate cutoff")
        ax_depth.set_ylabel("packing exponent D")
        ax_depth.set_title(f"{dim}+1: exponent vs cutoff depth")
        ax_depth.grid(True, alpha=0.3)
        ax_depth.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")
    for line in summary:
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("converge_grid.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
