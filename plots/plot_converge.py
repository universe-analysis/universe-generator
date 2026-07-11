"""CONVERGE campaign chart: where each headline exponent actually lands.

Left panel (3+1): the local two-point slope of N(T) between adjacent ladder
rungs, stitched across the PACK (T = 20-160) and CONVERGE (T = 200-360)
1e-6 stores. The question the campaign asked: does the rising slope flatten
onto D/d = 3/4 (D = 2.25)? The converged plateau (weighted fit over
T >= 160) is drawn with its value; the 3/4 and 7/3 references frame it.

Right panel (2+1): the exponent per cutoff decade — full-ladder weighted
fits (T >= 100) for 1e-6/1e-7/1e-8 and the two-point T = 140 -> 300 slope
for all four decades including the 1e-9 anchor. A converging exponent would
decelerate toward a plateau; the data show the drift persisting.

Usage::

    uv run python -m plots.plot_converge --out converge.png
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

#: (label, db path) per cutoff decade, 2+1. The e6/e7 rungs are PACK stores.
LADDERS_2D = (
    (6, "data/pack/pack2d_e6.db"),
    (7, "data/pack/pack2d_e7.db"),
    (8, "data/converge/converge2d_e8.db"),
    (9, "data/converge/converge2d_e9.db"),
)
#: (cutoff label, stores to stitch, color) per 3+1 arm.
LADDERS_3D = (
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
)
#: Converged-window start (3+1) and fit-window start (2+1).
WINDOW_3D = 160
WINDOW_2D = 100
TWO_POINT_2D = (140, 300)


def cells(db: str | Path) -> dict[int, tuple[float, float]]:
    """Per-T seed mean and SEM of the jam count (2-term rows)."""
    rows = (
        sqlite3.connect(db)
        .execute("select t, n_final from runs where status='done' and terms=2")
        .fetchall()
    )
    by_t: dict[int, list[int]] = {}
    for t, n in rows:
        by_t.setdefault(t, []).append(n)
    return {
        t: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
        for t, v in sorted(by_t.items())
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


def plot(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax3, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # -- 3+1: local slope vs T, one series per cutoff ----------------------
    d_convs: list[tuple[str, float, float]] = []
    for label, dbs, color in LADDERS_3D:
        c3 = cells(dbs[0])
        for db in dbs[1:]:
            c3.update(cells(db))
        mid, slope, err = local_slopes(c3)
        d_conv, d_err = ladder_fit(c3, WINDOW_3D)
        d_convs.append((label, d_conv, d_err))
        ax3.errorbar(
            mid,
            slope,
            yerr=err,
            fmt="o",
            color=color,
            ms=5,
            capsize=2,
            alpha=0.85,
            label=f"{label}: converged D = {d_conv:.4f} ± {d_err:.4f}",
        )
        ax3.axhline(d_conv, ls="--", color=color, lw=1.2, alpha=0.7)
    ax3.axhline(2.25, ls=":", color="darkgreen", lw=1.2)
    ax3.text(21, 2.253, "D/d = 3/4", color="darkgreen", fontsize=9)
    ax3.axhline(7 / 3, ls=":", color="gray", lw=1.2)
    ax3.text(21, 7 / 3 + 0.003, "7/3", color="gray", fontsize=9)
    ax3.legend(fontsize=9, loc="lower right")
    ax3.set_xscale("log")
    ax3.set_xlabel("T (geometric mean of the rung pair)")
    ax3.set_ylabel("local slope d log N / d log T")
    ax3.set_title("3+1: local packing exponent vs resolution, per cutoff")
    ax3.grid(True, which="both", alpha=0.3)

    # -- 2+1: exponent vs cutoff depth ------------------------------------
    decades, fits, fit_errs, twops, twop_errs = [], [], [], [], []
    for dec, db in LADDERS_2D:
        c2 = cells(db)
        decades.append(dec)
        lo, hi = TWO_POINT_2D
        (n1, e1), (n2, e2) = c2[lo], c2[hi]
        lr = np.log(hi / lo)
        twops.append(np.log(n2 / n1) / lr)
        twop_errs.append(np.hypot(e1 / n1, e2 / n2) / lr)
        if len(c2) > 2:  # full ladder available (not the 2-T anchor arm)
            d, e = ladder_fit(c2, WINDOW_2D)
            fits.append(d)
            fit_errs.append(e)
    ax2.errorbar(
        decades[: len(fits)],
        fits,
        yerr=fit_errs,
        fmt="s-",
        color="tab:blue",
        ms=6,
        capsize=2,
        label=f"full-ladder fit (T>={WINDOW_2D})",
    )
    ax2.errorbar(
        decades,
        twops,
        yerr=twop_errs,
        fmt="o--",
        color="tab:orange",
        ms=5,
        capsize=2,
        label=f"two-point {TWO_POINT_2D[0]}->{TWO_POINT_2D[1]}",
    )
    ax2.set_xticks(decades)
    ax2.set_xticklabels([f"1e-{d}" for d in decades])
    ax2.set_xlabel("acceptance-rate cutoff")
    ax2.set_ylabel("packing exponent D")
    ax2.set_title("2+1: exponent vs cutoff depth (no plateau)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")
    for label, d_conv, d_err in d_convs:
        print(
            f"  3+1 converged (T>={WINDOW_3D}) @{label}: "
            f"D = {d_conv:.4f} +/- {d_err:.4f}"
        )
    for dec, d, e in zip(decades, twops, twop_errs):
        print(f"  2+1 two-point @1e-{dec}: {d:.4f} +/- {e:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("converge.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
