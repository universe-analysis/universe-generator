"""Plot the approach-to-jamming curves (N vs attempts) across T.

Each timestep's run is a curve of accepted count N against attempts. As the
packing fills, N rises and then flattens onto its jammed ceiling. Plotting them
together shows both the ceiling growing with T (the D scaling) and how close
each run got to its own plateau.

All curves now use the full Nyquist frequency band. Caveat: the T=6/T=10 curves
are edge-weighted, run to true jamming; the T=18..60 curves are the uniform
1e-7-cutoff campaign (near, but not all the way to, jamming -- the higher T ones
stop further short). So the remaining mismatch is sampler (edge vs uniform, ~5%)
and depth, not the frequency band.

Usage::

    python plot_approach.py --out approach_jamming.png
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

DIAG = Path("data/diag")
CURVES = Path("data/curves")


def load_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in csv.DictReader(open(path)):
        pair = (int(r["attempts"]), int(r["n"]))
        if pair not in seen:
            seen.add(pair)
            rows.append(pair)
    rows.sort()
    return np.array([p[0] for p in rows]), np.array([p[1] for p in rows])


def seed_mean_curve(pattern: str) -> tuple[np.ndarray, np.ndarray]:
    """Average N across seeds on a common log-attempts grid."""
    curves = [load_curve(p) for p in sorted(glob.glob(pattern))]
    lo = max(a[0] for a, _ in curves)
    hi = min(a[-1] for a, _ in curves)
    grid = np.logspace(np.log10(lo), np.log10(hi), 60)
    stacked = [np.interp(grid, a, n) for a, n in curves]
    return grid, np.mean(stacked, axis=0)


def plot(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))

    # Deep, truly-jammed edge-weighted runs at the low end, full Nyquist band.
    a5, n5 = load_curve(str(DIAG / "t5_1e12_edge.csv"))
    ax.plot(a5, n5, lw=2, color="silver", label="T=5 (edge, full Nyquist -> jammed, N=26)")
    a6, n6 = load_curve(str(DIAG / "t6_nyq_edge.csv"))
    ax.plot(a6, n6, lw=2, color="dimgray", label="T=6 (edge, full Nyquist -> jammed, N=43)")
    a10, n10 = load_curve(str(DIAG / "t10_nyq_edge.csv"))
    ax.plot(a10, n10, lw=2, color="black", label="T=10 (edge, full Nyquist -> jammed, N=195)")

    # T=18..60: uniform 1e-7 campaign, seed-averaged.
    colors = ["tab:purple", "tab:blue", "tab:green", "tab:orange", "tab:red"]
    for t, color in zip([18, 24, 32, 44, 60], colors):
        grid, mean = seed_mean_curve(str(CURVES / f"d3_nyq_T{t}_s*.csv"))
        ax.plot(grid, mean, lw=2, color=color, label=f"T={t} (uniform nyq, 1e-7)")

    # T=100 and T=180: edge, full Nyquist, 1e-7 cutoff (new high-T points).
    a100, n100 = load_curve(str(DIAG / "t100_nyq_edge.csv"))
    ax.plot(a100, n100, lw=2, color="magenta", label="T=100 (edge nyq, 1e-7, N=57934)")
    a180, n180 = load_curve(str(DIAG / "ladder_T180.csv"))
    ax.plot(a180, n180, lw=2, color="darkviolet", label="T=180 (edge nyq, 1e-7, N=250248)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("attempts")
    ax.set_ylabel("N accepted (worldlines packed)")
    ax.set_title("Approach to jamming across T: N rises, then plateaus")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("approach_jamming.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
