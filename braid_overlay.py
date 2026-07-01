"""Step 2: drop the real braid acceptance curve between the two cell rails.

Step 1 (coupon_cells.py) established the 2D (M=T**2) and 3D (M=T**3) reference
decays. Here we put the braid's *measured* acceptance curve onto the same axes
at a single timestep T and see where it falls.

The fair axis is "given n worldlines already placed, what is the chance the
next attempt is accepted?" -- a function of the count n, not of raw attempts
(the braid needs ~1e9 attempts to jam vs ~1e4 for the cells, so an attempts
axis would just shove them apart). On this axis:

    2D cell rail:   p(n) = (T**2 - n) / T**2     # collapses to 0 at n = T**2
    3D cell rail:   p(n) = (T**3 - n) / T**3     # collapses to 0 at n = T**3
    braid (data):   p(n) = dn / d(attempts)      # from the run curve

Each process keeps accepting until its acceptance collapses; the *count at
which it collapses* is the dimensional signal. We expect the braid to die
between T**2 and T**3.

Usage::

    python braid_overlay.py --timestep 18 --curves data/curves --out braid_overlay.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def braid_rate_points(
    curve_paths: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (count, acceptance_rate) sampled from every seed's run curve.

    For each consecutive pair of distinct (attempts, n) checkpoints, the local
    acceptance rate is dn/d(attempts), plotted at the midpoint count.
    """
    counts: list[float] = []
    rates: list[float] = []
    for path in curve_paths:
        rows: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("attempts"):
                continue
            attempts_str, n_str = line.split(",")
            pair = (int(attempts_str), int(n_str))
            if pair not in seen:
                seen.add(pair)
                rows.append(pair)
        rows.sort()
        for (a0, n0), (a1, n1) in zip(rows, rows[1:]):
            if a1 == a0:
                continue
            counts.append(0.5 * (n0 + n1))
            rates.append((n1 - n0) / (a1 - a0))
    return np.array(counts), np.array(rates)


def cell_rail(timestep: int, power: int, max_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Acceptance probability vs count for an ideal M = T**power cell grid."""
    cells = timestep**power
    n = np.linspace(0, min(cells, max_count), 600)
    return n, (cells - n) / cells


def overlay(timestep: int, curves_dir: Path, out_path: Path) -> None:
    """Plot the braid acceptance curve against the 2D and 3D cell rails."""
    import matplotlib.pyplot as plt

    pattern = str(curves_dir / f"d3_nyq_T{timestep}_s*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no curves matched {pattern}")
    counts, rates = braid_rate_points(paths)
    jam_count = float(np.max(counts))

    two_d = timestep**2
    three_d = timestep**3
    dimension = np.log(jam_count) / np.log(timestep)
    print(f"\nBraid vs cell rails at T = {timestep}\n" + "-" * 48)
    print(f"2D rail collapses at n = T^2 = {two_d}")
    print(f"3D rail collapses at n = T^3 = {three_d}")
    print(f"braid jams at  n ~= {jam_count:.0f}  ({len(paths)} seeds)")
    print(f"=> single-T dimension ln(n_jam)/ln(T) = {dimension:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    x2, p2 = cell_rail(timestep, 2, three_d)
    ax.plot(x2, p2, color="tab:blue", lw=2, label=f"2D rail (T^2={two_d})")
    x3, p3 = cell_rail(timestep, 3, three_d)
    ax.plot(x3, p3, color="tab:red", lw=2, label=f"3D rail (T^3={three_d})")

    ax.scatter(
        counts,
        rates,
        s=14,
        color="black",
        alpha=0.45,
        label="braid (measured, all seeds)",
    )
    ax.axvline(jam_count, color="green", ls="--", lw=1.5, label=f"braid jam n~{jam_count:.0f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n = worldlines already placed")
    ax.set_ylabel("acceptance probability of next attempt")
    ax.set_title(f"Braid jams between the 2D and 3D cell rails (T={timestep})")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestep", type=int, default=18)
    parser.add_argument("--curves", type=Path, default=Path("data/curves"))
    parser.add_argument("--out", type=Path, default=Path("braid_overlay.png"))
    args = parser.parse_args()
    overlay(args.timestep, args.curves, args.out)


if __name__ == "__main__":
    main()
