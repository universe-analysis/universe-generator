"""Cross-check: is the count-scaling packing dimension (~2.46) a *geometric*
dimension of the worldline set?

Reconstructs every accepted worldline's comoving trajectory across all timesteps
(X(z) = a*sin(b*z)/sin(z) + a2, z_j = 0.01 + j*(pi-0.02)/(T-1)), pools them into
one cloud, and measures the box-counting dimension as a function of SCALE (local
slope). Compared against the single turnaround snapshot.

Finding: the pooled worldline cloud is a union of smooth 1D curves that
collectively fill 3D. Its local dimension runs from ~1 at the finest scale
(resolving individual trajectories) up to ~3 at coarse scales (space-filling),
with NO plateau at 2.46. So the count-scaling 2.46 is NOT a geometric dimension
of any static cloud -- it is a packing-NUMBER exponent, set by how the joint
across-time exclusion suppresses the achievable count below the 3D maximum, not
by spatial clustering (the arrangement itself is ~uniform, D->3).

Usage::

    python analyze_spacetime_dim.py --dumps data/corrdim/dumps --t 200 \
        --seeds 3 --out figures/spacetime_dim.png
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

from braidlab.corrdim import wrap_unit

COUNT_D = 2.46  # default count-scaling packing dimension (hard-wall model)


def load_params(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def trajectories(
    cols: dict[str, np.ndarray], t: int, torus: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return (snapshot Nx3 at turnaround, pooled (N*T)x3 over all timesteps)."""
    z = 0.01 + np.arange(t) * (np.pi - 0.02) / (t - 1)
    sinz = np.sin(z)
    half = int(np.argmin(np.abs(z - np.pi / 2)))

    def axis(a, b, a2):  # X(z) = a*sin(b*z)/sin(z) + a2, shape (N, T)
        out = a[:, None] * np.sin(np.outer(b, z)) / sinz + a2[:, None]
        return wrap_unit(out) if torus else out

    x = axis(cols["ax"], cols["bx"], cols["ax2"])
    y = axis(cols["ay"], cols["by"], cols["ay2"])
    w = axis(cols["aw"], cols["bw"], cols["aw2"])
    snap = np.stack([x[:, half], y[:, half], w[:, half]], axis=1)
    pooled = np.stack([x.ravel(), y.ravel(), w.ravel()], axis=1)
    return snap, pooled


def box_count(pts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Occupied-cell count at each edge length (int-hash for a fast unique)."""
    counts = []
    for s in sizes:
        c = np.floor(pts / s).astype(np.int64) + (1 << 20)
        key = c[:, 0] * (1 << 42) + c[:, 1] * (1 << 21) + c[:, 2]
        counts.append(len(np.unique(key)))
    return np.array(counts, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="data/corrdim/dumps")
    parser.add_argument("--t", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--torus",
        action="store_true",
        help="torus-model dumps: wrap reconstructed positions onto [-1, 1)",
    )
    parser.add_argument(
        "--count-d",
        type=float,
        default=COUNT_D,
        help="count-scaling packing dimension drawn as the reference line",
    )
    parser.add_argument("--out", type=Path, default=Path("figures/spacetime_dim.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell = 2.0 / args.t
    sizes = np.logspace(np.log10(cell), np.log10(0.2), 22)
    paths = sorted(glob.glob(f"{args.dumps}/d3_nyq_T{args.t}_s*.csv"))[: args.seeds]
    print(f"T={args.t}, CELL={cell:.4f}, {len(paths)} seeds")

    snap_counts, pool_counts = [], []
    for p in paths:
        snap, pooled = trajectories(load_params(p), args.t, torus=args.torus)
        snap_counts.append(box_count(snap, sizes))
        pool_counts.append(box_count(pooled, sizes))
        print(f"  {Path(p).name}: snapshot {len(snap):,}  pooled {len(pooled):,}")

    snap_n = np.mean(snap_counts, axis=0)
    pool_n = np.mean(pool_counts, axis=0)
    snap_d = -np.gradient(np.log(snap_n), np.log(sizes))
    pool_d = -np.gradient(np.log(pool_n), np.log(sizes))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(
        sizes,
        pool_d,
        "o-",
        color="tab:purple",
        label="pooled worldlines (all timesteps)",
    )
    ax.plot(
        sizes, snap_d, "s--", color="tab:gray", label="single snapshot (turnaround)"
    )
    ax.axhline(
        args.count_d,
        color="tab:red",
        ls="-",
        lw=1.5,
        label=f"count-scaling packing dim = {args.count_d}",
    )
    ax.axhline(3.0, color="black", ls=":", lw=1, label="space-filling (3)")
    ax.axhline(1.0, color="tab:green", ls=":", lw=1, label="single curve (1)")
    ax.axvspan(
        cell, 0.045, color="tab:gray", alpha=0.08, label="snapshot subsample-saturated"
    )
    ax.set_xscale("log")
    ax.set_xlabel("box scale")
    ax.set_ylabel("local box-counting dimension")
    ax.set_title(
        "Geometric dimension vs scale: the worldline set runs 1 -> 3, "
        f"with no {args.count_d} plateau"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.text(
        0.5,
        0.005,
        "The pooled worldlines are ~1D curves that collectively fill 3D "
        f"(local D: 1 at fine -> 3 at coarse). The count-scaling {args.count_d} is a "
        "packing-NUMBER exponent (count suppressed by the across-time "
        "constraint), not a geometric dimension of this cloud.",
        ha="center",
        fontsize=7.5,
        color="gray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)

    def at(sizes_arr, d_arr, lo, hi):
        m = (sizes_arr >= lo) & (sizes_arr <= hi)
        return float(np.mean(d_arr[m]))

    print(
        f"\npooled local D:  fine (~CELL) = {pool_d[0]:.2f}, "
        f"mid [0.02,0.04] = {at(sizes, pool_d, 0.02, 0.04):.2f}, "
        f"coarse [0.08,0.15] = {at(sizes, pool_d, 0.08, 0.15):.2f}"
    )
    print(f"count-scaling packing dimension = {args.count_d} (no matching plateau)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
