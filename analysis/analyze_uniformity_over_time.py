"""In-universe uniformity distribution over conformal time.

Not to be confused with the across-total-T jam scaling: this fixes ONE universe
(a single total-timestep count) and asks how the spatial slice changes from the
bang, through the turnaround, to the crunch. For each timestep z we reconstruct
the comoving positions X(z) = a*sin(b*z)/sin(z) + a2 (bounded in [-1, 1]) and
measure two things:

  rms spread  : sqrt(mean(X^2 + Y^2 + W^2) / 3) across worldlines.
  slice dim   : coarse box-counting dimension of the 3D comoving slice.

Finding: in comoving coordinates the slices are statistically identical at every
conformal time -- rms spread and slice box-dim are FLAT across the whole loop,
dipping only ~1% at the very endpoints (where sin(z) in the denominator mildly
compresses the reconstruction). The bang/crunch collapse lives entirely in
PHYSICAL coordinates (x = X*sin(z) -> 0), not the comoving packing. This is why a
single-slice correlation dimension returns one stable number regardless of which
timestep is sampled, and it is a separate phenomenon from the low-total-T jam
that saturates near 2D.

Usage::

    python analyze_uniformity_over_time.py --dumps data/corrdim/dumps --t 200 \
        --seeds 3 --out figures/uniformity_over_time.png
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

# Coarse spatial window for the per-slice box-counting slope (avoids the packing
# cell at the fine end and the box boundary at the coarse end).
SLICE_SIZES = np.array([0.05, 0.075, 0.1125, 0.169])


def load_params(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def slice_boxdim(pts3: np.ndarray) -> float:
    """Coarse box-counting dimension of one 3D comoving slice."""
    counts = []
    for s in SLICE_SIZES:
        c = np.floor(pts3 / s).astype(np.int64) + 64
        key = c[:, 0] * (1 << 16) + c[:, 1] * (1 << 8) + c[:, 2]
        counts.append(len(np.unique(key)))
    n = np.array(counts, dtype=float)
    return float(-np.polyfit(np.log(SLICE_SIZES), np.log(n), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="data/corrdim/dumps")
    parser.add_argument("--t", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=Path("figures/uniformity_over_time.png")
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = 0.01 + np.arange(args.t) * (np.pi - 0.02) / (args.t - 1)
    sinz = np.sin(z)
    paths = sorted(glob.glob(f"{args.dumps}/d3_nyq_T{args.t}_s*.csv"))[: args.seeds]
    print(f"T={args.t}, {len(paths)} seeds")

    def axis(cols, a, b, a2):  # (N, T)
        return (
            cols[a][:, None] * np.sin(np.outer(cols[b], z)) / sinz + cols[a2][:, None]
        )

    rms_all, dim_all = [], []
    for p in paths:
        cols = load_params(p)
        x = axis(cols, "ax", "bx", "ax2")
        y = axis(cols, "ay", "by", "ay2")
        w = axis(cols, "aw", "bw", "aw2")
        rms_all.append(np.sqrt((x**2 + y**2 + w**2).mean(axis=0) / 3.0))
        dim_all.append(
            np.array(
                [
                    slice_boxdim(np.stack([x[:, j], y[:, j], w[:, j]], axis=1))
                    for j in range(args.t)
                ]
            )
        )

    rms = np.mean(rms_all, axis=0)
    dim = np.mean(dim_all, axis=0)
    zpi = z / np.pi

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax0.plot(zpi, dim, "-", color="tab:purple", lw=2)
    ax0.axvline(0.5, color="gray", ls=":", lw=1)
    ax0.set_ylabel("slice box-counting dim")
    ax0.set_ylim(2.0, 3.0)
    ax0.set_title(
        "In-universe uniformity vs conformal time (comoving): "
        "flat across the whole loop"
    )
    ax0.grid(True, alpha=0.3)
    ax0.annotate(
        "turnaround", xy=(0.5, 2.62), xytext=(0.55, 2.85), fontsize=8, color="gray"
    )

    ax1.plot(zpi, rms, "-", color="tab:blue", lw=2)
    ax1.axvline(0.5, color="gray", ls=":", lw=1)
    ax1.set_ylabel("rms comoving spread")
    ax1.set_xlabel("conformal time  z / pi   (0 = bang, 1 = crunch)")
    ax1.set_ylim(0.55, 0.75)
    ax1.grid(True, alpha=0.3)

    fig.text(
        0.5,
        0.005,
        "Comoving slices are statistically identical at every conformal time; the "
        "bang/crunch collapse is entirely in physical coords (x = X*sin z -> 0). A "
        "single-slice correlation therefore returns one stable number, independent "
        "of timestep -- separate from the low-total-T jam that saturates near 2D.",
        ha="center",
        fontsize=7.5,
        color="gray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)

    half = int(np.argmin(np.abs(z - np.pi / 2)))
    print(f"turnaround : rms={rms[half]:.4f}  boxdim={dim[half]:.3f}")
    print(f"near bang   : rms={rms[2]:.4f}  boxdim={dim[2]:.3f}")
    print(f"near crunch : rms={rms[-3]:.4f}  boxdim={dim[-3]:.3f}")
    print(f"spread range over loop: {rms.min():.4f} - {rms.max():.4f}")
    print(f"boxdim range over loop: {dim.min():.3f} - {dim.max():.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
