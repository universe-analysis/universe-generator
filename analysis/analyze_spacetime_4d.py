"""4D box-counting of the worldline bundle -- time treated as a spatial axis.

Chris's framing: these are 4D waves where conformal time is spatial, so the honest
object to box-count is the full (x, y, w, z) bundle, not a 3D projection of it.
Each accepted worldline is a smooth 1D curve
X(z) = a*sin(b*z)/sin(z) + a2 (per axis), z_j = 0.01 + j*(pi-0.02)/(T-1); we pool
all timesteps into one 4D cloud and measure the box-counting dimension as a
function of SCALE (local slope). Two time-axis scalings are compared:

  native : z on its real [0, pi] range  -> geometric bundle dimension.
  packed : z' = z * (2/pi) in [0, 2] so one timestep equals one exclusion cell
           (dz_step == CELL == 2/T), i.e. an isotropic 4D collision cell.

Finding: under both scalings the local dimension sweeps ~1 (fine, resolving the
individual curves) up to ~4 (coarse, space-filling), crossing the count-scaling
2.46 only in passing -- there is NO plateau at 2.46. So 2.46 is not a geometric
dimension of the bundle in 4D any more than in 3D: it is a packing-NUMBER exponent
taken across the family of universes (box == CELL == 2/T shrinking as T grows),
not within a single fixed one. At the cell scale only a fraction of a percent of
the 4D cells are occupied, so the bundle is a thin near-1D set threading 4D
spacetime, not a solid.

Usage::

    python analyze_spacetime_4d.py --dumps data/corrdim/dumps --t 200 \
        --seeds 3 --out figures/spacetime_4d.png
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

from braidlab.corrdim import wrap_unit

COUNT_D = 2.46  # count-scaling packing dimension, for reference


def load_params(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def pooled_4d(
    cols: dict[str, np.ndarray], t: int, packed: bool, torus: bool = False
) -> np.ndarray:
    """Return the pooled (N*T)x4 (x, y, w, z) cloud for one dump."""
    z = 0.01 + np.arange(t) * (np.pi - 0.02) / (t - 1)
    sinz = np.sin(z)

    def axis(a, b, a2):  # X(z) = a*sin(b*z)/sin(z) + a2, shape (N, T)
        out = a[:, None] * np.sin(np.outer(b, z)) / sinz + a2[:, None]
        return wrap_unit(out) if torus else out

    x = axis(cols["ax"], cols["bx"], cols["ax2"]).ravel()
    y = axis(cols["ay"], cols["by"], cols["ay2"]).ravel()
    w = axis(cols["aw"], cols["bw"], cols["aw2"]).ravel()
    zt = np.tile(z, len(cols["ax"]))
    zc = zt * (2.0 / np.pi) if packed else zt
    return np.stack([x, y, w, zc], axis=1)


def box_count_4d(pts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Occupied-cell count at each edge length (int-hash for a fast unique)."""
    counts = []
    for s in sizes:
        c = np.floor(pts / s).astype(np.int64) + (1 << 18)
        key = c[:, 0] * (1 << 45) + c[:, 1] * (1 << 30) + c[:, 2] * (1 << 15) + c[:, 3]
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
    parser.add_argument("--out", type=Path, default=Path("figures/spacetime_4d.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell = 2.0 / args.t
    sizes = np.logspace(np.log10(cell), np.log10(0.4), 18)
    paths = sorted(glob.glob(f"{args.dumps}/d3_nyq_T{args.t}_s*.csv"))[: args.seeds]
    print(f"T={args.t}, CELL={cell:.4f}, {len(paths)} seeds")

    curves = {}
    for packed in (False, True):
        counts = []
        for p in paths:
            counts.append(
                box_count_4d(
                    pooled_4d(load_params(p), args.t, packed, torus=args.torus), sizes
                )
            )
        n = np.mean(counts, axis=0)
        curves["packed" if packed else "native"] = (
            n,
            -np.gradient(np.log(n), np.log(sizes)),
        )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(
        sizes,
        curves["native"][1],
        "o-",
        color="tab:purple",
        label="4D bundle, z native [0, pi]",
    )
    ax.plot(
        sizes,
        curves["packed"][1],
        "s--",
        color="tab:blue",
        label="4D bundle, z packed (dz_step = CELL)",
    )
    ax.axhline(
        COUNT_D,
        color="tab:red",
        ls="-",
        lw=1.5,
        label=f"count-scaling packing dim = {COUNT_D}",
    )
    ax.axhline(4.0, color="black", ls=":", lw=1, label="space-filling in 4D (4)")
    ax.axhline(1.0, color="tab:green", ls=":", lw=1, label="single curve (1)")
    ax.set_xscale("log")
    ax.set_xlabel("box scale")
    ax.set_ylabel("local box-counting dimension (4D)")
    ax.set_title(
        "4D box-count of the worldline bundle: local D runs 1 -> 4, no 2.46 plateau"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.text(
        0.5,
        0.005,
        "Time treated as a spatial axis. The bundle is a union of smooth 1D curves "
        "threading 4D; its geometric dimension sweeps 1 (fine) -> 4 (coarse) under "
        "both z scalings. The count-scaling 2.46 is a packing-NUMBER exponent taken "
        "across resolution T, not a geometric dimension of this 4D cloud.",
        ha="center",
        fontsize=7.5,
        color="gray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)

    n_native = curves["native"][0]
    occ = n_native[0] / ((2.0 / cell) ** 4)
    print(f"\ncell-scale occupancy (native): {occ * 100:.2f}% of 4D cells filled")
    for name, (_, d) in curves.items():
        print(
            f"{name:>7}: fine D = {d[0]:.2f}, "
            f"crosses {COUNT_D} near scale "
            f"{sizes[np.argmin(np.abs(d - COUNT_D))]:.3f}, coarse D = {d[-4]:.2f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
