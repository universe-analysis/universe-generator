"""Periodic (toroidal / minimum-image) correlation dimension vs T.

Measures D2 with wrapped distances: the comoving domain [-1, 1]^3 is treated as
a 3-torus (side length 2 per axis), so a neighborhood that runs off one face
re-enters the opposite face. With no boundary, there is no edge depletion, and a
uniform D=3 control returns exactly 3.000 for both the sphere (L2) and cube
(Linf) probe -- the boundary artifact that splits sphere/cube in the
non-periodic estimator vanishes.

CAVEAT (read this): applied here to the EXISTING hard-wall packing, which is NOT
periodic. The generator rejects any path crossing |X| = 1 and an edge-weighted
sampler piles density against both walls, so wrapping the *measurement* invents
spurious close pairs across the seam (dense +wall now adjacent to dense -wall).
These numbers are therefore NOT physically self-consistent -- this is a preview
of the estimator for a future genuinely-periodic space, run on request. The
reference non-periodic sphere line is included so the shift is visible.

Usage::

    python analyze_periodic_corrdim.py --dumps data/corrdim/dumps \
        --out figures/periodic_corrdim.png
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from braidlab.corrdim import FIT_HI, FIT_LO, R_MAX, fit_slope, load_turnaround_cloud

#: Comoving domain [-1, 1]^3 -> side length 2 per axis for the toroidal wrap.
BOX = 2.0
_NAME_RE = re.compile(r"_T(?P<t>\d+)_s(?P<seed>\d+)")


def corr_dimension(cloud: np.ndarray, t: int, p: float, periodic: bool) -> float:
    """Correlation D2 for one cloud; ``periodic`` wraps distances on the 2-torus."""
    cell = 2.0 / t
    radii = np.logspace(np.log10(cell), np.log10(R_MAX), 40)
    shifted = cloud + 1.0  # [-1, 1] -> [0, 2) as boxsize requires
    kw = {"boxsize": BOX} if periodic else {}
    tree = KDTree(shifted, **kw)
    n = len(shifted)
    step = max(1, n // min(4000, n))
    centers = KDTree(shifted[::step], **kw)
    n_c = centers.n
    counts = centers.count_neighbors(tree, radii, p=p)
    C = (counts - n_c) / (n_c * (n - 1.0))
    return fit_slope(radii, C, FIT_LO, FIT_HI)


def _mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), sem


def control_check() -> None:
    """Uniform D=3 cloud: periodic must return ~3.0 for both probes, gap ~0."""
    rng = np.random.default_rng(0)
    u = rng.uniform(-1.0, 1.0, size=(60000, 3))
    print("control -- uniform cloud (true D=3):")
    for p, name in [(2.0, "sphere"), (np.inf, "cube  ")]:
        npv = corr_dimension(u, 200, p, periodic=False)
        pv = corr_dimension(u, 200, p, periodic=True)
        print(f"  {name}: non-periodic={npv:.3f}   periodic={pv:.3f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="data/corrdim/dumps")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--out", type=Path, default=Path("figures/periodic_corrdim.png")
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control_check()

    by_t: dict[int, list[str]] = {}
    for path in sorted(glob.glob(f"{args.dumps}/d3_nyq_T*_s*.csv")):
        m = _NAME_RE.search(Path(path).name)
        if m:
            by_t.setdefault(int(m["t"]), []).append(path)

    ts = sorted(by_t)
    keys = ["per_sphere", "per_cube", "nonper_sphere"]
    mean: dict[str, list[float]] = {k: [] for k in keys}
    sem: dict[str, list[float]] = {k: [] for k in keys}
    print(f"{'T':>4} {'per_sphere':>11} {'per_cube':>10} {'nonper_sphere':>14}")
    for t in ts:
        clouds = [load_turnaround_cloud(p) for p in by_t[t][: args.seeds]]
        runs = {
            "per_sphere": [corr_dimension(c, t, 2.0, True) for c in clouds],
            "per_cube": [corr_dimension(c, t, np.inf, True) for c in clouds],
            "nonper_sphere": [corr_dimension(c, t, 2.0, False) for c in clouds],
        }
        for k in keys:
            m, s = _mean_sem(runs[k])
            mean[k].append(m)
            sem[k].append(s)
        print(f"{t:>4} {mean['per_sphere'][-1]:>11.3f} "
              f"{mean['per_cube'][-1]:>10.3f} {mean['nonper_sphere'][-1]:>14.3f}")

    style = {
        "per_sphere": ("tab:green", "o-", "periodic D2 (sphere)"),
        "per_cube": ("tab:olive", "s--", "periodic D2 (cube)"),
        "nonper_sphere": ("tab:blue", "o:", "non-periodic D2 (sphere), reference"),
    }
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for k in keys:
        color, fmt, label = style[k]
        ax.errorbar(ts, mean[k], yerr=sem[k], fmt=fmt, color=color, ms=4,
                    capsize=3, label=label)
        for t, v in zip(ts, mean[k]):
            ax.annotate(f"{v:.2f}", (t, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=6, color=color)
    ax.axhline(3.0, color="black", ls="--", lw=1, label="space-filling D = 3")
    ax.set_xlabel("T (resolution)")
    ax.set_ylabel("correlation dimension D2")
    ax.set_title("Periodic (wrapped) correlation dimension vs T (3+1)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5,
        0.005,
        "CAVEAT: wrapped measurement on the HARD-WALL packing (not periodic) -- "
        "the seam invents spurious cross-wall pairs. Preview of the estimator "
        "for a future periodic space; not physically self-consistent on this data.",
        ha="center",
        fontsize=7.5,
        color="firebrick",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)

    def tail(k: str) -> float:
        return float(np.mean([v for t, v in zip(ts, mean[k]) if t >= 100]))

    print("\nconverged (T>=100 mean):")
    for k in keys:
        print(f"  {style[k][2]:<38} = {tail(k):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
