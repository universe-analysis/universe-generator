"""Sphere vs cube for *both* estimator families, on the turnaround cloud.

The correlation dimension already has a sphere (L2) and cube (L-inf) version.
Box-counting so far only has a cube version (partition into axis-aligned cells).
Its natural sphere analog is a **ball covering**: greedily cover the points with
radius-eps balls and count the balls -- the Minkowski dimension via balls instead
of boxes -- so D0 gets a sphere/cube pair just like D2 does.

Four estimators, seed-averaged vs T (3+1, 1e-7 dumps):
  correlation D2 sphere | correlation D2 cube | box-count cube | box-count sphere

Usage::

    python analyze_box_geometry.py --out figures/box_geometry.png
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from braidlab.corrdim import (
    FIT_HI,
    FIT_LO,
    box_count,
    correlation_integral,
    fit_slope,
    load_turnaround_cloud,
)

_NAME_RE = re.compile(r"_T(?P<t>\d+)_s(?P<seed>\d+)")
# Both box-counting estimators run on the FULL cloud -- no subsampling (which
# suppresses box-counting D: 60k->2.71, 9k->2.32). The ball cover is only
# computed over the fit-window radii: the tiny radii (where you need ~N balls,
# one per point) are the expensive part and are not used in the fit, so dropping
# them makes the full-cloud cover ~0.3 s/dump.


def ball_cover_count(pts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Greedy covering number: how many radius-eps balls cover the points."""
    tree = KDTree(pts)
    n = len(pts)
    counts = []
    for r in sizes:
        covered = np.zeros(n, dtype=bool)
        n_balls = 0
        for i in range(n):
            if covered[i]:
                continue
            n_balls += 1
            covered[tree.query_ball_point(pts[i], r)] = True
        counts.append(n_balls)
    return np.array(counts, dtype=float)


def measure(path: str, t: int, rng: np.random.Generator) -> dict:
    cloud = load_turnaround_cloud(path)
    cell = 2.0 / t
    radii = np.logspace(np.log10(cell), np.log10(0.7), 40)
    sizes = np.logspace(np.log10(cell), np.log10(0.7), 18)
    # ball cover only over the fit window (skips the expensive ~N-ball tiny radii)
    ball_sizes = np.logspace(np.log10(FIT_LO * 0.85), np.log10(FIT_HI * 1.15), 14)
    return {
        "corr_sphere": fit_slope(
            radii, correlation_integral(cloud, radii, p=2.0), FIT_LO, FIT_HI),
        "corr_cube": fit_slope(
            radii, correlation_integral(cloud, radii, p=np.inf), FIT_LO, FIT_HI),
        "box_cube": -fit_slope(sizes, box_count(cloud, sizes), FIT_LO, FIT_HI),
        "box_sphere": -fit_slope(
            ball_sizes, ball_cover_count(cloud, ball_sizes), FIT_LO, FIT_HI),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="data/corrdim/dumps")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("figures/box_geometry.png"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    by_t: dict[int, list[str]] = {}
    for p in sorted(glob.glob(f"{args.dumps}/d3_nyq_T*_s*.csv")):
        m = _NAME_RE.search(Path(p).name)
        if m:
            by_t.setdefault(int(m["t"]), []).append(p)

    keys = ["corr_sphere", "corr_cube", "box_cube", "box_sphere"]
    ts = sorted(by_t)
    mean = {k: [] for k in keys}
    sem = {k: [] for k in keys}
    print(f"{'T':>4} {'corrSph':>8} {'corrCube':>9} {'boxCube':>8} {'boxSph':>8}")
    for t in ts:
        runs = [measure(p, t, rng) for p in by_t[t][: args.seeds]]
        for k in keys:
            vals = np.array([r[k] for r in runs])
            mean[k].append(float(vals.mean()))
            sem[k].append(float(vals.std(ddof=1) / np.sqrt(len(vals)))
                          if len(vals) > 1 else 0.0)
        print(f"{t:>4} {mean['corr_sphere'][-1]:>8.3f} {mean['corr_cube'][-1]:>9.3f} "
              f"{mean['box_cube'][-1]:>8.3f} {mean['box_sphere'][-1]:>8.3f}")

    # (color, marker, label, label y-offset in points)
    style = {
        "corr_sphere": ("tab:blue", "o-", "correlation D2 (spheres)", 9),
        "corr_cube": ("tab:red", "s-", "correlation D2 (cubes)", -13),
        "box_sphere": ("tab:green", "v-", "box-counting (ball cover, full cloud)", 9),
        "box_cube": ("tab:orange", "^-", "box-counting (cube cells, full cloud)", -13),
    }
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for k in ["corr_sphere", "corr_cube", "box_sphere", "box_cube"]:
        color, fmt, lab, dy = style[k]
        m = np.array(mean[k])
        ax.errorbar(ts, m, yerr=sem[k], fmt=fmt, color=color, ms=4, capsize=3,
                    label=lab)
        for t, v in zip(ts, m):
            ax.annotate(f"{v:.2f}", (t, v), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=6, color=color)
    ax.axhline(2.79, color="black", ls="--", lw=1, label="converged D ~ 2.79")
    clean_t = min((t for t in ts if 2.0 / t < FIT_LO / 2), default=max(ts))
    ax.axvspan(min(ts) - 5, clean_t, color="red", alpha=0.08,
               label="CELL intrudes on fit window")
    ax.set_xlabel("T (resolution)")
    ax.set_ylabel("dimension")
    ax.set_title("Sphere vs cube for correlation AND box-counting (3+1)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)

    def tail(k: str) -> float:
        return float(np.mean([m for t, m in zip(ts, mean[k]) if t >= 100]))

    print("\nconverged (T>=100 mean):")
    for k in keys:
        print(f"  {style[k][2]:<28} = {tail(k):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
