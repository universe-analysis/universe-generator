"""Collision geometry (cube vs sphere) for correlation AND box-counting (3+1).

This is the apples-to-apples chart: the *estimator* (and its probe) is held
fixed, and we vary the **collision geometry the data was generated with** --
the default Chebyshev cube exclusion vs the L2 "euclid" sphere exclusion.

Two estimators, each measured on both collision datasets, seed-averaged vs T:

  correlation D2 (L2 probe)   on cube-collision dumps   -> "correlation, cube"
  correlation D2 (L2 probe)   on sphere-collision dumps -> "correlation, sphere"
  box-counting D0 (cube cells) on cube-collision dumps  -> "box-counting, cube"
  box-counting D0 (cube cells) on sphere-collision dumps-> "box-counting, sphere"

Unlike analyze_box_geometry.py (which varies the *probe* L2/L-inf on one cloud),
here the probe is the natural one for each estimator and the axis being compared
is the packing exclusion shape used during generation.

Usage::

    python analyze_collision_geometry.py \
        --cube-dumps data/corrdim/dumps \
        --sphere-dumps data/corrdim3d_euclid/dumps \
        --out figures/collision_geometry.png
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

from braidlab.corrdim import (
    FIT_HI,
    FIT_LO,
    box_count,
    correlation_integral,
    fit_slope,
    load_turnaround_cloud,
)

_NAME_RE = re.compile(r"_T(?P<t>\d+)_s(?P<seed>\d+)")


def measure(path: str, t: int) -> dict[str, float]:
    """Correlation D2 (L2 probe) and box-counting D0 (cube cells) for one dump."""
    cloud = load_turnaround_cloud(path)
    cell = 2.0 / t
    radii = np.logspace(np.log10(cell), np.log10(0.7), 40)
    sizes = np.logspace(np.log10(cell), np.log10(0.7), 18)
    corr = fit_slope(radii, correlation_integral(cloud, radii, p=2.0), FIT_LO, FIT_HI)
    box = -fit_slope(sizes, box_count(cloud, sizes), FIT_LO, FIT_HI)
    return {"corr": corr, "box": box}


def collect(
    dumps: str, seeds: int
) -> tuple[list[int], dict[str, list], dict[str, list]]:
    """Seed-average both estimators across the T grid found under ``dumps``."""
    by_t: dict[int, list[str]] = {}
    for p in sorted(glob.glob(f"{dumps}/d3_nyq_T*_s*.csv")):
        m = _NAME_RE.search(Path(p).name)
        if m:
            by_t.setdefault(int(m["t"]), []).append(p)

    ts = sorted(by_t)
    mean = {"corr": [], "box": []}
    sem = {"corr": [], "box": []}
    for t in ts:
        runs = [measure(p, t) for p in by_t[t][:seeds]]
        for k in ("corr", "box"):
            vals = np.array([r[k] for r in runs])
            mean[k].append(float(vals.mean()))
            sem[k].append(
                float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            )
    return ts, mean, sem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-dumps", default="data/corrdim/dumps")
    parser.add_argument("--sphere-dumps", default="data/corrdim3d_euclid/dumps")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--out", type=Path, default=Path("figures/collision_geometry.png")
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("cube-collision dumps:")
    cube_ts, cube_mean, cube_sem = collect(args.cube_dumps, args.seeds)
    print("sphere-collision dumps:")
    sphere_ts, sphere_mean, sphere_sem = collect(args.sphere_dumps, args.seeds)

    # Labels name the fixed probe first, then the generation geometry, so this
    # chart is never confused with the probe-comparison charts (where
    # "cube/sphere" is the measurement neighborhood, not the packing exclusion).
    # (T grid, mean, sem, color, marker, label, label y-offset in points)
    series = [
        (cube_ts, cube_mean["corr"], cube_sem["corr"],
         "tab:blue", "o-", "correlation D2 (L2 probe), cube-generated", 9),
        (sphere_ts, sphere_mean["corr"], sphere_sem["corr"],
         "tab:cyan", "o--", "correlation D2 (L2 probe), sphere-generated", 9),
        (cube_ts, cube_mean["box"], cube_sem["box"],
         "tab:orange", "^-", "box-counting D0 (cube cells), cube-generated", -13),
        (sphere_ts, sphere_mean["box"], sphere_sem["box"],
         "tab:red", "^--", "box-counting D0 (cube cells), sphere-generated", -13),
    ]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    for ts, mean, sem, color, fmt, label, dy in series:
        m = np.array(mean)
        ax.errorbar(ts, m, yerr=sem, fmt=fmt, color=color, ms=4, capsize=3, label=label)
        for t, v in zip(ts, m):
            ax.annotate(f"{v:.2f}", (t, v), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=6, color=color)

    ax.axhline(2.79, color="black", ls="--", lw=1, label="converged D ~ 2.79")
    all_ts = sorted(set(cube_ts) | set(sphere_ts))
    clean_t = min((t for t in all_ts if 2.0 / t < FIT_LO / 2), default=max(all_ts))
    ax.axvspan(min(all_ts) - 5, clean_t, color="red", alpha=0.08,
               label="CELL intrudes on fit window")
    ax.set_xlabel("T (resolution)")
    ax.set_ylabel("dimension")
    ax.set_title(
        "Collision geometry: cube- vs sphere-GENERATED packing, probe fixed (3+1)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5,
        0.005,
        "Probe held fixed (correlation = L2 ball, box-counting = cube cells); "
        "only the generation collision shape varies. In the probe-comparison "
        "charts, 'cube/sphere' instead means the measurement neighborhood.",
        ha="center",
        fontsize=7.5,
        color="gray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)

    def tail(ts: list[int], mean: list[float]) -> float:
        return float(np.mean([v for t, v in zip(ts, mean) if t >= 100]))

    corr_cube = tail(cube_ts, cube_mean["corr"])
    corr_sphere = tail(sphere_ts, sphere_mean["corr"])
    box_cube = tail(cube_ts, cube_mean["box"])
    box_sphere = tail(sphere_ts, sphere_mean["box"])
    print("\nconverged (T>=100 mean):")
    print(f"  correlation D2  cube collision   = {corr_cube:.3f}")
    print(f"  correlation D2  sphere collision = {corr_sphere:.3f}")
    print(f"  box-counting D0 cube collision   = {box_cube:.3f}")
    print(f"  box-counting D0 sphere collision = {box_sphere:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
