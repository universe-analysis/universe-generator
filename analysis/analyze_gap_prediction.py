"""Experiment 1: predict the sphere-vs-cube dimension gap from the ramp.

The cube (L-inf) correlation integral is, *under isotropy*, fully determined by
the sphere (L2) one through pure geometry. A cube of half-side r reaches L2
radius r/m(u) in direction u, where m(u) = max_i|u_i| in [1/sqrt(d), 1] (=1 along
an axis, 1/sqrt(d) along the main diagonal). So for an isotropic pair
distribution

    C_cube(r) = < C_sphere( r / m(u) ) >   averaged over directions u.

For a *perfect* monofractal C_sphere ~ r^D, this gives C_cube ~ r^D too (same
exponent): no gap. A gap appears only because D2 drifts with scale (the ramp),
and the cube reaches to larger radii (corners) where D2 is lower -> its fitted
slope is pulled down.

We predict the cube slope from each packing's *measured* sphere curve and compare
to the *observed* cube slope:

    gap_observed  = D2_sphere - D2_cube_observed
    gap_predicted = D2_sphere - D2_cube_predicted   (the isotropic, ramp-only gap)
    residual      = gap_observed - gap_predicted     (genuine anisotropy)

If residual ~ 0 the gap is pure isotropic scale-dependence; a nonzero residual is
real axis-vs-diagonal anisotropy in the packing.

Usage::

    python analyze_gap_prediction.py --out figures/gap_prediction.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from braidlab.corrdim import (
    FIT_HI,
    FIT_LO,
    correlation_integral,
    fit_slope,
    load_turnaround_cloud,
)

#: (dim, dumps dir, T values in the converged plateau, seeds to average)
CASES = [
    (3, "data/corrdim/dumps", [100, 120, 140, 160, 180, 200], 5),
    (2, "data/corrdim2d/dumps", [240, 280, 320, 360, 400], 4),
]


def direction_max_norm(dim: int, n: int = 30000) -> np.ndarray:
    """m(u) = max_i|u_i| for n uniform random unit directions in d-space."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.max(np.abs(v), axis=1)


def loglog_interp(rq: np.ndarray, rg: np.ndarray, cg: np.ndarray) -> np.ndarray:
    """Interpolate C(r) in log-log space (C spans many decades)."""
    return np.exp(np.interp(np.log(rq), np.log(rg), np.log(cg)))


def predict_cube(radii: np.ndarray, c_sphere: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Isotropic prediction C_cube(r) = mean_u C_sphere(r / m(u))."""
    pred = np.empty_like(radii)
    for i, r in enumerate(radii):
        rr = np.clip(r / m, radii[0], radii[-1])
        pred[i] = loglog_interp(rr, radii, c_sphere).mean()
    return pred


def seed_mean_C(paths: list[str], radii: np.ndarray, p: float) -> np.ndarray:
    """Average the correlation integral across seeds."""
    return np.mean(
        [correlation_integral(load_turnaround_cloud(p_), radii, p=p) for p_ in paths],
        axis=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/gap_prediction.png"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(CASES), figsize=(13, 5.5))
    for ax, (dim, dumps_dir, ts, nseed) in zip(axes, CASES):
        m = direction_max_norm(dim)
        print(f"\n=== {dim}+1  (cube reach to sqrt({dim})={np.sqrt(dim):.3f} r) ===")
        print(f"{'T':>4} {'D2_sph':>7} {'cube_obs':>9} {'cube_pred':>10} "
              f"{'gap_obs':>8} {'gap_pred':>9} {'residual':>9}")
        rows = []
        for t in ts:
            cell = 2.0 / t
            # extend the radius grid to sqrt(d)*FIT_HI so r/m is covered.
            rmax = np.sqrt(dim) * FIT_HI * 1.05
            radii = np.logspace(np.log10(cell), np.log10(rmax), 60)
            paths = sorted(glob.glob(f"{dumps_dir}/d{dim}_nyq_T{t}_s*.csv"))[:nseed]
            c_sph = seed_mean_C(paths, radii, p=2.0)
            c_cube = seed_mean_C(paths, radii, p=np.inf)
            c_pred = predict_cube(radii, c_sph, m)
            d_sph = fit_slope(radii, c_sph, FIT_LO, FIT_HI)
            d_obs = fit_slope(radii, c_cube, FIT_LO, FIT_HI)
            d_pred = fit_slope(radii, c_pred, FIT_LO, FIT_HI)
            gap_obs = d_sph - d_obs
            gap_pred = d_sph - d_pred
            rows.append((t, gap_obs, gap_pred, gap_obs - gap_pred))
            print(f"{t:>4} {d_sph:>7.3f} {d_obs:>9.3f} {d_pred:>10.3f} "
                  f"{gap_obs:>8.3f} {gap_pred:>9.3f} {gap_obs - gap_pred:>9.3f}")

        rows = np.array(rows)
        ax.plot(rows[:, 0], rows[:, 1], "o-", color="tab:purple", label="gap observed")
        ax.plot(rows[:, 0], rows[:, 2], "s--", color="tab:green",
                label="gap predicted (ramp, isotropic)")
        ax.plot(rows[:, 0], rows[:, 3], "^:", color="tab:red",
                label="residual (anisotropy)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("T (resolution)")
        ax.set_ylabel("sphere - cube  D2 gap")
        ax.set_title(f"{dim}+1  gap decomposition")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
