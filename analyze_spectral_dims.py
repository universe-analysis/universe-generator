"""Two more independent angles on D: the Dq spectrum and the structure factor.

(1) Generalized dimensions Dq (Grassberger-Procaccia). For each point the local
    neighbour fraction n_i(r) is measured; the q-th moment gives

        C_q(r) = < n_i(r)^(q-1) >^(1/(q-1)) ~ r^Dq        (q != 1)
        D_1    = d < log n_i(r) > / d log r               (information dim)

    For a monofractal all Dq coincide; Dq decreasing in q is multifractal. q=2
    reproduces the ordinary correlation dimension (~2.79 in 3+1) as a check.

(2) Structure factor S(k) = <|sum_j exp(i k.r_j)|^2> / N -- the reciprocal-space
    view (a diffraction pattern). For a mass fractal S(k) ~ k^(-D) over the
    scaling window, so the slope of log S vs log k gives -D, with completely
    different systematics from the real-space methods. Computed isotropically and
    along axes vs diagonals (a direct reciprocal-space look at the anisotropy).

Usage::

    python analyze_spectral_dims.py --out figures/spectral_dims.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from braidlab.corrdim import FIT_HI, FIT_LO, load_turnaround_cloud

# 3+1 and 2+1: (dim, dumps dir, representative T, seeds to average, label).
CASES = [
    (3, "data/corrdim/dumps", 200, 3, "3+1"),
    (2, "data/corrdim2d/dumps", 400, 3, "2+1"),
]
QS = [1, 2, 3, 4, 5]


def fit_slope(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    mask = (x >= lo) & (x <= hi) & np.isfinite(y)
    return float(np.polyfit(np.log(x[mask]), y[mask], 1)[0])


def generalized_dims(pts: np.ndarray, cell: float, n_centers: int = 2500) -> dict:
    """Dq for each q in QS via the generalized correlation integral."""
    radii = np.logspace(np.log10(cell), np.log10(0.7), 40)
    tree = KDTree(pts)
    n = len(pts)
    centers = pts[:: max(1, n // n_centers)]
    counts = np.empty((len(centers), len(radii)))
    for ci, c in enumerate(centers):
        idx = tree.query_ball_point(c, radii[-1])
        d = np.sort(np.linalg.norm(pts[idx] - c, axis=1))
        counts[ci] = np.searchsorted(d, radii, side="right")
    frac = np.clip((counts - 1) / (n - 1), 1e-15, None)  # drop self, normalise

    dq = {}
    for q in QS:
        if q == 1:
            curve = np.mean(np.log(frac), axis=0)  # <log n>; D1 = its slope
        else:
            curve = np.log(np.mean(frac ** (q - 1), axis=0)) / (q - 1)
        dq[q] = fit_slope(radii, curve, FIT_LO, FIT_HI)
    return dq


def structure_factor(pts: np.ndarray, kmag: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """S(k) averaged over the supplied unit directions, per |k|."""
    n = len(pts)
    out = np.empty(len(kmag))
    for i, k in enumerate(kmag):
        s = 0.0
        for u in dirs:
            phase = pts @ (k * u)
            s += (np.cos(phase).sum() ** 2 + np.sin(phase).sum() ** 2) / n
        out[i] = s / len(dirs)
    return out


def axis_and_diagonal_dirs(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random (isotropic), axis-aligned, and body-diagonal unit directions."""
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((30, dim))
    iso /= np.linalg.norm(iso, axis=1, keepdims=True)
    axis = np.eye(dim)
    diag = np.array(np.meshgrid(*([[-1, 1]] * dim))).reshape(dim, -1).T / np.sqrt(dim)
    return iso, axis, diag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/spectral_dims.png"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # k window matching the real-space fit window r in [FIT_LO, FIT_HI].
    k_lo, k_hi = 2 * np.pi / FIT_HI, 2 * np.pi / FIT_LO
    kmag = np.logspace(np.log10(k_lo / 3), np.log10(k_hi * 2), 32)

    for dim, dumps_dir, t, nseed, label in CASES:
        cell = 2.0 / t
        paths = sorted(glob.glob(f"{dumps_dir}/d{dim}_nyq_T{t}_s*.csv"))[:nseed]

        # --- Dq spectrum (seed-averaged) ---
        dq_runs = [generalized_dims(load_turnaround_cloud(p), cell) for p in paths]
        dq_mean = {q: np.mean([r[q] for r in dq_runs]) for q in QS}
        dq_sem = {q: np.std([r[q] for r in dq_runs], ddof=1) / np.sqrt(nseed)
                  for q in QS}
        print(f"=== {label} Dq spectrum (T={t}) ===")
        for q in QS:
            print(f"  D{q} = {dq_mean[q]:.3f} +/- {dq_sem[q]:.3f}")
        axes[0].errorbar(QS, [dq_mean[q] for q in QS],
                         yerr=[dq_sem[q] for q in QS], fmt="o-", capsize=3,
                         label=f"{label} (D2={dq_mean[2]:.2f})")

        # --- structure factor S(k) ---
        cloud = load_turnaround_cloud(paths[0])
        iso, axis, diag = axis_and_diagonal_dirs(dim)
        s_iso = structure_factor(cloud, kmag, iso)
        s_axis = structure_factor(cloud, kmag, axis)
        s_diag = structure_factor(cloud, kmag, diag)
        d_iso = -fit_slope(kmag, np.log(s_iso), k_lo, k_hi)
        d_axis = -fit_slope(kmag, np.log(s_axis), k_lo, k_hi)
        d_diag = -fit_slope(kmag, np.log(s_diag), k_lo, k_hi)
        print(f"  S(k) slope -> D: iso={d_iso:.2f} axis={d_axis:.2f} "
              f"diag={d_diag:.2f}")

        ax = axes[1] if dim == 3 else axes[2]
        ax.loglog(kmag, s_iso, "o-", color="tab:blue",
                  label=f"isotropic (D={d_iso:.2f})")
        ax.loglog(kmag, s_axis, "s--", color="tab:red",
                  label=f"axis (D={d_axis:.2f})")
        ax.loglog(kmag, s_diag, "^:", color="tab:green",
                  label=f"diagonal (D={d_diag:.2f})")
        ax.axvspan(k_lo, k_hi, color="gray", alpha=0.12, label="fit window")
        ax.set_xlabel("k")
        ax.set_ylabel("S(k)")
        ax.set_title(f"({label}) structure factor  S(k) ~ k^-D")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_xlabel("q")
    axes[0].set_ylabel("Dq")
    axes[0].set_title("Generalized dimensions Dq (flat = monofractal)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
