"""The turnaround dent: a full-spectrum-only anomaly in slice box-occupancy.

The uniformity-over-time trace of the full-spectrum (terms = T) model shows a
small V-shaped dip in the per-slice coarse box-counting dimension centered on
the turnaround z = pi/2 (Kevin spotted it on the 2026-07-26 chart); the
baseline single-wiggle model instead has its broad maximum there. This script
quantifies it: the mean slice box-dim over a control band (z/pi within 0.05
of 0.30 or 0.70) minus the mid-slice value, per seed, with a paired t
statistic, for a ladder of (dim, T, suffix) cases.

Established facts (2026-07-26): the dip is highly significant (paired-t ~ 30
at 3+1 T = 100), grows with T in 3+1 (+0.002 at T = 60 to +0.009 at T = 100),
has a 2+1 twin, and persists at the exact turnaround. It is NOT the
b = 0 (mod 4) term silencing (removing those terms at a control z leaves the
box-dim unchanged), and it is invisible to the wrapped correlation dimension
(D2 = 3.026 at z = 0.3pi, pi/2, and 0.7pi alike) -- a coarse-occupancy
rearrangement, not a scaling-exponent change. Mechanism open; the leading
suspect is the turnaround velocity freeze-out (w = d/(6T): the exclusion
constraint binds a nearly static configuration there, while the baseline's
arcsine-fast movers smear it).

Usage::

    uv run python -m analysis.analyze_turnaround_dent \
        --out figures/turnaround_dent.png
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from braidlab.corrdim import comoving_trajectories, load_axis_terms

#: Coarse spatial window, matching analyze_uniformity_over_time.SLICE_SIZES.
SIZES = np.array([0.05, 0.075, 0.1125, 0.169])
#: Control band: slices with z/pi within this half-width of 0.30 or 0.70.
CTRL_HALFWIDTH = 0.05


@dataclass(frozen=True)
class Case:
    """One (dimension, T) dump family to measure."""

    dim: int
    t: int
    suffix: str
    label: str


CASES = (
    Case(3, 60, "_fs3e6", "3+1 T=60"),
    Case(3, 80, "_fs3e6x", "3+1 T=80"),
    Case(3, 100, "_fs3e6x", "3+1 T=100"),
    Case(2, 100, "_fs2e6", "2+1 T=100"),
    Case(2, 150, "_fs2e6x", "2+1 T=150"),
)


def slice_boxdim_nd(pts: np.ndarray) -> float:
    """Coarse box-counting dimension of one comoving slice (any dim)."""
    counts = []
    for s in SIZES:
        c = np.floor(pts / s).astype(np.int64) + 64
        key = c[:, 0]
        for k in range(1, pts.shape[1]):
            key = key * (1 << 16) + c[:, k]
        counts.append(len(np.unique(key)))
    return float(-np.polyfit(np.log(SIZES), np.log(counts), 1)[0])


def dent_traces(dumps: str, case: Case, seeds: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed slice box-dim traces: returns (z/pi, (seeds, T) array)."""
    pattern = f"{dumps}/d{case.dim}_nyq_T{case.t}_s*{case.suffix}.csv"
    paths = sorted(glob.glob(pattern))[:seeds]
    step = np.pi / (case.t + 1)
    z = step + np.arange(case.t) * step
    dims = []
    for p in paths:
        axs = comoving_trajectories(load_axis_terms(p), z, wrap=True)
        dims.append(
            [
                slice_boxdim_nd(np.stack([a[:, j] for a in axs], axis=1))
                for j in range(case.t)
            ]
        )
    return z / np.pi, np.array(dims)


def dent_stat(zpi: np.ndarray, dims: np.ndarray) -> tuple[float, float, float]:
    """(dip mean, dip sem, paired-t) of control-band minus mid-slice."""
    mid = int(np.argmin(np.abs(zpi - 0.5)))
    ctrl = (np.abs(zpi - 0.30) < CTRL_HALFWIDTH) | (np.abs(zpi - 0.70) < CTRL_HALFWIDTH)
    d = dims[:, ctrl].mean(axis=1) - dims[:, mid]
    sem = float(d.std(ddof=1) / np.sqrt(len(d)))
    return float(d.mean()), sem, float(d.mean() / sem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="data/fullspec/dumps")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("figures/turnaround_dent.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax3, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    panels = {3: ax3, 2: ax2}
    colors = {60: "tab:cyan", 80: "tab:blue", 100: "tab:purple", 150: "tab:red"}
    for case in CASES:
        zpi, dims = dent_traces(args.dumps, case, args.seeds)
        dip, sem, t_stat = dent_stat(zpi, dims)
        print(
            f"{case.label:10s}: dip = {dip:+.4f} +/- {sem:.4f}  paired-t = {t_stat:.1f}"
        )
        mean = dims.mean(axis=0)
        # Center each trace on its control-band level so the dips overlay.
        ctrl = (np.abs(zpi - 0.30) < CTRL_HALFWIDTH) | (
            np.abs(zpi - 0.70) < CTRL_HALFWIDTH
        )
        panels[case.dim].plot(
            zpi,
            mean - mean[ctrl].mean(),
            "-",
            lw=1.5,
            color=colors[case.t],
            label=f"{case.label}  (dip {dip:+.4f}, t={t_stat:.0f})",
        )
    for dim, ax in panels.items():
        ax.axvline(0.5, color="gray", ls=":", lw=1)
        ax.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlim(0.15, 0.85)
        ax.set_xlabel("conformal time z / pi")
        ax.set_title(f"{dim}+1 full-spectrum: slice box-dim, control-band centered")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    panels[3].set_ylabel("slice box-dim minus control-band mean")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
