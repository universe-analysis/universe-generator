"""Turnaround kinematics of the phase model: the arcsine speed law and w.

In the phase model the peculiar velocity of a worldline at the turnaround has
a closed form. Per axis, the comoving position is
X(z) = sum_j a_j [sin(b_j z + f_j) - sin f_j]/sin z + a2, and at z = pi/2
(where sin z = 1 and cos z = 0) the peculiar speed collapses to
|sum_j a_j b_j cos(b_j pi/2 + f_j)|. In the legacy single-wiggle model the
rapidity constraint |ab| = 1 makes that |cos(b pi/2 + f)|: odd frequencies
(f = 0) are exactly at rest and the (at most one, by coprimality) even axis
moves at |cos f|. With f ~ U[0, pi), the mover speed is then
arcsine-distributed,

    P(v) = 2 / (pi sqrt(1 - v^2)),   v in [0, 1),

with <v> = 2/pi and <v^2> = 1/2 -- a parameter-free prediction, including the
integrable divergence at the speed cap v = 1. The turnaround equation of
state follows: w = (1/3) <E v^2>/<E> = (1/6) x (mobile fraction) whenever the
energy dictionary is parity-blind.

Multi-term dumps (--terms, including the full-spectrum terms = T model) are
handled by the same estimator: the speed sums over every wiggle term, the
mobile set is "any even frequency on any axis", and the wave-dictionary
energy is the total frequency sum(b) over all terms. For terms = T every
worldline carries even frequencies, so the mobile fraction is 1 by
construction and the arcsine overlay probes whether the multi-term speed sum
keeps the single-term law.

Usage::

    python -m analysis.analyze_phase_kinematics --dumps data/torus/dumps \
        --dim 3 --suffix _tor_ph_e6 --t 200 --out figures/phase_kinematics.png
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

from braidlab.corrdim import AxisTerms, load_axis_terms

_NAME_RE = re.compile(r"_T(?P<t>\d+)_s(?P<seed>\d+)")
HALF_PI = np.pi / 2.0


def turnaround_speed(axes: list[AxisTerms]) -> np.ndarray:
    """Peculiar speed |dx_pec/dz| at z = pi/2 per worldline (Euclidean norm).

    The sin1 term's physical velocity a2 cos z vanishes at the turnaround, so
    only the wiggle terms contribute.
    """
    v2 = np.zeros(len(axes[0].a2))
    for ax in axes:
        v_axis = (ax.a * ax.b * np.cos(ax.b * HALF_PI + ax.f)).sum(axis=1)
        v2 += v_axis**2
    return np.sqrt(v2)


def mobile_mask(axes: list[AxisTerms]) -> np.ndarray:
    """Worldlines carrying an even frequency (the kinematically mobile set)."""
    mask = np.zeros(len(axes[0].a2), dtype=bool)
    for ax in axes:
        mask |= (ax.b.astype(np.int64) % 2 == 0).any(axis=1)
    return mask


def wave_energy(axes: list[AxisTerms]) -> np.ndarray:
    """E ~ total frequency: sum of b over every axis and term."""
    return np.sum([ax.b.sum(axis=1) for ax in axes], axis=0)


def arc_length(axes: list[AxisTerms], nz: int = 400) -> np.ndarray:
    """Physical arc length of each worldline's spatial path over the loop."""
    z = np.linspace(1e-3, np.pi - 1e-3, nz)
    dz = z[1] - z[0]
    speed2 = np.zeros((len(axes[0].a2), nz))
    for ax in axes:
        # physical x(z) = sum_j a_j[sin(b_j z + f_j) - sin f_j] + a2 sin z;
        # the constant -a sin f offsets differentiate away. Terms accumulate
        # one at a time to keep the (N, nz) working set flat in nw.
        dx = ax.a2[:, None] * np.cos(z)
        for j in range(ax.a.shape[1]):
            dx += (
                ax.a[:, j : j + 1]
                * ax.b[:, j : j + 1]
                * np.cos(np.outer(ax.b[:, j], z) + ax.f[:, j : j + 1])
            )
        speed2 += dx**2
    return np.sqrt(speed2).sum(axis=1) * dz


def eos_w(energy: np.ndarray, v: np.ndarray) -> float:
    """w = P/rho = (1/3) <E v^2> / <E>."""
    return float((energy * v**2).sum() / (3.0 * energy.sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="data/torus/dumps")
    parser.add_argument("--dim", type=int, default=3, choices=(2, 3))
    parser.add_argument("--suffix", default="_tor_ph_e6")
    parser.add_argument("--t", type=int, default=200, help="T for the speed panel")
    parser.add_argument(
        "--out", type=Path, default=Path("figures/phase_kinematics.png")
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- speed distribution + string-dictionary cross-check at one T ----
    paths_t = sorted(
        glob.glob(f"{args.dumps}/d{args.dim}_nyq_T{args.t}_s*{args.suffix}.csv")
    )
    speeds, w_wave_t, w_string_t = [], [], []
    for p in paths_t:
        axes = load_axis_terms(p)
        v = turnaround_speed(axes)
        speeds.append(v[mobile_mask(axes)])
        w_wave_t.append(eos_w(wave_energy(axes), v))
        w_string_t.append(eos_w(arc_length(axes), v))
    movers = np.concatenate(speeds)
    print(
        f"T={args.t}: {len(paths_t)} seeds, {len(movers):,} movers; "
        f"<v>={movers.mean():.4f} (arcsine: {2 / np.pi:.4f}), "
        f"<v^2>={np.mean(movers**2):.4f} (arcsine: 0.5)"
    )
    print(
        f"w(turnaround, T={args.t}) = {np.mean(w_wave_t):.4f} (E~b) / "
        f"{np.mean(w_string_t):.4f} (E~length)"
    )

    # ---- w and mobile fraction vs T (wave dictionary) ----
    by_t: dict[int, list[tuple[float, float]]] = {}
    for p in sorted(glob.glob(f"{args.dumps}/d{args.dim}_nyq_T*_s*{args.suffix}.csv")):
        m = _NAME_RE.search(Path(p).name)
        if not m:
            continue
        axes = load_axis_terms(p)
        v = turnaround_speed(axes)
        frac = float(mobile_mask(axes).mean())
        by_t.setdefault(int(m["t"]), []).append((eos_w(wave_energy(axes), v), frac))
    ts = sorted(by_t)
    w_mean = [float(np.mean([x[0] for x in by_t[t]])) for t in ts]
    frac_mean = [float(np.mean([x[1] for x in by_t[t]])) for t in ts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(
        movers,
        bins=60,
        range=(0, 1),
        density=True,
        alpha=0.65,
        color="steelblue",
        label=f"measured movers (T={args.t})",
    )
    vv = np.linspace(0, 0.9995, 400)
    ax1.plot(
        vv,
        2.0 / (np.pi * np.sqrt(1 - vv**2)),
        "-",
        color="tab:red",
        lw=2,
        label=r"arcsine law  $2/(\pi\sqrt{1-v^2})$",
    )
    ax1.set_xlabel("peculiar speed v at the turnaround")
    ax1.set_ylabel("probability density")
    ax1.set_title("Mover speed distribution vs the parameter-free prediction")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, w_mean, "o-", color="tab:green", label=r"measured $w$ ($E\sim b$)")
    ax2.plot(
        ts,
        [f / 6.0 for f in frac_mean],
        "s--",
        color="tab:gray",
        label=r"prediction $\frac{1}{6}\times$ mobile fraction",
    )
    ax2.set_xlabel("T (resolution)")
    ax2.set_ylabel(r"$w = P/\rho$ at the turnaround")
    ax2.set_title("Equation of state vs the arcsine prediction")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
