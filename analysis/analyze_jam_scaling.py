"""Jam-count scaling N_sat ~ T^D -- the direct packing dimension.

This is the measurement of record: how the jammed worldline count grows with the
total resolution T (a T=10 universe and a T=100 universe jam at very different
counts). The slope of log<N_sat> vs log T is D. Unlike the single-timestep
correlation dimension, this reads the packing directly and needs no snapshot.

For each dataset it reports, via braidlab.analyze.measure_d (weighted log-log
slope + seed bootstrap):
  * D +/- error over the full T ladder,
  * the power-law constancy check (local slopes; do they hold or drift?),
  * D over the high-T half alone (robustness to the low-T finite-size bend),
and it compares the acceptance-rate stopping cutoffs (1e-7 vs 1e-6) so the
cutoff sensitivity of D is explicit.

Usage::

    python analyze_jam_scaling.py --root . --out figures/jam_scaling.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from braidlab.analyze import measure_d
from braidlab.store import Store

#: (label, db path relative to --root, dim, color)
DATASETS = [
    ("3+1  cutoff 1e-7", "data/corrdim/run.db", 3, "tab:blue"),
    ("3+1  cutoff 1e-6", "data/corrdim3d_e6/run.db", 3, "tab:orange"),
    ("2+1  cutoff 1e-7", "data/corrdim2d/run.db", 2, "tab:green"),
    # Torus (new-dogma) model -- each compared against the SAME-cutoff hard-wall
    # dataset above (3+1 at 1e-6, 2+1 at 1e-7); missing dbs are skipped so the
    # script runs before/while the torus campaigns collect.
    ("3+1  torus 1e-6", "data/torus/run3d_e6.db", 3, "tab:red"),
    ("2+1  torus 1e-7", "data/torus/run2d.db", 2, "tab:purple"),
    # Phase schema (engine --phase) on the torus model, plus the 1e-6 no-phase
    # 2+1 baseline it is compared against (one-knob: phase on/off at the same
    # cutoff). The 3+1 phase dataset joins this list when its campaign lands.
    ("2+1  torus 1e-6", "data/torus/run2d_e6.db", 2, "tab:olive"),
    ("2+1  torus+phase 1e-6", "data/torus/run2d_phase_e6.db", 2, "tab:cyan"),
    ("3+1  torus+phase 1e-6", "data/torus/run3d_phase_e6.db", 3, "tab:pink"),
]
BAND = "nyq"
HIGH_T = 100  # high-T half cut for the robustness fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="dir holding data/*/run.db")
    parser.add_argument("--out", type=Path, default=Path("figures/jam_scaling.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 2]}
    )

    for label, db, dim, color in DATASETS:
        if not (root / db).exists():
            print(f"\n== {label} ==  (no db at {db}; skipped)")
            continue
        results = Store(root / db).results(dim, BAND)
        full = measure_d(results, dim, BAND)
        high = measure_d([r for r in results if r.t >= HIGH_T], dim, BAND)

        ts = np.array(full.t_values, dtype=float)
        n = np.array(full.n_mean)
        half = len(full.local_slopes) // 2
        lo_half = float(np.mean(full.local_slopes[:half]))
        hi_half = float(np.mean(full.local_slopes[half:]))
        print(f"\n== {label} ==  ({len(ts):.0f} T in {ts.min():.0f}-{ts.max():.0f}, "
              f"{full.n_seeds[0]} seeds)")
        print(f"  D (full ladder)   = {full.d:.3f} +/- {full.d_err:.3f}")
        print(f"  D (T >= {HIGH_T})     = {high.d:.3f} +/- {high.d_err:.3f}")
        print(f"  local slope: low-T half {lo_half:.2f}, high-T half {hi_half:.2f}, "
              f"scatter {full.local_slope_std:.3f}")

        # top: log-log N vs T with the full-ladder fit line
        ax.errorbar(ts, n, yerr=full.n_sem, fmt="o", color=color, ms=5, capsize=3,
                    label=f"{label}:  D = {full.d:.2f} +/- {full.d_err:.2f}")
        fit = np.exp(full.intercept) * ts**full.d
        ax.plot(ts, fit, "-", color=color, lw=1, alpha=0.8)

        # bottom: local slope vs T (midpoints) -- the power-law constancy check
        mids = np.sqrt(ts[:-1] * ts[1:])
        ax2.plot(mids, full.local_slopes, "o-", color=color, ms=4, label=label)
        ax2.axhline(full.d, color=color, ls=":", lw=1, alpha=0.6)

    # reference slopes anchored to the 3+1 1e-7 low-T point
    ref = measure_d(Store(root / DATASETS[0][1]).results(3, BAND), 3, BAND)
    t0, n0 = ref.t_values[0], ref.n_mean[0]
    tt = np.array([t0, ref.t_values[-1]], dtype=float)
    for power, name in [(2.0, "T^2"), (3.0, "T^3")]:
        ax.plot(tt, n0 * (tt / t0) ** power, "--", color="gray", lw=1, alpha=0.5)
        ax.annotate(name, (tt[-1], n0 * (tt[-1] / t0) ** power), color="gray",
                    fontsize=9, va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("jammed count  N_sat")
    ax.set_title("Jam-count scaling  N_sat ~ T^D  (the packing dimension)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("T (total timestep count)")
    ax2.set_ylabel("local slope\n(power-law constancy)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
