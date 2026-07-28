"""FULLSUB campaign summary: capacity, growth decay, and dust-law stability.

Three panels comparing the full-spectrum subpath campaign (fullsub2d_e6,
terms = T) with the 2-term baseline (subpath2d_e6):

  1. Subpath capacity Nsub/N at the collected depth vs T (log y).
  2. Tail growth exponent gamma of Nsub(attempts) vs T (subpaths never jam:
     gamma near 1 means growth barely decays at achievable depth).
  3. Turnaround w: uniques-only vs all-paths (uniques + subpaths) vs the
     d/(6T) dust law -- the trend-stability check (does filling the packing
     with arbitrarily many subpaths move the ensemble kinematics?).

Usage::

    uv run python -m plots.plot_fullsub_summary --out fullsub_summary.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import sqlite3
from pathlib import Path

import numpy as np

from analysis.analyze_phase_kinematics import eos_w, turnaround_speed, wave_energy
from braidlab.corrdim import load_axis_terms
from plots.plot_subpath_decay import tail_gamma

FULLSUB_DB = Path("data/fullspec/fullsub2d_e6.db")
BASELINE_DB = Path("data/converge/subpath2d_e6.db")
DUMPS = Path("data/fullspec/dumps")
#: T rungs for the (expensive) kinematics panel.
W_T = (20, 40, 60, 80, 100)


def curve_stats(db: Path) -> dict[int, dict[str, float]]:
    """Per-T seed means of the final Nsub/N ratio and the tail gamma."""
    conn = sqlite3.connect(db)
    ratios: dict[int, list[float]] = {}
    gammas: dict[int, list[float]] = {}
    for t, cp in conn.execute(
        "select t, curve_path from runs where status='done' order by t, seed"
    ):
        try:
            rows = list(csv.DictReader(open(cp)))
        except FileNotFoundError:
            continue
        if not rows or "nsub" not in rows[0]:
            continue
        att = np.array([float(r["attempts"]) for r in rows])
        n = float(rows[-1]["n"])
        nsub = np.array([float(r["nsub"]) for r in rows])
        if n > 0:
            ratios.setdefault(t, []).append(nsub[-1] / n)
        g = tail_gamma(att, nsub)
        if np.isfinite(g):
            gammas.setdefault(t, []).append(g)
    out: dict[int, dict[str, float]] = {}
    for t in ratios:
        out[t] = {
            "ratio": float(np.mean(ratios[t])),
            "gamma": float(np.mean(gammas.get(t, [np.nan]))),
            "gamma_sem": float(np.std(gammas[t], ddof=1) / np.sqrt(len(gammas[t])))
            if len(gammas.get(t, [])) > 1
            else 0.0,
        }
    return out


def w_stats(pattern: str) -> tuple[float, float]:
    """Seed mean and SEM of turnaround w (E ~ b) over a dump glob."""
    ws = []
    for p in sorted(glob.glob(pattern)):
        axes = load_axis_terms(p)
        ws.append(eos_w(wave_energy(axes), turnaround_speed(axes)))
    return float(np.mean(ws)), float(np.std(ws, ddof=1) / np.sqrt(len(ws)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("fullsub_summary.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = curve_stats(FULLSUB_DB)
    base = curve_stats(BASELINE_DB)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    for stats, color, label in (
        (full, "tab:red", "terms = T (fullsub, 1e9 budget)"),
        (base, "tab:blue", "terms = 2 (baseline, 1e10 budget)"),
    ):
        ts = sorted(stats)
        ax1.semilogy(
            ts, [stats[t]["ratio"] for t in ts], "o-", color=color, label=label
        )
        ax2.errorbar(
            ts,
            [stats[t]["gamma"] for t in ts],
            yerr=[stats[t]["gamma_sem"] for t in ts],
            fmt="o-",
            color=color,
            capsize=3,
            label=label,
        )
    ax1.set_xlabel("T")
    ax1.set_ylabel("subpaths per unique  Nsub/N")
    ax1.set_title("Subpath capacity at collected depth")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(1.0, color="gray", ls=":", lw=1)
    ax2.set_xlabel("T")
    ax2.set_ylabel(r"tail exponent  $\gamma$:  $N_{sub} \sim$ attempts$^\gamma$")
    ax2.set_title("Subpath growth decay (no jam in either model)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    tt = np.array(W_T)
    uniq = [w_stats(str(DUMPS / f"d2_nyq_T{t}_s*_ph_tm{t}_fs2e6.csv")) for t in W_T]
    allp = [w_stats(str(DUMPS / f"d2_nyq_T{t}_s*_sub_fsub2e6.csv")) for t in W_T]
    ax3.errorbar(
        tt,
        [u[0] for u in uniq],
        yerr=[u[1] for u in uniq],
        fmt="s",
        color="tab:blue",
        capsize=3,
        label="MEASURED w, uniques only",
    )
    ax3.errorbar(
        tt,
        [a[0] for a in allp],
        yerr=[a[1] for a in allp],
        fmt="o",
        color="tab:red",
        capsize=3,
        label="MEASURED w, all paths (uniques + subpaths)",
    )
    tfine = np.linspace(tt.min(), tt.max(), 200)
    ax3.plot(
        tfine,
        2.0 / (6.0 * tfine),
        "-",
        color="tab:green",
        lw=1.2,
        label="dust law d/(6T)",
    )
    ax3.set_xlabel("T")
    ax3.set_ylabel("w at the turnaround")
    ax3.set_title("Dust law survives subpath filling")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
