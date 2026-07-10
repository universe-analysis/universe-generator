"""PACK campaign chart: is the packing law N(T) invariant under the cutoff?

Overlays the seed-averaged jam count N(T) from the 1e-6 and 1e-7 arms of a
PACK campaign on a log-log grid, with an independent power-law fit per arm.
The exponent is the packing dimension; if the two fits agree, the exponent is
cutoff-robust and the cutoff only moves the prefactor (reported as the mean
per-T ratio N_e7/N_e6, i.e. the per-decade depth gain).

Usage::

    uv run python -m plots.plot_pack_nvt \
        --db-e6 data/pack/pack3d_e6.db --db-e7 data/pack/pack3d_e7.db \
        --dim 3 --out pack3d_nvt.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from braidlab.store import Store


def _seed_stats(
    db_path: Path, dim: int, band: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-T seed mean and SEM of the jam count (2-term runs only)."""
    by_t: dict[int, list[int]] = defaultdict(list)
    for r in Store(db_path).results(dim, band):
        if r.terms == 2 and r.n_final is not None:
            by_t[r.t].append(r.n_final)
    if not by_t:
        raise SystemExit(f"no completed 2-term runs for dim={dim} in {db_path}")
    t = np.array(sorted(by_t))
    mean = np.array([np.mean(by_t[v]) for v in t])
    sem = np.array([np.std(by_t[v], ddof=1) / np.sqrt(len(by_t[v])) for v in t])
    return t, mean, sem


def plot(db_e6: Path, db_e7: Path, out_path: Path, dim: int, band: str = "nyq") -> None:
    import matplotlib.pyplot as plt

    arms = [
        ("1e-6 cutoff", db_e6, "tab:blue", "s"),
        ("1e-7 cutoff", db_e7, "tab:red", "o"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    fits: dict[str, tuple[float, float]] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for label, db, color, marker in arms:
        t, mean, sem = _seed_stats(db, dim, band)
        curves[label] = (t, mean)
        slope, logc = np.polyfit(np.log(t), np.log(mean), 1)
        fits[label] = (slope, logc)
        ax.errorbar(
            t,
            mean,
            yerr=sem,
            fmt=marker,
            color=color,
            ms=6,
            lw=0,
            elinewidth=1.2,
            capsize=2,
            zorder=5,
            label=label,
        )
        fit_t = np.array([t[0] * 0.85, t[-1] * 1.18])
        ax.plot(
            fit_t,
            np.exp(logc) * fit_t**slope,
            "--",
            color=color,
            lw=1.3,
            label=f"{label} fit:  N ~ T^{slope:.3f}",
        )

    # Prefactor shift: mean per-T ratio on the common T grid.
    (t6, n6), (t7, n7) = curves["1e-6 cutoff"], curves["1e-7 cutoff"]
    common = np.intersect1d(t6, t7)
    ratio = np.mean([n7[t7 == t][0] / n6[t6 == t][0] for t in common])
    ax.text(
        0.03,
        0.97,
        f"mean N(1e-7)/N(1e-6) on shared T grid: {ratio:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T (timesteps)")
    ax.set_ylabel("jam count N (seed mean)")
    ax.set_title(f"{dim}+1 PACK: N(T) under cutoff depth ({band} band, 2 terms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")
    for label, (slope, _) in fits.items():
        print(f"  {label}: exponent {slope:.4f}")
    print(f"  prefactor ratio e7/e6: {ratio:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-e6", type=Path, required=True)
    parser.add_argument("--db-e7", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("pack_nvt.png"))
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--band", default="nyq")
    args = parser.parse_args()
    plot(args.db_e6, args.db_e7, args.out, args.dim, args.band)


if __name__ == "__main__":
    main()
