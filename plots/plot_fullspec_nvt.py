"""FULLSPEC chart: the jam count N(T) of the terms = T model vs the baseline.

Overlays the seed-averaged jam count of a full-spectrum campaign (terms
tracks T: every axis carries the whole frequency pool) on the legacy 2-term
baseline from a PACK store, log-log, with an independent power-law fit per
arm and the fullspec arm's adjacent-rung local slopes printed to stdout
(the fullspec exponent is still drifting upward at the ladder top, so the
single fit is a window average, not a converged limit).

Usage::

    uv run python -m plots.plot_fullspec_nvt \
        --db-fullspec data/fullspec/fullspec3d_e6.db \
        --db-baseline data/pack/pack3d_e6.db --dim 3 --out fullspec3d_nvt.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from braidlab.store import Store


def seed_stats(
    db_path: Path, dim: int, band: str, fullspec: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-T seed mean and SEM of the jam count.

    ``fullspec`` selects the terms == T rows; otherwise the 2-term baseline.
    """
    by_t: dict[int, list[int]] = defaultdict(list)
    for r in Store(db_path).results(dim, band):
        if r.n_final is None:
            continue
        if (fullspec and r.terms == r.t) or (not fullspec and r.terms == 2):
            by_t[r.t].append(r.n_final)
    if not by_t:
        kind = "fullspec" if fullspec else "2-term"
        raise SystemExit(f"no completed {kind} runs for dim={dim} in {db_path}")
    t = np.array(sorted(by_t))
    mean = np.array([np.mean(by_t[v]) for v in t])
    sem = np.array([np.std(by_t[v], ddof=1) / np.sqrt(len(by_t[v])) for v in t])
    return t, mean, sem


def plot(
    db_fullspec: Path, db_baseline: Path, out_path: Path, dim: int, band: str = "nyq"
) -> None:
    import matplotlib.pyplot as plt

    arms = [
        ("terms = T (full spectrum)", db_fullspec, True, "tab:red", "o"),
        ("terms = 2 (baseline)", db_baseline, False, "tab:blue", "s"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for label, db, fullspec, color, marker in arms:
        t, mean, sem = seed_stats(db, dim, band, fullspec)
        slope, intercept = np.polyfit(np.log(t), np.log(mean), 1)
        ax.errorbar(
            t,
            mean,
            yerr=sem,
            fmt=marker,
            color=color,
            ms=5,
            capsize=3,
            lw=0,
            elinewidth=1,
        )
        tt = np.linspace(t.min(), t.max(), 100)
        ax.plot(
            tt,
            np.exp(intercept) * tt**slope,
            "-",
            color=color,
            lw=1.5,
            label=f"{label}:  N ~ T^{slope:.3f}",
        )
        if fullspec:
            local = np.log(mean[1:] / mean[:-1]) / np.log(t[1:] / t[:-1])
            print(f"dim={dim} fullspec local slopes (adjacent rungs):")
            for a, b, s in zip(t[:-1], t[1:], local):
                print(f"  T {a:>3} -> {b:>3}: {s:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T (timesteps)")
    ax.set_ylabel("jam count N")
    ax.set_title(f"{dim}+1 packing number: full-spectrum (terms = T) vs baseline")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-fullspec", type=Path, required=True)
    parser.add_argument("--db-baseline", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=3, choices=(2, 3))
    parser.add_argument("--band", default="nyq")
    parser.add_argument("--out", type=Path, required=True)
    plot_args = parser.parse_args()
    plot(
        plot_args.db_fullspec,
        plot_args.db_baseline,
        plot_args.out,
        plot_args.dim,
        plot_args.band,
    )


if __name__ == "__main__":
    main()
