"""FREQ campaign chart: the terms effect separated by T, and a pool-fraction
collapse test.

At fixed T, the jam-count enhancement from extra terms is shown as
N(T, terms) / N(T, terms=2), one series per T (left panel). The right panel
replots the same ratios against the term count as a fraction of the
frequency pool available at that resolution ((terms-1)/(T-1) unique wiggle
frequencies at the nyq band): if the per-T curves collapse there, the
controlling variable is the pool fraction rather than the raw term count.

Usage::

    uv run python -m plots.plot_terms_by_t --db data/freq/freq3d_e6.db --out out.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from braidlab.store import Store


def plot(db_path: Path, out_path: Path, dim: int = 3, band: str = "nyq") -> None:
    import matplotlib.pyplot as plt

    results = Store(db_path).results(dim, band)
    by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in results:
        if r.n_final is not None:
            by_cell[(r.t, r.terms)].append(r.n_final)
    t_values = sorted({t for t, _ in by_cell})
    term_counts = sorted({k for _, k in by_cell})
    if not t_values or 2 not in term_counts:
        raise SystemExit(f"no usable (T, terms) grid in {db_path}")

    fig, (ax_raw, ax_frac) = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    cmap = plt.get_cmap("Blues")
    shades = [cmap(x) for x in np.linspace(0.35, 0.95, len(t_values))]

    for t, color in zip(t_values, shades):
        base = np.mean(by_cell[(t, 2)])
        ks = [k for k in term_counts if (t, k) in by_cell]
        ratio = np.array([np.mean(by_cell[(t, k)]) / base for k in ks])
        # SEM of the ratio: seed scatter of numerator and denominator in
        # quadrature (5 seeds per cell).
        err = np.array(
            [
                ratio[i]
                * np.hypot(
                    np.std(by_cell[(t, k)], ddof=1)
                    / np.sqrt(len(by_cell[(t, k)]))
                    / np.mean(by_cell[(t, k)]),
                    np.std(by_cell[(t, 2)], ddof=1)
                    / np.sqrt(len(by_cell[(t, 2)]))
                    / base,
                )
                for i, k in enumerate(ks)
            ]
        )
        frac = np.array([(k - 1) / (t - 1) for k in ks])
        ax_raw.errorbar(
            ks, ratio, yerr=err, fmt="o-", color=color, lw=1.4, ms=4, label=f"T={t}"
        )
        ax_frac.errorbar(frac, ratio, yerr=err, fmt="o-", color=color, lw=1.4, ms=4)

    ax_raw.set_xlabel("terms per axis (incl. sin1)")
    ax_raw.set_ylabel("N(T, terms) / N(T, terms=2)")
    ax_raw.set_title(f"{dim}+1: terms effect at fixed T")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(fontsize=8.5)

    ax_frac.set_xscale("log")
    ax_frac.set_xlabel("wiggle terms as fraction of frequency pool  (terms-1)/(T-1)")
    ax_frac.set_title("Same data vs pool fraction (collapse test)")
    ax_frac.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("terms_by_t.png"))
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--band", default="nyq")
    args = parser.parse_args()
    plot(args.db, args.out, dim=args.dim, band=args.band)


if __name__ == "__main__":
    main()
