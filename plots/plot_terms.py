"""FREQ campaign chart: packing exponent vs worldline term count.

Two panels from a freq-campaign store (T x seeds x terms at one cutoff):

  * left  -- seed-averaged N vs T (log-log), one series per term count with
             its fitted power law; term count is an ordered quantity, so the
             series use a single-hue light-to-dark ramp;
  * right -- the fitted exponent D (bootstrap error bars) vs term count.

Usage::

    uv run python -m plots.plot_terms --db data/freq/freq3d_e6.db --out terms.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from braidlab.analyze import measure_d
from braidlab.store import Store


def plot(db_path: Path, out_path: Path, dim: int = 3, band: str = "nyq") -> None:
    import matplotlib.pyplot as plt

    results = Store(db_path).results(dim, band)
    term_counts = sorted({r.terms for r in results})
    if not term_counts:
        raise SystemExit(f"no completed runs in {db_path}")

    fig, (ax_n, ax_d) = plt.subplots(
        1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )
    cmap = plt.get_cmap("Blues")
    shades = [cmap(x) for x in np.linspace(0.45, 0.95, len(term_counts))]

    d_by_terms = {}
    for terms, color in zip(term_counts, shades):
        subset = [r for r in results if r.terms == terms]
        fit = measure_d(subset, dim, band)
        d_by_terms[terms] = fit

        by_t: dict[int, list[int]] = defaultdict(list)
        for r in subset:
            if r.n_final is not None:
                by_t[r.t].append(r.n_final)
        ts = np.array(sorted(by_t))
        means = np.array([np.mean(by_t[t]) for t in ts])
        ax_n.plot(
            ts,
            means,
            "o-",
            color=color,
            lw=1.6,
            ms=5,
            label=f"terms={terms}  (D={fit.d:.3f})",
        )
        ax_n.annotate(
            f" {terms}", (ts[-1], means[-1]), color=color, fontsize=9, va="center"
        )

    ax_n.set_xscale("log")
    ax_n.set_yscale("log")
    ax_n.set_xlabel("T (timesteps = resolution)")
    ax_n.set_ylabel("N (worldlines packed, seed mean)")
    ax_n.set_title(f"{dim}+1 torus+phase, 1e-6 cutoff: N ~ T^D per term count")
    ax_n.legend(loc="upper left", fontsize=8.5)
    ax_n.grid(True, which="both", alpha=0.3)

    terms_arr = np.array(term_counts)
    d_arr = np.array([d_by_terms[k].d for k in term_counts])
    e_arr = np.array([d_by_terms[k].d_err for k in term_counts])
    ax_d.errorbar(
        terms_arr,
        d_arr,
        yerr=e_arr,
        fmt="o-",
        color=shades[-1],
        lw=1.6,
        ms=6,
        capsize=3,
    )
    for k, d in zip(terms_arr, d_arr):
        ax_d.annotate(
            f"{d:.3f}",
            (k, d),
            textcoords="offset points",
            xytext=(8, -11),
            fontsize=8.5,
            color="dimgray",
        )
    ax_d.set_xlabel("terms per axis (incl. sin1; 2 = legacy)")
    ax_d.set_ylabel("packing exponent D")
    ax_d.set_title("Exponent rises with term count")
    ax_d.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    for k in term_counts:
        f = d_by_terms[k]
        print(f"terms={k}: D = {f.d:.3f} +/- {f.d_err:.3f}")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("terms.png"))
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--band", default="nyq")
    args = parser.parse_args()
    plot(args.db, args.out, dim=args.dim, band=args.band)


if __name__ == "__main__":
    main()
