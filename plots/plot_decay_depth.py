"""FREQ decay chart: exponent and jam count vs cutoff depth.

Reads the six freqdecay stores (two dims x cutoffs 1e-6/7/8) and plots the
two-point exponent between the arm's T pair against cutoff depth, per term
count, plus the per-decade jam-count growth. Quantifies how cutoff-limited
each dimension's exponent is (3+1: barely; 2+1: strongly).

Usage::

    uv run python -m plots.plot_decay_depth --out decay_depth.png
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

CUTS = ["e6", "e7", "e8"]


def _cells(dim: int) -> dict[tuple[int, int, str], float]:
    out: dict[tuple[int, int, str], float] = {}
    for tag in CUTS:
        conn = sqlite3.connect(f"data/freq/freqdecay{dim}d_{tag}.db")
        rows = conn.execute(
            "select t, terms, avg(n_final) from runs where status='done' "
            "group by t, terms"
        )
        for t, k, n in rows:
            out[(t, k, tag)] = n
        conn.close()
    return out


def plot(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    x = np.arange(len(CUTS))
    for ax, dim in zip(axes, (3, 2)):
        cells = _cells(dim)
        ts = sorted({t for t, _, _ in cells})
        for k, style in ((2, "o-"), (10, "s--")):
            d = [
                np.log(cells[(ts[1], k, tag)] / cells[(ts[0], k, tag)])
                / np.log(ts[1] / ts[0])
                for tag in CUTS
            ]
            ax.plot(x, d, style, lw=1.6, ms=6, label=f"terms={k}")
            for xi, di in zip(x, d):
                ax.annotate(
                    f"{di:.3f}",
                    (xi, di),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    color="dimgray",
                )
        ax.set_xticks(x, ["1e-6", "1e-7", "1e-8"])
        ax.set_xlabel("acceptance-rate cutoff")
        ax.set_ylabel(f"two-point exponent (T={ts[0]} vs {ts[1]})")
        ax.set_title(f"{dim}+1: exponent vs cutoff depth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("decay_depth.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
