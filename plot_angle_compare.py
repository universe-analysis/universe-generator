"""Compare the angle (edge-weighted) sampler against the uniform sampler.

Two panels:
  (left)  occupancy folded to |X| -- shows the angle draw pulls packing off the
          center and onto the edges, exactly as intended.
  (right) N_sat vs T on log-log -- the fitted slope D is the fractal dimension;
          the two samplers land on top of each other, so the edge reweighting
          does not change D.

Usage::

    python plot_angle_compare.py --out angle_compare.png
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

DIAG = Path("data/diag")
T_VALUES = [18, 24, 32, 40]


def folded_occupancy(path: Path) -> tuple[list[float], list[int]]:
    """Return (|x| bin centers, summed counts) folded across the midline."""
    folded: dict[float, int] = collections.defaultdict(int)
    for row in csv.DictReader(open(path)):
        folded[round(abs(float(row["x_center"])), 2)] += int(row["count"])
    xs = sorted(folded)
    return xs, [folded[x] for x in xs]


def final_counts(mode: str) -> list[int]:
    """Final N for each T from the comparison run curves."""
    counts = []
    for t in T_VALUES:
        rows = list(csv.DictReader(open(DIAG / f"cmp_{mode}_T{t}.csv")))
        counts.append(int(rows[-1]["n"]))
    return counts


# (label, scan-name, occupancy file at T=40, color)
SAMPLERS = [
    ("uniform", "uniform", "d40uni_occ", "tab:blue"),
    ("edge-weighted", "edge", "d40edge_occ", "tab:orange"),
    ("center-weighted", "center", "d40cen_occ", "tab:green"),
]


def scan_counts() -> dict[tuple[str, int], list[int]]:
    """Parse the combined seed scan into {(sampler, T): [N per seed]}."""
    out: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for line in open(DIAG / "seedscan_all.txt"):
        parts = line.split()
        if len(parts) == 4 and parts[3].startswith("N="):
            out[(parts[0], int(parts[1]))].append(int(parts[3][2:]))
    return out


def plot(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_occ, ax_d) = plt.subplots(1, 2, figsize=(13, 5))

    for label, _scan, occ_file, color in SAMPLERS:
        ox, oc = folded_occupancy(DIAG / f"{occ_file}.csv")
        ax_occ.plot(ox, oc, "-o", color=color, ms=4, label=label)
    ax_occ.set_xlabel("|comoving X|  (0 = center, 1 = wall)")
    ax_occ.set_ylabel("accepted worldline occupancy")
    ax_occ.set_title("Sampler shifts where worldlines pack (T=40)")
    ax_occ.legend()
    ax_occ.grid(True, alpha=0.3)

    counts = scan_counts()
    ts = np.array(T_VALUES)
    for label, scan, _occ, color in SAMPLERS:
        means = np.array([np.mean(counts[(scan, t)]) for t in ts])
        stds = np.array([np.std(counts[(scan, t)]) for t in ts])
        per_seed_d = [
            np.polyfit(np.log(ts), np.log([counts[(scan, t)][s] for t in ts]), 1)[0]
            for s in range(len(counts[(scan, ts[0])]))
        ]
        d_mean, d_err = np.mean(per_seed_d), np.std(per_seed_d)
        ax_d.errorbar(
            ts, means, yerr=stds, fmt="o", color=color, ms=7, capsize=4,
            label=f"{label}: D={d_mean:.2f}+/-{d_err:.2f}",
        )
        coef = np.polyfit(np.log(ts), np.log(means), 1)
        fit_t = np.linspace(ts[0] * 0.95, ts[-1] * 1.05, 50)
        ax_d.plot(fit_t, np.exp(coef[1]) * fit_t ** coef[0], "--", color=color, lw=1.2)

    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("T (timesteps = resolution)")
    ax_d.set_ylabel("N_sat (worldlines at accept-rate 1e-6, 5 seeds)")
    ax_d.set_title("Same fractal dimension D ~ 2.6 every way")
    ax_d.legend()
    ax_d.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Sin1 sampling: edge / uniform / center -- packing moves, D does not",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("angle_compare.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
