"""Group multiplicity vs equation of state on subpath (two-phase) dumps.

A subpath campaign packs in two phases: unique worldlines jam first, then
phase-2 subpaths accrete onto them (a subpath must touch exactly one
existing group and joins it; accepted subpaths can seed further accretion).
A group = one unique plus everything that accreted onto it, recorded by the
dump's trailing ``gid`` column.

Question (Chris, 2026-08-05): do sterile groups (no subpaths) and fertile
groups (many subpaths) contribute differently to w? Groups are binned by
subpath count and each bin's turnaround w is measured under the standard
E ~ sum(b) dictionary, w = <E v^2>/<E>/3 (same conventions as
analyze_eos_history). The group-size distribution itself (a mass-function
analog) is reported alongside.

Usage::

    python -m analysis.analyze_subpath_groups \
        --params data/fullspec/dumps/d2_nyq_T{20,40,100}_s*_sub_fsub2e6.csv \
        --out-bins subpath_w_bins.png --out-sizes subpath_group_sizes.png
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.analyze_eos_history import velocity2
from braidlab.corrdim import load_axis_terms

HALF_PI = np.pi / 2.0
BINS: list[tuple[int, int, str]] = [
    (0, 0, "0 (sterile)"),
    (1, 1, "1 subpath"),
    (2, 9, "2-9"),
    (10, 99, "10-99"),
    (100, 10**9, "100+"),
]


def _t_of(path: Path) -> int:
    m = re.search(r"_T(\d+)_", path.name)
    if m is None:
        raise ValueError(f"no _T<N>_ in dump name: {path}")
    return int(m.group(1))


@dataclass
class CellGroups:
    """Pooled per-path state for one T cell of a subpath campaign."""

    t: int
    n_seeds: int = 0
    e: np.ndarray = field(default_factory=lambda: np.empty(0))
    v2: np.ndarray = field(default_factory=lambda: np.empty(0))
    subs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    group_sizes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def w_ensemble(self) -> float:
        return float((self.e * self.v2).sum() / self.e.sum() / 3.0)

    def bin_rows(self) -> list[tuple[str, int, int, float, float]]:
        """(label, n_groups, n_paths, energy_share, w) per multiplicity bin."""
        rows = []
        tot_e = self.e.sum()
        for lo, hi, label in BINS:
            gmask = (self.group_sizes - 1 >= lo) & (self.group_sizes - 1 <= hi)
            pmask = (self.subs >= lo) & (self.subs <= hi)
            if not pmask.any():
                rows.append((label, int(gmask.sum()), 0, 0.0, float("nan")))
                continue
            e, v2 = self.e[pmask], self.v2[pmask]
            rows.append(
                (
                    label,
                    int(gmask.sum()),
                    int(pmask.sum()),
                    float(e.sum() / tot_e),
                    float((e * v2).sum() / e.sum() / 3.0),
                )
            )
        return rows


def load_cells(paths: list[Path]) -> list[CellGroups]:
    """Pool dumps into per-T cells (seeds of one T are combined)."""
    by_t: dict[int, CellGroups] = {}
    for path in sorted(paths):
        t = _t_of(path)
        cell = by_t.setdefault(t, CellGroups(t=t))
        axes = load_axis_terms(path)
        with open(path) as f:
            gid = np.array([int(r["gid"]) for r in csv.DictReader(f)])
        v2 = velocity2(axes, HALF_PI)
        e = np.sum([ax.b.sum(axis=1) for ax in axes], axis=0)
        sizes = np.bincount(gid)
        cell.e = np.concatenate([cell.e, e])
        cell.v2 = np.concatenate([cell.v2, v2])
        cell.subs = np.concatenate([cell.subs, sizes[gid] - 1])
        cell.group_sizes = np.concatenate([cell.group_sizes, sizes])
        cell.n_seeds += 1
    return [by_t[t] for t in sorted(by_t)]


def report(cell: CellGroups) -> None:
    gs = cell.group_sizes
    d = 2  # subpaths are a 2+1 engine feature
    print(
        f"\n=== 2+1 T={cell.t} ({cell.n_seeds} seeds): {len(gs)} groups, "
        f"{gs.sum()} paths; ensemble w = {cell.w_ensemble:.5f} "
        f"(d/(6T) = {d / (6 * cell.t):.5f}) ==="
    )
    print(
        f"  group sizes: median {int(np.median(gs))}, max {gs.max()}, "
        f"sterile {(gs == 1).sum()}/{len(gs)} ({(gs == 1).mean():.0%})"
    )
    print(f"  {'bin':<14}{'groups':>7}{'paths':>9}{'E share':>9}{'w':>10}{'w/ens':>7}")
    for label, n_g, n_p, share, w in cell.bin_rows():
        ratio = w / cell.w_ensemble if np.isfinite(w) else float("nan")
        print(f"  {label:<14}{n_g:>7}{n_p:>9}{share:>9.2%}{w:>10.5f}{ratio:>7.2f}")


def plot_bins(cells: list[CellGroups], out: Path) -> None:
    """w/ensemble per multiplicity bin, one bar series per T."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = [b[2] for b in BINS]
    width = 0.8 / len(cells)
    for k, cell in enumerate(cells):
        ratios = [
            w / cell.w_ensemble if np.isfinite(w) else 0.0
            for _, _, _, _, w in cell.bin_rows()
        ]
        x = np.arange(len(labels)) + (k - (len(cells) - 1) / 2) * width
        ax.bar(x, ratios, width=width * 0.92, label=f"T={cell.t}")
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("group subpath count")
    ax.set_ylabel("bin w / ensemble w  (turnaround)")
    ax.set_title("Accretion multiplicity carries no w signal (2+1, terms=T)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def plot_sizes(cells: list[CellGroups], out: Path) -> None:
    """Complementary CDF of group sizes per T (the mass-function analog)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for cell in cells:
        sizes = np.sort(cell.group_sizes)[::-1]
        ccdf = np.arange(1, len(sizes) + 1) / len(sizes)
        ax.loglog(sizes, ccdf, drawstyle="steps-post", label=f"T={cell.t}")
    ax.set_xlabel("group size s (unique + subpaths)")
    ax.set_ylabel("fraction of groups with size >= s")
    ax.set_title("Group mass function: sterile majority, megagroup tail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params", nargs="+", required=True, help="subpath dumps (with gid column)"
    )
    parser.add_argument("--out-bins", type=Path, default=None)
    parser.add_argument("--out-sizes", type=Path, default=None)
    args = parser.parse_args()
    cells = load_cells([Path(p) for p in args.params])
    for cell in cells:
        report(cell)
    if args.out_bins is not None:
        plot_bins(cells, args.out_bins)
    if args.out_sizes is not None:
        plot_sizes(cells, args.out_sizes)


if __name__ == "__main__":
    main()
