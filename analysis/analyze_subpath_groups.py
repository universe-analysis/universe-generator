"""Group multiplicity vs equation of state on subpath (two-phase) dumps.

A subpath campaign packs in two phases: unique worldlines jam first, then
phase-2 subpaths accrete onto them (a subpath must touch exactly one
existing group and joins it; accepted subpaths can seed further accretion).
A group = one unique plus everything that accreted onto it, recorded by the
dump's trailing ``gid`` column.

Question (Chris, 2026-08-05): do sterile groups (no subpaths) and fertile
groups (many subpaths) contribute differently to w? Groups are binned by
subpath count and each bin's w is measured under the standard E ~ sum(b)
dictionary, w = <E v^2>/<E>/3 (same conventions as analyze_eos_history) --
at a chosen conformal time ``--z`` (default: the turnaround pi/2), and
optionally across the whole cycle (``--out-wz``): away from the turnaround
the comoving-anchor velocity a2 cos(z) dominates v^2, so a mid-cycle z
probes anchor differences the turnaround is blind to. The group-size
distribution itself (a mass-function analog) is reported alongside.

Two energy dictionaries are reported (Chris, 2026-08-05): the per-path
dictionary above (a group's budget share grows with its member count), and
a per-group dictionary in which every group represents an EQUAL share of
the total energy -- a sterile group counts as much as a megagroup. Under
the group dictionary a group's own w is its internal E-weighted mean and
the ensemble/bin w is the plain mean over groups.

Usage::

    python -m analysis.analyze_subpath_groups \
        --params data/fullspec/dumps/d2_nyq_T{20,40,100}_s*_sub_fsub2e6.csv \
        --z 0.3927 \
        --out-bins subpath_w_bins.png --out-sizes subpath_group_sizes.png \
        --out-wz subpath_w_of_z.png
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
#: default multiplicity brackets; override with --bins
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
    """Pooled per-bin state for one T cell, resolved over a z grid.

    Per-path energies are z-independent; only v^2(z) varies, so each bin
    accumulates a sum(E v^2) array over ``zgrid`` next to its scalar sum(E).
    """

    t: int
    zgrid: np.ndarray
    n_seeds: int = 0
    group_sizes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    bin_groups: dict[str, int] = field(default_factory=dict)
    bin_paths: dict[str, int] = field(default_factory=dict)
    bin_e: dict[str, float] = field(default_factory=dict)
    bin_ev2: dict[str, np.ndarray] = field(default_factory=dict)
    tot_e: float = 0.0
    tot_ev2: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: per-group internal w(z), rows aligned with group_sizes (group dict.)
    group_wz: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    #: multiplicity brackets (lo, hi, label) used for binning
    bins: list[tuple[int, int, str]] = field(default_factory=lambda: list(BINS))
    #: per-path anchor energy sum(a2^2) and global group index (anchor stats)
    path_a2sq: np.ndarray = field(default_factory=lambda: np.empty(0))
    path_group: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    def group_anchor_stats(self, min_size: int = 10) -> tuple[float, float]:
        """(ICC, size-corr) of the anchor energy across groups.

        ICC: 1 - within-group variance / total variance over members of
        groups with >= min_size members (1 = perfectly coherent groups).
        size-corr: Pearson corr of log group size vs group-mean anchor
        energy over all groups (0 = fertility carries no anchor bias).
        """
        sizes_of = self.group_sizes[self.path_group]
        m = sizes_of >= min_size
        g, x = self.path_group[m], self.path_a2sq[m]
        gm = np.bincount(g, weights=x, minlength=len(self.group_sizes))
        gn = np.bincount(g, minlength=len(self.group_sizes))
        mean_of = (gm / np.maximum(gn, 1))[g]
        icc = 1.0 - ((x - mean_of) ** 2).mean() / x.var()
        all_gm = np.bincount(
            self.path_group, weights=self.path_a2sq, minlength=len(self.group_sizes)
        ) / np.maximum(np.bincount(self.path_group, minlength=len(self.group_sizes)), 1)
        corr = float(np.corrcoef(np.log(self.group_sizes), all_gm)[0, 1])
        return float(icc), corr

    def zi(self, z: float) -> int:
        return int(np.argmin(np.abs(self.zgrid - z)))

    def w_ensemble(self, z: float) -> float:
        return float(self.tot_ev2[self.zi(z)] / self.tot_e / 3.0)

    def w_ensemble_group(self, z: float) -> float:
        """Ensemble w under the equal-energy-per-group dictionary."""
        return float(self.group_wz[:, self.zi(z)].mean())

    def _bin_gmask(self, lo: int, hi: int) -> np.ndarray:
        return (self.group_sizes - 1 >= lo) & (self.group_sizes - 1 <= hi)

    def bin_rows(self, z: float) -> list[tuple[str, int, int, float, float, float]]:
        """(label, n_groups, n_paths, E_share, w_path, w_group) at z per bin."""
        rows = []
        for lo, hi, label in self.bins:
            gmask = self._bin_gmask(lo, hi)
            if self.bin_paths.get(label, 0) == 0:
                rows.append((label, int(gmask.sum()), 0, 0.0, np.nan, np.nan))
                continue
            w = float(self.bin_ev2[label][self.zi(z)] / self.bin_e[label] / 3.0)
            w_grp = float(self.group_wz[gmask, self.zi(z)].mean())
            rows.append(
                (
                    label,
                    self.bin_groups[label],
                    self.bin_paths[label],
                    self.bin_e[label] / self.tot_e,
                    w,
                    w_grp,
                )
            )
        return rows


DUMP_ROW_CAP = 60_000  # campaign-era engines stop dumping past this many rows


def load_cells(
    paths: list[Path],
    zgrid: np.ndarray,
    complete_only: bool = False,
    bins: list[tuple[int, int, str]] | None = None,
) -> list[CellGroups]:
    """Pool dumps into per-T cells (seeds of one T are combined).

    A dump that hit the engine's row cap is truncated: group sizes are
    censored and gids can appear with zero dumped rows (phantom groups).
    Truncated dumps are skipped under ``complete_only``, else warned about;
    phantom groups are always excluded.
    """
    by_t: dict[int, CellGroups] = {}
    for path in sorted(paths):
        t = _t_of(path)
        axes = load_axis_terms(path)
        with open(path) as f:
            gid = np.array([int(r["gid"]) for r in csv.DictReader(f)])
        sizes = np.bincount(gid)
        truncated = len(gid) >= DUMP_ROW_CAP or (sizes == 0).any()
        if truncated:
            note = "skipping" if complete_only else "sizes censored, keeping"
            print(f"WARNING: {path.name} is truncated at the dump row cap; {note}")
            if complete_only:
                continue
        cell = by_t.setdefault(
            t, CellGroups(t=t, zgrid=zgrid, bins=list(bins if bins else BINS))
        )
        if cell.tot_ev2.size == 0:
            cell.tot_ev2 = np.zeros(len(zgrid))
        e = np.sum([ax.b.sum(axis=1) for ax in axes], axis=0)
        v2_z = np.stack([velocity2(axes, float(z)) for z in zgrid])  # (nz, N)
        subs = sizes[gid] - 1
        occupied = sizes > 0  # phantom (fully truncated) groups drop out
        remap = np.cumsum(occupied) - 1  # old gid -> occupied-only index
        offset = len(cell.group_sizes)
        cell.path_a2sq = np.concatenate(
            [cell.path_a2sq, np.sum([ax.a2**2 for ax in axes], axis=0)]
        )
        cell.path_group = np.concatenate([cell.path_group, remap[gid] + offset])
        cell.group_sizes = np.concatenate([cell.group_sizes, sizes[occupied]])
        cell.tot_e += e.sum()
        cell.tot_ev2 += v2_z @ e
        # per-group internal w(z): E-weighted within the group (group dict.)
        e_g = np.bincount(gid, weights=e)
        ev2_g = np.stack(
            [np.bincount(gid, weights=e * v2_z[k]) for k in range(len(zgrid))],
            axis=1,
        )
        wz_g = (ev2_g / np.where(e_g > 0, e_g, 1.0)[:, None] / 3.0)[occupied]
        if cell.group_wz.size == 0:
            cell.group_wz = wz_g
        else:
            cell.group_wz = np.concatenate([cell.group_wz, wz_g])
        for lo, hi, label in cell.bins:
            pmask = (subs >= lo) & (subs <= hi)
            gmask = (sizes - 1 >= lo) & (sizes - 1 <= hi)
            cell.bin_groups[label] = cell.bin_groups.get(label, 0) + int(gmask.sum())
            cell.bin_paths[label] = cell.bin_paths.get(label, 0) + int(pmask.sum())
            if not pmask.any():
                continue
            cell.bin_e[label] = cell.bin_e.get(label, 0.0) + e[pmask].sum()
            prev = cell.bin_ev2.get(label, np.zeros(len(zgrid)))
            cell.bin_ev2[label] = prev + v2_z[:, pmask] @ e[pmask]
        cell.n_seeds += 1
    return [by_t[t] for t in sorted(by_t)]


def report(cell: CellGroups, z: float) -> None:
    gs = cell.group_sizes
    d = 2  # subpaths are a 2+1 engine feature
    w_ens = cell.w_ensemble(z)
    w_ens_g = cell.w_ensemble_group(z)
    print(
        f"\n=== 2+1 T={cell.t} ({cell.n_seeds} seeds): {len(gs)} groups, "
        f"{gs.sum()} paths; z = {z:.4f} ({z / np.pi:.3f} pi); "
        f"ensemble w(z) = {w_ens:.5f} path-dict / {w_ens_g:.5f} group-dict ==="
    )
    print(
        f"  [turnaround w = {cell.w_ensemble(HALF_PI):.5f}, "
        f"d/(6T) = {d / (6 * cell.t):.5f}]  group sizes: median "
        f"{int(np.median(gs))}, max {gs.max()}, "
        f"sterile {(gs == 1).sum()}/{len(gs)} ({(gs == 1).mean():.0%})"
    )
    icc, corr = cell.group_anchor_stats()
    print(
        f"  anchor coherence: ICC = {icc:.3f} (groups >= 10 members; "
        f"shuffled null ~ 0), corr(log size, group anchor energy) = {corr:+.3f}"
    )
    c = ("bin", "groups", "paths", "E share", "w path", "/ens", "w group", "/ens")
    print(
        f"  {c[0]:<14}{c[1]:>7}{c[2]:>9}{c[3]:>9}{c[4]:>10}{c[5]:>6}{c[6]:>10}{c[7]:>6}"
    )
    for label, n_g, n_p, share, w, w_grp in cell.bin_rows(z):
        r = w / w_ens if np.isfinite(w) else float("nan")
        r_g = w_grp / w_ens_g if np.isfinite(w_grp) else float("nan")
        print(
            f"  {label:<14}{n_g:>7}{n_p:>9}{share:>9.2%}"
            f"{w:>10.5f}{r:>6.2f}{w_grp:>10.5f}{r_g:>6.2f}"
        )


def plot_bins(cells: list[CellGroups], z: float, out: Path) -> None:
    """w/ensemble per multiplicity bin at conformal time z, one series per T."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = [b[2] for b in cells[0].bins]
    width = 0.8 / len(cells)
    for k, cell in enumerate(cells):
        w_ens = cell.w_ensemble(z)
        ratios = [
            w / w_ens if np.isfinite(w) else 0.0
            for _, _, _, _, w, _ in cell.bin_rows(z)
        ]
        x = np.arange(len(labels)) + (k - (len(cells) - 1) / 2) * width
        ax.bar(x, ratios, width=width * 0.92, label=f"T={cell.t}")
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("group subpath count")
    ax.set_ylabel(f"bin w / ensemble w  at z = {z / np.pi:.3f} pi")
    ax.set_title("Accretion multiplicity vs w, off-turnaround (2+1, terms=T)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def plot_wz(
    cells: list[CellGroups],
    out: Path,
    mark_z: float | None = None,
    dictionary: str = "path",
) -> None:
    """Per-bin w(z)/ensemble-w(z) ratio curves across the cycle, one panel/T."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 4.2), sharey=True)
    for ax, cell in zip(np.atleast_1d(axs), cells):
        for lo, hi, label in cell.bins:
            if cell.bin_paths.get(label, 0) == 0:
                continue
            if dictionary == "group":
                gmask = cell._bin_gmask(lo, hi)
                ratio = cell.group_wz[gmask].mean(axis=0) / cell.group_wz.mean(axis=0)
            else:
                ratio = (cell.bin_ev2[label] / cell.bin_e[label]) / (
                    cell.tot_ev2 / cell.tot_e
                )
            ax.plot(cell.zgrid / np.pi, ratio, label=label)
        ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
        if mark_z is not None:
            ax.axvline(mark_z / np.pi, color="0.6", lw=0.8, ls=":")
        ax.axvline(0.5, color="0.85", lw=0.8)
        ax.set_xlabel("z / pi")
        ax.set_title(f"T={cell.t}")
    np.atleast_1d(axs)[0].set_ylabel("bin w(z) / ensemble w(z)")
    np.atleast_1d(axs)[0].legend(fontsize=8)
    name = "equal-energy-per-group" if dictionary == "group" else "per-path E"
    fig.suptitle(f"Group-multiplicity w across the whole cycle ({name} dictionary)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def plot_anchor(cells: list[CellGroups], out: Path) -> None:
    """Group anchor energy vs group size: coherence bars, no size trend."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.4), sharey=True)
    for ax, cell in zip(np.atleast_1d(axs), cells):
        n_g = len(cell.group_sizes)
        gsum = np.bincount(cell.path_group, weights=cell.path_a2sq, minlength=n_g)
        gsumsq = np.bincount(cell.path_group, weights=cell.path_a2sq**2, minlength=n_g)
        cnt = np.maximum(np.bincount(cell.path_group, minlength=n_g), 1)
        gmean = gsum / cnt
        gstd = np.sqrt(np.maximum(gsumsq / cnt - gmean**2, 0.0))
        ax.scatter(cell.group_sizes, gmean, s=7, alpha=0.3, color="C0", lw=0)
        big = cell.group_sizes >= 10
        ax.errorbar(
            cell.group_sizes[big],
            gmean[big],
            yerr=gstd[big],
            fmt="none",
            ecolor="C3",
            alpha=0.5,
            lw=0.9,
        )
        icc, corr = cell.group_anchor_stats()
        ax.set_xscale("log")
        ax.set_xlabel("group size (unique + subpaths)")
        ax.set_title(f"T={cell.t}   ICC {icc:.3f}, size-corr {corr:+.3f}")
    np.atleast_1d(axs)[0].set_ylabel("group mean anchor energy  sum a2^2")
    fig.suptitle(
        "Accretion groups are iso-anchor: tight internal spread (red bars), "
        "no size trend"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def parse_bins(spec: str) -> list[tuple[int, int, str]]:
    """Turn "0,1,2,10,100,1000" into (lo, hi, label) brackets."""
    los = sorted(int(x) for x in spec.split(","))
    out: list[tuple[int, int, str]] = []
    for i, lo in enumerate(los):
        hi = (los[i + 1] - 1) if i + 1 < len(los) else 10**9
        if lo == 0 and hi == 0:
            label = "0 (sterile)"
        elif lo == hi:
            label = f"{lo} subpath" + ("s" if lo != 1 else "")
        elif hi == 10**9:
            label = f"{lo}+"
        else:
            label = f"{lo}-{hi}"
        out.append((lo, hi, label))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params", nargs="+", required=True, help="subpath dumps (with gid column)"
    )
    parser.add_argument(
        "--z",
        type=float,
        default=HALF_PI,
        help="conformal time for the table/bar chart (default: turnaround pi/2)",
    )
    parser.add_argument("--out-bins", type=Path, default=None)
    parser.add_argument("--out-sizes", type=Path, default=None)
    parser.add_argument("--out-wz", type=Path, default=None)
    parser.add_argument(
        "--out-wz-group",
        type=Path,
        default=None,
        help="w(z) curves under the equal-energy-per-group dictionary",
    )
    parser.add_argument("--out-anchor", type=Path, default=None)
    parser.add_argument(
        "--bins",
        default=None,
        help="comma-separated bracket lower bounds, e.g. 0,1,2,10,100,1000 "
        "(each bracket runs to the next bound; the last is open-ended)",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="skip dumps truncated at the engine row cap (censored group sizes)",
    )
    args = parser.parse_args()
    base = np.linspace(0.02 * np.pi, 0.98 * np.pi, 49)
    zgrid = np.unique(np.concatenate([base, [args.z, HALF_PI]]))
    bins = parse_bins(args.bins) if args.bins else None
    cells = load_cells(
        [Path(p) for p in args.params],
        zgrid,
        complete_only=args.complete_only,
        bins=bins,
    )
    for cell in cells:
        report(cell, args.z)
    if args.out_bins is not None:
        plot_bins(cells, args.z, args.out_bins)
    if args.out_sizes is not None:
        plot_sizes(cells, args.out_sizes)
    if args.out_wz is not None:
        plot_wz(cells, args.out_wz, mark_z=args.z)
    if args.out_wz_group is not None:
        plot_wz(cells, args.out_wz_group, mark_z=args.z, dictionary="group")
    if args.out_anchor is not None:
        plot_anchor(cells, args.out_anchor)


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


if __name__ == "__main__":
    main()
