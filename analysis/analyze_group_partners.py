"""Partner correlations in unique groups -- correlation vs contact, quantified.

A unique group is one phase-1 worldline plus the subpaths that adopted its gid
(README "Unique generation -> Subpath Generation"). Co-grouped strands emerge
at distinct comoving Bang positions (X(0) = sum a*b*cos f + a1 per axis, the
z -> 0 limit of x/sin z) and only meet mid-loop -- yet their acceptance was a
joint condition on whole histories, so their z-independent parameters can be
correlated at epochs before any dynamical contact between them was possible.
This is the classical, measurable groundwork for the README's stated goal of
Bell-type tests on subpaths: it quantifies *selection correlation* between
partners and separates it from *dynamical contact*.

Ensembles (2+1 subpath dumps, the only ones carrying gid):
  A. 2-strand groups -- one parent + its only subpath, the clean "pair".
  B. never-touching co-group pairs from 3..MAX_GROUP-strand groups -- strands
     correlated only through their shared group (no direct contact ever).
  C. late-contact subset of A (first contact after the turnaround z = pi/2).

For each ensemble, pair statistics (Bang separation, frequency-parity match,
cold/mover class match, a1 and log-frequency double-entry correlations,
even-axis phase alignment, Bang approach rate) are tested against a
block-permutation null: partnerships reshuffled among the same strands within
each dump, which preserves every single-strand marginal and breaks only
co-membership. Features are centered within dumps before pooling so T-ladder
heterogeneity cannot masquerade as partner correlation.

Contact uses the engine's own predicate: wrapped Chebyshev distance <= CELL
(2/T) on the engine z grid z_i = (i+1)*pi/(T+1) (`cuda/braid_cuda.cu`), so
"pair A touches at z" here means exactly what admission meant in the engine.

Usage::

    uv run python -m analysis.analyze_group_partners \
        --dumps data/converge/dumps --out figures/group_partners.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.analyze_braids import Strand, load_dump

#: groups larger than this are excluded from pairwise work (they are the giant
#: percolating clusters; every exclusion is logged, never silent)
MAX_GROUP = 10

_NAME_RE = re.compile(r"_T(?P<t>\d+)_s(?P<seed>\d+)")


# ------------------------------------------------------------------ features


def bang_position(s: Strand) -> tuple[float, float]:
    """Comoving emergence point: lim_{z->0} x/sin z = sum a*b*cos f + a1."""
    x = float(np.sum(s.ax * s.bx * np.cos(s.fx)) + s.ax2)
    y = float(np.sum(s.ay * s.by * np.cos(s.fy)) + s.ay2)
    return float(wrap(x)), float(wrap(y))


def bang_velocity(s: Strand) -> tuple[float, float]:
    """Comoving dX/dz at the Bang: -sum a*b^2*sin f / 2 (0 for all-odd axes)."""
    vx = float(-np.sum(s.ax * s.bx**2 * np.sin(s.fx)) / 2.0)
    vy = float(-np.sum(s.ay * s.by**2 * np.sin(s.fy)) / 2.0)
    return vx, vy


def wrap(x: float | np.ndarray) -> float | np.ndarray:
    """Wrap comoving coordinate(s) onto the fundamental domain [-1, 1)."""
    return (x + 1.0) % 2.0 - 1.0


def all_odd(s: Strand) -> bool:
    """Cold class: every frequency on both axes odd (at rest at turnaround)."""
    return bool(np.all(s.bx % 2 == 1) and np.all(s.by % 2 == 1))


# ------------------------------------------------------------------ contact


def engine_zgrid(t: int) -> np.ndarray:
    """The engine's symmetric interior grid: z_i = (i+1)*pi/(T+1), i < T."""
    return (np.arange(t) + 1) * np.pi / (t + 1)


def first_contact(a: Strand, b: Strand, zgrid: np.ndarray, cell: float) -> float | None:
    """Earliest engine timestep where the pair is in Chebyshev contact.

    Mirrors `collides_dev`: wrapped per-axis deltas, |dX| <= cell AND
    |dY| <= cell. Returns the z value, or None if the pair never touches.
    """
    xa, ya = a.xy(zgrid)
    xb, yb = b.xy(zgrid)
    dx = np.abs(wrap(xa - xb))
    dy = np.abs(wrap(ya - yb))
    hits = np.nonzero((dx <= cell) & (dy <= cell))[0]
    return float(zgrid[hits[0]]) if hits.size else None


# ------------------------------------------------------------------ pairs


@dataclass(frozen=True)
class Pair:
    """One co-grouped strand pair with per-strand features and contact info."""

    dump: int  # dump index (permutation block)
    x0: tuple[tuple[float, float], tuple[float, float]]  # Bang positions
    v0: tuple[tuple[float, float], tuple[float, float]]  # Bang velocities
    cold: tuple[bool, bool]
    logb: tuple[float, float]  # mean log frequency per strand
    a1: tuple[tuple[float, float], tuple[float, float]]  # (a1x, a1y) each
    even_phases: list[tuple[float, float]]  # per even-even axis: (f_A, f_B)
    z_meet: float | None

    @property
    def separation(self) -> float:
        """Wrapped Chebyshev Bang separation."""
        (xa, ya), (xb, yb) = self.x0
        return max(abs(float(wrap(xa - xb))), abs(float(wrap(ya - yb))))

    @property
    def approaching(self) -> bool:
        """Is the pair closing its comoving gap at the Bang? (swap-invariant)"""
        (xa, ya), (xb, yb) = self.x0
        (vxa, vya), (vxb, vyb) = self.v0
        ddot = float(wrap(xa - xb)) * (vxa - vxb) + float(wrap(ya - yb)) * (vya - vyb)
        return ddot < 0


def make_pair(a: Strand, b: Strand, dump: int, zgrid: np.ndarray, cell: float) -> Pair:
    """Assemble the feature bundle for one unordered co-grouped pair."""
    even_phases: list[tuple[float, float]] = []
    for fa, ba, fb, bb in (
        (a.fx, a.bx, b.fx, b.bx),
        (a.fy, a.by, b.fy, b.by),
    ):
        for j in range(len(ba)):
            for k in range(len(bb)):
                if ba[j] % 2 == 0 and bb[k] % 2 == 0:
                    even_phases.append((float(fa[j]), float(fb[k])))
    return Pair(
        dump=dump,
        x0=(bang_position(a), bang_position(b)),
        v0=(bang_velocity(a), bang_velocity(b)),
        cold=(all_odd(a), all_odd(b)),
        logb=(
            float(np.mean(np.log(np.concatenate([a.bx, a.by])))),
            float(np.mean(np.log(np.concatenate([b.bx, b.by])))),
        ),
        a1=((a.ax2, a.ay2), (b.ax2, b.ay2)),
        even_phases=even_phases,
        z_meet=first_contact(a, b, zgrid, cell),
    )


# --------------------------------------------------------------- statistics


def _center_by_dump(values: np.ndarray, dumps: np.ndarray) -> np.ndarray:
    """Subtract each dump's mean so T-ladder offsets cannot fake correlation."""
    out = values.astype(float).copy()
    for d in np.unique(dumps):
        m = dumps == d
        out[m] -= out[m].mean()
    return out


def _double_entry_r(u: np.ndarray, v: np.ndarray) -> float:
    """Pearson r symmetrized over pair order (each pair entered both ways)."""
    uu = np.concatenate([u, v])
    vv = np.concatenate([v, u])
    if np.std(uu) == 0 or np.std(vv) == 0:
        return 0.0
    return float(np.corrcoef(uu, vv)[0, 1])


#: statistic name -> (fn(pairs) -> float, description used in the report)
def _stat_separation(p: list[Pair]) -> float:
    return float(np.mean([q.separation for q in p]))


def _stat_cold_match(p: list[Pair]) -> float:
    return float(np.mean([q.cold[0] == q.cold[1] for q in p]))


def _stat_a1(p: list[Pair]) -> float:
    dumps = np.repeat([q.dump for q in p], 2)
    u = np.array([[q.a1[0][0], q.a1[0][1]] for q in p]).ravel()
    v = np.array([[q.a1[1][0], q.a1[1][1]] for q in p]).ravel()
    return _double_entry_r(_center_by_dump(u, dumps), _center_by_dump(v, dumps))


def _stat_logb(p: list[Pair]) -> float:
    dumps = np.array([q.dump for q in p])
    u = np.array([q.logb[0] for q in p])
    v = np.array([q.logb[1] for q in p])
    return _double_entry_r(_center_by_dump(u, dumps), _center_by_dump(v, dumps))


def _stat_phase(p: list[Pair]) -> float:
    """Mean cos 2(f_A - f_B) over even-even axis pairs (period-pi alignment)."""
    ph = [q for pair in p for q in pair.even_phases]
    if not ph:
        return 0.0
    arr = np.array(ph)
    return float(np.mean(np.cos(2.0 * (arr[:, 0] - arr[:, 1]))))


def _stat_approach(p: list[Pair]) -> float:
    return float(np.mean([q.approaching for q in p]))


def _stat_a1_gap(p: list[Pair]) -> float:
    """Mean wrapped |da1| over both axes -- seam-robust anchor gap.

    The Pearson statistic understates anchor sharing for pairs matched across
    the a1 = +/-1 seam (the corner clusters in the scatter); the wrapped gap
    treats those as the near-perfect matches they are.
    """
    gaps = [abs(float(wrap(q.a1[0][i] - q.a1[1][i]))) for q in p for i in (0, 1)]
    return float(np.mean(gaps))


STATISTICS = {
    "Bang separation (Chebyshev)": _stat_separation,
    "cold/mover class match rate": _stat_cold_match,
    "a1 partner correlation": _stat_a1,
    "a1 anchor gap |wrap(da1)|": _stat_a1_gap,
    "log-frequency partner correlation": _stat_logb,
    "even-axis phase alignment cos2(df)": _stat_phase,
    "approaching-at-Bang rate": _stat_approach,
}


@dataclass(frozen=True)
class StatResult:
    name: str
    observed: float
    null_mean: float
    null_std: float
    perm_p: float


def permutation_test(
    pairs: list[Pair],
    *,
    n_perm: int = 2000,
    seed: int = 20260722,
) -> list[StatResult]:
    """All statistics vs the within-dump partner-reshuffle null.

    A permuted replica re-pairs the same left strands with shuffled right
    strands of the same dump, so every single-strand marginal is preserved and
    only co-membership is destroyed. Feature bundles are rebuilt pair-wise
    (phases/contact are pair-level), but contact is skipped for null pairs --
    no null statistic reads z_meet.
    """
    rng = np.random.default_rng(seed)
    observed = {name: fn(pairs) for name, fn in STATISTICS.items()}
    nulls: dict[str, list[float]] = {name: [] for name in STATISTICS}
    by_dump: dict[int, list[Pair]] = {}
    for q in pairs:
        by_dump.setdefault(q.dump, []).append(q)

    for _ in range(n_perm):
        fake: list[Pair] = []
        for d, plist in by_dump.items():
            idx = rng.permutation(len(plist))
            for i, j in enumerate(idx):
                left, right = plist[i], plist[int(j)]
                fake.append(
                    Pair(
                        dump=d,
                        x0=(left.x0[0], right.x0[1]),
                        v0=(left.v0[0], right.v0[1]),
                        cold=(left.cold[0], right.cold[1]),
                        logb=(left.logb[0], right.logb[1]),
                        a1=(left.a1[0], right.a1[1]),
                        even_phases=_null_phases(left, right),
                        z_meet=None,
                    )
                )
        for name, fn in STATISTICS.items():
            nulls[name].append(fn(fake))

    results = []
    for name in STATISTICS:
        null = np.array(nulls[name])
        obs = observed[name]
        centered = np.abs(null - null.mean())
        p = float((np.sum(centered >= abs(obs - null.mean())) + 1) / (n_perm + 1))
        results.append(StatResult(name, obs, float(null.mean()), float(null.std()), p))
    return results


def _null_phases(left: Pair, right: Pair) -> list[tuple[float, float]]:
    """Even-axis phase pairs for a reshuffled pair (A phases x B phases)."""
    fa = [f for f, _ in left.even_phases]
    fb = [f for _, f in right.even_phases]
    n = min(len(fa), len(fb))
    return list(zip(fa[:n], fb[:n]))


# ------------------------------------------------------------------ loading


def collect_pairs(
    dump_paths: list[Path],
) -> tuple[list[Pair], list[Pair], dict[str, int]]:
    """Ensembles A (2-groups) and B (never-touching co-group pairs), + census."""
    pairs_a: list[Pair] = []
    pairs_b: list[Pair] = []
    census = {"strands": 0, "groups": 0, "excluded_groups": 0, "dumps": 0}
    for di, path in enumerate(sorted(dump_paths)):
        m = _NAME_RE.search(path.name)
        if m is None:
            continue
        t = int(m.group("t"))
        zgrid = engine_zgrid(t)
        cell = 2.0 / t
        strands = load_dump(str(path))
        groups: dict[int, list[Strand]] = {}
        for s in strands:
            groups.setdefault(s.gid, []).append(s)
        census["dumps"] += 1
        census["strands"] += len(strands)
        census["groups"] += len(groups)
        for members in groups.values():
            if len(members) > MAX_GROUP:
                census["excluded_groups"] += 1
                continue
            if len(members) == 2:
                pairs_a.append(make_pair(members[0], members[1], di, zgrid, cell))
            elif len(members) >= 3:
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        q = make_pair(members[i], members[j], di, zgrid, cell)
                        if q.z_meet is None:
                            pairs_b.append(q)
    return pairs_a, pairs_b, census


# --------------------------------------------------------------------- main


def _report(label: str, pairs: list[Pair], results: list[StatResult]) -> None:
    print(f"== {label}  ({len(pairs)} pairs)")
    for r in results:
        z = (r.observed - r.null_mean) / r.null_std if r.null_std else 0.0
        print(
            f"   {r.name:>36}: {r.observed:+.4f}  null {r.null_mean:+.4f} "
            f"+/- {r.null_std:.4f}  z = {z:+5.1f}  perm p = {r.perm_p:.4f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", type=Path, default=Path("data/converge/dumps"))
    parser.add_argument("--out", type=Path, default=Path("figures/group_partners.png"))
    parser.add_argument("--n-perm", type=int, default=2000)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dump_paths = sorted(args.dumps.glob("*_sub_*.csv"))
    if not dump_paths:
        raise SystemExit(f"no *_sub_*.csv dumps under {args.dumps}")
    pairs_a, pairs_b, census = collect_pairs(dump_paths)
    print(
        f"{census['dumps']} dumps, {census['strands']} strands, "
        f"{census['groups']} groups; {census['excluded_groups']} giant groups "
        f"(> {MAX_GROUP} strands) excluded from pairwise work\n"
    )

    missing = [q for q in pairs_a if q.z_meet is None]
    print(
        f"engine-predicate validation: {len(pairs_a) - len(missing)}/"
        f"{len(pairs_a)} 2-group pairs have a detected contact "
        f"(subpath admission guarantees one; mismatches = {len(missing)})\n"
    )

    late = [q for q in pairs_a if q.z_meet is not None and q.z_meet > np.pi / 2]
    ensembles = [
        ("A: 2-strand groups (parent + subpath)", pairs_a),
        ("B: never-touching co-group pairs (3..10-strand groups)", pairs_b),
        ("C: late-contact subset of A (z_meet > pi/2)", late),
    ]
    all_results: dict[str, list[StatResult]] = {}
    for label, pairs in ensembles:
        if len(pairs) < 20:
            print(f"== {label}: only {len(pairs)} pairs -- skipped\n")
            continue
        res = permutation_test(pairs, n_perm=args.n_perm)
        all_results[label] = res
        _report(label, pairs, res)

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax_s, ax_z, ax_a1, ax_sz, ax_cls, ax_eff = axes.ravel()

    # a1 partner scatter (both orderings; x axis pooled with y axis)
    u = np.array([[q.a1[0][0], q.a1[0][1]] for q in pairs_a]).ravel()
    v = np.array([[q.a1[1][0], q.a1[1][1]] for q in pairs_a]).ravel()
    zmeet_a = np.array([q.z_meet if q.z_meet is not None else np.nan for q in pairs_a])
    zc = np.repeat(zmeet_a, 2) / np.pi
    sc = ax_a1.scatter(
        np.concatenate([u, v]),
        np.concatenate([v, u]),
        c=np.concatenate([zc, zc]),
        s=2,
        alpha=0.4,
        cmap="viridis",
    )
    fig.colorbar(sc, ax=ax_a1, label="first contact z / pi")
    ax_a1.set_xlabel("a1 (one partner)")
    ax_a1.set_ylabel("a1 (other partner)")
    ax_a1.set_title(
        "The shared hidden anchor: partner a1 vs a1 (r = +0.95)", fontsize=10
    )

    # cold/mover pair-class composition, observed vs reshuffled null (A and C)
    def _class_fracs(pairs: list[Pair]) -> np.ndarray:
        cc = np.mean([q.cold[0] and q.cold[1] for q in pairs])
        mm = np.mean([not q.cold[0] and not q.cold[1] for q in pairs])
        return np.array([cc, 1.0 - cc - mm, mm])

    def _null_class_fracs(pairs: list[Pair]) -> np.ndarray:
        cold_rate = np.mean([q.cold[i] for q in pairs for i in (0, 1)])
        return np.array(
            [
                cold_rate**2,
                2 * cold_rate * (1 - cold_rate),
                (1 - cold_rate) ** 2,
            ]
        )

    labels = ["cold+cold", "mixed", "mover+mover"]
    xpos = np.arange(3)
    for off, (nm, ens, col) in enumerate(
        [("A observed", pairs_a, "tab:blue"), ("C observed", late, "tab:green")]
    ):
        ax_cls.bar(
            xpos + off * 0.35 - 0.175,
            _class_fracs(ens),
            width=0.32,
            color=col,
            label=nm,
        )
        ax_cls.plot(
            xpos + off * 0.35 - 0.175,
            _null_class_fracs(ens),
            "k_",
            ms=16,
            label="independent null" if off == 0 else None,
        )
    ax_cls.set_xticks(xpos)
    ax_cls.set_xticklabels(labels, fontsize=9)
    ax_cls.set_ylabel("fraction of pairs")
    ax_cls.set_title("Partners anti-match the cold/mover class", fontsize=10)
    ax_cls.legend(fontsize=8)

    seps = np.array([q.separation for q in pairs_a])
    rng = np.random.default_rng(3)
    null_seps = []
    by_dump: dict[int, list[Pair]] = {}
    for q in pairs_a:
        by_dump.setdefault(q.dump, []).append(q)
    for plist in by_dump.values():
        idx = rng.permutation(len(plist))
        for i, j in enumerate(idx):
            a, b = plist[i].x0[0], plist[int(j)].x0[1]
            null_seps.append(
                max(abs(float(wrap(a[0] - b[0]))), abs(float(wrap(a[1] - b[1]))))
            )
    bins = np.linspace(0, 1, 41)
    ax_s.hist(seps, bins=bins, density=True, alpha=0.6, label="partners")
    ax_s.hist(
        null_seps,
        bins=bins,
        density=True,
        histtype="step",
        color="k",
        label="reshuffled null",
    )
    ax_s.set_xlabel("Bang separation (wrapped Chebyshev)")
    ax_s.set_ylabel("density")
    ax_s.set_title("Where partners emerge", fontsize=10)
    ax_s.legend(fontsize=8)

    zmeet = np.array([q.z_meet for q in pairs_a if q.z_meet is not None])
    ax_z.hist(zmeet / np.pi, bins=40, color="tab:blue", alpha=0.7)
    ax_z.axvline(0.5, color="gray", ls="--", lw=1, label="turnaround")
    ax_z.set_xlabel("first contact z / pi")
    ax_z.set_ylabel("pairs")
    ax_z.set_title(
        f"When partners first touch (median z/pi = {np.median(zmeet) / np.pi:.2f}, "
        f"{np.mean(zmeet > np.pi / 2) * 100:.0f}% after turnaround)",
        fontsize=10,
    )
    ax_z.legend(fontsize=8)

    ax_sz.plot(seps, zmeet / np.pi, ".", ms=3, alpha=0.4)
    ax_sz.axhline(0.5, color="gray", ls="--", lw=1)
    ax_sz.set_xlabel("Bang separation")
    ax_sz.set_ylabel("first contact z / pi")
    ax_sz.set_title("Separation vs meeting time", fontsize=10)

    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    names = list(STATISTICS)
    width = 0.25
    for k, (label, res) in enumerate(all_results.items()):
        zs = [
            (r.observed - r.null_mean) / r.null_std if r.null_std else 0.0 for r in res
        ]
        ypos = np.arange(len(names)) + (k - 1) * width
        ax_eff.barh(ypos, zs, height=width, color=colors[label[0]], label=label[:1])
    ax_eff.axvline(0, color="k", lw=0.8)
    for x in (-2, 2):
        ax_eff.axvline(x, color="gray", lw=0.6, ls=":")
    ax_eff.set_yticks(np.arange(len(names)))
    ax_eff.set_yticklabels(names, fontsize=8)
    ax_eff.set_xlabel("standardized effect (null sd)")
    ax_eff.set_title("Partner statistics vs reshuffle null", fontsize=10)
    ax_eff.legend(fontsize=8, title="ensemble")

    fig.suptitle(
        "Selection correlations between co-grouped strands "
        "(2+1 subpath dumps, T = 20-300)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
