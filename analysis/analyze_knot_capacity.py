"""Knot capacity census (2+1): the largest set of mutually linked strands.

Chris's conjectured maximum-knottable-strand sequence (2, 8, 24 for
d = 1, 2, 3 -- d*2^d, the directed-edge count of the d-hypercube; lab note
2026-08-01) is exploratory: no automated measurement backs it. This census
tests the 2+1 anchor (8) against packed universes.

Operational definition (stated so the number means something):

  * Strand pair "linking": the signed sum of x-projection crossings over the
    conformal interval, sign = sign(DeltaY at the crossing) x sign(DeltaX
    just before it) -- the same chirality convention as the braid census
    (analyze_braids), under which a rotating pair accumulates same-sign
    crossings. A pair is LINKED iff the sum is nonzero; the stricter
    |sum| >= 2 variant (a full mutual wind under the census closure) is
    reported alongside.
  * Knot capacity of a dump: the size of the largest CLIQUE of mutually
    pairwise-linked strands ("n mutually knotted paths"), computed per
    unique group (linking requires braiding, braiding requires the group's
    common neighbourhood). The largest linked connected COMPONENT is
    reported for reference (a weaker, chain-like notion).
  * Groups larger than --max-group strands are skipped for cost and
    COUNTED: a skipped giant could in principle hide a larger clique, so
    the verdict line reports the skip count.

The test verifies "8" if max cliques reach 8 and never exceed it across
deep-filled universes. 3+1 note: point-strand braids are topologically
trivial in three spatial dimensions (strands slide past each other), so
this census cannot test the 3+1 value (24) -- that number needs a framed
(ribbon) or otherwise different operational definition before any
verification is possible.

Usage::

    uv run python -m analysis.analyze_knot_capacity \
        --dumps 'data/fullspec/dumps/d2_nyq_T100_s*_sub_fsub2e6.csv'
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from math import pi

import numpy as np

from analysis.analyze_braids import Strand, load_dump, pair_crossings

_T_RE = __import__("re").compile(r"_T(\d+)")


def pair_link_sum(
    sa: Strand,
    sb: Strand,
    zgrid: np.ndarray,
    contact: float,
    xya: tuple[np.ndarray, np.ndarray],
    xyb: tuple[np.ndarray, np.ndarray],
) -> tuple[int, int]:
    """Signed crossing sums of one pair (census chirality convention).

    Returns (sum over all crossings, sum excluding FRAGILE crossings --
    those with |DeltaY| below the contact scale 2/T, where over-vs-under
    is beneath the packing's own resolution).
    """
    xa = xya[0]
    xb = xyb[0]
    dx = xa - xb
    crossings = pair_crossings(sa, sb, zgrid, contact, xya, xyb)
    if not crossings:
        return 0, 0
    # Sign of DeltaX just before each crossing; flips order is z order,
    # matching pair_crossings' output order.
    flips = np.nonzero(np.sign(dx[:-1]) * np.sign(dx[1:]) < 0)[0]
    pre_signs = np.sign(dx[flips]).astype(int)
    total = robust = 0
    for c, s_pre in zip(crossings, pre_signs):
        term = c.dy_sign * int(s_pre)
        total += term
        if not c.fragile:
            robust += term
    return total, robust


def max_clique(adj: dict[int, set[int]]) -> int:
    """Exact maximum clique size (Bron-Kerbosch with pivoting).

    Group linked-graphs are small and sparse; this is fast enough here.
    """
    best = 0

    def bk(r: set[int], p: set[int], x: set[int]) -> None:
        nonlocal best
        if not p and not x:
            best = max(best, len(r))
            return
        if len(r) + len(p) <= best:
            return
        pivot = max(p | x, key=lambda v: len(adj[v] & p), default=None)
        if pivot is None:
            return
        for v in list(p - adj[pivot]):
            bk(r | {v}, p & adj[v], x & adj[v])
            p = p - {v}
            x = x | {v}

    bk(set(), set(adj), set())
    return best


def components(adj: dict[int, set[int]]) -> list[int]:
    seen: set[int] = set()
    sizes = []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            stack.extend(adj[v] - comp)
        seen |= comp
        sizes.append(len(comp))
    return sizes


def census_dump(
    path: str, supersample: int, max_group: int
) -> tuple[int, int, int, int, int, int]:
    """(clique sum!=0, clique |sum|>=2, clique robust!=0, max comp, groups, skipped)."""
    t = int(_T_RE.search(path).group(1))  # type: ignore[union-attr]
    contact = 2.0 / t
    zgrid = np.linspace(pi / (t + 1), t * pi / (t + 1), (t - 1) * supersample + 1)
    groups: dict[int, list[Strand]] = defaultdict(list)
    for s in load_dump(path):
        groups[s.gid].append(s)

    best1 = best2 = best3 = best_comp = skipped = 0
    n_groups = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        n_groups += 1
        if len(members) > max_group:
            skipped += 1
            continue
        cache = [s.xy(zgrid) for s in members]
        adj1: dict[int, set[int]] = {i: set() for i in range(len(members))}
        adj2: dict[int, set[int]] = {i: set() for i in range(len(members))}
        adj3: dict[int, set[int]] = {i: set() for i in range(len(members))}
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                w, w_rob = pair_link_sum(
                    members[i], members[j], zgrid, contact, cache[i], cache[j]
                )
                if w != 0:
                    adj1[i].add(j)
                    adj1[j].add(i)
                if abs(w) >= 2:
                    adj2[i].add(j)
                    adj2[j].add(i)
                if w_rob != 0:
                    adj3[i].add(j)
                    adj3[j].add(i)
        best1 = max(best1, max_clique(adj1))
        best2 = max(best2, max_clique(adj2))
        best3 = max(best3, max_clique(adj3))
        comp_sizes = components({k: v for k, v in adj1.items() if v})
        if comp_sizes:
            best_comp = max(best_comp, max(comp_sizes))
    return best1, best2, best3, best_comp, n_groups, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", required=True, help="glob of subpath dumps")
    parser.add_argument("--supersample", type=int, default=6)
    parser.add_argument("--max-group", type=int, default=400)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.dumps))
    if not paths:
        raise SystemExit(f"no dumps match {args.dumps}")
    print(
        f"{'dump':52s} {'clique!=0':>9} {'clique>=2':>9} {'robust':>7} "
        f"{'comp':>6} {'groups':>6} {'skip':>5}"
    )
    overall1 = overall2 = overall3 = 0
    for p in paths:
        c1, c2, c3, comp, ng, skip = census_dump(p, args.supersample, args.max_group)
        overall1 = max(overall1, c1)
        overall2 = max(overall2, c2)
        overall3 = max(overall3, c3)
        name = p.split("/")[-1]
        print(
            f"{name:52s} {c1:>9} {c2:>9} {c3:>7} {comp:>6} {ng:>6} {skip:>5}",
            flush=True,
        )
    print(
        f"\nVERDICT: max mutually-linked clique = {overall1} (sum != 0), "
        f"{overall2} (|sum| >= 2), {overall3} (fragile-excluded) "
        f"across {len(paths)} dumps"
    )


if __name__ == "__main__":
    main()
