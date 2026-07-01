"""Exact non-intersection solver for the constrained braiding model.

Model
-----
Time ``t`` runs over the open loop interval ``(0, pi)`` (the Bang/Crunch
endpoints are excluded by convention -- every path sits at the origin there).

A *path* in ``n`` spatial dimensions assigns to each dimension a single
component ``s * sin(f * t)`` where ``s`` is a sign (+1/-1) and ``f`` is one of
the allowed integer frequencies (frequency 1 is disallowed in this model).

Two paths *intersect* if there is a time ``t`` in ``(0, pi)`` where every
coordinate matches simultaneously.  We want the largest set of paths that are
pairwise non-intersecting.

Why this is exact
-----------------
For integer frequencies, ``a*sin(f t) = b*sin(g t)`` holds only at times that
are *rational multiples of pi*.  So the per-coordinate equality set is a finite
set of fractions ``r = t/pi in (0, 1)`` (or "all t" when the components are
identical).  Two paths collide iff the intersection of these sets over all
coordinates contains an interior point.  Everything is computed with
``fractions.Fraction`` -- no floating point, so "never overlaps" is provable.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations, product
from typing import Optional

# A single coordinate component: (sign, frequency).
Component = tuple[int, int]
# A path: one component per spatial dimension.
Path = tuple[Component, ...]

# Sentinel meaning "this coordinate matches for every t" (identical components).
ALL = "ALL"


@dataclass(frozen=True)
class SolveResult:
    """Outcome of a maximum non-intersecting search."""

    paths: list[Path]
    count: int


def _interior_fracs(numer_step: int, denom: int, offset: int) -> set[Fraction]:
    """Return fractions ``(offset + numer_step*k)/denom`` lying in ``(0, 1)``.

    ``numer_step`` is 1 (for ``k`` ranging over all integers) or 2 (even/odd
    families).  ``denom`` may be negative; it is normalised here.
    """
    if denom == 0:
        return set()
    if denom < 0:
        denom = -denom
        offset = -offset
        # numer_step magnitude is unchanged by the sign flip.
    out: set[Fraction] = set()
    # k such that 0 < (offset + numer_step*k)/denom < 1.
    # => -offset < numer_step*k < denom - offset.
    lo = -offset
    hi = denom - offset
    k = -(-lo // numer_step) if numer_step else 0  # ceil(lo/step) start point
    # Iterate a safe range; denom is small so this is cheap.
    k_min = (lo // numer_step) - 1
    k_max = (hi // numer_step) + 1
    for k in range(k_min, k_max + 1):
        val = offset + numer_step * k
        frac = Fraction(val, denom)
        if 0 < frac < 1:
            out.add(frac)
    return out


def coordinate_equality_set(a: int, f: int, b: int, g: int) -> object:
    """Times ``r = t/pi in (0, 1)`` where ``a*sin(f t) == b*sin(g t)``.

    Returns either the sentinel :data:`ALL` (identical components, matches
    everywhere) or a ``set`` of :class:`~fractions.Fraction` interior times.
    """
    same_sign = a == b
    if f == g:
        if same_sign:
            return ALL
        # a*sin(f t) = -a*sin(f t) -> sin(f t) = 0 -> r = k/f.
        return _interior_fracs(1, f, 0)

    out: set[Fraction] = set()
    if same_sign:
        # sin(f t)=sin(g t): r = 2k/(f-g)  or  r = (2k+1)/(f+g).
        out |= _interior_fracs(2, f - g, 0)
        out |= _interior_fracs(2, f + g, 1)
    else:
        # sin(f t)=-sin(g t): r = 2k/(f+g)  or  r = (2k+1)/(f-g).
        out |= _interior_fracs(2, f + g, 0)
        out |= _interior_fracs(2, f - g, 1)
    return out


def paths_intersect(p: Path, q: Path) -> bool:
    """True if paths ``p`` and ``q`` coincide at some interior time."""
    if p == q:
        return True
    common: Optional[set[Fraction]] = None
    for (a, f), (b, g) in zip(p, q):
        s = coordinate_equality_set(a, f, b, g)
        if s is ALL:
            continue  # this coordinate never constrains the collision time
        assert isinstance(s, set)
        if not s:
            return False  # this coordinate can never match -> no collision
        common = s if common is None else (common & s)
        if not common:
            return False
    # common is None only if every coordinate was ALL -> identical functions.
    return bool(common) if common is not None else True


def generate_candidates(
    n: int, frequencies: list[int], *, permutation: bool
) -> list[Path]:
    """Enumerate candidate paths for ``n`` dimensions.

    ``permutation=True`` assigns the ``n`` frequencies bijectively across the
    ``n`` dimensions (requires ``len(frequencies) == n``).  ``permutation=False``
    lets every dimension pick any frequency independently (the "free" model).
    """
    signs = (1, -1)
    if permutation:
        if len(frequencies) != n:
            raise ValueError("permutation mode needs len(frequencies) == n")
        freq_orders = list(permutations(frequencies))
    else:
        freq_orders = list(product(frequencies, repeat=n))
    out: list[Path] = []
    for order in freq_orders:
        for sign_combo in product(signs, repeat=n):
            path: Path = tuple(zip(sign_combo, order))
            out.append(path)
    # Dedupe (permutation of equal freqs can repeat).
    return sorted(set(out))


def _max_clique(nodes: list[int], adj: list[set[int]]) -> list[int]:
    """Maximum clique via Bron-Kerbosch with pivoting (small graphs)."""
    best: list[int] = []

    def expand(r: list[int], p: set[int], x: set[int]) -> None:
        nonlocal best
        if not p and not x:
            if len(r) > len(best):
                best = list(r)
            return
        # Bound: cannot beat best.
        if len(r) + len(p) <= len(best):
            return
        pivot = max(p | x, key=lambda u: len(adj[u] & p)) if (p | x) else None
        candidates = list(p - adj[pivot]) if pivot is not None else list(p)
        for v in candidates:
            expand(r + [v], p & adj[v], x & adj[v])
            p = p - {v}
            x = x | {v}

    expand([], set(nodes), set())
    return best


def max_non_intersecting(candidates: list[Path]) -> SolveResult:
    """Largest pairwise non-intersecting subset of ``candidates`` (exact)."""
    m = len(candidates)
    adj: list[set[int]] = [set() for _ in range(m)]
    for i, j in combinations(range(m), 2):
        if not paths_intersect(candidates[i], candidates[j]):
            adj[i].add(j)
            adj[j].add(i)
    clique = _max_clique(list(range(m)), adj)
    return SolveResult(paths=[candidates[i] for i in clique], count=len(clique))


def solve(n: int, frequencies: list[int], *, permutation: bool = True) -> SolveResult:
    """Convenience: enumerate candidates then return the maximum set."""
    return max_non_intersecting(
        generate_candidates(n, frequencies, permutation=permutation)
    )


def format_path(p: Path) -> str:
    """Render a path like ``(+sin3t, -sin10t)``."""
    axes = ["x", "y", "z", "w"]
    parts = []
    for i, (s, f) in enumerate(p):
        sign = "+" if s > 0 else "-"
        axis = axes[i] if i < len(axes) else f"d{i}"
        parts.append(f"{axis}={sign}sin{f}t")
    return "(" + ", ".join(parts) + ")"
