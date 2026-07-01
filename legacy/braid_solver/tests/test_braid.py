"""Tests for the exact braiding solver."""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path as FsPath

sys.path.insert(0, str(FsPath(__file__).resolve().parent.parent))

from braid import (  # noqa: E402
    ALL,
    coordinate_equality_set,
    generate_candidates,
    paths_intersect,
    solve,
)


def _numeric_min_gap(p, q, n: int = 20000, margin: float = 0.01) -> float:
    """Min interior L-inf gap, endpoints excluded."""
    best = math.inf
    lo, hi = int(n * margin), int(n * (1 - margin))
    for k in range(lo, hi + 1):
        t = math.pi * k / n
        gap = max(
            abs(a * math.sin(f * t) - b * math.sin(g * t))
            for (a, f), (b, g) in zip(p, q)
        )
        best = min(best, gap)
    return best


def test_identical_components_match_everywhere() -> None:
    assert coordinate_equality_set(1, 5, 1, 5) is ALL


def test_opposite_sign_same_freq_zeros() -> None:
    # +sin(3t) == -sin(3t) only at zeros r = k/3 in (0,1).
    s = coordinate_equality_set(1, 3, -1, 3)
    assert s == {Fraction(1, 3), Fraction(2, 3)}


def test_1d_two_paths_intersect() -> None:
    # +sin and -sin cross at interior zeros.
    assert paths_intersect(((1, 100),), ((-1, 100),))


def test_1d_max_is_one() -> None:
    assert solve(1, [100], permutation=True).count == 1


def test_2d_max_is_four() -> None:
    assert solve(2, [3, 10], permutation=True).count == 4


def test_3d_permutation_max_is_twelve() -> None:
    assert solve(3, [3, 7, 10], permutation=True).count == 12


def test_3d_two_freq_free_max_is_eight() -> None:
    assert solve(3, [3, 10], permutation=False).count == 8


def test_selected_set_is_numerically_collision_free() -> None:
    res = solve(2, [3, 10], permutation=True)
    for i in range(len(res.paths)):
        for j in range(i + 1, len(res.paths)):
            assert _numeric_min_gap(res.paths[i], res.paths[j]) > 1e-2


def test_exact_matches_numeric_over_all_pairs() -> None:
    cands = generate_candidates(2, [3, 10], permutation=True)
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            exact = paths_intersect(cands[i], cands[j])
            numeric = _numeric_min_gap(cands[i], cands[j]) < 1e-2
            assert exact == numeric, (cands[i], cands[j])
