"""Tests for analysis.analyze_braids: B_3 algebra + geometric braid extraction."""

from __future__ import annotations

from math import pi

import numpy as np

from analysis.analyze_braids import (
    Strand,
    b3_invariant,
    build_b3_dictionary,
    group_braid,
    word_str,
)

# ---------------------------------------------------------------------------
# B_3 algebra: the Burau-at-(-1) invariant
# ---------------------------------------------------------------------------


def test_braid_relation_holds() -> None:
    # sigma1 sigma2 sigma1 = sigma2 sigma1 sigma2
    left = [(1, 1), (2, 1), (1, 1)]
    right = [(2, 1), (1, 1), (2, 1)]
    assert b3_invariant(left) == b3_invariant(right)


def test_free_reduction_is_identity() -> None:
    assert b3_invariant([(1, 1), (1, -1)]) == b3_invariant([])
    assert b3_invariant([(2, -1), (2, 1)]) == b3_invariant([])


def test_bt_templates_are_distinct_nontrivial_classes() -> None:
    lh = b3_invariant([(1, 1), (2, -1)])  # sigma1 sigma2^-1 (left-handed)
    rh = b3_invariant([(2, -1), (1, 1)])  # sigma2^-1 sigma1 (right-handed)
    assert lh != rh
    assert lh != b3_invariant([])
    assert rh != b3_invariant([])


def test_dictionary_finds_shortest_words() -> None:
    d = build_b3_dictionary(4)
    assert d[b3_invariant([])] == []
    # A padded word reduces to its 2-letter class representative.
    padded = [(1, 1), (2, 1), (2, -1), (2, -1)]
    short = d[b3_invariant(padded)]
    assert len(short) == 2
    assert b3_invariant(short) == b3_invariant(padded)


def test_word_str() -> None:
    assert word_str([]) == "1 (trivial)"
    assert word_str([(1, 1), (2, -1)]) == "s1.s2'"


# ---------------------------------------------------------------------------
# geometric extraction on synthetic strands
# ---------------------------------------------------------------------------


class Orbit:
    """A strand moving on a circle: center + radius + phase, angle = rate*z."""

    def __init__(self, cx: float, cy: float, r: float, rate: float, phase: float):
        self.cx, self.cy, self.r, self.rate, self.phase = cx, cy, r, rate, phase

    def xy(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        th = self.rate * z + self.phase
        return self.cx + self.r * np.cos(th), self.cy + self.r * np.sin(th)


class Still:
    """A strand parked at a fixed point."""

    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def xy(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full_like(z, self.x), np.full_like(z, self.y)


class Mirror:
    """y-mirror of another strand (flips every crossing's chirality)."""

    def __init__(self, inner):
        self.inner = inner

    def xy(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.inner.xy(z)
        return x, -y


ZGRID = np.linspace(0.01, pi - 0.01, 4001)


def test_two_strands_winding() -> None:
    # Two strands on opposite sides of a circle making 3 full relative turns:
    # 2 crossings per turn, all the same sign -> |exp_sum| = 6.
    rate = 6 * 2 * pi / (ZGRID[-1] - ZGRID[0]) / 2  # 3 relative turns
    a = Orbit(0, 0, 1, rate, 0.0)
    b = Orbit(0, 0, 1, rate, pi)
    gb = group_braid([a, b], ZGRID, contact=1e-6)  # type: ignore[arg-type]
    assert gb.n == 2
    assert abs(gb.exp_sum) == 6
    assert len(gb.word) == 6
    assert gb.fragile == 0


def test_mirror_flips_chirality() -> None:
    rate = 2 * pi / (ZGRID[-1] - ZGRID[0])  # 1 relative turn
    a = Orbit(0, 0, 1, rate, 0.0)
    b = Orbit(0, 0, 1, rate, pi)
    gb = group_braid([a, b], ZGRID, contact=1e-6)  # type: ignore[arg-type]
    gm = group_braid([Mirror(a), Mirror(b)], ZGRID, contact=1e-6)  # type: ignore[arg-type]
    assert gb.exp_sum == -gm.exp_sum != 0


def test_bystander_strand_stays_out_of_the_word() -> None:
    # A far-right parked strand never crosses; the pair braids as sigma1^k.
    rate = 2 * pi / (ZGRID[-1] - ZGRID[0])
    a = Orbit(0, 0, 1, rate, 0.0)
    b = Orbit(0, 0, 1, rate, pi)
    c = Still(10.0, 0.0)
    gb = group_braid([a, b, c], ZGRID, contact=1e-6)  # type: ignore[arg-type]
    assert gb.n == 3
    assert all(k == 1 for k, _ in gb.word)  # only sigma1 appears
    assert abs(gb.exp_sum) == 2
    assert gb.perm[2] == 2  # bystander keeps its slot


def test_rigid_half_turn_is_garside_element() -> None:
    # Three strands rigidly rotating by a half-turn trace the Garside braid
    # Delta = sigma1 sigma2 sigma1 (3 crossings, all one sign, full reversal).
    rate = pi / (ZGRID[-1] - ZGRID[0])
    # Distinct phases with distinct x-projections (cos 0, cos 2, cos 4).
    strands = [Orbit(0, 0, 1, rate, ph) for ph in (0.0, 2.0, 4.0)]
    gb = group_braid(strands, ZGRID, contact=1e-6)  # type: ignore[arg-type]
    assert len(gb.word) == 3
    assert abs(gb.exp_sum) == 3
    assert tuple(gb.perm) == tuple(reversed(np.argsort([1.0, np.cos(2), np.cos(4)])))
    d = build_b3_dictionary(4)
    short = d[b3_invariant(gb.word)]
    assert len(short) == 3


def test_real_strand_parametrization_smoke() -> None:
    # A real Strand evaluates finitely across the engine grid interval and is
    # pinned near its comoving offset at both ends (wiggle/sin z stays bounded).
    s = Strand(
        ax=np.array([0.5]),
        bx=np.array([2]),
        fx=np.array([0.7]),
        ay=np.array([1.0 / 3.0]),
        by=np.array([3]),
        fy=np.array([0.0]),
        ax2=0.25,
        ay2=-0.5,
        gid=0,
    )
    t = 40
    z = np.linspace(pi / (t + 1), t * pi / (t + 1), 4 * t)
    x, y = s.xy(z)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    assert np.all(np.abs(x) < 4) and np.all(np.abs(y) < 4)
