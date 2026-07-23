"""Tests for the group-partner correlation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.analyze_braids import Strand
from analysis.analyze_group_partners import (
    Pair,
    bang_position,
    bang_velocity,
    engine_zgrid,
    first_contact,
    make_pair,
    permutation_test,
    wrap,
)


def strand(
    ax: float = 0.01,
    bx: int = 3,
    fx: float = 0.0,
    ay: float = 0.01,
    by: int = 5,
    fy: float = 0.0,
    ax2: float = 0.0,
    ay2: float = 0.0,
    gid: int = 0,
) -> Strand:
    return Strand(
        ax=np.array([ax]),
        bx=np.array([bx]),
        fx=np.array([fx]),
        ay=np.array([ay]),
        by=np.array([by]),
        fy=np.array([fy]),
        ax2=ax2,
        ay2=ay2,
        gid=gid,
    )


def test_bang_position_matches_trajectory_limit() -> None:
    s = strand(ax=0.05, bx=4, fx=1.1, ay=-0.03, by=7, ax2=0.4, ay2=-0.8)
    x0, y0 = bang_position(s)
    z = np.array([1e-6])
    x, y = s.xy(z)
    assert wrap(x[0]) == pytest.approx(x0, abs=1e-4)
    assert wrap(y[0]) == pytest.approx(y0, abs=1e-4)


def test_bang_velocity_matches_trajectory_derivative() -> None:
    s = strand(ax=0.05, bx=4, fx=1.1, ay=0.02, by=6, fy=2.0)
    vx, vy = bang_velocity(s)
    z = np.array([1e-4, 2e-4])
    x, y = s.xy(z)
    assert (x[1] - x[0]) / 1e-4 == pytest.approx(vx, rel=1e-2)
    assert (y[1] - y[0]) / 1e-4 == pytest.approx(vy, rel=1e-2)
    cold = strand(bx=3, by=5)  # all-odd: phases are 0, so V0 must vanish
    assert bang_velocity(cold) == (0.0, 0.0)


def test_first_contact_identical_and_disjoint() -> None:
    t = 50
    zgrid = engine_zgrid(t)
    a = strand(ax2=0.3, ay2=0.3)
    assert first_contact(a, a, zgrid, 2.0 / t) == pytest.approx(zgrid[0])
    b = strand(ax2=-0.7, ay2=-0.7)  # parked 1.0 away on both axes, tiny wiggle
    assert first_contact(a, b, zgrid, 2.0 / t) is None


def test_first_contact_wraps_the_torus() -> None:
    t = 50
    zgrid = engine_zgrid(t)
    a = strand(ax=0.0, ay=0.0, ax2=-0.999, ay2=0.0)
    b = strand(ax=0.0, ay=0.0, ax2=0.999, ay2=0.0)
    # Comoving X = a1 (constant): the unwrapped gap 1.998 exceeds the cell,
    # but the wrapped gap is 0.002 -- in contact from the first timestep.
    z = first_contact(a, b, zgrid, 2.0 / t)
    assert z == pytest.approx(zgrid[0])


def _paired_ensemble(rho: float, n: int = 400, seed: int = 5) -> list[Pair]:
    """Pairs whose a1 features share correlation rho; all else independent."""
    rng = np.random.default_rng(seed)
    out = []
    zgrid = engine_zgrid(20)
    for i in range(n):
        base = rng.uniform(-1, 1, size=2)
        noise = rng.uniform(-1, 1, size=2)
        a1_b = rho * base + (1 - rho) * noise
        a = strand(ax2=float(base[0]), ay2=float(base[1]), gid=i)
        b = strand(ax2=float(a1_b[0]), ay2=float(a1_b[1]), gid=i)
        out.append(make_pair(a, b, dump=i % 4, zgrid=zgrid, cell=0.1))
    return out


def test_permutation_detects_planted_a1_correlation() -> None:
    res = permutation_test(_paired_ensemble(0.8), n_perm=300)
    a1 = next(r for r in res if "a1" in r.name)
    assert a1.observed > 0.5
    assert a1.perm_p < 0.02


def test_permutation_null_on_independent_pairs() -> None:
    res = permutation_test(_paired_ensemble(0.0), n_perm=300)
    a1 = next(r for r in res if "a1" in r.name)
    assert abs(a1.observed) < 0.15
    assert a1.perm_p > 0.05


def test_pair_symmetry_of_approach_and_separation() -> None:
    zgrid = engine_zgrid(30)
    a = strand(ax=0.04, bx=4, fx=0.7, ax2=0.2)
    b = strand(ax=0.04, bx=6, fx=2.1, ax2=-0.5)
    p1 = make_pair(a, b, 0, zgrid, 2.0 / 30)
    p2 = make_pair(b, a, 0, zgrid, 2.0 / 30)
    assert p1.separation == pytest.approx(p2.separation)
    assert p1.approaching == p2.approaching
    assert p1.z_meet == p2.z_meet
