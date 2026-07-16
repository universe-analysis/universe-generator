"""Closed-form identities of the wiggle-budget causal reach."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.analyze_wiggle_reach import (
    plateau_constants,
    reach_from_bang,
    reach_two_time,
)

PI = np.pi


def test_pi_half_phase_free_is_four_thirds() -> None:
    """b=3, f=0 gives exactly |sin(3pi/2)/3 - 1| = 4/3 at the turnaround."""
    r, b = reach_from_bang(PI / 2, bmax=400, phases=False)
    assert b == 3
    assert r == pytest.approx(4 / 3, abs=1e-12)


def test_pi_half_with_phase_is_sqrt2() -> None:
    """b=2 with f=pi/4 gives exactly sqrt(2) at the turnaround."""
    r, b = reach_from_bang(PI / 2, bmax=400, phases=True)
    assert b == 2
    assert r == pytest.approx(np.sqrt(2), abs=1e-12)


def test_crunch_reach_is_two() -> None:
    """Every even frequency approaches displacement 2 at the Crunch."""
    r, _ = reach_from_bang(PI - 1e-6, bmax=400, phases=False)
    assert r == pytest.approx(2.0, abs=1e-4)


def test_b2_envelope_matches_2sin_half_z() -> None:
    """The phased b=2 term alone reaches exactly 2 sin(z/2)."""
    for z in (0.3, 1.0, 2.0, 2.9):
        r, _ = reach_from_bang(z, bmax=2, phases=True)
        assert r == pytest.approx(2 * np.sin(z / 2), abs=1e-12)


def test_bang_plateaus() -> None:
    """Plateaus: 1 + |min sinc| = 1.21723 (f=0) and 1.25959 (phased)."""
    p_f0, u_f0, p_ph, u_ph = plateau_constants(n=1_000_000)
    # tan u = u root and its sinc value
    assert u_f0 == pytest.approx(4.4934, abs=2e-3)
    assert p_f0 == pytest.approx(1.21723, abs=1e-4)
    assert p_ph == pytest.approx(1.25959, abs=1e-4)
    assert u_ph == pytest.approx(4.0856, abs=2e-3)


def test_early_time_reach_beats_antipode() -> None:
    """No horizon: with b up to ~4.5/z available the reach exceeds chi=1
    at any early time (antipodes stay connectable toward the Bang)."""
    for zf in (0.02, 0.05, 0.1, 0.2):
        r, _ = reach_from_bang(zf * PI, bmax=400, phases=True)
        assert r > 1.0


def test_two_time_reach_limits() -> None:
    """R(z1 -> z2) vanishes as z1 -> z2 and recovers the Bang reach as
    z1 -> 0."""
    zo = 0.6 * PI
    assert reach_two_time(zo - 1e-9, zo, bmax=400) == pytest.approx(0.0, abs=1e-6)
    r_bang, _ = reach_from_bang(zo, bmax=400, phases=True)
    assert reach_two_time(1e-7, zo, bmax=400) == pytest.approx(r_bang, abs=1e-5)
