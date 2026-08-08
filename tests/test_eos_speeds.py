"""Tests for the three-velocity equation-of-state analysis."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.analyze_eos_speeds import speeds_squared, wz_curves
from braidlab.corrdim import AxisTerms


def axis(a2: float, a: float = 0.0, b: int = 3, f: float = 0.0) -> AxisTerms:
    return AxisTerms(
        a=np.array([[a]]),
        b=np.array([[float(b)]]),
        f=np.array([[f]]),
        a2=np.array([a2]),
    )


def test_pure_anchor_has_zero_peculiar_speed() -> None:
    """A comoving anchor (X = const) moves only with the Hubble flow."""
    axes = [axis(a2=0.7), axis(a2=-0.3)]
    for z in (0.3, 1.1, 2.6):
        v_phys, v_pec, v_com = speeds_squared(axes, z)
        # physical slope is the recession a2*cos(z), summed in quadrature
        assert v_phys[0] == pytest.approx(
            (0.7 * np.cos(z)) ** 2 + (0.3 * np.cos(z)) ** 2
        )
        assert v_pec[0] == pytest.approx(0.0, abs=1e-12)
        assert v_com[0] == pytest.approx(0.0, abs=1e-12)


def test_peculiar_equals_physical_at_turnaround() -> None:
    """cos(pi/2) = 0 kills the Hubble term: the two EOS speeds coincide."""
    axes = [axis(a2=0.5, a=0.2, b=4, f=0.9), axis(a2=-0.8, a=0.1, b=7)]
    v_phys, v_pec, v_com = speeds_squared(axes, np.pi / 2)
    assert v_pec[0] == pytest.approx(v_phys[0], rel=1e-12)
    # and the comoving speed equals the peculiar one there (sin = 1)
    assert v_com[0] == pytest.approx(v_pec[0], rel=1e-12)


def test_comoving_is_peculiar_over_sin_squared() -> None:
    axes = [axis(a2=0.4, a=0.15, b=5)]
    for z in (0.4, 2.0):
        _, v_pec, v_com = speeds_squared(axes, z)
        assert v_com[0] == pytest.approx(v_pec[0] / np.sin(z) ** 2, rel=1e-12)


def test_wz_curves_shapes_and_turnaround_identity() -> None:
    rng = np.random.default_rng(3)
    n, nw = 40, 3
    axes = [
        AxisTerms(
            a=rng.normal(0, 0.05, (n, nw)),
            b=rng.integers(2, 9, (n, nw)).astype(float),
            f=rng.uniform(0, np.pi, (n, nw)),
            a2=rng.uniform(-1, 1, n),
        )
        for _ in range(3)
    ]
    zgrid = np.array([0.3, np.pi / 2, 2.8])
    w = wz_curves(axes, zgrid)
    assert set(w) == {"phys", "pec", "com"}
    assert all(len(v) == 3 for v in w.values())
    assert w["pec"][1] == pytest.approx(w["phys"][1], rel=1e-10)
    assert w["com"][1] == pytest.approx(w["pec"][1], rel=1e-10)
