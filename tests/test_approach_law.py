"""Tests for the Feder-law tail fit."""

import numpy as np

from analysis.analyze_approach_law import fit_feder


def _synth(n_inf: float, c: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.logspace(4, 10, 120)
    return t, n_inf - c * t ** (-p)


def test_fit_recovers_synthetic_ceiling() -> None:
    t, n = _synth(5000.0, 20000.0, 0.4)
    n_inf, p, interior = fit_feder(t, n)
    assert interior
    assert abs(p - 0.4) < 0.02
    assert abs(n_inf - 5000.0) / 5000.0 < 0.01


def test_flat_tail_reads_as_railed_not_wrong() -> None:
    # Nearly pure power growth (no visible ceiling): the free fit must flag
    # itself as railed rather than return a confident bogus ceiling.
    t = np.logspace(4, 10, 120)
    n = 100.0 * (t / t[0]) ** 0.05
    _, _, interior = fit_feder(t, n)
    assert not interior


def test_fixed_p_reports_interior() -> None:
    t, n = _synth(5000.0, 20000.0, 0.12)
    n_inf, p, interior = fit_feder(t, n, p_fixed=0.12)
    assert interior
    assert p == 0.12
    assert abs(n_inf - 5000.0) / 5000.0 < 0.01
