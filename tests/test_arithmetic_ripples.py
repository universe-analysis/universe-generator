"""Tests for the arithmetic-ripple analysis (analysis/analyze_arithmetic_ripples)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.analyze_arithmetic_ripples import (
    abundance,
    big_omega,
    divisor_count,
    factorize,
    fit_ripples,
    is_prime,
    lag1_autocorr,
    largest_prime_factor,
    omega,
    regress_covariate,
)


def test_arithmetic_functions() -> None:
    assert factorize(360) == {2: 3, 3: 2, 5: 1}
    assert divisor_count(360) == 24
    assert divisor_count(100) == 9
    assert omega(360) == 3
    assert big_omega(360) == 6
    assert largest_prime_factor(520) == 13
    assert abundance(6) == pytest.approx(2.0)  # perfect number
    assert is_prime(13) and not is_prime(1) and not is_prime(360)


def test_factorize_rejects_zero() -> None:
    with pytest.raises(ValueError):
        factorize(0)


def _synthetic(ripple: dict[int, float] | None = None) -> dict[int, list[float]]:
    """Seed ensembles on an exact N = 3 T^2.3 law, optional per-T ripple."""
    rng = np.random.default_rng(7)
    by_t: dict[int, list[float]] = {}
    for t in range(20, 340, 20):
        n = 3.0 * t**2.3 * np.exp((ripple or {}).get(t, 0.0))
        by_t[t] = list(n * np.exp(rng.normal(0, 1e-3, size=6)))
    return by_t


def test_fit_ripples_recovers_pure_power_law() -> None:
    rip = fit_ripples(_synthetic(), degree=2)
    assert rip.slope == pytest.approx(2.3, abs=0.01)
    assert np.max(np.abs(rip.resid)) < 5e-3
    assert rip.chi2_p > 1e-4  # residuals consistent with seed noise


def test_planted_divisor_signal_is_detected() -> None:
    ts = list(range(20, 340, 20))
    planted = {t: 0.02 * np.log(divisor_count(t)) for t in ts}
    rip = fit_ripples(_synthetic(planted), degree=2)
    vals = np.array([np.log(divisor_count(t)) for t in rip.t_values])
    res = regress_covariate(rip, "log d(T)", vals, n_perm=2000)
    assert res is not None
    assert res.perm_p < 0.01
    assert res.wls_slope == pytest.approx(0.02, rel=0.2)


def test_pure_noise_gives_null() -> None:
    rip = fit_ripples(_synthetic(), degree=2)
    vals = np.array([np.log(divisor_count(t)) for t in rip.t_values])
    res = regress_covariate(rip, "log d(T)", vals, n_perm=2000)
    assert res is not None
    assert res.perm_p > 0.05


def test_constant_covariate_returns_none() -> None:
    rip = fit_ripples(_synthetic(), degree=2)
    assert regress_covariate(rip, "const", np.ones(len(rip.t_values))) is None


def test_lag1_autocorr_flags_smooth_structure() -> None:
    ts = list(range(20, 340, 20))
    smooth = {t: 0.02 * np.sin(2 * np.pi * (t - 20) / 320) for t in ts}
    rip = fit_ripples(_synthetic(smooth), degree=1)
    ac, p = lag1_autocorr(rip, n_perm=2000)
    assert ac > 0.3
    assert p < 0.05
