"""Tests for the D estimator, cost exponent, and curve loading."""

import numpy as np

from braidlab.analyze import cost_exponent, formula_ceiling, load_curve, measure_d
from braidlab.store import RunResult


def _result(t: int, seed: int, n: int) -> RunResult:
    return RunResult(
        dim=3,
        band="nyq",
        t=t,
        seed=seed,
        accept_rate=1e-7,
        terms=2,
        status="done",
        n_final=n,
        attempts=1_000_000,
        curve_path=None,
        host="h",
    )


def test_measure_d_recovers_known_slope() -> None:
    # synthetic power law N = 2 * T**2.5 with small per-seed jitter
    rng = np.random.default_rng(0)
    results = []
    for t in (16, 24, 36, 54, 80):
        base = 2.0 * t**2.5
        for seed in range(8):
            n = base * (1 + 0.02 * rng.standard_normal())
            results.append(_result(t, seed, int(n)))
    r = measure_d(results, 3, "nyq")
    assert abs(r.d - 2.5) < 0.05
    assert r.d_err < 0.05
    assert r.local_slope_std < 0.1  # clean power law -> consistent local slopes


def test_cost_exponent_recovers_slope() -> None:
    # attempts to reach fraction f scale as T**3 by construction
    curves = {}
    for t in (10, 20, 40, 80):
        attempts = np.array([t**3 * x for x in (0.1, 0.5, 1.0, 5.0, 10.0)])
        n = np.array([10.0, 50.0, 90.0, 99.0, 100.0])  # reaches 100
        curves[t] = (attempts, n)
    k = cost_exponent(curves, fractions=(0.5,), ref={t: 100.0 for t in curves})
    assert abs(k[0.5] - 3.0) < 0.2


def test_formula_ceiling() -> None:
    assert formula_ceiling(3, 10) == 125.0
    assert formula_ceiling(2, 40) == 400.0


def test_load_curve(tmp_path) -> None:
    p = tmp_path / "c.csv"
    p.write_text("attempts,n\n100,5\n200,9\n")
    a, n = load_curve(p)
    assert list(a) == [100.0, 200.0]
    assert list(n) == [5.0, 9.0]
