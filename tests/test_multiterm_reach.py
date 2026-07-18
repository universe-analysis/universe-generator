"""Multi-term configs must never out-reach the single-term envelope."""

from __future__ import annotations

import numpy as np

from analysis.analyze_multiterm_reach import (
    optimize_at,
    random_sweep,
    split_scan,
    top_examples,
)
from analysis.analyze_wiggle_reach import reach_from_bang

PI = np.pi
BMAX = 120


def test_random_multiterm_never_beats_envelope() -> None:
    """A random ensemble sweep (k = 2..4) stays at or below the blue line."""
    rng = np.random.default_rng(7)
    z_grid = np.linspace(0.02 * PI, 0.98 * PI, 40)
    envelope = np.array([reach_from_bang(float(z), BMAX)[0] for z in z_grid])
    for k in (2, 3, 4):
        best = random_sweep(z_grid, k, n_configs=3000, bmax=BMAX, rng=rng)
        assert np.all(best <= envelope + 1e-12)


def test_optimizer_saturates_at_envelope() -> None:
    """The adversarial optimizer recovers the envelope exactly (budget
    concentration on the best single frequency) but cannot exceed it."""
    rng = np.random.default_rng(7)
    for zf in (0.1, 0.5, 0.9):
        z = zf * PI
        env = reach_from_bang(z, BMAX)[0]
        for k in (2, 3):
            found = optimize_at(z, k, BMAX, rng, n_freq_sets=6, n_starts=4)
            assert found <= env + 1e-6
            assert found >= 0.95 * env  # it does find near-optimal configs


def test_examples_spend_exactly_unit_budget() -> None:
    """Every sampled config satisfies sum |a_i * b_i| = 1 and its trajectory
    at z_star reproduces the recorded displacement."""
    rng = np.random.default_rng(11)
    z_star = PI / 2
    env = reach_from_bang(z_star, BMAX)[0]
    for cfg in top_examples(z_star, 2, n_configs=2000, bmax=BMAX, rng=rng):
        assert abs(cfg.budget() - 1.0) < 1e-12
        traj = cfg.trajectory(np.array([z_star]))
        assert abs(abs(traj[0]) - cfg.disp) < 1e-12
        assert cfg.disp <= env + 1e-12


def test_split_scan_is_linear_in_the_budget() -> None:
    """The phase-maximized reach of a fixed frequency pair interpolates
    linearly in the budget split — the geometric heart of the claim."""
    for pair in ((2, 3), (2, 4)):
        w_grid, numeric, predicted = split_scan(PI / 2, pair)
        assert np.allclose(numeric, predicted, atol=2e-5)
        assert float(numeric.max()) <= np.sqrt(2) + 1e-12


def test_pi_half_optimum_is_sqrt2() -> None:
    """At the turnaround the 2-term optimizer lands on sqrt(2) — the b=2,
    f=pi/4 single-term optimum — to high precision."""
    rng = np.random.default_rng(3)
    found = optimize_at(PI / 2, 2, BMAX, rng, n_freq_sets=8, n_starts=8)
    assert abs(found - np.sqrt(2)) < 1e-4
