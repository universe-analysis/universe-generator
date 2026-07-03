"""Tests for the correlation-dimension core and seed-aggregation."""

import csv
from pathlib import Path

import numpy as np

from braidlab import corrdim


def test_fit_slope_recovers_a_known_exponent() -> None:
    x = np.logspace(-2, 0, 50)
    y = 3.0 * x**2.5
    assert abs(corrdim.fit_slope(x, y, 0.02, 0.5) - 2.5) < 1e-9


def test_correlation_integral_on_uniform_cube_is_3d() -> None:
    rng = np.random.default_rng(0)
    pts = rng.random((15000, 3))
    radii = np.logspace(np.log10(0.02), np.log10(0.12), 25)
    c = corrdim.correlation_integral(pts, radii)
    # A uniform 3D cloud has correlation dimension 3 (window kept off the walls).
    assert abs(corrdim.fit_slope(radii, c, 0.03, 0.1) - 3.0) < 0.25


def _write_dump(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cols = ["ax", "ay", "aw", "bx", "by", "bw", "ax2", "ay2", "aw2"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for _ in range(n):
            a2 = rng.uniform(-1, 1, 3)  # spreads positions across [-1, 1]^3
            w.writerow([0.001, 0.001, 0.001, 101, 103, 107, *a2])


def _write_dump_2d(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cols = ["ax", "ay", "bx", "by", "ax2", "ay2"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for _ in range(n):
            a2 = rng.uniform(-1, 1, 2)
            w.writerow([0.001, 0.001, 101, 103, *a2])


def test_load_turnaround_cloud_shape(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    _write_dump(p, 200, seed=1)
    cloud = corrdim.load_turnaround_cloud(p)
    assert cloud.shape == (200, 3)


def test_load_turnaround_cloud_infers_2d(tmp_path: Path) -> None:
    p = tmp_path / "d2.csv"
    _write_dump_2d(p, 150, seed=1)
    cloud = corrdim.load_turnaround_cloud(p)
    assert cloud.shape == (150, 2)  # no w-axis column -> 2+1


def test_aggregate_groups_by_t_and_averages(tmp_path: Path) -> None:
    for seed in (1, 2, 3):
        _write_dump(tmp_path / f"d3_nyq_T40_s{seed}.csv", 4000, seed)
    stats = corrdim.aggregate(tmp_path, dim=3, band="nyq")
    assert len(stats) == 1
    s = stats[0]
    assert s.t == 40 and s.n_seeds == 3
    assert s.sphere_sem >= 0.0 and np.isfinite(s.sphere_mean)


def test_wrap_unit_maps_onto_fundamental_domain() -> None:
    x = np.array([-2.0, -1.2, -1.0, -0.3, 0.0, 0.99, 1.0, 1.2, 1.9])
    w = corrdim.wrap_unit(x)
    assert np.all(w >= -1.0) and np.all(w < 1.0)
    # A full period (2) away maps to the same point; interior points unmoved.
    assert np.allclose(corrdim.wrap_unit(x + 2.0), w)
    assert np.allclose(w[3:6], x[3:6])


def test_load_turnaround_cloud_wrap(tmp_path: Path) -> None:
    p = tmp_path / "d3_tor.csv"
    _write_dump(p, 200, seed=1)
    wrapped = corrdim.load_turnaround_cloud(p, wrap=True)
    assert np.all(wrapped >= -1.0) and np.all(wrapped < 1.0)
