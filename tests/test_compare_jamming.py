"""Tests for the engine A/B jamming-comparison tool."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from analysis.compare_jamming import (
    compare_curves,
    load_curve,
    main,
    seed_spread,
    spread_verdict,
)


def _write_curve(path: Path, rows: list[tuple[int, int]]) -> Path:
    path.write_text("attempts,n\n" + "\n".join(f"{a},{n}" for a, n in rows) + "\n")
    return path


def _saturating_rows(n_final: int) -> list[tuple[int, int]]:
    """A jamming-shaped curve: N rises then saturates at n_final."""
    attempts = np.logspace(3, 9, 40)
    n = n_final * (1.0 - np.exp(-attempts / 1e6))
    return [(int(a), int(v)) for a, v in zip(attempts, n)]


def test_load_curve_dedupes_and_sorts(tmp_path: Path) -> None:
    p = _write_curve(tmp_path / "c.csv", [(1000, 5), (1000, 7), (500, 2)])
    curve = load_curve(p)
    assert curve.tolist() == [[500, 2], [1000, 7]]  # sorted, last dup kept


def test_compare_curves_identical(tmp_path: Path) -> None:
    rows = _saturating_rows(5000)
    a = load_curve(_write_curve(tmp_path / "a.csv", rows))
    cmp_ = compare_curves(a, a)
    assert cmp_.final_rel_diff == 0.0
    assert cmp_.tail_max_rel_diff == 0.0


def test_compare_curves_offset(tmp_path: Path) -> None:
    a = load_curve(_write_curve(tmp_path / "a.csv", _saturating_rows(5000)))
    b = load_curve(_write_curve(tmp_path / "b.csv", _saturating_rows(5050)))
    cmp_ = compare_curves(a, b)
    assert abs(cmp_.final_n_a - 5000) <= 1
    assert 0.005 < cmp_.final_rel_diff < 0.015
    assert cmp_.tail_max_rel_diff < 0.02


def test_compare_curves_disjoint_ranges_raise(tmp_path: Path) -> None:
    a = load_curve(_write_curve(tmp_path / "a.csv", [(100, 1), (200, 2)]))
    b = load_curve(_write_curve(tmp_path / "b.csv", [(300, 1), (400, 2)]))
    with pytest.raises(ValueError):
        compare_curves(a, b)


def _make_ref_db(path: Path, n_values: list[int]) -> Path:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE runs (dim INTEGER, band TEXT, t INTEGER, seed INTEGER,"
        " accept_rate REAL, status TEXT, n_final INTEGER, attempts INTEGER,"
        " curve_path TEXT, host TEXT, updated_at TEXT)"
    )
    for seed, n in enumerate(n_values, start=1):
        con.execute(
            "INSERT INTO runs VALUES (3, 'nyq', 60, ?, 1e-7, 'done', ?,"
            " 1000000, NULL, 'h', NULL)",
            (seed, n),
        )
    con.commit()
    con.close()
    return path


def test_seed_spread_reads_done_rows(tmp_path: Path) -> None:
    db = _make_ref_db(tmp_path / "run.db", [17060, 16929, 17223])
    assert seed_spread(db, 3, "nyq", 60, 1e-7) == [16929, 17060, 17223]
    assert seed_spread(db, 3, "nyq", 80, 1e-7) == []


def test_spread_verdict_tolerance() -> None:
    ok, _ = spread_verdict(17000, [16929, 17223], tolerance=0.0)
    assert ok
    ok, _ = spread_verdict(16800, [16929, 17223], tolerance=0.0)
    assert not ok
    ok, _ = spread_verdict(16800, [16929, 17223], tolerance=0.01)
    assert ok  # 16929 * 0.99 < 16800


def test_main_pass_and_fail(tmp_path: Path) -> None:
    a = _write_curve(tmp_path / "a.csv", _saturating_rows(17000))
    b = _write_curve(tmp_path / "b.csv", _saturating_rows(17100))
    db = _make_ref_db(tmp_path / "run.db", [16929, 17060, 17223])
    common = ["--curve-a", str(a), "--curve-b", str(b)]
    ref = [
        "--ref-db",
        str(db),
        "--dim",
        "3",
        "--band",
        "nyq",
        "--t",
        "60",
        "--accept-rate",
        "1e-7",
    ]
    assert main(common + ref + ["--max-rel-diff", "0.02"]) == 0
    # A tight final-N budget fails the ~0.6% offset between the curves.
    assert main(common + ["--max-rel-diff", "0.001"]) == 1
    # A curve far outside the stored seed spread fails.
    far = _write_curve(tmp_path / "far.csv", _saturating_rows(20000))
    assert main(["--curve-a", str(far)] + ref) == 1


def test_sparse_job_name_and_command() -> None:
    """Sparse jobs get the _sp suffix (workspace collision guard) and --sparse."""
    from braidlab.config import Job
    from braidlab.engine import build_command

    j = Job(
        dim=3,
        band="nyq",
        t=60,
        seed=1,
        accept_rate=1e-7,
        max_attempts=3e12,
        sparse=True,
    )
    assert j.name == "d3_nyq_T60_s1_ph_sp"
    cmd = build_command(j, "braid_cuda3d", "curves/x.csv")
    assert "--sparse" in cmd
