"""Tests for the subpath group analysis (analyze_subpath_groups)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from analysis.analyze_subpath_groups import load_cells, plot_wz_overlay

HALF_PI = np.pi / 2.0


def write_dump(
    path: Path, gids: list[int], a2s: list[tuple[float, float]] | None = None
) -> None:
    """Write a minimal 2+1 legacy-layout dump with a gid column.

    One wiggle term per axis with fixed (a, b, f); a2 anchors vary per row so
    per-group w(z) curves are distinguishable.
    """
    a2s = a2s if a2s is not None else [(0.1 * (i + 1), 0.0) for i in range(len(gids))]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["ax", "bx", "fx", "ax2", "ay", "by", "fy", "ay2", "gid"])
        for gid, (ax2, ay2) in zip(gids, a2s):
            wr.writerow([0.02, 3, 0.0, ax2, 0.02, 5, 0.0, ay2, gid])


def test_load_cells_group_fields(tmp_path: Path) -> None:
    # seed 1: groups {0: 3 paths, 1: 1 path}; seed 2: {0: 1, 1: 2}
    write_dump(tmp_path / "d2_nyq_T40_s1_sub.csv", [0, 0, 0, 1])
    write_dump(tmp_path / "d2_nyq_T40_s2_sub.csv", [0, 1, 1])
    zgrid = np.linspace(0.1, 3.0, 7)
    (cell,) = load_cells(sorted(tmp_path.glob("*.csv")), zgrid)
    assert cell.dim == 2
    assert cell.n_seeds == 2
    assert cell.group_sizes.tolist() == [3, 1, 1, 2]
    assert cell.group_seed.tolist() == [0, 0, 1, 1]
    assert cell.group_e.shape == (4,)
    assert cell.group_wz.shape == (4, len(zgrid))
    # per-path dictionary: total = sum of group contributions
    assert cell.tot_e == pytest.approx(cell.group_e.sum())
    ens = cell.tot_ev2 / cell.tot_e / 3.0
    from_groups = (cell.group_wz * cell.group_e[:, None]).sum(axis=0) / cell.tot_e
    assert ens == pytest.approx(from_groups)


def test_sterile_aggregate_matches_bin_accumulator(tmp_path: Path) -> None:
    write_dump(tmp_path / "d2_nyq_T40_s1_sub.csv", [0, 0, 1, 2])
    zgrid = np.linspace(0.1, 3.0, 5)
    (cell,) = load_cells(sorted(tmp_path.glob("*.csv")), zgrid)
    sterile = cell.group_sizes == 1
    w_sterile = (cell.group_wz[sterile] * cell.group_e[sterile, None]).sum(
        axis=0
    ) / cell.group_e[sterile].sum()
    w_bin = cell.bin_ev2["0 (sterile)"] / cell.bin_e["0 (sterile)"] / 3.0
    assert w_sterile == pytest.approx(w_bin)


def test_plot_wz_overlay_writes_figure(tmp_path: Path) -> None:
    write_dump(tmp_path / "d2_nyq_T40_s1_sub.csv", [0, 0, 0, 1, 2])
    write_dump(tmp_path / "d2_nyq_T40_s2_sub.csv", [0, 1, 1, 1, 2])
    zgrid = np.linspace(0.1, 3.0, 9)
    cells = load_cells(sorted(tmp_path.glob("*.csv")), zgrid)
    out = tmp_path / "overlay.png"
    plot_wz_overlay(cells, out, mark_z=HALF_PI)
    assert out.stat().st_size > 0


def test_mixed_dimension_pooling_rejected(tmp_path: Path) -> None:
    write_dump(tmp_path / "d2_nyq_T40_s1_sub.csv", [0, 1])
    # a 3+1-layout dump for the same T: add the w-axis quad
    path3 = tmp_path / "d3_nyq_T40_s2_sub.csv"
    with open(path3, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "ax",
                "bx",
                "fx",
                "ax2",
                "ay",
                "by",
                "fy",
                "ay2",
                "aw",
                "bw",
                "fw",
                "aw2",
                "gid",
            ]
        )
        wr.writerow([0.02, 3, 0.0, 0.1, 0.02, 5, 0.0, 0.0, 0.02, 7, 0.0, 0.0, 0])
    with pytest.raises(ValueError, match="axes"):
        load_cells(sorted(tmp_path.glob("*.csv")), np.linspace(0.1, 3.0, 5))


def test_report_heading_uses_dump_dimension(tmp_path: Path, capsys) -> None:
    path3 = tmp_path / "d3_nyq_T40_s1_sub.csv"
    with open(path3, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "ax",
                "bx",
                "fx",
                "ax2",
                "ay",
                "by",
                "fy",
                "ay2",
                "aw",
                "bw",
                "fw",
                "aw2",
                "gid",
            ]
        )
        for gid in (0, 0, 1):
            wr.writerow([0.02, 3, 0.0, 0.2, 0.02, 5, 0.0, 0.1, 0.02, 7, 0.0, 0.3, gid])
    from analysis.analyze_subpath_groups import report

    (cell,) = load_cells([path3], np.linspace(0.1, 3.0, 5))
    assert cell.dim == 3
    report(cell, HALF_PI)
    assert "=== 3+1 T=40" in capsys.readouterr().out
