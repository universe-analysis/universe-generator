"""Tests for the per-worker fleet leaderboard."""

from pathlib import Path

from braidlab.config import Job
from braidlab.leaderboard import format_board, gather
from braidlab.store import Store


def _mark(store: Store, t: int, seed: int, host: str, n: int, attempts: int) -> None:
    job = Job(dim=3, band="nyq", t=t, seed=seed, accept_rate=1e-6, max_attempts=3e12)
    store.register(job)
    store.mark(job, "done", n_final=n, attempts=attempts, host=host)


def test_gather_ranks_by_attempts_across_stores(tmp_path: Path) -> None:
    a = Store(tmp_path / "a.db")
    b = Store(tmp_path / "b.db")
    _mark(a, 20, 1, "mother", n=100, attempts=1_000)
    _mark(a, 40, 1, "vast:1", n=200, attempts=5_000)
    _mark(b, 60, 1, "vast:1", n=300, attempts=7_000)
    _mark(b, 80, 1, "kitt", n=50, attempts=2_000)
    stats = gather([a.path, b.path])
    assert [s.host for s in stats] == ["vast:1", "kitt", "mother"]
    top = stats[0]
    assert top.jobs == 2 and top.attempts == 12_000 and top.worldlines == 500


def test_gather_skips_pending_rows(tmp_path: Path) -> None:
    store = Store(tmp_path / "a.db")
    job = Job(dim=3, band="nyq", t=20, seed=1, accept_rate=1e-6, max_attempts=3e12)
    store.register(job)  # pending, no host
    assert gather([store.path]) == []


def test_format_board_lists_workers_in_order(tmp_path: Path) -> None:
    store = Store(tmp_path / "a.db")
    _mark(store, 20, 1, "mother", n=100, attempts=2_000_000_000)
    _mark(store, 40, 1, "kitt", n=100, attempts=1_000)
    board = format_board(gather([store.path]))
    assert board.index("mother") < board.index("kitt")
    assert "2.00G" in board  # humanized attempts
    assert format_board([]) == "no completed jobs yet"
