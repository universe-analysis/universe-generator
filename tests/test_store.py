"""Tests for the resumable result store."""

from pathlib import Path

from braidlab.config import Job
from braidlab.store import Store


def _job(t: int, seed: int) -> Job:
    return Job(dim=3, band="nyq", t=t, seed=seed, accept_rate=1e-7, max_attempts=3e12)


def test_register_and_pending(tmp_path: Path) -> None:
    store = Store(tmp_path / "x.db")
    jobs = [_job(20, 1), _job(20, 2), _job(40, 1)]
    for j in jobs:
        store.register(j)
    assert len(store.pending(jobs)) == 3
    store.mark(jobs[0], "done", n_final=100, attempts=5_000_000)
    pending = store.pending(jobs)
    assert jobs[0] not in pending and len(pending) == 2


def test_register_idempotent(tmp_path: Path) -> None:
    store = Store(tmp_path / "x.db")
    j = _job(20, 1)
    store.register(j)
    store.mark(j, "done", n_final=42)
    store.register(j)  # must not clobber the completed row
    row = store.get(j)
    assert row is not None and row.status == "done" and row.n_final == 42


def test_terms_variants_are_distinct_rows(tmp_path: Path) -> None:
    """Jobs differing only in terms must not collide in the store."""
    store = Store(tmp_path / "x.db")
    j2 = _job(20, 1)
    j10 = Job(
        dim=3, band="nyq", t=20, seed=1, accept_rate=1e-7, max_attempts=3e12, terms=10
    )
    store.register(j2)
    store.register(j10)
    store.mark(j2, "done", n_final=100)
    assert store.pending([j2, j10]) == [j10]
    row = store.get(j10)
    assert row is not None and row.status == "pending" and row.terms == 10


def test_pre_terms_db_is_migrated(tmp_path: Path) -> None:
    """Opening a pre---terms database adds the terms column (default 2)."""
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE runs (
            dim INTEGER NOT NULL, band TEXT NOT NULL, t INTEGER NOT NULL,
            seed INTEGER NOT NULL, accept_rate REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', n_final INTEGER,
            attempts INTEGER, curve_path TEXT, host TEXT, updated_at TEXT,
            PRIMARY KEY (dim, band, t, seed, accept_rate)
        );
        INSERT INTO runs (dim, band, t, seed, accept_rate, status, n_final)
        VALUES (3, 'nyq', 20, 1, 1e-07, 'done', 77);
        """
    )
    conn.commit()
    conn.close()
    store = Store(path)
    row = store.get(_job(20, 1))
    assert row is not None and row.terms == 2 and row.n_final == 77


def test_results_ordering(tmp_path: Path) -> None:
    store = Store(tmp_path / "x.db")
    for t, s, n in [(40, 1, 9), (20, 2, 5), (20, 1, 4)]:
        j = _job(t, s)
        store.register(j)
        store.mark(j, "done", n_final=n)
    res = store.results(3, "nyq")
    assert [(r.t, r.seed) for r in res] == [(20, 1), (20, 2), (40, 1)]
