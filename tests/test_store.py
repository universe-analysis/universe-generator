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


def test_results_ordering(tmp_path: Path) -> None:
    store = Store(tmp_path / "x.db")
    for t, s, n in [(40, 1, 9), (20, 2, 5), (20, 1, 4)]:
        j = _job(t, s)
        store.register(j)
        store.mark(j, "done", n_final=n)
    res = store.results(3, "nyq")
    assert [(r.t, r.seed) for r in res] == [(20, 1), (20, 2), (40, 1)]
