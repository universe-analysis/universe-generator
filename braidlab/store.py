"""SQLite-backed result store: resumable, one record per job key.

The store is the source of truth for what has been computed. Re-running a
campaign diffs its job list against the store and only dispatches what is
missing, so the orchestrator survives interruption. Kinetic curves are written
as CSV files alongside the database and referenced by path.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from braidlab.config import Job

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    dim          INTEGER NOT NULL,
    band         TEXT    NOT NULL,
    t            INTEGER NOT NULL,
    seed         INTEGER NOT NULL,
    accept_rate  REAL    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',
    n_final      INTEGER,
    attempts     INTEGER,
    curve_path   TEXT,
    host         TEXT,
    updated_at   TEXT,
    PRIMARY KEY (dim, band, t, seed, accept_rate)
);
"""


@dataclass(frozen=True)
class RunResult:
    """A completed (or in-progress) run row."""

    dim: int
    band: str
    t: int
    seed: int
    accept_rate: float
    status: str
    n_final: int | None
    attempts: int | None
    curve_path: str | None
    host: str | None


class Store:
    """Thin SQLite wrapper keyed by job identity."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.curves_dir = self.path.parent / "curves"
        self.curves_dir.mkdir(exist_ok=True)
        self.dumps_dir = self.path.parent / "dumps"
        self.dumps_dir.mkdir(exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def register(self, job: Job) -> None:
        """Insert a job as pending if it is not already present."""
        self._conn.execute(
            "INSERT OR IGNORE INTO runs (dim, band, t, seed, accept_rate, "
            "status, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', "
            "datetime('now'))",
            job.key,
        )
        self._conn.commit()

    def mark(
        self,
        job: Job,
        status: str,
        *,
        n_final: int | None = None,
        attempts: int | None = None,
        curve_path: str | None = None,
        host: str | None = None,
    ) -> None:
        """Update a job's status and (optionally) its results."""
        self._conn.execute(
            "UPDATE runs SET status=?, n_final=COALESCE(?, n_final), "
            "attempts=COALESCE(?, attempts), curve_path=COALESCE(?, curve_path),"
            " host=COALESCE(?, host), updated_at=datetime('now') "
            "WHERE dim=? AND band=? AND t=? AND seed=? AND accept_rate=?",
            (status, n_final, attempts, curve_path, host, *job.key),
        )
        self._conn.commit()

    def get(self, job: Job) -> RunResult | None:
        """Return the stored row for a job, or None."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE dim=? AND band=? AND t=? AND seed=? "
            "AND accept_rate=?",
            job.key,
        ).fetchone()
        return _row_to_result(row) if row else None

    def completed_keys(self) -> set[tuple[int, str, int, int, float]]:
        """Keys of all runs with status 'done'."""
        rows = self._conn.execute(
            "SELECT dim, band, t, seed, accept_rate FROM runs WHERE status='done'"
        ).fetchall()
        return {tuple(r) for r in rows}

    def results(self, dim: int, band: str) -> list[RunResult]:
        """All completed runs for a (dim, band), ordered by T then seed."""
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE dim=? AND band=? AND status='done' "
            "ORDER BY t, seed",
            (dim, band),
        ).fetchall()
        return [_row_to_result(r) for r in rows]

    def pending(self, jobs: list[Job]) -> list[Job]:
        """Subset of `jobs` not yet marked done in the store."""
        done = self.completed_keys()
        return [j for j in jobs if j.key not in done]


def _row_to_result(row: sqlite3.Row) -> RunResult:
    return RunResult(
        dim=row["dim"],
        band=row["band"],
        t=row["t"],
        seed=row["seed"],
        accept_rate=row["accept_rate"],
        status=row["status"],
        n_final=row["n_final"],
        attempts=row["attempts"],
        curve_path=row["curve_path"],
        host=row["host"],
    )
