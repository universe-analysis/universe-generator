"""Per-worker fleet leaderboard over one or more campaign stores.

Aggregates completed runs by the worker that ran them (the store's ``host``
column, which for multi-GPU boxes carries the ``alias:N`` token) and formats
a ranked board for the terminal or a Discord embed. Ranking is by total
attempts crunched — the fairest single work proxy across mixed job sizes,
since a T=160 job costs orders of magnitude more attempts than a T=20 one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerStat:
    """Aggregate of one worker's completed jobs across the given stores."""

    host: str
    jobs: int
    attempts: int
    worldlines: int
    last_done: str  # most recent updated_at (UTC, from the store)


def _human_count(n: int) -> str:
    """1234567890 -> '1.23G' (engineering-style thousands units)."""
    for cut, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= cut:
            return f"{n / cut:.2f}{suffix}"
    return str(n)


def gather(db_paths: list[str | Path]) -> list[WorkerStat]:
    """Aggregate completed runs by worker across stores, best first."""
    totals: dict[str, list] = {}  # host -> [jobs, attempts, worldlines, last]
    for path in db_paths:
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute(
                "SELECT host, COUNT(*), COALESCE(SUM(attempts), 0), "
                "COALESCE(SUM(n_final), 0), MAX(updated_at) FROM runs "
                "WHERE status='done' AND host IS NOT NULL GROUP BY host"
            ).fetchall()
        finally:
            conn.close()
        for host, jobs, attempts, worldlines, last in rows:
            t = totals.setdefault(host, [0, 0, 0, ""])
            t[0] += jobs
            t[1] += attempts
            t[2] += worldlines
            t[3] = max(t[3], last or "")
    stats = [
        WorkerStat(host=h, jobs=t[0], attempts=t[1], worldlines=t[2], last_done=t[3])
        for h, t in totals.items()
    ]
    return sorted(stats, key=lambda s: -s.attempts)


def format_board(stats: list[WorkerStat]) -> str:
    """Monospace leaderboard block (Discord- and terminal-friendly)."""
    if not stats:
        return "no completed jobs yet"
    medals = ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8."]
    lines = [f"{'':3} {'worker':<14} {'jobs':>5} {'attempts':>9} {'worldlines':>10}"]
    for i, s in enumerate(stats):
        rank = medals[i] if i < len(medals) else f"{i + 1}."
        lines.append(
            f"{rank:3} {s.host:<14} {s.jobs:>5} {_human_count(s.attempts):>9} "
            f"{_human_count(s.worldlines):>10}"
        )
    return "```\n" + "\n".join(lines) + "\n```"
