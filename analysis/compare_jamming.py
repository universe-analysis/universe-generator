"""Compare jamming runs: engine A/B validation against a reference experiment.

Validates an engine change (e.g. the sparse collision grid) by comparing its
jamming kinetics against a baseline. Two comparisons are supported, separately
or together:

  * curve vs curve: two ``attempts,n`` kinetic-curve CSVs (same T / seed /
    flags, different engine build). Reports the final-N relative difference
    and the maximum relative deviation over the overlapping tail of the
    curves. The engine is not bit-deterministic (survivor admission order
    comes from atomicAdd slots), so identical builds differ run-to-run by
    a fraction of a percent; an engine change should sit inside that noise.

  * curve vs stored experiment: a curve's final N against the seed spread of
    a previously collected campaign in a braidlab run.db (same dim / band /
    T / accept-rate). A valid engine change lands inside (or within
    tolerance of) the min-max spread across seeds.

Usage::

    uv run python -m analysis.compare_jamming \\
        --curve-a dense_T60_s1.csv --curve-b sparse_T60_s1.csv \\
        --ref-db data/corrdim/run.db --dim 3 --band nyq --t 60 \\
        --accept-rate 1e-7 --max-rel-diff 0.02
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Fraction of the attempts axis (from the end, log-scale) treated as the
#: comparison tail. The early curve is tiny-N and noisy; the tail is the
#: physics (approach to jamming).
TAIL_DECADE_FRACTION = 0.5


def load_curve(path: str | Path) -> np.ndarray:
    """Load an ``attempts,n`` kinetic curve as a (K, 2) float array.

    Duplicate attempts rows (the engine emits milestone rows in bursts early
    on) are collapsed keeping the last, and the curve is sorted by attempts.
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    # Keep the last row of each attempts value.
    keep = np.append(raw[1:, 0] != raw[:-1, 0], True)
    return raw[keep]


@dataclass(frozen=True)
class CurveComparison:
    """Result of comparing two kinetic curves."""

    final_n_a: int
    final_n_b: int
    final_rel_diff: float
    #: Max |n_b - n_a| / n_a over the shared tail (see TAIL_DECADE_FRACTION).
    tail_max_rel_diff: float


def compare_curves(a: np.ndarray, b: np.ndarray) -> CurveComparison:
    """Compare two kinetic curves over their shared attempts range.

    Curve B's N is interpolated onto curve A's attempt milestones (linear in
    log-attempts; the curves are monotone counting processes) and compared
    over the tail of the shared range.
    """
    final_a, final_b = float(a[-1, 1]), float(b[-1, 1])
    lo = max(a[0, 0], b[0, 0])
    hi = min(a[-1, 0], b[-1, 0])
    if hi <= lo:
        raise ValueError("curves have no overlapping attempts range")
    tail_lo = np.exp(
        np.log(lo) + (1.0 - TAIL_DECADE_FRACTION) * (np.log(hi) - np.log(lo))
    )
    at = a[(a[:, 0] >= tail_lo) & (a[:, 0] <= hi)]
    if len(at) == 0:  # degenerate short curve: fall back to the last point
        at = a[-1:]
    n_b = np.interp(np.log(at[:, 0]), np.log(b[:, 0]), b[:, 1])
    rel = np.abs(n_b - at[:, 1]) / np.maximum(at[:, 1], 1.0)
    return CurveComparison(
        final_n_a=int(final_a),
        final_n_b=int(final_b),
        final_rel_diff=abs(final_b - final_a) / max(final_a, 1.0),
        tail_max_rel_diff=float(rel.max()),
    )


def seed_spread(
    db: str | Path, dim: int, band: str, t: int, accept_rate: float
) -> list[int]:
    """Final-N values across seeds for one stored (dim, band, T, accept_rate).

    Reads a braidlab run.db (the ``runs`` table). Only ``done`` rows count.
    """
    con = sqlite3.connect(f"file:{Path(db)}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT n_final FROM runs WHERE dim=? AND band=? AND t=? AND "
            "accept_rate=? AND status='done' AND n_final IS NOT NULL",
            (dim, band, t, accept_rate),
        ).fetchall()
    finally:
        con.close()
    return sorted(r[0] for r in rows)


def spread_verdict(n: int, spread: list[int], tolerance: float) -> tuple[bool, str]:
    """Judge a final N against a stored seed spread, padded by `tolerance`."""
    lo, hi = min(spread), max(spread)
    ok = lo * (1.0 - tolerance) <= n <= hi * (1.0 + tolerance)
    return ok, (
        f"N={n} vs stored spread [{lo}, {hi}] over {len(spread)} seeds "
        f"(tolerance {tolerance:.1%}): {'OK' if ok else 'OUTSIDE'}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code (0 = comparison passed)."""
    ap = argparse.ArgumentParser(
        description="Compare jamming runs: engine A/B validation."
    )
    ap.add_argument("--curve-a", required=True, help="baseline kinetic curve CSV")
    ap.add_argument("--curve-b", help="candidate kinetic curve CSV")
    ap.add_argument("--label-a", default="A", help="display label for curve A")
    ap.add_argument("--label-b", default="B", help="display label for curve B")
    ap.add_argument("--ref-db", help="braidlab run.db with the stored experiment")
    ap.add_argument("--dim", type=int, default=3, help="spatial dimension (ref-db)")
    ap.add_argument("--band", default="nyq", help="frequency band (ref-db)")
    ap.add_argument("--t", type=int, help="timesteps T (ref-db)")
    ap.add_argument(
        "--accept-rate", type=float, default=1e-7, help="stop threshold (ref-db)"
    )
    ap.add_argument(
        "--max-rel-diff",
        type=float,
        default=0.0,
        help="fail (exit 1) if curve A/B final N differ by more than this "
        "fraction; 0 disables the check",
    )
    ap.add_argument(
        "--spread-tolerance",
        type=float,
        default=0.01,
        help="pad the stored min-max seed spread by this fraction",
    )
    args = ap.parse_args(argv)

    failed = False
    curve_a = load_curve(args.curve_a)

    if args.curve_b:
        cmp_ = compare_curves(curve_a, load_curve(args.curve_b))
        print(
            f"{args.label_a}: final N = {cmp_.final_n_a}   "
            f"{args.label_b}: final N = {cmp_.final_n_b}"
        )
        print(
            f"final rel diff = {cmp_.final_rel_diff:.3%}   "
            f"tail max rel diff = {cmp_.tail_max_rel_diff:.3%}"
        )
        if args.max_rel_diff > 0 and cmp_.final_rel_diff > args.max_rel_diff:
            print(f"FAIL: final rel diff exceeds {args.max_rel_diff:.1%}")
            failed = True

    if args.ref_db:
        if args.t is None:
            ap.error("--ref-db requires --t")
        spread = seed_spread(args.ref_db, args.dim, args.band, args.t, args.accept_rate)
        if not spread:
            print(
                f"no stored runs for dim={args.dim} band={args.band} "
                f"T={args.t} accept_rate={args.accept_rate} in {args.ref_db}"
            )
            failed = True
        else:
            curves = [(args.label_a, curve_a)]
            if args.curve_b:
                curves.append((args.label_b, load_curve(args.curve_b)))
            for label, curve in curves:
                ok, text = spread_verdict(
                    int(curve[-1, 1]), spread, args.spread_tolerance
                )
                print(f"{label}: {text}")
                failed |= not ok

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
