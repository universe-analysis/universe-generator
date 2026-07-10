"""Approach-to-jamming (Feder-law) fit: extrapolate the jam ceiling N_inf.

Cutoff states are not jams: N keeps growing ~10-16% per cutoff decade with no
plateau in reach (PHYSICS_FINDINGS section 3), so exponents fitted on cutoff
N values inherit a depth systematic — fatally so in 2+1, where the exponent
climbs through 1e-9 with no sign of a limit.

This tool goes after the limit directly. Continuum RSA approaches its jam as
a power law of attempts t (Feder's law):

    N(t) = N_inf - c * t^(-p)        (d_eff = 1/p)

Each run's kinetic curve (attempts, n CSV referenced by the store) is fitted
on its late tail by scanning p and solving the linear subproblem
(N_inf, c | p) exactly; N_inf per (T, seed) then gives a seed-averaged
extrapolated ceiling per T, and the ladder of ceilings gives the
jamming-limit packing exponent D_inf — the number the cutoff ladders could
not reach.

Usage::

    uv run python -m analysis.analyze_approach_law \
        --db data/converge/converge2d_e8.db --db data/converge/converge2d_e9.db \
        --dim 2 --fit-tmin 100 --out figures/approach_law_2d.png
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np

#: Fit window: the last `TAIL_DECADES` decades of attempts of each curve.
TAIL_DECADES = 2.0
#: Scanned approach exponents p (d_eff = 1/p from ~0.7 to 20).
P_GRID = np.linspace(0.05, 1.5, 146)


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an (attempts, n) kinetic curve, deduplicated and sorted."""
    rows: set[tuple[int, int]] = set()
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.add((int(r["attempts"]), int(r["n"])))
    srt = sorted(rows)
    return np.array([a for a, _ in srt], float), np.array([n for _, n in srt], float)


def fit_feder(
    t: np.ndarray, n: np.ndarray, p_fixed: float | None = None
) -> tuple[float, float, bool]:
    """Fit N(t) = N_inf - c * t^(-p) on the curve tail.

    For each p on a grid the remaining problem is linear least squares; the
    best (N_inf, c, p) by residual wins. With `p_fixed`, only that p is
    used. Returns (n_inf, p, interior): `interior` is False when the best p
    railed at the grid edge, i.e. the tail alone does not constrain the
    approach exponent and the extrapolated ceiling is unreliable.
    """
    tail = t >= t[-1] / 10**TAIL_DECADES
    tt, nn = t[tail], n[tail]
    grid = P_GRID if p_fixed is None else np.array([p_fixed])
    best: tuple[float, float, float] | None = None
    for p in grid:
        x = tt ** (-p)
        a = np.vstack([np.ones_like(x), -x]).T
        (n_inf, c), res, *_ = np.linalg.lstsq(a, nn, rcond=None)
        rms = float(np.sqrt(res[0] / len(nn))) if res.size else 0.0
        if c > 0 and (best is None or rms < best[2]):
            best = (float(n_inf), float(p), rms)
    if best is None:  # every c <= 0: curve still convex-up; no fit
        raise ValueError("no decaying-tail fit found")
    interior = p_fixed is not None or (grid[0] < best[1] < grid[-1])
    return best[0], best[1], interior


def ceilings(
    dbs: list[Path], dim: int, band: str, p_override: float | None = None
) -> dict[int, tuple[float, float, float, int]]:
    """Seed-averaged N_inf and d_eff per T across the given stores.

    Two passes. First every curve is fitted with p free; the runs whose best
    p is interior (the tail genuinely constrains the approach exponent —
    in practice the small-T runs, which sit near true jamming) vote on a
    shared p by median, honoring Feder's structure: p is a property of the
    process, not of T. Every curve is then refitted with that p fixed, so
    the large-T ceilings are extrapolated with a measured exponent instead
    of a railed one.

    Returns {T: (n_inf_mean, n_inf_sem, d_eff_shared, n_runs)}. When several
    stores carry the same (T, seed) at different cutoffs, every run
    contributes — a deeper curve is simply a better-constrained fit of the
    same ceiling.
    """
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    for db in dbs:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "select t, curve_path from runs where status='done' and terms=2 "
            "and dim=? and band=?",
            (dim, band),
        ).fetchall()
        for r in rows:
            t_arr, n_arr = load_curve(Path(r["curve_path"]))
            curves.append((r["t"], t_arr, n_arr))

    if p_override is not None:
        p_shared = p_override
        print(f"approach exponent OVERRIDDEN: p = {p_shared:.4f}")
    else:
        interior_ps: list[float] = []
        for _, t_arr, n_arr in curves:
            try:
                _, p, interior = fit_feder(t_arr, n_arr)
            except ValueError:
                continue
            if interior:
                interior_ps.append(p)
        if not interior_ps:
            raise SystemExit("no curve constrains the approach exponent (all railed)")
        p_shared = float(np.median(interior_ps))
        q1, q3 = np.percentile(interior_ps, [25, 75])
        print(
            f"shared approach exponent: p = {p_shared:.4f} "
            f"(d_eff = {1 / p_shared:.2f}, from {len(interior_ps)} interior fits, "
            f"IQR {q1:.3f}-{q3:.3f})"
        )

    by_t: dict[int, list[float]] = {}
    for t, t_arr, n_arr in curves:
        try:
            n_inf, _, _ = fit_feder(t_arr, n_arr, p_fixed=p_shared)
        except ValueError:
            continue
        by_t.setdefault(t, []).append(n_inf)
    return {
        t: (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
            1.0 / p_shared,
            len(vals),
        )
        for t, vals in sorted(by_t.items())
        if len(vals) > 1
    }


def ladder_exponent(
    ceil: dict[int, tuple[float, float, float, int]], tmin: int
) -> tuple[float, float]:
    """Weighted log-log fit of N_inf over T >= tmin."""
    ts = np.array([t for t in ceil if t >= tmin])
    ns = np.array([ceil[t][0] for t in ts])
    sig = np.array([ceil[t][1] / ceil[t][0] for t in ts])
    p, cov = np.polyfit(np.log(ts), np.log(ns), 1, w=1 / sig, cov=True)
    return float(p[0]), float(np.sqrt(cov[0][0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, action="append", required=True)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--band", default="nyq")
    parser.add_argument(
        "--fit-tmin", type=int, default=100, help="ladder-fit window start"
    )
    parser.add_argument(
        "--p-fixed",
        type=float,
        default=None,
        help="skip the shared-p vote and use this approach exponent "
        "(for bracketing the p systematic)",
    )
    parser.add_argument("--out", type=Path, default=None, help="optional chart")
    args = parser.parse_args()

    ceil = ceilings(args.db, args.dim, args.band, p_override=args.p_fixed)
    print(f"{'T':>5} {'N_inf':>10} {'sem':>7} {'d_eff':>7} {'runs':>5}")
    for t, (n_inf, sem, d_eff, k) in ceil.items():
        print(f"{t:>5} {n_inf:>10.0f} {sem:>7.0f} {d_eff:>7.2f} {k:>5}")
    d, err = ladder_exponent(ceil, args.fit_tmin)
    print(f"jamming-limit exponent (T>={args.fit_tmin}): D_inf = {d:.4f} +/- {err:.4f}")

    if args.out is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = sorted(ceil)
        ns = [ceil[t][0] for t in ts]
        es = [ceil[t][1] for t in ts]
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.errorbar(ts, ns, yerr=es, fmt="o", color="tab:purple", capsize=2)
        fit_t = np.array([min(ts) * 0.85, max(ts) * 1.18])
        anchor = next(t for t in ts if t >= args.fit_tmin)
        ax.plot(
            fit_t,
            ceil[anchor][0] * (fit_t / anchor) ** d,
            "--",
            color="black",
            lw=1.3,
            label=f"N_inf ~ T^{d:.3f}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("T")
        ax.set_ylabel("extrapolated jam ceiling N_inf")
        ax.set_title(f"{args.dim}+1: Feder-law ceilings ({args.band})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=160)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
