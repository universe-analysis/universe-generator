"""Exact invariant checker for subpath (two-phase) parameter dumps.

Validates that a ``--subpaths`` engine run obeyed the phase-2 accretion
semantics, by reconstructing every dumped worldline's comoving trajectory on
the engine z grid and re-testing all pairwise contacts. Used to gate the 3+1
subpath engine port (2026-08-06) against the 2+1 reference semantics; works
on dumps from either engine (dimension inferred from the columns).

Invariants checked (exact, not statistical):

1. gid layout: rows 0..N-1 (the uniques) have gid == row index, every
   subpath's gid refers to a unique (gid < N).
2. Uniques are pairwise non-contacting over all timesteps.
3. Every contact edge in the final packing connects same-gid paths (no
   subpath touches a second group).
4. Every subpath touches at least one member of its own group (no orphans).
5. Each group's contact graph is connected (accretion chains are real).
6. With ``--curve``: the curve's last row has nsub == dumped rows - N and a
   non-decreasing ``filled`` column bounded by T * gw^3. (Skip on subsampled
   dumps -- row counts are censored.)

Contacts are recomputed from %.10g-rounded parameters, so marginal pairs at
the exclusion boundary can flip: violations are only flagged when CERTAIN
(inside cell - eps), and existence checks (4, 5) accept loose contacts
(within cell + eps).

Usage::

    uv run python -m analysis.verify_subpath_dump \\
        --params dump_T20_s1.csv --curve curve_T20_s1.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from analysis.analyze_subpath_groups import _t_of
from braidlab.corrdim import load_axis_terms

#: Boundary tolerance absorbing the %.10g dump rounding (see module docstring).
EPS = 1e-6


def engine_zgrid(t: int) -> np.ndarray:
    """The engine's z grid: T interior points of (0, pi), step pi/(T+1)."""
    step = np.pi / (t + 1)
    return step * np.arange(1, t + 1)


def comoving_positions(path: Path, zgrid: np.ndarray) -> np.ndarray:
    """(n_paths, T, dim) wrapped comoving positions of every dumped path."""
    axes = load_axis_terms(path)
    cols = []
    for ax in axes:
        # x(z) = sum_j a_j (sin(b_j z + f_j) - sin f_j) + a2 sin z, X = x/sin z
        x = np.einsum(
            "nj,njt->nt", ax.a, np.sin(np.multiply.outer(ax.b, zgrid) + ax.f[..., None])
        )
        x -= (ax.a * np.sin(ax.f)).sum(axis=1)[:, None]
        x += ax.a2[:, None] * np.sin(zgrid)[None, :]
        cols.append(x / np.sin(zgrid)[None, :])
    pos = np.stack(cols, axis=-1)
    return pos - 2.0 * np.floor((pos + 1.0) / 2.0)  # wrap onto [-1, 1)


def contact_matrices(
    pos: np.ndarray, cell: float, euclid: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """(certain, loose) boolean any-timestep contact matrices."""
    n, t, _ = pos.shape
    certain = np.zeros((n, n), dtype=bool)
    loose = np.zeros((n, n), dtype=bool)
    for i in range(t):
        d = pos[:, None, i, :] - pos[None, :, i, :]
        d -= 2.0 * np.round(d / 2.0)  # minimum image on the period-2 torus
        r = np.linalg.norm(d, axis=-1) if euclid else np.abs(d).max(axis=-1)
        certain |= r <= cell - EPS
        loose |= r <= cell + EPS
    np.fill_diagonal(certain, False)
    np.fill_diagonal(loose, False)
    return certain, loose


def group_connected(loose: np.ndarray, members: np.ndarray) -> bool:
    """Is the loose-contact graph over ``members`` connected? (BFS)"""
    if len(members) <= 1:
        return True
    sub = loose[np.ix_(members, members)]
    seen = np.zeros(len(members), dtype=bool)
    seen[0] = True
    frontier = np.array([0])
    while frontier.size:
        nxt = sub[frontier].any(axis=0) & ~seen
        seen |= nxt
        frontier = np.flatnonzero(nxt)
    return bool(seen.all())


def verify(params: Path, curve: Path | None = None, euclid: bool = False) -> list[str]:
    """Run all invariants; returns a list of violation messages (empty = pass)."""
    t = _t_of(params)
    cell = 2.0 / t
    with open(params) as f:
        gid = np.array([int(r["gid"]) for r in csv.DictReader(f)])
    n_rows = len(gid)
    errors: list[str] = []

    # 1. gid layout: the unique prefix, then subpaths referring into it.
    is_unique = gid == np.arange(n_rows)
    n_unique = int(is_unique.sum())
    if not is_unique[:n_unique].all() or is_unique[n_unique:].any():
        errors.append("gid layout: uniques (gid == row) are not a prefix")
    if (gid >= n_unique).any():
        errors.append(f"gid range: max gid {gid.max()} >= N={n_unique}")

    pos = comoving_positions(params, engine_zgrid(t))
    certain, loose = contact_matrices(pos, cell, euclid=euclid)

    # 2. Uniques pairwise non-contacting.
    uu = certain[:n_unique, :n_unique]
    if uu.any():
        i, j = np.argwhere(uu)[0]
        errors.append(f"unique-unique contact: rows {i} and {j}")

    # 3. Every certain contact edge is same-gid.
    for i, j in np.argwhere(np.triu(certain, k=1)):
        if gid[i] != gid[j]:
            errors.append(
                f"cross-group contact: row {i} (gid {gid[i]}) vs row {j} (gid {gid[j]})"
            )
            break

    # 4. Every subpath touches its own group (loosely).
    for r in np.flatnonzero(~is_unique):
        mates = np.flatnonzero(gid == gid[r])
        if not loose[r, mates[mates != r]].any():
            errors.append(
                f"orphan subpath: row {r} (gid {gid[r]}) touches no groupmate"
            )
            break

    # 5. Group contact graphs connected.
    for g in np.unique(gid):
        members = np.flatnonzero(gid == g)
        if not group_connected(loose, members):
            errors.append(
                f"group {g}: contact graph disconnected ({len(members)} members)"
            )
            break

    # 6. Curve consistency (complete dumps only).
    if curve is not None:
        rows = np.loadtxt(curve, delimiter=",", skiprows=1, ndmin=2)
        if rows.shape[1] != 4:
            errors.append(
                f"curve {curve.name}: expected 4 columns, got {rows.shape[1]}"
            )
        else:
            nsub = int(rows[-1, 2])
            if n_unique + nsub != n_rows:
                errors.append(
                    f"curve/dump mismatch: N={n_unique} + nsub={nsub} != rows={n_rows}"
                )
            filled = rows[:, 3]
            if (np.diff(filled) < 0).any():
                errors.append("curve: filled column decreases")
            if filled[-1] > t * float(t) ** pos.shape[2]:
                errors.append("curve: filled exceeds T * gw^d cells")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True, help="subpath dump (gid)")
    parser.add_argument("--curve", type=Path, default=None, help="4-column curve CSV")
    parser.add_argument("--euclid", action="store_true", help="L2 exclusion rule")
    args = parser.parse_args()
    errors = verify(args.params, curve=args.curve, euclid=args.euclid)
    with open(args.params) as f:
        n = sum(1 for _ in f) - 1
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        sys.exit(1)
    print(f"OK: {args.params.name} ({n} rows) passed all subpath invariants")


if __name__ == "__main__":
    main()
