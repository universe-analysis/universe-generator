"""Cell-fill reference model: the ideal acceptance-decay curve.

This is step 1 of building the D bracket. Before comparing the braid to a 2D
(T**2 cells) and 3D (T**3 cells) reference, we first establish what those
references *are* -- and show they need no simulation at all.

The model: take M cells, pick one uniformly at random, and
accept it only if it has not been picked before. Keep going until every cell
has been picked once. M = T**2 for the square, M = T**3 for the cube.

Closed form
-----------
When n of the M cells are already filled, the chance the next random pick is
new (accepted) is simply the fraction still empty::

    p_accept(n) = (M - n) / M

Treating attempts as continuous, dn/da = (M - n) / M, which integrates to::

    n(a)        = M * (1 - exp(-a / M))      # cells filled after a attempts
    rate(a)     = dn/da = exp(-a / M)        # acceptance rate (a pure decay)
    empty(a)    = 1 - n(a)/M = exp(-a / M)   # fraction still empty

So the acceptance rate is a clean exponential with decay constant 1/M, and the
mean attempts to fill *everything* is the coupon-collector number
M * H_M ~= M * (ln M + 0.5772). No simulation required.

This script prints those numbers and verifies the closed form against a direct
Monte-Carlo run, then plots the two on top of each other so the match is visible.

Usage::

    python coupon_cells.py --timestep 18 --out coupon_cells.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EULER_MASCHERONI = 0.5772156649


def analytic_empty_fraction(attempts: np.ndarray, cells: int) -> np.ndarray:
    """Return the fraction of cells still empty after each attempt count.

    This is also exactly the acceptance probability at that point, since a pick
    is accepted iff it lands on an empty cell.
    """
    return np.exp(-attempts / cells)


def mean_attempts_to_fill(cells: int) -> float:
    """Expected attempts to pick every cell at least once (coupon collector)."""
    harmonic = np.log(cells) + EULER_MASCHERONI + 1.0 / (2.0 * cells)
    return cells * harmonic


def simulate_cell_fill(cells: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the exact process: random pick, accept if unseen, until all seen.

    Returns (attempts, empty_fraction) sampled once per attempt. This is a
    direct numerical check that the closed form above is right.
    """
    rng = np.random.default_rng(seed)
    seen = np.zeros(cells, dtype=bool)
    filled = 0
    attempts: list[int] = []
    empty: list[float] = []
    a = 0
    while filled < cells:
        a += 1
        pick = int(rng.integers(cells))
        if not seen[pick]:
            seen[pick] = True
            filled += 1
        attempts.append(a)
        empty.append((cells - filled) / cells)
    return np.array(attempts), np.array(empty)


def compare(timestep: int, out_path: Path) -> None:
    """Verify the closed form vs simulation for the 2D and 3D cell counts."""
    import matplotlib.pyplot as plt

    cases = [
        ("2D: M = T^2", timestep**2, "tab:blue"),
        ("3D: M = T^3", timestep**3, "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    print(f"\nCell-fill reference at T = {timestep}\n" + "-" * 48)
    for label, cells, color in cases:
        sim_a, sim_empty = simulate_cell_fill(cells, seed=1)
        model_empty = analytic_empty_fraction(sim_a, cells)
        max_abs_err = float(np.max(np.abs(sim_empty - model_empty)))
        print(
            f"{label:<14} M={cells:>8}  "
            f"mean attempts to fill ~= {mean_attempts_to_fill(cells):>12.0f}  "
            f"max|sim-model| = {max_abs_err:.4f}"
        )
        # Thin the simulation points so the marker cloud stays readable.
        step = max(1, len(sim_a) // 400)
        ax.scatter(
            sim_a[::step],
            sim_empty[::step],
            s=10,
            color=color,
            alpha=0.5,
            label=f"{label} (simulation)",
        )
        smooth_a = np.linspace(1, sim_a[-1], 400)
        ax.plot(
            smooth_a,
            analytic_empty_fraction(smooth_a, cells),
            color=color,
            lw=2,
            label=f"{label}  exp(-a/M)",
        )

    ax.set_yscale("log")
    ax.set_xlabel("attempts a")
    ax.set_ylabel("acceptance probability = fraction empty")
    ax.set_title(
        f"Ideal cell-fill decay is exp(-a/M), no simulation needed (T={timestep})"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timestep",
        type=int,
        default=18,
        help="T; the references use T^2 and T^3 cells (default: 18)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("coupon_cells.png"),
        help="output figure path (default: coupon_cells.png)",
    )
    args = parser.parse_args()
    compare(args.timestep, args.out)


if __name__ == "__main__":
    main()
