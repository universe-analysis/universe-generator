"""Physics extraction at the turnaround z = pi/2 (Path B + velocity check).

From the dumped worldline parameters we reconstruct, analytically at z = pi/2:
  - comoving position X = ax*sin(bx*pi/2) + ax2   (per axis)
  - comoving velocity  dX/dz = ax*bx*cos(bx*pi/2) (per axis)

Then:
  (Path B) pair-correlation g(r) in the collision (Chebyshev) metric, in CELL
           units, checked for collapse across T (universal => real; T-locked =>
           resolution artifact). NOTE: exclusion is enforced over ALL timesteps,
           not just this slice, so the turnaround snapshot may show only soft
           suppression rather than a hard hole at CELL -- that's the thing to see.
  (Path A check) the velocity structure: the odd/even-frequency split, the
           at-rest fraction, and how the coprimality constraint shapes it.

Usage::

    python analyze_turnaround.py --out turnaround.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PARAMS = Path("data/params")
T_VALUES = [18, 32, 60]
HALF_PI = np.pi / 2.0


def load(t: int) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(PARAMS / f"params_T{t}.csv")))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols


def turnaround_state(c: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Positions and velocities at z=pi/2 for all worldlines."""
    pos, vel = {}, {}
    for a, b, a2, key in [
        ("ax", "bx", "ax2", "x"),
        ("ay", "by", "ay2", "y"),
        ("aw", "bw", "aw2", "w"),
    ]:
        pos[key] = c[a] * np.sin(c[b] * HALF_PI) + c[a2]
        vel[key] = c[a] * c[b] * np.cos(c[b] * HALF_PI)
    return {"pos": np.stack([pos["x"], pos["y"], pos["w"]], axis=1),
            "vel": np.stack([vel["x"], vel["y"], vel["w"]], axis=1),
            "freq": np.stack([c["bx"], c["by"], c["bw"]], axis=1)}


def pair_correlation(points: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """g(r) in the Chebyshev metric, with r in units of CELL."""
    tree = cKDTree(points)
    edges_cell = np.linspace(0.1, 6.0, 40)         # r / CELL
    radii = edges_cell * cell
    counts = tree.count_neighbors(tree, radii, p=np.inf).astype(float)
    counts -= len(points)                          # drop self-pairs
    shell = np.diff(counts)                         # pairs in each (r, r+dr) shell
    n = len(points)
    edge = 1.0 - 0.5 * cell
    volume = (2.0 * edge) ** 3
    density = n / volume
    # ideal shell volume for a Chebyshev ball is d((2r)^3) = 24 r^2 dr
    r_mid = 0.5 * (radii[1:] + radii[:-1])
    dr = np.diff(radii)
    ideal = n * density * 24.0 * r_mid**2 * dr
    g = shell / ideal
    return edges_cell[1:] - 0.5 * np.diff(edges_cell)[0], g


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("turnaround.png"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    fig, (ax_g, ax_v) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["tab:blue", "tab:green", "tab:red"]

    for t, color in zip(T_VALUES, colors):
        st = turnaround_state(load(t))
        cell = 2.0 / t
        r_cell, g = pair_correlation(st["pos"], cell)
        ax_g.plot(r_cell, g, "-o", ms=3, color=color, label=f"T={t}")

        # velocity structure
        speed = np.linalg.norm(st["vel"], axis=1)
        even = (st["freq"].astype(int) % 2 == 0)
        n_even = even.sum(axis=1)               # even frequencies per worldline
        at_rest = (n_even == 0)                 # all-odd => v=0 at turnaround
        print(f"T={t}: N={len(speed)}  at-rest frac={at_rest.mean():.3f}  "
              f"mean|v|={speed.mean():.3f}  even-freq/worldline hist="
              f"{[int((n_even==k).sum()) for k in range(4)]}")
        ax_v.hist(speed, bins=40, histtype="step", color=color, density=True,
                  label=f"T={t}")

    ax_g.axhline(1.0, color="gray", ls=":", lw=1)
    ax_g.axvline(1.0, color="black", ls="--", lw=1, label="r = CELL")
    ax_g.set_xlabel("pair separation r / CELL  (Chebyshev)")
    ax_g.set_ylabel("g(r)")
    ax_g.set_title("Pair correlation at z=pi/2 (does it collapse across T?)")
    ax_g.legend()
    ax_g.grid(True, alpha=0.3)

    ax_v.set_xlabel("speed |dX/dz| at z=pi/2")
    ax_v.set_ylabel("density")
    ax_v.set_title("Velocity distribution at the turnaround")
    ax_v.legend()
    ax_v.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
