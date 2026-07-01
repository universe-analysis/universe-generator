"""Warm-fraction-vs-T (step 1) and the moving-component speed distribution (step 2).

From the dumped parameters, at z=pi/2:
  - a worldline is "at rest" iff all three frequencies are odd (symmetric about
    the turnaround); "moving" iff it has an even frequency (anti-symmetric, swings
    through). Coprimality => at most one even, so movers move along a single axis.
  - speed |dX/dz| = |a_even * b_even| = (1 - |a2_even|), the slope-1 budget.

Left panel:  moving ("warm") fraction vs T -- does it converge or keep growing?
Right panel: speed distribution of the moving component at the largest T.

Usage::

    python analyze_velocity.py --out velocity.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np

PARAMS = Path("data/params")
HALF_PI = np.pi / 2.0


def load(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def stats(path: str) -> tuple[int, float, np.ndarray]:
    """Return (N, moving_fraction, moving_speeds) for one packing."""
    c = load(path)
    freq = np.stack([c["bx"], c["by"], c["bw"]], axis=1).astype(int)
    moving = (freq % 2 == 0).any(axis=1)
    vx = c["ax"] * c["bx"] * np.cos(c["bx"] * HALF_PI)
    vy = c["ay"] * c["by"] * np.cos(c["by"] * HALF_PI)
    vw = c["aw"] * c["bw"] * np.cos(c["bw"] * HALF_PI)
    speed = np.sqrt(vx**2 + vy**2 + vw**2)
    return len(freq), float(moving.mean()), speed[moving]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("velocity.png"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(str(PARAMS / "params_T*.csv")),
                   key=lambda p: int(re.search(r"_T(\d+)", p).group(1)))
    ts, fracs, speed_sets = [], [], {}
    print(f"{'T':>4} {'N':>7} {'moving %':>9}")
    for f in files:
        t = int(re.search(r"_T(\d+)", f).group(1))
        n, frac, speeds = stats(f)
        ts.append(t)
        fracs.append(frac)
        speed_sets[t] = speeds
        print(f"{t:>4} {n:>7} {frac:>8.1%}")

    fig, (ax_f, ax_s) = plt.subplots(1, 2, figsize=(13, 5))

    ax_f.plot(ts, np.array(fracs) * 100, "-o", color="tab:red")
    ax_f.set_xlabel("T (timesteps = resolution)")
    ax_f.set_ylabel("moving / 'warm' fraction (%)")
    ax_f.set_title("Warm fraction vs resolution (does it converge?)")
    ax_f.grid(True, alpha=0.3)

    # Speed distribution of the moving component at the largest T that has movers.
    movers_t = [t for t in ts if len(speed_sets[t]) > 20]
    if movers_t:
        tmax = movers_t[-1]
        ax_s.hist(speed_sets[tmax], bins=40, color="tab:purple", alpha=0.8)
        ax_s.set_xlabel("speed |dX/dz| at z=pi/2  (slope-1 capped at 1)")
        ax_s.set_ylabel("count")
        ax_s.set_title(f"Moving-component speeds, T={tmax} "
                       f"({len(speed_sets[tmax])} movers)")
        ax_s.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
