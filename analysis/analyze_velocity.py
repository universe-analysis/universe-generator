"""Warm-fraction-vs-T (step 1) and the moving-component speed distribution (step 2).

From the dumped parameters, at z=pi/2:
  - a worldline is "at rest" iff all its frequencies are odd (symmetric about
    the turnaround); "moving" iff it has an even frequency (anti-symmetric,
    swings through). In the legacy model coprimality => at most one even
    frequency, so movers move along a single axis with speed
    |a_even * b_even| = (1 - |a2_even|), the slope-1 budget. Multi-term dumps
    sum every term: speed per axis = |sum_j a_j b_j cos(b_j pi/2 + f_j)|.

Left panel:  moving ("warm") fraction vs T -- does it converge or keep growing?
Right panel: speed distribution of the moving component at the largest T.

Usage::

    python -m analysis.analyze_velocity --params "data/params/params_T*.csv" \
        --out velocity.png
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

from braidlab.corrdim import load_axis_terms

PARAMS = Path("data/params")
HALF_PI = np.pi / 2.0
_T_RE = re.compile(r"_T(\d+)")


def _t_of(path: str) -> int:
    m = _T_RE.search(path)
    if m is None:
        raise ValueError(f"no _T<N> in dump name: {path}")
    return int(m.group(1))


def stats(path: str) -> tuple[int, float, np.ndarray]:
    """Return (N, moving_fraction, moving_speeds) for one packing."""
    axes = load_axis_terms(path)
    moving = np.zeros(len(axes[0].a2), dtype=bool)
    v2 = np.zeros(len(axes[0].a2))
    for ax in axes:
        moving |= (ax.b.astype(np.int64) % 2 == 0).any(axis=1)
        v2 += (ax.a * ax.b * np.cos(ax.b * HALF_PI + ax.f)).sum(axis=1) ** 2
    speed = np.sqrt(v2)
    return len(moving), float(moving.mean()), speed[moving]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        default=str(PARAMS / "params_T*.csv"),
        help="glob of parameter dumps (legacy or multi-term layout)",
    )
    parser.add_argument("--out", type=Path, default=Path("velocity.png"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(args.params), key=_t_of)
    ts, fracs, speed_sets = [], [], {}
    print(f"{'T':>4} {'N':>7} {'moving %':>9}")
    for f in files:
        t = _t_of(f)
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
        ax_s.set_title(
            f"Moving-component speeds, T={tmax} ({len(speed_sets[tmax])} movers)"
        )
        ax_s.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
