"""Step 3: the braid's jam count vs T, bracketed by the T**2 and T**3 rails.

Steps 1-2 showed, at a single timestep, that the braid jams at a count between
the 2D (T**2) and 3D (T**3) cell rails. Here we do it for every T at once.

Plotting jam count vs T on log-log axes turns each rail into a straight line:
T**2 has slope 2, T**3 has slope 3. The braid's jam counts form a line of
slope D running between them -- so D is read off as a single fitted slope,
which is Chris's "T^2.X" made into one number instead of one point per T.

Jam count per T is the mean final n across seeds; the error bar is the seed
spread. We also print the slope between each consecutive pair of T so the
high-T sag (runs that stopped short of true jamming) is visible.

Usage::

    python multi_t_bracket.py --curves data/curves --out multi_t_bracket.png
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def final_counts_by_t(curves_dir: Path) -> dict[int, list[float]]:
    """Return {T: [final n for each seed]} from the run curves."""
    by_t: dict[int, list[float]] = defaultdict(list)
    for path in sorted(glob.glob(str(curves_dir / "d3_nyq_T*_s*.csv"))):
        match = re.search(r"_T(\d+)_s", path)
        if not match:
            continue
        timestep = int(match.group(1))
        last_n: float | None = None
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("attempts"):
                continue
            _, n_str = line.split(",")
            last_n = float(n_str)
        if last_n is not None:
            by_t[timestep].append(last_n)
    return dict(by_t)


def fit_slope(ts: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept of log(count) vs log(T)."""
    slope, intercept = np.polyfit(np.log(ts), np.log(counts), 1)
    return float(slope), float(intercept)


def bracket(curves_dir: Path, out_path: Path) -> None:
    """Plot jam count vs T with the T**2 and T**3 rails and the fitted D."""
    import matplotlib.pyplot as plt

    by_t = final_counts_by_t(curves_dir)
    ts = np.array(sorted(by_t))
    means = np.array([np.mean(by_t[t]) for t in ts])
    stds = np.array([np.std(by_t[t]) for t in ts])

    slope, intercept = fit_slope(ts, means)
    print("\nMulti-T bracket\n" + "-" * 48)
    print(f"{'T':>4} {'n_jam':>10} {'T^2':>8} {'T^3':>9} {'lnN/lnT':>8}")
    for t, m in zip(ts, means):
        print(f"{t:>4} {m:>10.1f} {t**2:>8} {t**3:>9} {np.log(m) / np.log(t):>8.3f}")
    print(f"\nfitted slope across all T:  D = {slope:.3f}")
    print("pairwise slopes (watch the high-T sag):")
    log_t = np.log(ts)
    log_n = np.log(means)
    for i in range(len(ts) - 1):
        d = (log_n[i + 1] - log_n[i]) / (log_t[i + 1] - log_t[i])
        print(f"  T{ts[i]} -> T{ts[i + 1]}:  {d:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 6))

    rail_t = np.linspace(ts[0] * 0.9, ts[-1] * 1.1, 100)
    ax.plot(rail_t, rail_t**2, color="tab:blue", lw=2, label="2D rail: T^2 (slope 2)")
    ax.plot(rail_t, rail_t**3, color="tab:red", lw=2, label="3D rail: T^3 (slope 3)")

    fit_line = np.exp(intercept) * rail_t**slope
    ax.plot(
        rail_t,
        fit_line,
        color="green",
        ls="--",
        lw=2,
        label=f"braid fit: T^{slope:.2f}",
    )
    ax.errorbar(
        ts,
        means,
        yerr=stds,
        fmt="o",
        color="black",
        ms=7,
        capsize=4,
        label="braid jam count (mean over seeds)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T (timesteps = resolution)")
    ax.set_ylabel("n = worldlines packed at jamming")
    ax.set_title(f"Braid packs as T^{slope:.2f}, between the T^2 and T^3 rails")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=Path("data/curves"))
    parser.add_argument("--out", type=Path, default=Path("multi_t_bracket.png"))
    args = parser.parse_args()
    bracket(args.curves, args.out)


if __name__ == "__main__":
    main()
