"""Why step 2 switched axes: braid vs cells on the *attempts* axis.

Step 1 plotted the cell rails against attempts; step 2 plotted the braid
against count. This script puts all three on the attempts axis at once to show
the honest reason for the switch: the cells saturate in ~1e4 attempts while the
braid needs ~1e9, so on this axis they are ~5 decades apart and cannot be
compared. The count axis (step 2) is what brings them onto the same chart.

Usage::

    python attempts_overlay.py --timestep 18 --curves data/curves --out attempts_overlay.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from plots.braid_overlay import braid_rate_points


def rate_vs_attempts(paths: list[str]) -> tuple[list[float], list[float]]:
    """Braid acceptance rate dn/d(attempts) keyed on the attempt of each step."""
    attempts: list[float] = []
    rates: list[float] = []
    for path in paths:
        rows: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("attempts"):
                continue
            a_str, n_str = line.split(",")
            pair = (int(a_str), int(n_str))
            if pair not in seen:
                seen.add(pair)
                rows.append(pair)
        rows.sort()
        for (a0, n0), (a1, n1) in zip(rows, rows[1:]):
            if a1 == a0:
                continue
            attempts.append(0.5 * (a0 + a1))
            rates.append((n1 - n0) / (a1 - a0))
    return attempts, rates


def overlay(
    timestep: int,
    braid_glob: str,
    compare_glob: str | None,
    braid_label: str,
    compare_label: str,
    out_path: Path,
    logx: bool,
) -> None:
    """Plot braid (and optional comparison braid) vs cell decays on attempts."""
    import matplotlib.pyplot as plt

    paths = sorted(glob.glob(braid_glob))
    if not paths:
        raise SystemExit(f"no curves matched {braid_glob}")
    attempts, rates = rate_vs_attempts(paths)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x_max = max(attempts)
    for power, color, label in [(2, "tab:blue", "2D cell"), (3, "tab:red", "3D cell")]:
        cells = timestep**power
        a = np.logspace(1, np.log10(x_max), 600) if logx else np.linspace(1, x_max, 2000)
        ax.plot(a, np.exp(-a / cells), color=color, lw=2, label=f"{label} exp(-a/M), M=T^{power}")

    # Optional comparison braid (e.g. uniform), drawn faint behind the main one.
    if compare_glob:
        cpaths = sorted(glob.glob(compare_glob))
        if cpaths:
            ca, cr = rate_vs_attempts(cpaths)
            ax.scatter(ca, cr, s=14, color="tab:gray", alpha=0.3, label=compare_label)

    ax.scatter(attempts, rates, s=16, color="black", alpha=0.55, label=braid_label)

    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 2)
    ax.set_xlabel("attempts a")
    ax.set_ylabel("acceptance probability of next attempt")
    ax.set_title(f"Braid vs cell decays on the attempts axis (T={timestep})")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"braid attempts span: {min(attempts):.3g} .. {max(attempts):.3g}")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestep", type=int, default=18)
    parser.add_argument(
        "--braid-glob",
        default="data/curves/d3_nyq_T18_s*.csv",
        help="glob for the primary braid curves",
    )
    parser.add_argument("--braid-label", default="braid (measured)")
    parser.add_argument(
        "--compare-glob", default=None, help="optional second braid series (drawn faint)"
    )
    parser.add_argument("--compare-label", default="comparison")
    parser.add_argument("--out", type=Path, default=Path("attempts_overlay.png"))
    parser.add_argument(
        "--logx", action="store_true", help="use log x-axis (recommended)"
    )
    args = parser.parse_args()
    overlay(
        args.timestep,
        args.braid_glob,
        args.compare_glob,
        args.braid_label,
        args.compare_label,
        args.out,
        args.logx,
    )


if __name__ == "__main__":
    main()
