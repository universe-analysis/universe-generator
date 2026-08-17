"""Ricci scalar of the sin-scale-factor cosmos under two time-coordinate readings.

Chris's derivation (2026-08-17): for a flat 3-torus with a(t) = sin t in COSMIC
(proper) time, R = 6/a^2 - 12, which changes sign at t = pi/4 and 3pi/4 -- the
middle half of the history has negative scalar curvature. His comparison closed
universe, a 3-sphere with a(t) = 2 sin(t/2), gives R = 12/a^2 - 3 >= 0.

Both formulas are correct, but the model's ground truth (PHYSICS_FINDINGS.md)
defines z in (0, pi) as CONFORMAL time with a(z) = sin z, and under that reading
R = -6/a^2: negative for the whole history, no sign change. The two readings are
different spacetimes, not different coordinates on one -- the cosmic-time sin
universe has infinite conformal time and cannot carry the closed z-loop.

One panel per dimension (3+1 and 2+1), three curves each, x = loop fraction:

  * conformal reading (the model): a(z) = sin z, k = 0;
  * cosmic reading (Chris):        a(t) = sin t, k = 0, closed form
    R = d(d-1)/a^2 - d(d+1);
  * closed-sphere comparison:      a(t) = 2 sin(t/2), k = +1, closed form
    R = 2d(d-1)/a^2 - d(d+1)/4.

The convention-independent fact both panels show: at the turnaround H = 0, so
any FLAT universe that turns around has R = 2d * (addot/a) < 0 there (both
torus readings meet at R = -2d), while the closed sphere in 3+1 stays >= 0.

Usage::

    uv run python -m plots.plot_ricci_readings --out ricci_readings.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def ricci_cosmic(
    d: int, k: int, a: np.ndarray, adot: np.ndarray, addot: np.ndarray
) -> np.ndarray:
    """Ricci scalar of a (d+1)-dimensional FRW metric, cosmic-time derivatives.

    R = 2d (addot/a) + d(d-1) [(adot/a)^2 + k/a^2], signature (-,+,...,+).
    """
    return 2 * d * addot / a + d * (d - 1) * ((adot / a) ** 2 + k / a**2)


def curves(d: int, n: int = 2001) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The three R(loop fraction) curves for spatial dimension d."""
    eps = 1e-3
    frac = np.linspace(eps, 1 - eps, n)

    # Conformal reading (the model): a(z) = sin z, z in (0, pi), k = 0.
    # Convert conformal derivatives a' = cos z, a'' = -sin z to cosmic ones.
    z = np.pi * frac
    a = np.sin(z)
    ap = np.cos(z)
    app = -np.sin(z)
    adot = ap / a
    addot = app / a**2 - ap**2 / a**3
    r_conformal = ricci_cosmic(d, 0, a, adot, addot)

    # Cosmic reading (Chris): a(t) = sin t, t in (0, pi), k = 0.
    t = np.pi * frac
    a = np.sin(t)
    r_cosmic = ricci_cosmic(d, 0, a, np.cos(t), -np.sin(t))

    # Closed-sphere comparison: a(t) = 2 sin(t/2), t in (0, 2 pi), k = +1.
    t = 2 * np.pi * frac
    a = 2 * np.sin(t / 2)
    r_sphere = ricci_cosmic(d, 1, a, np.cos(t / 2), -np.sin(t / 2) / 2)

    return {
        "conformal": (frac, r_conformal),
        "cosmic": (frac, r_cosmic),
        "sphere": (frac, r_sphere),
    }


def plot(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    ylim = (-30.0, 20.0)

    for ax, d in zip(axes, (3, 2)):
        c = curves(d)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0.5, color="gray", lw=0.8, ls=":", alpha=0.7)

        frac, r = c["cosmic"]
        ax.plot(
            frac,
            r,
            color="crimson",
            lw=2.0,
            label=(
                f"cosmic reading: a(t) = sin t, k=0   "
                f"R = {d * (d - 1)}/a² − {d * (d + 1)}"
            ),
        )
        frac, r = c["conformal"]
        ax.plot(
            frac,
            r,
            color="navy",
            lw=2.0,
            label="conformal reading (the model): a(z) = sin z, k=0   "
            + ("R = −6/a²" if d == 3 else "R = −4/a² − 2cos²z/a⁴"),
        )
        frac, r = c["sphere"]
        ax.plot(
            frac,
            r,
            color="seagreen",
            lw=1.6,
            ls="--",
            label=(
                f"closed sphere: a(t) = 2 sin(t/2), k=1   "
                f"R = {2 * d * (d - 1)}/a² − {d * (d + 1) / 4:g}"
            ),
        )

        # Sign-change window of the cosmic reading: R < 0 for a^2 > (d-1)/(d+1).
        lo = np.arcsin(np.sqrt((d - 1) / (d + 1))) / np.pi
        ax.axvspan(lo, 1 - lo, color="crimson", alpha=0.07)
        for x in (lo, 1 - lo):
            ax.axvline(x, color="crimson", lw=0.8, ls=":", alpha=0.8)
        ax.annotate(
            f"R < 0 for t/π ∈ ({lo:.3g}, {1 - lo:.3g})",
            (0.5, ylim[0] + 2.5),
            ha="center",
            fontsize=9,
            color="crimson",
        )

        # Both flat readings meet at the turnaround: H = 0 forces R = -2d there.
        ax.plot([0.5], [-2 * d], "ko", ms=5, zorder=5)
        ax.annotate(
            f"turnaround: R = −{2 * d}\n(both flat readings)",
            (0.5, -2 * d),
            textcoords="offset points",
            xytext=(0, -58),
            ha="center",
            fontsize=9,
        )

        ax.set_ylim(*ylim)
        ax.set_xlabel("loop fraction (t or z over its full span)")
        ax.set_title(f"{d}+1")
        ax.legend(loc="upper center", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Ricci scalar R")
    fig.suptitle(
        "Ricci scalar of the sin-scale-factor cosmos: "
        "cosmic-time reading changes sign, the model's conformal reading never does"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)

    for d in (3, 2):
        lo = np.arcsin(np.sqrt((d - 1) / (d + 1))) / np.pi
        print(
            f"{d}+1: cosmic R = {d * (d - 1)}/a^2 - {d * (d + 1)}, "
            f"sign change at t/pi = {lo:.4f} and {1 - lo:.4f}; "
            f"conformal R < 0 always; turnaround R = -{2 * d}"
        )
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ricci_readings.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
