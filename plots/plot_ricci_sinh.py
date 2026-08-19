"""Ricci scalar of the sin cosmos continued to an imaginary scale factor.

Chris's follow-up (2026-08-18) to the 2026-08-17 Ricci computation: repeat the
calculation with the turnaround scale factor 1 replaced by i = sqrt(-1). In the
family a(t) = A sin(t/A) (turnaround at t = A pi/2 with a = A), setting A = i
gives i sin(t/i) = sinh t -- a REAL, forever-expanding scale factor whose formal
turnaround sits at imaginary time t = i pi/2, where a = sinh(i pi/2) = i. So
"scale factor i" is exactly the Wick partner of last time's universe.

The closed forms just flip the constant term, R = d(d-1)/a^2 - d(d+1)/A^2;
concretely (flat k = 0, cosmic time):

    A = 1:  R = 6/a^2 - 12   (3+1)    2/a^2 - 6   (2+1)   -- recollapses
    A = i:  R = 6/a^2 + 12   (3+1)    2/a^2 + 6   (2+1)   -- inflates forever

i.e. the -12 was a NEGATIVE cosmological constant (R_Lambda = 4 Lambda with
Lambda = -3/A^2): sin t is the flat universe with Lambda = -3 plus a curvature-
like w = -1/3 fluid, and the i-continuation keeps the fluid while flipping
Lambda to +3, so R -> 12 = the de Sitter value and never changes sign. At the
complex turnaround a = i the scalar is R = -6/A^2 = +6 (it was -6).

One panel per dimension, three curves, mirroring plot_ricci_readings:

  * cosmic reading:    a(t) = sinh t, k = 0;
  * conformal reading: a(z) = sinh z, k = 0 -- R = +6/a^2 in 3+1 (the exact
    sign flip of the model's -6/a^2); in 2+1, R = 2(sinh^2 z - 1)/a^4, which
    still changes sign once, at z = ln(1 + sqrt 2);
  * open-sphere comparison: the k = +1 sphere a = 2 sin(t/2) continues to
    k = -1, a = 2 sinh(t/2) -- which is exactly de Sitter space in the open
    chart: R = d(d+1)/A^2 = 3 (3+1) / 1.5 (2+1), constant.

Usage::

    uv run python -m plots.plot_ricci_sinh --out ricci_sinh.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plots.plot_ricci_readings import ricci_cosmic


def curves(d: int, n: int = 2001) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The three R(t/pi) curves for spatial dimension d, over the old window."""
    eps = 1e-3
    frac = np.linspace(eps, 1.0, n)

    # Cosmic reading: a(t) = sinh t, k = 0 (the A = i member of A sin(t/A)).
    t = np.pi * frac
    a = np.sinh(t)
    r_cosmic = ricci_cosmic(d, 0, a, np.cosh(t), np.sinh(t))

    # Conformal reading: a(z) = sinh z, k = 0.
    # Convert conformal derivatives a' = cosh z, a'' = sinh z to cosmic ones.
    z = np.pi * frac
    a = np.sinh(z)
    ap = np.cosh(z)
    app = np.sinh(z)
    adot = ap / a
    addot = app / a**2 - ap**2 / a**3
    r_conformal = ricci_cosmic(d, 0, a, adot, addot)

    # Open-sphere comparison: a(t) = 2 sinh(t/2), k = -1 (de Sitter, open chart).
    t = 2 * np.pi * frac
    a = 2 * np.sinh(t / 2)
    r_sphere = ricci_cosmic(d, -1, a, np.cosh(t / 2), np.sinh(t / 2) / 2)

    return {
        "cosmic": (frac, r_cosmic),
        "conformal": (frac, r_conformal),
        "sphere": (frac, r_sphere),
    }


def plot(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    ylim = (-12.0, 30.0)

    for ax, d in zip(axes, (3, 2)):
        c = curves(d)
        ax.axhline(0, color="gray", lw=0.8)

        # The cosmic reading's late-time de Sitter asymptote (H -> 1, so
        # R -> d(d+1) and Lambda = d(d-1)/2).
        de_sitter = d * (d + 1)
        ax.axhline(de_sitter, color="crimson", lw=0.8, ls=":", alpha=0.8)
        ax.annotate(
            f"R → {de_sitter} (de Sitter, Λ = +{d * (d - 1) // 2})",
            (0.99, de_sitter),
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="crimson",
        )

        frac, r = c["cosmic"]
        ax.plot(
            frac,
            r,
            color="crimson",
            lw=2.0,
            label=(
                f"cosmic reading: a(t) = sinh t, k=0   "
                f"R = {d * (d - 1)}/a² + {d * (d + 1)}"
            ),
        )
        frac, r = c["conformal"]
        ax.plot(
            frac,
            r,
            color="navy",
            lw=2.0,
            label="conformal reading: a(z) = sinh z, k=0   "
            + ("R = +6/a²" if d == 3 else "R = 2(sinh²z − 1)/a⁴"),
        )
        frac, r = c["sphere"]
        ax.plot(
            frac,
            r,
            color="seagreen",
            lw=1.6,
            ls="--",
            label=(
                f"open sphere: a(t) = 2 sinh(t/2), k=−1   "
                f"R = {d * (d + 1) / 4:g} (de Sitter, open chart)"
            ),
        )

        if d == 2:
            # The one surviving sign change: 2+1 conformal, at sinh z = 1.
            x0 = float(np.log(1 + np.sqrt(2)) / np.pi)
            ax.plot([x0], [0.0], "o", color="navy", ms=5, zorder=5)
            ax.annotate(
                "R = 0 at z = ln(1+√2)",
                (x0, 0.0),
                textcoords="offset points",
                xytext=(10, -14),
                fontsize=9,
                color="navy",
            )
        else:
            ax.annotate(
                "the turnaround moved to imaginary time:\n"
                "t = iπ/2, a = sinh(iπ/2) = i, R = +6\n"
                "(it was a = 1, R = −6)",
                (0.5, -7.5),
                ha="center",
                fontsize=9,
            )

        ax.set_ylim(*ylim)
        ax.set_xlabel("t/π or z/π (the old loop window; history now extends to ∞)")
        ax.set_title(f"{d}+1")
        ax.legend(loc="upper center", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Ricci scalar R")
    fig.suptitle(
        "Scale factor i: the A → i continuation turns sin into sinh — "
        "Λ flips −3 → +3 and the Ricci scalar turns positive"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)

    for d in (3, 2):
        extra = (
            "conformal R = +6/a^2 > 0 always"
            if d == 3
            else "conformal sign change at z = ln(1+sqrt2) = "
            f"{np.log(1 + np.sqrt(2)):.4f}"
        )
        print(
            f"{d}+1: cosmic R = {d * (d - 1)}/a^2 + {d * (d + 1)} > 0 always; "
            f"{extra}; open sphere R = {d * (d + 1) / 4:g} const; "
            f"complex turnaround (t = i pi/2, a = i): R = +{2 * d}"
        )
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ricci_sinh.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
