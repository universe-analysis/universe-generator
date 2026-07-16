"""Causal reach of the wiggle family: max comoving displacement vs time.

The unit rapidity budget |a*b| = 1 defines the model's own causal structure.
A unit-budget term (a = 1/b) has comoving position

    X(z) = [sin(b z + f) - sin f] / (b sin z)
         = [ alpha(z) sin f + beta(z) cos f ]

with alpha(z) = (cos bz - 1)/(b sin z) and beta(z) = sin bz/(b sin z), i.e.
X is *linear* in (sin f, cos f). The max comoving displacement between two
times is therefore exact:

    even b (free phase):  sqrt(d_alpha^2 + d_beta^2)
    odd  b (f in {0,pi}): |d_beta|

and, the budget being linear across terms, concentrating it in the single
best term is always optimal -- multi-term worldlines cannot reach farther.

Closed-form landmarks (see PHYSICS_FINDINGS / lab notes 2026-07-16):

  * Bang plateau, phase-free: 1 + |min sinc| = 1.21723 at bz = 4.4934
    (the root of tan u = u); cutting the below-zero overshoot gives
    exactly 1 at bz = pi.
  * Bang plateau, phases on:  1.25959 at bz = 4.0856.
  * At z = pi/2: exactly 4/3 (b = 3, f = 0) phase-free; exactly sqrt(2)
    (b = 2, f = pi/4) with phases.
  * Crunch: every even-frequency term reaches displacement exactly 2.
  * Late growth follows the b = 2 envelope 2 sin(z/2), overtaking the
    plateau at z = 2 arcsin(plateau/2) ~ 0.43 pi.

The plateau sits above the antipode distance chi = 1, so the family has no
horizon at any epoch: antipodal comoving points are connectable arbitrarily
close to the Bang (needs b ~ 4.1/z). The reach is per-axis, i.e. natively a
Chebyshev square, and dimension-independent (2+1 and 3+1 share this chart).

Usage::

    python -m analysis.analyze_wiggle_reach --out figures/wiggle_reach.png
"""

from __future__ import annotations

import argparse

import numpy as np

PI = np.pi
#: Bang-limit plateau of the phase-free reach, 1 + |min sinc|.
PLATEAU_PHASE_FREE = 1.2172336
#: Bang-limit plateau with free phases on even frequencies.
PLATEAU_WITH_PHASE = 1.2595906


def alpha_beta(b: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Comoving basis pair (alpha, beta) of a unit-budget term at time z."""
    s = np.sin(z)
    return (np.cos(b * z) - 1.0) / (b * s), np.sin(b * z) / (b * s)


def reach_two_time(z1: float, z2: float, bmax: int, phases: bool = True) -> float:
    """Max |X(z2) - X(z1)| over integer frequencies b in [2, bmax]."""
    b = np.arange(2, bmax + 1, dtype=float)
    a1, b1 = alpha_beta(b, z1)
    a2, b2 = alpha_beta(b, z2)
    d_f0 = np.abs(b2 - b1)
    if phases:
        d_ph = np.hypot(a2 - a1, b2 - b1)
        even = np.arange(2, bmax + 1) % 2 == 0
        d = np.where(even, d_ph, d_f0)
    else:
        d = d_f0
    return float(np.max(d))


def reach_from_bang(z: float, bmax: int, phases: bool = True) -> tuple[float, int]:
    """Max |X(z) - X(0)| over integer b in [2, bmax]; returns (reach, argmax b).

    From the Bang alpha -> 0 and beta -> 1, so the displacement is
    hypot(alpha, beta - 1) (even, phased) or |beta - 1| (f = 0).
    """
    b = np.arange(2, bmax + 1, dtype=float)
    a, be = alpha_beta(b, z)
    d_f0 = np.abs(be - 1.0)
    if phases:
        d_ph = np.hypot(a, be - 1.0)
        even = np.arange(2, bmax + 1) % 2 == 0
        d = np.where(even, d_ph, d_f0)
    else:
        d = d_f0
    i = int(np.argmax(d))
    return float(d[i]), i + 2


def plateau_constants(n: int = 3_000_000) -> tuple[float, float, float, float]:
    """Bang-limit (z -> 0, b -> inf, u = bz continuous) reach plateaus.

    Returns (plateau_f0, argmax_u_f0, plateau_phase, argmax_u_phase).
    """
    u = np.linspace(1e-4, 30.0, n)
    sinc = np.sin(u) / u
    d_f0 = np.abs(sinc - 1.0)
    d_ph = np.hypot((np.cos(u) - 1.0) / u, sinc - 1.0)
    i0, i1 = int(np.argmax(d_f0)), int(np.argmax(d_ph))
    return float(d_f0[i0]), float(u[i0]), float(d_ph[i1]), float(u[i1])


def conformal_fit_speed(z_floor: float) -> float:
    """The viewer's fit speed c = -1/eta(floor) (antipode at pi/2)."""
    return -1.0 / float(np.log(np.tan(z_floor / 2)))


def make_figure(out: str, bmax: int, floor_frac: float) -> None:
    """Two-panel chart: reach from the Bang, and two-time reach vs the
    conformal front the viewer used before the wiggle-budget mode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_f0, _, p_ph, _ = plateau_constants()
    zg = np.linspace(0.005 * PI, 0.995 * PI, 800)
    r_f0 = np.array([reach_from_bang(z, bmax, phases=False)[0] for z in zg])
    r_ph = np.array([reach_from_bang(z, bmax, phases=True)[0] for z in zg])

    c_fit = conformal_fit_speed(floor_frac * PI)
    eta = lambda z: np.log(np.tan(z / 2))  # noqa: E731
    obs_list = (0.25 * PI, 0.50 * PI, 0.75 * PI, 0.98 * PI)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.2))

    axa.plot(
        zg / PI,
        r_ph,
        lw=2,
        color="tab:blue",
        label=f"reach, phases on (current ensemble), bmax={bmax}",
    )
    axa.plot(zg / PI, r_f0, lw=1.6, color="tab:orange", label="reach, phase-free f=0")
    axa.plot(
        zg / PI,
        2 * np.sin(zg / 2),
        "--",
        lw=1,
        color="gray",
        label="b=2 envelope  2 sin(z/2)",
    )
    axa.axhline(1.0, color="k", lw=0.8, ls=":", label="antipode  $\\chi=1$")
    axa.axhline(2.0, color="k", lw=0.8, ls="-.", label="full lap  $\\chi=2$")
    axa.axhline(p_f0, color="tab:orange", lw=0.7, ls=":", alpha=0.7)
    axa.axhline(p_ph, color="tab:blue", lw=0.7, ls=":", alpha=0.7)
    axa.plot([0.5], [4 / 3], "o", color="tab:orange", ms=6)
    axa.annotate(
        "4/3 (b=3)",
        (0.5, 4 / 3),
        textcoords="offset points",
        xytext=(8, -14),
        fontsize=9,
        color="tab:orange",
    )
    axa.plot([0.5], [np.sqrt(2)], "o", color="tab:blue", ms=6)
    axa.annotate(
        "$\\sqrt{2}$ (b=2, f=$\\pi$/4)",
        (0.5, np.sqrt(2)),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=9,
        color="tab:blue",
    )
    axa.set_xlabel("z / $\\pi$")
    axa.set_ylabel("max comoving displacement from Bang (per axis = Chebyshev)")
    axa.set_title("Wiggle-budget causal reach from the Bang")
    axa.legend(fontsize=8, loc="upper left")
    axa.set_ylim(0, 2.15)
    axa.grid(alpha=0.25)

    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(obs_list)))
    for zo, col in zip(obs_list, colors):
        ze = np.linspace(0.005 * PI, zo * 0.999, 500)
        rt = np.array([reach_two_time(z, zo, bmax) for z in ze])
        chi = np.clip(c_fit * (eta(zo) - eta(ze)), 0, None)
        axb.plot(
            ze / PI,
            rt,
            lw=2,
            color=col,
            label=f"model reach to $z_{{obs}}$ = {zo / PI:.2f} $\\pi$",
        )
        axb.plot(ze / PI, chi, "--", lw=1.2, color=col)
    axb.axhline(1.0, color="k", lw=0.8, ls=":")
    axb.axhline(2.0, color="k", lw=0.8, ls="-.")
    axb.set_xlabel("emission time $z_e$ / $\\pi$")
    axb.set_ylabel("max comoving separation traversable ($z_e \\to z_{obs}$)")
    axb.set_title(f"Two-time reach (solid) vs conformal front c={c_fit:.3f} (dashed)")
    axb.legend(fontsize=8)
    axb.set_ylim(0, 2.6)
    axb.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--out", default="figures/wiggle_reach.png", help="output figure path"
    )
    ap.add_argument(
        "--bmax", type=int, default=400, help="largest integer frequency in the family"
    )
    ap.add_argument(
        "--floor",
        type=float,
        default=0.02,
        help="rewind floor as a fraction of pi (viewer default)",
    )
    args = ap.parse_args()

    p_f0, u_f0, p_ph, u_ph = plateau_constants()
    print(f"Bang plateau, phase-free : {p_f0:.6f} at u = b*z = {u_f0:.4f}")
    print(f"Bang plateau, with phase : {p_ph:.6f} at u = b*z = {u_ph:.4f}")
    print(
        f"conformal fit speed at floor {args.floor}pi: "
        f"{conformal_fit_speed(args.floor * PI):.4f}"
    )
    for zf in (0.02, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.98):
        z = zf * PI
        r0, b0 = reach_from_bang(z, args.bmax, phases=False)
        r1, b1 = reach_from_bang(z, args.bmax, phases=True)
        print(
            f"z = {zf:.2f}pi  reach f0 = {r0:.4f} (b={b0:3d})"
            f"   with-phase = {r1:.4f} (b={b1:3d})"
        )
    make_figure(args.out, args.bmax, args.floor)


if __name__ == "__main__":
    main()
