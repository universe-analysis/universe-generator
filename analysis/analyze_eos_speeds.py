"""Equation of state under the three velocity definitions.

The codebase carries three related speeds for a worldline (see the box
viewer's paint modes and ``analyze_eos_history``):

1. **physical slope** dx/dz -- the standard w-dictionary velocity,
   peculiar motion PLUS the Hubble recession of the comoving anchor
   (the ``a2 cos z`` term). This is what every published w(z) uses.
2. **proper peculiar speed** sin(z) * dX/dz -- motion relative to the
   comoving frame in physical units (the scale factor carries the
   comoving coordinate rate back to physical space). A legitimate
   EOS velocity: bounded, chart-independent, and equal to the physical
   slope at the turnaround (cos z = 0 kills the Hubble term there).
3. **comoving coordinate speed** dX/dz -- peculiar motion per comoving
   grid unit. NOT a valid EOS velocity: it is chart-dependent and
   unbounded (per axis dX/dz = v_pec/sin z), so its "w" is exactly the
   peculiar w divided by sin^2(z), diverging at the Bang/Crunch as a
   pure coordinate artifact. It is drawn anyway, clipped, to show the
   pathology.

All three are combined into w = <E v^2> / <E> / 3 under the standard
E ~ sum(b) dictionary (uniform per-path weights at terms = T).

Usage::

    uv run python -m analysis.analyze_eos_speeds \
        --params3 data/fullspec/dumps/d3_nyq_T40_s*_sub_fsub3e6.csv \
        --params2 data/fullspec/dumps/d2_nyq_T100_s*_sub_fsub2e6.csv \
        --out eos_speeds.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from braidlab.corrdim import AxisTerms, load_axis_terms

HALF_PI = np.pi / 2.0


def _t_of(path: Path) -> int:
    m = re.search(r"_T(\d+)_", path.name)
    if m is None:
        raise ValueError(f"no _T<N>_ in dump name: {path}")
    return int(m.group(1))


def axis_kinematics(ax: AxisTerms, z: float) -> tuple[np.ndarray, np.ndarray]:
    """(physical position x, physical slope dx/dz) of one axis at z."""
    s, c = np.sin(z), np.cos(z)
    phase = ax.b * z + ax.f
    x = ax.a2 * s + np.einsum("nj,nj->n", ax.a, np.sin(phase) - np.sin(ax.f))
    dx = ax.a2 * c + np.einsum("nj,nj->n", ax.a, ax.b * np.cos(phase))
    return x, dx


def speeds_squared(
    axes: list[AxisTerms], z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(v_phys^2, v_pec^2, v_com^2) per path at z, summed over axes.

    Per axis: v_phys = dx/dz; the comoving coordinate rate is
    dX/dz = (dx*s - x*c)/s^2, so v_pec = s*dX/dz = dx - x*c/s and
    v_com = dX/dz = v_pec/s.
    """
    s, c = np.sin(z), np.cos(z)
    v_phys = np.zeros(len(axes[0].a2))
    v_pec = np.zeros_like(v_phys)
    for ax in axes:
        x, dx = axis_kinematics(ax, z)
        v_phys += dx**2
        v_pec += (dx - x * c / s) ** 2
    return v_phys, v_pec, v_pec / s**2


def wz_curves(axes: list[AxisTerms], zgrid: np.ndarray) -> dict[str, np.ndarray]:
    """w(z) = <E v^2>/<E>/3 under E ~ sum(b), for the three speeds."""
    e = np.sum([ax.b.sum(axis=1) for ax in axes], axis=0)
    esum = e.sum()
    out = {k: np.empty(len(zgrid)) for k in ("phys", "pec", "com")}
    for i, z in enumerate(zgrid):
        vp, vq, vc = speeds_squared(axes, float(z))
        out["phys"][i] = float(e @ vp / esum / 3.0)
        out["pec"][i] = float(e @ vq / esum / 3.0)
        out["com"][i] = float(e @ vc / esum / 3.0)
    return out


def load_pooled(paths: list[Path]) -> tuple[list[AxisTerms], int]:
    """Pool the dumps of one cell (same T) into concatenated axis terms."""
    ts = {_t_of(p) for p in paths}
    if len(ts) != 1:
        raise ValueError(f"pool one T at a time, got {sorted(ts)}")
    per_file = [load_axis_terms(p) for p in sorted(paths)]
    n_axes = len(per_file[0])
    axes = []
    for k in range(n_axes):
        axes.append(
            AxisTerms(
                a=np.concatenate([f[k].a for f in per_file]),
                b=np.concatenate([f[k].b for f in per_file]),
                f=np.concatenate([f[k].f for f in per_file]),
                a2=np.concatenate([f[k].a2 for f in per_file]),
            )
        )
    return axes, ts.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params3", nargs="+", default=[], help="3+1 dumps (one T)")
    parser.add_argument("--params2", nargs="+", default=[], help="2+1 dumps (one T)")
    parser.add_argument("--zpoints", type=int, default=33)
    parser.add_argument("--out", type=Path, default=Path("eos_speeds.png"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    cells = []
    for group in (args.params3, args.params2):
        if group:
            cells.append(load_pooled([Path(p) for p in group]))
    if not cells:
        raise SystemExit("no dumps given")

    base = np.linspace(0.02 * np.pi, 0.98 * np.pi, args.zpoints)
    zgrid = np.unique(np.concatenate([base, [HALF_PI]]))
    fig, axs = plt.subplots(1, len(cells), figsize=(5.4 * len(cells), 4.4))
    for axp, (axes, t) in zip(np.atleast_1d(axs), cells):
        d = len(axes)
        w = wz_curves(axes, zgrid)
        zp = zgrid / np.pi
        w_pred = (d / 3.0) * (np.cos(zgrid) ** 2 / 3.0 + 1.0 / t)
        axp.plot(zp, w["phys"], color="k", lw=2.0, label="physical slope dx/dz")
        axp.plot(zp, w_pred, color="0.4", lw=1.0, ls="--", label="closed form")
        axp.plot(
            zp,
            w["pec"],
            color="C0",
            lw=1.8,
            label="proper peculiar  sin z · dX/dz",
        )
        axp.plot(
            zp,
            w["com"],
            color="C3",
            lw=1.2,
            ls=":",
            label="comoving dX/dz (not an EOS)",
        )
        i0 = int(np.argmin(np.abs(zgrid - HALF_PI)))
        print(
            f"{d}+1 T={t}: turnaround w  phys={w['phys'][i0]:.5f}  "
            f"pec={w['pec'][i0]:.5f}  com={w['com'][i0]:.5f}  "
            f"d/(6T)={d / (6 * t):.5f}"
        )
        axp.axvline(0.5, color="0.85", lw=0.8)
        # the comoving curve diverges ~1/sin^2 z; clip the view to the
        # physical curves' scale so the pathology reads as "leaves the top"
        top = 1.4 * max(w["phys"].max(), w["pec"].max())
        axp.set_ylim(0, top)
        axp.set_xlabel("z / pi")
        axp.set_title(f"{d}+1, T={t}")
    np.atleast_1d(axs)[0].set_ylabel("w(z)")
    np.atleast_1d(axs)[0].legend(fontsize=8)
    fig.suptitle(
        "w(z) under the three velocity definitions (E ~ Σb dictionary); "
        "all meet at the turnaround"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
