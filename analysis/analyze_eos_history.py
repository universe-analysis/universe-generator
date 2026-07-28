"""Equation-of-state history w(z) across the expand-contract cycle.

For one fixed packing, evaluate the kinetic pressure at every conformal
time z in (0, pi):

    v_i^2(z) = sum_axis (sum_j a_j b_j cos(b_j z + f_j) + a2 cos z)^2
    w(z)     = P/rho = (1/3) * sum_n E_n v_n^2(z) / sum_n E_n

(physical velocity dx/dz, summed over every wiggle term -- the legacy
single-wiggle dumps are the nw = 1 case) under both mass dictionaries
(E ~ sum(b) and E ~ proper length), to see whether the dictionary-robustness
found at the turnaround holds across the whole history.

Expectation: fast (stiff/relativistic) near the bang/crunch z->0, pi; slow
(matter-like) at the turnaround z=pi/2.

Usage::

    python -m analysis.analyze_eos_history --params <dump.csv> --out out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from braidlab.corrdim import AxisTerms, load_axis_terms

PARAMS = Path("data/params")
HALF_PI = np.pi / 2.0


def velocity2(axes: list[AxisTerms], z: float) -> np.ndarray:
    """Physical speed^2 (dx/dz)^2 per worldline at conformal time z."""
    v2 = np.zeros(len(axes[0].a2))
    for ax in axes:
        v_axis = (ax.a * ax.b * np.cos(ax.b * z + ax.f)).sum(axis=1)
        v2 += (v_axis + ax.a2 * np.cos(z)) ** 2
    return v2


def _t_of(path: str | Path) -> int:
    import re

    m = re.search(r"_T(\d+)", Path(path).name)
    if m is None:
        raise ValueError(f"no _T<N> in dump name: {path}")
    return int(m.group(1))


def measured_wz(
    axes: list[AxisTerms], zgrid: np.ndarray, string_dict: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """MEASURED w(z) from one packing, under the E ~ b dictionary (and,
    optionally, the E ~ arc-length dictionary — the slow one)."""
    e_b = np.sum([ax.b.sum(axis=1) for ax in axes], axis=0)
    e_len = None
    if string_dict:
        zs_l = np.linspace(0.0, HALF_PI, 600)
        dz_l = zs_l[1] - zs_l[0]
        e_len = np.zeros(len(e_b))
        for z in zs_l:
            e_len += np.sqrt(velocity2(axes, float(z))) * dz_l
    w_b_list, w_l_list = [], []
    for z in zgrid:
        v2 = velocity2(axes, float(z))
        w_b_list.append((e_b * v2).sum() / e_b.sum() / 3.0)
        if e_len is not None:
            w_l_list.append((e_len * v2).sum() / e_len.sum() / 3.0)
    return np.array(w_b_list), (np.array(w_l_list) if e_len is not None else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        nargs="+",
        default=[str(PARAMS / "params_T120.csv")],
        help="one dump: measured w(z) under both energy dictionaries; several "
        "dumps: one measured E~b curve per dump, labeled by its T (the "
        "timestep-invariance view)",
    )
    parser.add_argument(
        "--predict-fullspec",
        action="store_true",
        help="overlay the full-spectrum closed form w(z) = (d/3)(cos^2 z / 3 "
        "+ 1/T) per dump (T read from the filename): the comoving-offset "
        "velocity a2 cos z dominates away from the turnaround (radiation-like "
        "w -> d/9 at the bang/crunch) while the ~T incoherent wiggle terms "
        "set the O(1/T) dust floor",
    )
    parser.add_argument("--out", type=Path, default=Path("eos_history.png"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    zgrid = np.linspace(0.01, np.pi - 0.01, 240)
    multi = len(args.params) > 1
    fig, ax = plt.subplots(figsize=(9, 5.5))
    d = 0
    w_b = w_L = None
    for i, params in enumerate(args.params):
        axes = load_axis_terms(params)
        d = len(axes)
        t = _t_of(params)
        # The arc-length dictionary is the slow one; the multi-T view uses
        # E ~ b alone (the dictionaries were shown to agree on the single-T
        # chart).
        w_b, w_L = measured_wz(axes, zgrid, string_dict=not multi)
        color = f"C{i}"
        if multi:
            ax.plot(zgrid, w_b, color=color, lw=2, label=f"MEASURED w(z), T={t}")
        else:
            ax.plot(
                zgrid,
                w_b,
                color="tab:blue",
                lw=2,
                label=r"MEASURED w(z), energy dictionary E $\propto$ b",
            )
            assert w_L is not None
            ax.plot(
                zgrid,
                w_L,
                color="tab:orange",
                lw=2,
                ls="--",
                label=r"MEASURED w(z), energy dictionary E $\propto$ arc length",
            )
        if args.predict_fullspec:
            w_pred = (d / 3.0) * (np.cos(zgrid) ** 2 / 3.0 + 1.0 / t)
            ax.plot(
                zgrid,
                w_pred,
                color=color if multi else "tab:green",
                lw=1.2,
                ls=":",
                label=rf"closed-form approximation, T={t}"
                if multi
                else r"closed-form approximation $(d/3)(\cos^2 z/3 + 1/T)$",
            )
    ax.axhline(1 / 3, color="tab:red", ls=":", lw=1, label="radiation w=1/3")
    ax.axhline(0, color="gray", ls=":", lw=1, label="dust w=0")
    ax.axvline(HALF_PI, color="black", ls="--", lw=1, label="turnaround z=pi/2")
    ax.set_xlabel("conformal time z  (bang 0 -> turnaround pi/2 -> crunch pi)")
    ax.set_ylabel("equation of state  w = P/rho")
    stem = ", ".join(Path(p).stem for p in args.params[:1])
    title = (
        "Equation-of-state history — MEASURED, one curve per T"
        if multi
        else f"Equation-of-state history — MEASURED ({stem})"
    )
    ax.set_title(title)
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    if w_b is not None and w_L is not None:
        iz = int(np.argmin(np.abs(zgrid - HALF_PI)))
        print(f"w at turnaround:  E~b={w_b[iz]:.4f}  E~L={w_L[iz]:.4f}")
        print(f"w near bang (z~0.01):  E~b={w_b[0]:.4f}  E~L={w_L[0]:.4f}")
        print(f"w max over cycle: E~b={w_b.max():.4f} at z={zgrid[w_b.argmax()]:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
