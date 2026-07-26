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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(PARAMS / "params_T120.csv"))
    parser.add_argument("--out", type=Path, default=Path("eos_history.png"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    axes = load_axis_terms(args.params)

    # mass weights
    E_b = np.sum([ax.b.sum(axis=1) for ax in axes], axis=0)
    zs_L = np.linspace(0.0, HALF_PI, 600)
    dz_L = zs_L[1] - zs_L[0]
    E_L = np.zeros(len(E_b))
    for z in zs_L:
        E_L += np.sqrt(velocity2(axes, float(z))) * dz_L

    # w(z) under each dictionary
    zgrid = np.linspace(0.01, np.pi - 0.01, 240)
    w_b_list, w_L_list = [], []
    for z in zgrid:
        v2 = velocity2(axes, float(z))
        w_b_list.append((E_b * v2).sum() / E_b.sum() / 3.0)
        w_L_list.append((E_L * v2).sum() / E_L.sum() / 3.0)
    w_b, w_L = np.array(w_b_list), np.array(w_L_list)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(zgrid, w_b, color="tab:blue", lw=2, label="E ~ b (Quantum Wave)")
    ax.plot(zgrid, w_L, color="tab:orange", lw=2, ls="--", label="E ~ length (String)")
    ax.axhline(1 / 3, color="tab:red", ls=":", lw=1, label="radiation w=1/3")
    ax.axhline(0, color="gray", ls=":", lw=1, label="dust w=0")
    ax.axvline(HALF_PI, color="black", ls="--", lw=1, label="turnaround z=pi/2")
    ax.set_xlabel("conformal time z  (bang 0 -> turnaround pi/2 -> crunch pi)")
    ax.set_ylabel("equation of state  w = P/rho")
    ax.set_title(f"Equation-of-state history ({Path(args.params).stem})")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    iz = np.argmin(np.abs(zgrid - HALF_PI))
    print(f"w at turnaround:  E~b={w_b[iz]:.4f}  E~L={w_L[iz]:.4f}")
    print(f"w near bang (z~0.01):  E~b={w_b[0]:.4f}  E~L={w_L[0]:.4f}")
    print(f"w max over cycle: E~b={w_b.max():.4f} at z={zgrid[w_b.argmax()]:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
