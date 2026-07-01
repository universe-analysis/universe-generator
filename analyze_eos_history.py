"""Equation-of-state history w(z) across the expand-contract cycle.

For the fixed T=120 packing, evaluate the kinetic pressure at every conformal
time z in (0, pi):

    v_i^2(z) = sum_axis (a*b*cos(b z) + a2*cos(z))^2     (physical velocity dx/dz)
    w(z)     = P/rho = (1/3) * sum_n E_n v_n^2(z) / sum_n E_n

under both mass dictionaries (E ~ b and E ~ proper length), to see whether the
dictionary-robustness found at the turnaround holds across the whole history.

Expectation: fast (stiff/relativistic) near the bang/crunch z->0, pi; slow
(matter-like) at the turnaround z=pi/2.

Usage::

    python analyze_eos_history.py --out eos_history.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PARAMS = Path("data/params")
HALF_PI = np.pi / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(PARAMS / "params_T120.csv"))
    parser.add_argument("--out", type=Path, default=Path("eos_history.png"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(args.params)))
    g = lambda k: np.array([float(r[k]) for r in rows])
    a = [g("ax"), g("ay"), g("aw")]
    b = [g("bx"), g("by"), g("bw")]
    a2 = [g("ax2"), g("ay2"), g("aw2")]

    # mass weights
    E_b = b[0] + b[1] + b[2]
    zs_L = np.linspace(0.0, HALF_PI, 600)
    dz_L = zs_L[1] - zs_L[0]
    E_L = np.zeros(len(E_b))
    for z in zs_L:
        s = sum((a[i] * b[i] * np.cos(b[i] * z) + a2[i] * np.cos(z)) ** 2 for i in range(3))
        E_L += np.sqrt(s) * dz_L

    # w(z) under each dictionary
    zgrid = np.linspace(0.01, np.pi - 0.01, 240)
    w_b, w_L = [], []
    for z in zgrid:
        v2 = sum((a[i] * b[i] * np.cos(b[i] * z) + a2[i] * np.cos(z)) ** 2 for i in range(3))
        w_b.append((E_b * v2).sum() / E_b.sum() / 3.0)
        w_L.append((E_L * v2).sum() / E_L.sum() / 3.0)
    w_b, w_L = np.array(w_b), np.array(w_L)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(zgrid, w_b, color="tab:blue", lw=2, label="E ~ b (Quantum Wave)")
    ax.plot(zgrid, w_L, color="tab:orange", lw=2, ls="--", label="E ~ length (String)")
    ax.axhline(1 / 3, color="tab:red", ls=":", lw=1, label="radiation w=1/3")
    ax.axhline(0, color="gray", ls=":", lw=1, label="dust w=0")
    ax.axvline(HALF_PI, color="black", ls="--", lw=1, label="turnaround z=pi/2")
    ax.set_xlabel("conformal time z  (bang 0 -> turnaround pi/2 -> crunch pi)")
    ax.set_ylabel("equation of state  w = P/rho")
    ax.set_title("Equation-of-state history of the braided universe (T=120)")
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
