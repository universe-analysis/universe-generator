"""N-vs-T chart: packed count vs resolution across the full T ladder.

One coherent edge-sampled dataset now:
  - jammed anchors (T=5,6,10), run to true jamming,
  - 1e-7-cutoff ladder (T=18..180), full Nyquist, identical code.

The slope is the packing dimension D. Light T^2 and T^3 guide lines show the
dimension is bracketed between the 2D and 3D references. The fit is taken on the
high-T cutoff window (T>=100), where the local slope is cleanest (~2.50).

Usage::

    python plot_nvt.py --out nvt.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Edge sampler throughout. Jammed anchors (true jamming) and 1e-7-cutoff ladder.
JAMMED = {2: 1, 3: 8, 4: 15, 5: 26, 6: 43, 10: 195}
CUTOFF = {
    18: 802, 24: 1792, 32: 3697, 44: 8524, 60: 17049, 80: 34003,
    100: 58369, 120: 92008, 140: 135351, 160: 188940, 180: 250248,
    200: 324080,
}


def plot(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Bracket guides: slope-2 and slope-3 references (prefactors chosen to frame).
    tline = np.array([1.8, 220])
    ax.plot(tline, 0.45 * tline**2, ":", color="gray", lw=1, label="slope 2 (T^2)")
    ax.plot(tline, 0.04 * tline**3, ":", color="darkgray", lw=1, label="slope 3 (T^3)")

    # Power-law fit on the clean high-T cutoff window (T >= 100).
    hi_t = np.array([t for t in CUTOFF if t >= 100])
    hi_n = np.array([CUTOFF[t] for t in hi_t])
    d, logc = np.polyfit(np.log(hi_t), np.log(hi_n), 1)
    fit_t = np.array([14, 220])
    ax.plot(
        fit_t, np.exp(logc) * fit_t**d, "--", color="black", lw=1.5,
        label=f"fit T>=100 (cutoff):  N ~ T^{d:.2f}",
    )

    ax.scatter(
        list(JAMMED), list(JAMMED.values()), s=95, color="tab:green",
        marker="o", zorder=5, label="jammed (edge, true jamming)",
    )
    ax.scatter(
        list(CUTOFF), list(CUTOFF.values()), s=60, color="tab:blue",
        marker="s", zorder=5, label="1e-7 cutoff (edge)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T (timesteps = resolution)")
    ax.set_ylabel("N (worldlines packed)")
    ax.set_title("Packing count vs resolution (edge sampler): N ~ T^D")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"high-T cutoff fit D (T>=100) = {d:.3f}")
    print(f"global cutoff fit (T=18..180) = "
          f"{np.polyfit(np.log(list(CUTOFF)), np.log(list(CUTOFF.values())), 1)[0]:.3f}")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("nvt.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
