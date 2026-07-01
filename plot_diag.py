"""Plot the engine's center-bias diagnostics: occupancy and probe acceptance.

Two panels from one short run:
  (left)  occupancy -- where accepted worldlines actually sit in comoving X.
  (right) probe -- acceptance rate of fresh proposals vs their X-center, i.e.
          how much room is left as a function of position.

If the uniform proposal over-filled the center and starved the edges, the left
panel would peak in the middle and the right panel would be higher at the
edges. Neither happens, which is the point.

Usage::

    python plot_diag.py --prefix data/diag/d18b --out diag_T18.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(open(path)))


def plot(prefix: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    occ = load(Path(f"{prefix}_occ.csv"))
    probe = load(Path(f"{prefix}_probe.csv"))

    occ_x = [float(r["x_center"]) for r in occ]
    occ_c = [int(r["count"]) for r in occ]

    probe_x = [float(r["x_center"]) for r in probe]
    probe_rate = [
        int(r["accepted"]) / int(r["proposed"]) if int(r["proposed"]) else 0.0
        for r in probe
    ]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    ax_left.bar(occ_x, occ_c, width=0.035, color="tab:purple", alpha=0.8)
    ax_left.set_xlabel("comoving X (center = 0, walls at +/-1)")
    ax_left.set_ylabel("accepted worldline occupancy")
    ax_left.set_title("Where worldlines sit: flat, edge peaks (not center-piled)")
    ax_left.grid(True, axis="y", alpha=0.3)

    ax_right.scatter(probe_x, probe_rate, s=18, color="tab:green")
    ax_right.set_xlabel("proposal X-center")
    ax_right.set_ylabel("probe acceptance rate (room left)")
    ax_right.set_title("Room left vs position: flat (edges not starved)")
    ax_right.set_ylim(bottom=0)
    ax_right.grid(True, alpha=0.3)

    fig.suptitle(
        "Center-bias diagnostics, T=18 (uniform proposal): no center crowding",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path("data/diag/d18b"))
    parser.add_argument("--out", type=Path, default=Path("diag_T18.png"))
    args = parser.parse_args()
    plot(args.prefix, args.out)


if __name__ == "__main__":
    main()
