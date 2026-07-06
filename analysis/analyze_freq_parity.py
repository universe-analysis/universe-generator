"""Frequency-parity structure of the packing (the even/odd analysis).

Physics context (PHYSICS_FINDINGS): sin(b z)/sin(z) is symmetric about the
turnaround for odd b (at rest at z = pi/2) and anti-symmetric for even b
(swings through). Without phases only high, small-swing even frequencies can
thread the packing. The phase schema hands even frequencies a free phase, so
this structure is expected to move -- these plots measure how much.

Three views:

  1. Per-worldline parity vs T: % of packed worldlines entirely odd-frequency
     vs carrying >= 1 even frequency (the two sum to 100%).
  2. Per-frequency parity vs T: fraction of all accepted b values that are odd.
  3. Stage-resolved even acceptance vs b: among frequencies accepted in the
     first 25% (sparse) vs the last 25% (near-jam) of the acceptance sequence,
     the % even, binned by frequency value b.

Views 1-2 read the campaign's collected dumps; parity fractions are
order-free, so the runner's shuffled subsampling is harmless. View 3 is NOT
order-free: it needs full ``--dump-params`` files written directly by the
engine (acceptance order preserved), supplied via ``--stage``.

Usage::

    # parity vs T, phased vs no-phase overlay
    python -m analysis.analyze_freq_parity --dumps data/torus/dumps --dim 3 \
        --suffix _tor_ph_e6 --baseline-suffix _tor_e6 --out figures/parity.png

    # stage-resolved even acceptance from ordered dumps
    python -m analysis.analyze_freq_parity --dim 3 \
        --stage "with phases=/path/ordered_ph.csv" \
        --stage "no phases=/path/ordered_noph.csv" \
        --stage-out figures/parity_stage.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np

_NAME_RE = re.compile(r"_T(?P<t>\d+)_s(?P<seed>\d+)")
#: Minimum accepted b values in a bin before we quote a % even for it.
MIN_BIN_COUNT = 30


def freq_cols(dim: int) -> list[str]:
    return ["bx", "by", "bw"][: dim if dim == 3 else 2]


def load_freqs(path: str | Path, dim: int) -> np.ndarray:
    """(N, dim-ish) integer frequency matrix from a dump CSV, in file order."""
    rows = list(csv.DictReader(open(path)))
    cols = freq_cols(dim)
    return np.array([[int(float(r[c])) for c in cols] for r in rows])


def parity_vs_t(
    dumps_dir: str, dim: int, suffix: str
) -> tuple[list[int], list[float], list[float]]:
    """Seed-averaged (%% all-odd worldlines, %% odd frequencies) per T."""
    by_t: dict[int, list[tuple[float, float]]] = {}
    for path in sorted(glob.glob(f"{dumps_dir}/d{dim}_nyq_T*_s*{suffix}.csv")):
        m = _NAME_RE.search(Path(path).name)
        if not m:
            continue
        b = load_freqs(path, dim)
        all_odd = float((b % 2 == 1).all(axis=1).mean() * 100.0)
        odd_freq = float((b % 2 == 1).mean() * 100.0)
        by_t.setdefault(int(m["t"]), []).append((all_odd, odd_freq))
    ts = sorted(by_t)
    world = [float(np.mean([v[0] for v in by_t[t]])) for t in ts]
    freq = [float(np.mean([v[1] for v in by_t[t]])) for t in ts]
    return ts, world, freq


def stage_even_by_b(b: np.ndarray, n_bins: int = 24) -> dict[str, np.ndarray]:
    """%% even vs b for the first and last acceptance quartile of one dump."""
    n = len(b)
    quarters = {"first 25% (sparse)": b[: n // 4], "last 25% (near-jam)": b[-n // 4 :]}
    lo, hi = 2, int(b.max())
    edges = np.linspace(lo, hi + 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out: dict[str, np.ndarray] = {"centers": centers}
    for label, part in quarters.items():
        flat = part.ravel()
        pct = np.full(n_bins, np.nan)
        for i in range(n_bins):
            sel = flat[(flat >= edges[i]) & (flat < edges[i + 1])]
            if len(sel) >= MIN_BIN_COUNT:
                pct[i] = (sel % 2 == 0).mean() * 100.0
        out[label] = pct
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="data/torus/dumps")
    parser.add_argument("--dim", type=int, default=3, choices=(2, 3))
    parser.add_argument("--suffix", default="", help="primary dataset suffix")
    parser.add_argument(
        "--baseline-suffix",
        default=None,
        help="optional second dataset overlaid dashed (e.g. the no-phase run)",
    )
    parser.add_argument("--label", default="with phases")
    parser.add_argument("--baseline-label", default="no phases")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="ordered full dump for the stage-resolved view (repeatable)",
    )
    parser.add_argument("--stage-out", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.out is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        datasets = [(args.label, args.suffix, "-")]
        if args.baseline_suffix is not None:
            datasets.append((args.baseline_label, args.baseline_suffix, "--"))
        for label, suffix, ls in datasets:
            ts, world_odd, freq_odd = parity_vs_t(args.dumps, args.dim, suffix)
            ax1.plot(
                ts,
                world_odd,
                "o" + ls,
                color="tab:green",
                label=f"entirely odd-frequency ({label})",
            )
            ax1.plot(
                ts,
                [100.0 - w for w in world_odd],
                "o" + ls,
                color="steelblue",
                label=f">= 1 even ({label})",
            )
            ax2.plot(ts, freq_odd, "o" + ls, color="tab:green", label=label)
            print(
                f"[{label}] T={ts[-1]}: all-odd worldlines {world_odd[-1]:.1f}%, "
                f"odd frequencies {freq_odd[-1]:.1f}%"
            )
        ax1.set_title("Per-worldline parity (the two sum to 100%)")
        ax1.set_xlabel("T (resolution)")
        ax1.set_ylabel("% of packed worldlines")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax2.set_title("Per-frequency: fraction of all b values that are odd")
        ax2.set_xlabel("T (resolution)")
        ax2.set_ylabel("% of accepted frequencies that are odd")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"wrote {args.out}")

    if args.stage:
        if args.stage_out is None:
            parser.error("--stage requires --stage-out")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        colors = {"first 25% (sparse)": "tab:blue", "last 25% (near-jam)": "tab:orange"}
        styles = ["-", "--", ":"]
        for spec, ls in zip(args.stage, styles):
            label, _, path = spec.partition("=")
            b = load_freqs(path, args.dim)
            res = stage_even_by_b(b)
            for stage_label, color in colors.items():
                ax.plot(
                    res["centers"],
                    res[stage_label],
                    "o" + ls,
                    ms=3.5,
                    color=color,
                    label=f"{stage_label} — {label}",
                )
        ax.set_xlabel("frequency value b")
        ax.set_ylabel("% even among frequencies accepted in that window")
        ax.set_title("Even acceptance vs frequency, by packing stage")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        args.stage_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.stage_out, dpi=130)
        print(f"wrote {args.stage_out}")


if __name__ == "__main__":
    main()
