"""SUBPATH campaign chart: growth curves and capacity ratios.

Subpaths do not jam (Kevin's design point): phase 2 has no ceiling to
converge to, so every Nsub is conditional on where the run stopped — the
T >= 60 cells stop when the admission rate decays through the campaign
cutoff (1e-6), while T = 20's rate never decays that far (it floors near
7e-6) and is stopped by a fixed 1e10-attempt budget instead. What is
portable is the *shape*: the growth law of Nsub(attempts) and how the
admission rate decays.

Left: Nsub vs attempts per T (seed 1), log-log — the growth-law view.
Right: the stop-state capacity ratio Nsub/N per T (rate-stopped tiers
only; T = 20 is budget-stopped, so its ratio is not comparable and is
drawn hollow), plus the tail growth exponent gamma per T.

No 3+1 twin exists for this chart: subpaths are a 2+1 engine feature
(the dimension-symmetric-plots convention notes the exception).

Usage::

    uv run python -m plots.plot_subpath_decay --db data/converge/subpath2d_e6.db \
        --out subpath_decay.png
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np

#: Budget-stopped tiers (admission rate floors above the cutoff; Nsub there
#: is set by the --sub-attempts budget, not by rate convergence).
BUDGET_STOPPED = {20}


def load(db: Path) -> dict[int, list[dict[str, np.ndarray]]]:
    """Per-T list of curve dicts (attempts, n, nsub arrays)."""
    out: dict[int, list[dict[str, np.ndarray]]] = {}
    conn = sqlite3.connect(db)
    for t, cp in conn.execute(
        "select t, curve_path from runs where status='done' order by t, seed"
    ):
        rows = list(csv.DictReader(open(cp)))
        cur = {
            k: np.array([float(r[k]) for r in rows]) for k in ("attempts", "n", "nsub")
        }
        out.setdefault(t, []).append(cur)
    return out


def tail_gamma(att: np.ndarray, nsub: np.ndarray) -> float:
    """Log-log tail slope of Nsub(attempts) over its last decade."""
    grow = nsub > 0
    a, s = att[grow], nsub[grow]
    tail = a >= a[-1] / 10
    if tail.sum() < 3:
        return float("nan")
    p = np.polyfit(np.log(a[tail]), np.log(s[tail]), 1)
    return float(p[0])


def plot(db: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = load(db)
    ts = sorted(curves)
    cmap = plt.get_cmap("viridis")
    shades = {t: cmap(x) for t, x in zip(ts, np.linspace(0.1, 0.9, len(ts)))}

    fig, (ax_g, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    for t in ts:
        c = curves[t][0]  # seed 1 for the growth view
        grow = c["nsub"] > 0
        style = "--" if t in BUDGET_STOPPED else "-"
        ax_g.plot(
            c["attempts"][grow],
            c["nsub"][grow],
            style,
            color=shades[t],
            lw=1.4,
            label=f"T={t}" + (" (budget-stopped)" if t in BUDGET_STOPPED else ""),
        )
    ax_g.set_xscale("log")
    ax_g.set_yscale("log")
    ax_g.set_xlabel("attempts")
    ax_g.set_ylabel("Nsub")
    ax_g.set_title("2+1 subpath growth (seed 1 per T)")
    ax_g.grid(True, which="both", alpha=0.3)
    ax_g.legend(fontsize=8)

    ratios, ratio_errs, gammas = [], [], []
    for t in ts:
        rr = [c["nsub"][-1] / c["n"][-1] for c in curves[t]]
        gg = [tail_gamma(c["attempts"], c["nsub"]) for c in curves[t]]
        ratios.append(float(np.mean(rr)))
        ratio_errs.append(float(np.std(rr, ddof=1) / np.sqrt(len(rr))))
        gammas.append(float(np.nanmean(gg)))
    solid = [t not in BUDGET_STOPPED for t in ts]
    ax_r.errorbar(
        [t for t, s in zip(ts, solid) if s],
        [r for r, s in zip(ratios, solid) if s],
        yerr=[e for e, s in zip(ratio_errs, solid) if s],
        fmt="o-",
        color="tab:blue",
        capsize=2,
        label="Nsub/N at the 1e-6 rate stop",
    )
    for t, r, s in zip(ts, ratios, solid):
        if not s:
            ax_r.plot(t, r, "o", mfc="none", color="tab:blue")
            ax_r.annotate(
                f"budget-stopped\n({r:.0f}x)",
                (t, r),
                fontsize=8,
                textcoords="offset points",
                xytext=(8, -12),
            )
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("T")
    ax_r.set_ylabel("Nsub / N at stop")
    ax_r.set_title("2+1 subpath capacity ratio vs resolution")
    ax_r.grid(True, which="both", alpha=0.3)
    ax_r.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")
    for t, r, g in zip(ts, ratios, gammas):
        stop = "budget" if t in BUDGET_STOPPED else "rate"
        print(f"  T={t:>3} ({stop}-stopped): Nsub/N = {r:8.2f}  tail gamma = {g:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("subpath_decay.png"))
    args = parser.parse_args()
    plot(args.db, args.out)


if __name__ == "__main__":
    main()
