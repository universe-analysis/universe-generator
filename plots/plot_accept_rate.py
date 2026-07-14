"""Acceptance-rate view of the packing kinetics, per dimension.

Companion to `plots.plot_converge`: that chart shows the stopped *state*
(N vs T); this one shows the *process* that produced it, in acceptance-rate
terms. A symmetric 2x2 grid (rows = dimension, 3+1 top):

  left column  = instantaneous acceptance rate dN/d(attempts) between curve
                 checkpoints vs attempts, one curve per selected T (seed 1),
                 log-log, with the campaign stop rate marked. The decay law
                 is the approach law in rate form: logarithmic N(t) in 3+1
                 means rate ~ 1/t (slope -1); a 2+1 Feder law
                 N_inf - c t^(-p) means rate ~ t^(-1-p).
  right column = cost curve: attempts needed to reach the stop rate vs T
                 (seed mean +/- sem), log-log, with a weighted power-law fit
                 over the converged window. Under *pure* log kinetics
                 (N = b ln t, rate = b/t) the stop time would be
                 t_stop = b(T)/rate_stop and this exponent would equal the
                 rate exponent (2.336 in 3+1); it measures 2.223 +/- 0.012
                 instead -- a third exponent out of the same process, and
                 more evidence that the kinetics carry a T-dependent piece
                 beyond the pure log.

Both columns read the 1e-6 stores (the deepest ladders with full T reach,
including the 3+1 T = 400-520 extension).

Usage::

    uv run python -m plots.plot_accept_rate --out accept_rate.png
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from analysis.analyze_approach_law import load_curve

#: Per dimension: stores stitched into the 1e-6 ladder.
STORES = {
    3: (
        "data/pack/pack3d_e6.db",
        "data/converge/converge3d_e6.db",
        "data/converge/converge3d_e6ext.db",
    ),
    2: ("data/pack/pack2d_e6.db",),
}
#: T values drawn in the rate-decay panel (all T clutters it unreadably).
RATE_T = {3: (40, 80, 160, 240, 360, 520), 2: (20, 60, 100, 180, 300)}
#: Window start for the cost-curve fit (matches plot_converge).
WINDOW = {3: 160, 2: 100}
#: The campaign stop rate for these stores.
STOP_RATE = 1e-6
#: Readable T ticks for the (log-x) cost panel.
COST_TICKS = {3: (20, 40, 80, 160, 240, 360, 520), 2: (20, 40, 80, 140, 220, 300)}


def curves_by_t(dbs: tuple[str, ...]) -> dict[int, list[Path]]:
    """Per-T curve paths (2-term rows), stitched across stores."""
    out: dict[int, list[Path]] = {}
    for db in dbs:
        rows = sqlite3.connect(db).execute(
            "select t, curve_path from runs "
            "where status='done' and terms=2 order by t, seed"
        )
        for t, path in rows:
            out.setdefault(t, []).append(Path(path))
    return out


def rate_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous acceptance rate between checkpoints, at mid-attempts."""
    att, n = load_curve(path)
    d_att = np.diff(att)
    d_n = np.diff(n)
    keep = d_att > 0
    mid = np.sqrt(att[:-1] * att[1:])[keep]
    rate = d_n[keep] / d_att[keep]
    grow = rate > 0
    return mid[grow], rate[grow]


def stop_attempts(paths: list[Path]) -> tuple[float, float]:
    """Seed mean and SEM of the attempts at the stop, in log10."""
    stops = np.array([load_curve(p)[0][-1] for p in paths], float)
    logs = np.log10(stops)
    return float(np.mean(logs)), float(np.std(logs, ddof=1) / np.sqrt(len(logs)))


def cost_fit(
    ts: np.ndarray, log_stop: np.ndarray, sem: np.ndarray, tmin: int
) -> tuple[float, float]:
    """Weighted power-law fit of t_stop(T) over T >= tmin."""
    win = ts >= tmin
    p, cov = np.polyfit(
        np.log10(ts[win]), log_stop[win], 1, w=1 / np.maximum(sem[win], 1e-4), cov=True
    )
    return float(p[0]), float(np.sqrt(cov[0][0]))


def plot(out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))

    for row, dim in enumerate((3, 2)):
        ax_rate, ax_cost = axes[row]
        curves = curves_by_t(STORES[dim])
        cmap = plt.get_cmap("viridis")
        sel = [t for t in RATE_T[dim] if t in curves]
        shades = {t: cmap(x) for t, x in zip(sel, np.linspace(0.1, 0.9, len(sel)))}

        for t in sel:
            mid, rate = rate_curve(curves[t][0])
            ax_rate.plot(mid, rate, "-", color=shades[t], lw=1.4, label=f"T={t}")
        ax_rate.axhline(STOP_RATE, ls=":", color="gray", lw=1.2)
        ax_rate.annotate(
            "stop rate 1e-6",
            (0.02, STOP_RATE),
            xycoords=("axes fraction", "data"),
            fontsize=8.5,
            color="gray",
            va="bottom",
        )
        ax_rate.set_xscale("log")
        ax_rate.set_yscale("log")
        ax_rate.set_xlabel("attempts")
        ax_rate.set_ylabel("acceptance rate dN/d(attempts)")
        ax_rate.set_title(f"{dim}+1: acceptance-rate decay per T (seed 1, 1e-6)")
        ax_rate.grid(True, which="both", alpha=0.3)
        ax_rate.legend(fontsize=8.5)

        ts = np.array(sorted(curves))
        stats = [stop_attempts(curves[t]) for t in ts]
        log_stop = np.array([m for m, _ in stats])
        sem = np.array([e for _, e in stats])
        d, err = cost_fit(ts, log_stop, sem, WINDOW[dim])
        ax_cost.errorbar(
            ts,
            10**log_stop,
            yerr=np.log(10) * sem * 10**log_stop,
            fmt="o",
            color="tab:blue",
            ms=4.5,
            capsize=2,
            label=f"attempts at stop; fit T>={WINDOW[dim]}: ~T^{d:.3f}±{err:.3f}",
        )
        win = ts >= WINDOW[dim]
        anchor = log_stop[win][0]
        ax_cost.plot(
            ts[win],
            10 ** (anchor + d * np.log10(ts[win] / ts[win][0])),
            "--",
            color="tab:blue",
            lw=1.0,
            alpha=0.6,
        )
        ax_cost.set_xscale("log")
        ax_cost.set_yscale("log")
        ax_cost.xaxis.set_major_locator(mticker.FixedLocator(COST_TICKS[dim]))
        ax_cost.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_cost.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax_cost.set_xlabel("T")
        ax_cost.set_ylabel("attempts to reach the stop rate")
        ax_cost.set_title(f"{dim}+1: cost to converge vs resolution")
        ax_cost.grid(True, which="both", alpha=0.3)
        ax_cost.legend(fontsize=8.5, loc="lower right")
        print(f"{dim}+1 cost exponent (T>={WINDOW[dim]}): {d:.3f} +/- {err:.3f}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("accept_rate.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
