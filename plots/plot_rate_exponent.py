"""Rate-exponent systematics: is the process exponent a constant at all?

Follow-up to the state-vs-rate question. The quoted rate exponent
(b(T) ~ T^2.336(8) in 3+1) carried only the fit error; this chart makes the
two real systematics visible, per dimension (3+1 top, 2+1 bottom):

  left column  = local two-point slope of the process observable vs rung T
                 (3+1: the log-growth rate b = dN/d ln t; 2+1: the Feder
                 ceiling N_inf from analyze_approach_law), with the relevant
                 references. In 3+1 the local rate slope *decays* with T —
                 2.37 at T~180 down to ~2.33 at T~500, converging onto 7/3
                 from above while the state exponent sits below at 2.318.
  right column = the fitted process exponent under both knobs: the curve-tail
                 length used to measure the observable (x axis) and the
                 T window fitted (series). The tail knob barely matters
                 (~0.006); the T window moves the answer by ~0.02 because of
                 the left panel's drift — a single "rate exponent +/- err"
                 was never well-posed.

Usage::

    uv run python -m plots.plot_rate_exponent --out rate_exponent.png
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from analysis.analyze_approach_law import ceilings, load_curve

#: 3+1 stores feeding the log-rate observable (the full 1e-6 ladder).
RATE_DBS_3D = (
    "data/pack/pack3d_e6.db",
    "data/converge/converge3d_e6.db",
    "data/converge/converge3d_e6ext.db",
)
#: 2+1 stores feeding the Feder-ceiling observable.
CEIL_DBS_2D = ("data/converge/converge2d_e8.db", "data/converge/converge2d_e9.db")
#: T windows fitted, per dim (label, tmin, tmax).
WINDOWS = {
    3: (("160-360", 160, 360), ("160-520", 160, 520), ("400-520 ext", 400, 520)),
    2: (("100-300", 100, 300), ("180-300", 180, 300)),
}
#: Curve-tail lengths (decades of attempts) over which the observable is fit.
TAILS = (1.0, 1.5, 2.0, 2.5, 3.0)
#: The convention tail used by the left panel and prior quotes.
TAIL_DEFAULT = 2.0
#: State-exponent window fits (plots.plot_converge) drawn for contrast.
STATE_FIT = {3: 2.3177, 2: 1.3927}
#: Readable T ticks for the (log-x) left panels.
T_TICKS = {3: (20, 40, 80, 160, 240, 360, 520), 2: (20, 40, 80, 140, 220, 300)}


def lograte_cells(tail_decades: float) -> dict[int, tuple[float, float]]:
    """Per-T mean/SEM of the 3+1 log-growth rate b = dN/d ln t."""
    by_t: dict[int, list[float]] = {}
    for db in RATE_DBS_3D:
        rows = sqlite3.connect(db).execute(
            "select t, curve_path from runs where status='done' and terms=2"
        )
        for t, path in rows:
            tt, nn = load_curve(Path(path))
            tail = tt >= tt[-1] / 10**tail_decades
            a_mat = np.vstack([np.ones_like(tt[tail]), np.log(tt[tail])]).T
            (_, b), *_ = np.linalg.lstsq(a_mat, nn[tail], rcond=None)
            by_t.setdefault(t, []).append(float(b))
    return {
        t: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
        for t, v in sorted(by_t.items())
        if len(v) > 1
    }


def ceiling_cells(tail_decades: float) -> dict[int, tuple[float, float]]:
    """Per-T mean/SEM of the 2+1 Feder ceiling N_inf."""
    ceil = ceilings(
        [Path(db) for db in CEIL_DBS_2D], 2, "nyq", tail_decades=tail_decades
    )
    return {t: (v[0], v[1]) for t, v in ceil.items()}


def local_slopes(
    c: dict[int, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-point slope between adjacent rungs, at the geometric-mean T."""
    ts = sorted(c)
    mid, slope, err = [], [], []
    for a, b in zip(ts, ts[1:]):
        (n1, e1), (n2, e2) = c[a], c[b]
        lr = np.log(b / a)
        mid.append(np.sqrt(a * b))
        slope.append(np.log(n2 / n1) / lr)
        err.append(np.hypot(e1 / n1, e2 / n2) / lr)
    return np.array(mid), np.array(slope), np.array(err)


def window_fit(
    c: dict[int, tuple[float, float]], tmin: int, tmax: int
) -> tuple[float, float] | None:
    """Weighted log-log fit of the observable over tmin <= T <= tmax."""
    ts = np.array([t for t in c if tmin <= t <= tmax])
    if len(ts) < 3:
        return None
    ns = np.array([c[t][0] for t in ts])
    sig = np.array([c[t][1] / c[t][0] for t in ts])
    p, cov = np.polyfit(np.log(ts), np.log(ns), 1, w=1 / sig, cov=True)
    return float(p[0]), float(np.sqrt(cov[0][0]))


def plot(out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))

    for row, dim in enumerate((3, 2)):
        ax_slope, ax_sys = axes[row]
        cells_of = lograte_cells if dim == 3 else ceiling_cells
        obs = "log-rate b(T)" if dim == 3 else "Feder ceiling N∞(T)"

        c = cells_of(TAIL_DEFAULT)
        mid, slope, err = local_slopes(c)
        ax_slope.errorbar(
            mid,
            slope,
            yerr=err,
            fmt="s",
            color="tab:green",
            ms=5,
            capsize=2,
            label=f"local slope of {obs} (tail={TAIL_DEFAULT} decades)",
        )
        if dim == 3:
            ax_slope.axhline(7 / 3, ls=":", color="gray", lw=1.2)
            ax_slope.annotate(
                "7/3",
                (0.02, 7 / 3),
                xycoords=("axes fraction", "data"),
                fontsize=8.5,
                color="gray",
                va="bottom",
            )
        else:
            ax_slope.axhline(1.434, ls=":", color="gray", lw=1.2)
            ax_slope.annotate(
                "D∞ = 1.434 (Feder)",
                (0.02, 1.434),
                xycoords=("axes fraction", "data"),
                fontsize=8.5,
                color="gray",
                va="bottom",
            )
        ax_slope.axhline(
            STATE_FIT[dim],
            ls="--",
            color="tab:blue",
            lw=1.0,
            alpha=0.6,
        )
        ax_slope.annotate(
            f"state window fit {STATE_FIT[dim]:.3f} (1e-6)",
            (0.02, STATE_FIT[dim]),
            xycoords=("axes fraction", "data"),
            fontsize=8.5,
            color="tab:blue",
            va="top",
        )
        ax_slope.set_xscale("log")
        ax_slope.xaxis.set_major_locator(mticker.FixedLocator(T_TICKS[dim]))
        ax_slope.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_slope.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax_slope.set_xlabel("T (geometric mean of the rung pair)")
        ax_slope.set_ylabel("local exponent")
        ax_slope.set_title(f"{dim}+1: PROCESS local exponent vs resolution")
        ax_slope.grid(True, which="both", alpha=0.3)
        ax_slope.legend(fontsize=8.5, loc="lower right")

        for (label, tmin, tmax), color in zip(
            WINDOWS[dim], ("tab:blue", "tab:red", "tab:purple")
        ):
            xs, ys, es = [], [], []
            for td in TAILS:
                f = window_fit(cells_of(td), tmin, tmax)
                if f is None:
                    continue
                xs.append(td)
                ys.append(f[0])
                es.append(f[1])
                if td == TAIL_DEFAULT:
                    print(
                        f"{dim}+1 {obs} T={label} tail={td}: {f[0]:.4f} +/- {f[1]:.4f}"
                    )
            ax_sys.errorbar(
                xs, ys, yerr=es, fmt="o-", color=color, ms=4.5, capsize=2, label=label
            )
        if dim == 3:
            ax_sys.axhline(7 / 3, ls=":", color="gray", lw=1.2)
            ax_sys.annotate(
                "7/3",
                (0.02, 7 / 3),
                xycoords=("axes fraction", "data"),
                fontsize=8.5,
                color="gray",
                va="bottom",
            )
        ax_sys.axhline(STATE_FIT[dim], ls="--", color="tab:blue", lw=1.0, alpha=0.6)
        ax_sys.set_xlabel("curve-tail length used for the observable (decades)")
        ax_sys.set_ylabel("fitted process exponent")
        ax_sys.set_title(f"{dim}+1: PROCESS exponent — tail and T-window knobs")
        ax_sys.grid(True, alpha=0.3)
        ax_sys.legend(fontsize=8.5, title="T window")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rate_exponent.png"))
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
