"""Wave test: is the exponent plateau a plateau, or the crest of a slow wave?

Chris's challenge on the CONVERGE grid: the local state exponent
d log N / d log T could be a slow oscillation in log T that happens to
crest inside our window, making the "converged" value an accident of where
the ladder ends. Three attacks, one dimension-symmetric figure:

1. **Window flatness.** On the converged window only (where the plateau is
   claimed), compare a constant against the best sinusoid in log10 T
   (periods 0.3-3 decades, all phases). If no wave beats the constant by
   dchi2 > 4, any wave crest within the window is bounded by the best-fit
   |A|.

2. **Transient vs transient+wave on the full range.** The low-T rise is
   equally consistent with a saturating stopping transient
   slope(T) = D - c T^(-k) or with the rising flank of a wave. Fit both;
   if the saturating model already gives chi2/dof ~ 1, there is no
   residual structure left for a wave to explain.

3. **Process-level observable.** The stopped N(T) carries stopping
   kinematics; the process-level object does not. In 3+1 (logarithmic
   approach) that is the log-growth rate b(T) = dN/d ln t; in 2+1 (Feder
   approach) it is the extrapolated ceiling N_inf(T) from
   analyze_approach_law. Same window test applied there.

What this cannot do: exclude waves with period much longer than the
observed window. The T = 400-520 extension (converge3d_e6ext, stitched
into the 1e-6 arm) widened the 3+1 window to half a decade; beyond that,
only more T can.

Usage::

    uv run python -m analysis.analyze_wave_test --out figures/wave_test.png
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from analysis.analyze_approach_law import ceilings, load_curve

#: Per dimension: (cutoff label, stores to stitch, color) for the state view.
LADDERS: dict[int, tuple[tuple[str, tuple[str, ...], str], ...]] = {
    3: (
        (
            "1e-6",
            (
                "data/pack/pack3d_e6.db",
                "data/converge/converge3d_e6.db",
                "data/converge/converge3d_e6ext.db",
            ),
            "tab:blue",
        ),
        (
            "1e-7",
            ("data/pack/pack3d_e7.db", "data/converge/converge3d_e7.db"),
            "tab:red",
        ),
    ),
    2: (
        ("1e-6", ("data/pack/pack2d_e6.db",), "tab:blue"),
        ("1e-7", ("data/pack/pack2d_e7.db",), "tab:red"),
        ("1e-8", ("data/converge/converge2d_e8.db",), "tab:purple"),
    ),
}
#: Stores feeding the process-level observable per dimension.
PROCESS_3D = (
    "1e-6 log-rate",
    ("data/pack/pack3d_e6.db", "data/converge/converge3d_e6.db"),
    "tab:green",
)
PROCESS_2D = (
    "Feder ceilings (e8+e9)",
    ("data/converge/converge2d_e8.db", "data/converge/converge2d_e9.db"),
    "tab:green",
)
#: Converged/claimed-plateau window start per dimension.
WINDOW = {3: 160, 2: 100}
#: Readable T ticks for the (log-x) panels, matching plots.plot_converge.
T_TICKS = {3: (20, 40, 80, 160, 240, 360, 520), 2: (20, 40, 80, 140, 220, 300)}
#: Tail window (decades of attempts) for the 3+1 log-rate estimate.
RATE_TAIL_DECADES = 2.0
#: Scanned wave periods, in decades of T.
PERIODS = np.linspace(0.3, 3.0, 55)


def state_cells(dbs: tuple[str, ...]) -> dict[int, tuple[float, float]]:
    """Per-T mean/SEM of the stopped count N."""
    by_t: dict[int, list[float]] = {}
    for db in dbs:
        for t, n in sqlite3.connect(db).execute(
            "select t, n_final from runs where status='done' and terms=2"
        ):
            by_t.setdefault(t, []).append(float(n))
    return _mean_sem(by_t)


def lograte_cells(dbs: tuple[str, ...]) -> dict[int, tuple[float, float]]:
    """Per-T mean/SEM of the log-growth rate b = dN/d ln t (3+1 only)."""
    by_t: dict[int, list[float]] = {}
    for db in dbs:
        rows = sqlite3.connect(db).execute(
            "select t, curve_path from runs where status='done' and terms=2"
        )
        for t, path in rows:
            tt, nn = load_curve(Path(path))
            tail = tt >= tt[-1] / 10**RATE_TAIL_DECADES
            a_mat = np.vstack([np.ones_like(tt[tail]), np.log(tt[tail])]).T
            (_, b), *_ = np.linalg.lstsq(a_mat, nn[tail], rcond=None)
            by_t.setdefault(t, []).append(float(b))
    return _mean_sem(by_t)


def ceiling_cells(dbs: tuple[str, ...]) -> dict[int, tuple[float, float]]:
    """Per-T mean/SEM of the Feder-extrapolated jam ceiling (2+1 only)."""
    ceil = ceilings([Path(db) for db in dbs], 2, "nyq")
    return {t: (v[0], v[1]) for t, v in ceil.items()}


def _mean_sem(by_t: dict[int, list[float]]) -> dict[int, tuple[float, float]]:
    return {
        t: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
        for t, v in sorted(by_t.items())
        if len(v) > 1
    }


def local_slopes(
    c: dict[int, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-point log-slope between adjacent rungs at the geometric-mean T."""
    ts = sorted(c)
    mid, slope, err = [], [], []
    for a, b in zip(ts, ts[1:]):
        (n1, e1), (n2, e2) = c[a], c[b]
        lr = np.log(b / a)
        mid.append(np.sqrt(a * b))
        slope.append(np.log(n2 / n1) / lr)
        err.append(np.hypot(e1 / n1, e2 / n2) / lr)
    return np.array(mid), np.array(slope), np.array(err)


def best_wave(
    mid: np.ndarray, slope: np.ndarray, err: np.ndarray
) -> tuple[float, float]:
    """Best sinusoid vs constant: (best-fit |A|, its dchi2 over constant)."""
    x = np.log10(mid)
    w = 1 / err**2
    const = np.sum(w * slope) / np.sum(w)
    chi2_const = float(np.sum(w * (slope - const) ** 2))
    best_a, best_dchi = 0.0, 0.0
    for period in PERIODS:
        for phase in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            s = np.sin(2 * np.pi * x / period + phase)
            sw = np.sum(w * s)
            a_den = float(np.sum(w * s * s) - sw**2 / np.sum(w))
            if a_den <= 1e-12:
                continue
            amp = float(np.sum(w * s * slope) - sw * const) / a_den
            resid = slope - (const + amp * (s - sw / np.sum(w)))
            dchi = chi2_const - float(np.sum(w * resid**2))
            if dchi > best_dchi:
                best_dchi, best_a = dchi, abs(amp)
    return best_a, best_dchi


def saturating_fit(
    mid: np.ndarray, slope: np.ndarray, err: np.ndarray
) -> tuple[float, float]:
    """Fit slope(T) = D - c T^(-k) (k scanned): (chi2/dof, fitted D)."""
    w = 1 / err**2
    best: tuple[float, float] | None = None
    for k in np.linspace(0.2, 2.5, 47):
        x = mid ** (-k)
        a_mat = np.vstack([np.ones_like(x), -x]).T
        coef, _, _, _ = np.linalg.lstsq(
            a_mat * np.sqrt(w)[:, None], slope * np.sqrt(w), rcond=None
        )
        d_fit, c = float(coef[0]), float(coef[1])
        if c <= 0:
            continue
        chi2 = float(np.sum(w * (slope - (d_fit - c * x)) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, d_fit)
    if best is None:
        return float("nan"), float("nan")
    dof = max(len(mid) - 3, 1)
    return best[0] / dof, best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("wave_test.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))

    for row, dim in enumerate((3, 2)):
        ax_state, ax_proc = axes[row]
        window = WINDOW[dim]

        for label, dbs, color in LADDERS[dim]:
            c = state_cells(dbs)
            mid, slope, err = local_slopes(c)
            in_win = mid >= window * 0.9
            amp_w, dchi_w = best_wave(mid[in_win], slope[in_win], err[in_win])
            chi2dof_sat, d_sat = saturating_fit(mid, slope, err)
            verdict = "structure" if dchi_w > 4 else "no detectable wave"
            ax_state.errorbar(
                mid,
                slope,
                yerr=err,
                fmt="o",
                color=color,
                ms=4.5,
                capsize=2,
                alpha=0.85,
                label=f"{label}: window {verdict} (dchi2={dchi_w:.1f}); "
                f"smooth transient chi2/dof={chi2dof_sat:.1f} "
                "(asymptote unconstrained)",
            )
            print(
                f"{dim}+1 state @{label}: window best |A|={amp_w:.4f} "
                f"dchi2={dchi_w:.1f} ({verdict}); saturating-transient "
                f"chi2/dof={chi2dof_sat:.2f} D={d_sat:.4f}"
            )
        ax_state.axvline(window, ls=":", color="gray", lw=1)
        ax_state.annotate(
            "window",
            (window, 0.02),
            xycoords=("data", "axes fraction"),
            fontsize=8,
            color="gray",
            rotation=90,
            va="bottom",
        )
        ax_state.set_xscale("log")
        ax_state.xaxis.set_major_locator(mticker.FixedLocator(T_TICKS[dim]))
        ax_state.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_state.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax_state.set_xlabel("T (geometric mean of the rung pair)")
        ax_state.set_ylabel("local exponent")
        ax_state.set_title(f"{dim}+1: STATE N(T) — window wave test")
        ax_state.grid(True, which="both", alpha=0.3)
        ax_state.legend(fontsize=7.5, loc="lower right")

        proc_label, proc_dbs, proc_color = PROCESS_3D if dim == 3 else PROCESS_2D
        c = lograte_cells(proc_dbs) if dim == 3 else ceiling_cells(proc_dbs)
        mid, slope, err = local_slopes(c)
        amp_f, dchi_f = best_wave(mid, slope, err)
        verdict = "structure" if dchi_f > 4 else "no detectable wave"
        ax_proc.errorbar(
            mid,
            slope,
            yerr=err,
            fmt="s",
            color=proc_color,
            ms=5,
            capsize=2,
            label=f"{proc_label}: {verdict} (dchi2={dchi_f:.1f})",
        )
        print(
            f"{dim}+1 process ({proc_label}): full-range best "
            f"|A|={amp_f:.4f} dchi2={dchi_f:.1f} ({verdict})"
        )
        ax_proc.set_xscale("log")
        ax_proc.xaxis.set_major_locator(mticker.FixedLocator(T_TICKS[dim]))
        ax_proc.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_proc.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax_proc.set_xlabel("T (geometric mean of the rung pair)")
        ax_proc.set_ylabel("local exponent")
        ax_proc.set_title(
            f"{dim}+1: PROCESS "
            f"({'log-rate b(T)' if dim == 3 else 'Feder ceiling N∞(T)'})"
        )
        ax_proc.grid(True, which="both", alpha=0.3)
        ax_proc.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
