"""Turnaround correlation dimension -- core math, seed-averaging, and report.

This is the canonical home for the jamming-free fractal-dimension measurement
(see PHYSICS_FINDINGS section 8). The box-counting exponent N_sat ~ T^D needs
the jammed count, which is cutoff-limited at high T; the correlation dimension
instead reads the spatial arrangement of a single packing's turnaround cloud,
so it never touches the jamming wall.

For a packing's turnaround positions (each worldline's comoving X at z=pi/2),
the correlation integral

    C(r) = (pairs within r) / (all pairs) ~ r^D2

over a fixed physical window gives D2. ``measure_dump`` reads one parameter
dump and returns sphere (L2), cube (L-inf), and box-counting estimates;
``aggregate`` seed-averages across a campaign's dumps to mean +/- SEM per T;
``plot_convergence`` draws the error-barred convergence.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

#: Fixed physical scale window (same band of r for every T) and probe ceiling.
FIT_LO, FIT_HI = 0.08, 0.5
R_MAX = 0.7
#: Random center points for the correlation-integral estimator.
N_CENTERS = 4000
#: Dump filenames look like d3_nyq_T120_s2.csv (or ..._s2_eu.csv for variants).
_NAME_RE = re.compile(r"_T(?P<t>\d+)_s(?P<seed>\d+)")


def wrap_unit(x: np.ndarray) -> np.ndarray:
    """Wrap comoving coordinates onto the torus fundamental domain [-1, 1).

    Torus-model (``--torus``) worldlines can excurse beyond |X| = 1 (the free
    sin1 offset plus a full-budget wiggle); the engine stores them wrapped, so
    any reconstruction from raw dump parameters must apply the same wrap.
    """
    return x - 2.0 * np.floor((x + 1.0) * 0.5)


def load_turnaround_cloud(path: str | Path, wrap: bool = False) -> np.ndarray:
    """Load a parameter dump and return its z=pi/2 comoving cloud (N, d).

    Dimension is inferred from the columns: a 3+1 dump carries the third (w)
    axis, a 2+1 dump does not. At the turnaround sin(z)=1, so the comoving
    position is just X = a*sin(b*pi/2) + a2 on each axis. ``wrap`` applies the
    torus wrap onto [-1, 1) (required for torus-model dumps).
    """
    rows = list(csv.DictReader(open(path)))
    fields = set(rows[0].keys()) if rows else set()
    half_pi = np.pi / 2.0

    def col(k: str) -> np.ndarray:
        return np.array([float(r[k]) for r in rows])

    triples = [("ax", "bx", "ax2"), ("ay", "by", "ay2")]
    if "aw" in fields:  # 3+1 dump
        triples.append(("aw", "bw", "aw2"))
    axes = [col(a) * np.sin(col(b) * half_pi) + col(a2) for a, b, a2 in triples]
    cloud = np.column_stack(axes)
    return wrap_unit(cloud) if wrap else cloud


def correlation_integral(
    pts: np.ndarray, radii: np.ndarray, n_centers: int = N_CENTERS, p: float = 2.0
) -> np.ndarray:
    """C(r): fraction of point pairs within distance r, by center-sampling.

    ``p`` selects the probe shape: 2 = Euclidean spheres, inf = max-norm cubes.
    The fractal exponent is metric-independent; only the prefactor differs.
    """
    tree = KDTree(pts)
    n = len(pts)
    step = max(1, n // min(n_centers, n))
    centers = KDTree(pts[::step])
    n_c = centers.n
    counts = centers.count_neighbors(tree, radii, p=p)  # ordered, incl. self
    return (counts - n_c) / (n_c * (n - 1.0))


def box_count(pts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Occupied-cell count at each cubic edge length (single-packing ruler)."""
    return np.array(
        [len(np.unique(np.floor(pts / s).astype(np.int64), axis=0)) for s in sizes],
        dtype=float,
    )


def fit_slope(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Least-squares slope of log y vs log x over lo <= x <= hi."""
    mask = (x >= lo) & (x <= hi) & (y > 0)
    slope, _ = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(slope)


@dataclass(frozen=True)
class DumpMeasure:
    """The three dimension estimators for one packing."""

    t: int
    seed: int
    d2_sphere: float
    d2_cube: float
    box_d: float


def measure_dump(path: str | Path, t: int, seed: int = 0) -> DumpMeasure:
    """Measure sphere-D2, cube-D2, and box-D for a single parameter dump."""
    cloud = load_turnaround_cloud(path)
    cell = 2.0 / t
    radii = np.logspace(np.log10(cell), np.log10(R_MAX), 40)
    c_sphere = correlation_integral(cloud, radii, p=2.0)
    c_cube = correlation_integral(cloud, radii, p=np.inf)
    sizes = np.logspace(np.log10(cell), np.log10(R_MAX), 18)
    box_d = -fit_slope(sizes, box_count(cloud, sizes), FIT_LO, FIT_HI)
    return DumpMeasure(
        t=t,
        seed=seed,
        d2_sphere=fit_slope(radii, c_sphere, FIT_LO, FIT_HI),
        d2_cube=fit_slope(radii, c_cube, FIT_LO, FIT_HI),
        box_d=box_d,
    )


@dataclass(frozen=True)
class TStat:
    """Seed-averaged estimators at one T: mean and standard error."""

    t: int
    n_seeds: int
    sphere_mean: float
    sphere_sem: float
    cube_mean: float
    cube_sem: float
    box_mean: float
    box_sem: float


def _mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), sem


def aggregate(dumps_dir: str | Path, dim: int = 3, band: str = "nyq") -> list[TStat]:
    """Seed-average every ``d{dim}_{band}_T*_s*.csv`` dump in a directory."""
    by_t: dict[int, list[DumpMeasure]] = {}
    for path in sorted(Path(dumps_dir).glob(f"d{dim}_{band}_T*_s*.csv")):
        m = _NAME_RE.search(path.name)
        if not m:
            continue
        t, seed = int(m["t"]), int(m["seed"])
        by_t.setdefault(t, []).append(measure_dump(path, t, seed))

    stats: list[TStat] = []
    for t in sorted(by_t):
        ms = by_t[t]
        sphere = _mean_sem([m.d2_sphere for m in ms])
        cube = _mean_sem([m.d2_cube for m in ms])
        box = _mean_sem([m.box_d for m in ms])
        stats.append(
            TStat(t, len(ms), sphere[0], sphere[1], cube[0], cube[1], box[0], box[1])
        )
    return stats


def converged_value(stats: list[TStat]) -> float:
    """Mean sphere-D2 over the clean high-T tail (CELL below the fit window)."""
    clean = [s for s in stats if 2.0 / s.t < FIT_LO / 2.0]
    tail = clean[-3:] if len(clean) >= 3 else clean
    return float(np.mean([s.sphere_mean for s in tail])) if tail else 0.0


def plot_convergence(
    stats: list[TStat], out: str | Path, labels: bool = False
) -> Path:
    """Error-barred convergence of the three estimators vs T.

    With ``labels`` set, each point is annotated with its value -- staggered per
    series (spheres above, cubes and box below) so the three converging curves
    stay legible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.array([s.t for s in stats])
    fig, ax = plt.subplots(figsize=(13 if labels else 10, 6))
    # (mean, sem, color, marker, label, label-y-offset in points)
    series = [
        ("sphere_mean", "sphere_sem", "tab:blue", "o", "correlation D2 (spheres)", 9),
        ("cube_mean", "cube_sem", "tab:red", "s", "correlation D2 (cubes)", -13),
        ("box_mean", "box_sem", "tab:orange", "^", "box-counting D (cubes)", -25),
    ]
    for mean_k, sem_k, color, marker, label, dy in series:
        means = np.array([getattr(s, mean_k) for s in stats])
        sems = np.array([getattr(s, sem_k) for s in stats])
        ax.errorbar(ts, means, yerr=sems, fmt=marker + "-", color=color,
                    capsize=3, ms=5, label=label)
        if labels:
            for t, m in zip(ts, means):
                ax.annotate(f"{m:.2f}", (t, m), textcoords="offset points",
                            xytext=(0, dy), ha="center", fontsize=6, color=color)

    converged = converged_value(stats)
    ax.axhline(converged, color="black", ls="--", lw=1.2,
               label=f"converged D ~ {converged:.2f}")
    clean_ts = [s.t for s in stats if 2.0 / s.t < FIT_LO / 2.0]
    if clean_ts:
        ax.axvspan(ts.min() - 2, min(clean_ts), color="red", alpha=0.08,
                   label="CELL intrudes on fit window")
    ax.set_xlabel("T (resolution)")
    ax.set_ylabel("dimension")
    ax.set_title("Turnaround correlation dimension vs resolution "
                 "(seed-averaged, +/- SEM)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
