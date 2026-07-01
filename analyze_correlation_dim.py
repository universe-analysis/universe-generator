"""Correlation dimension of the braid -- a jamming-free fractal dimension.

The box-counting dimension D (N_sat ~ T^D) needs the *jammed count*, which is
permanently cutoff-limited at high T. The correlation dimension is a second,
independent definition of fractal dimension that reads the *spatial arrangement*
of a single packing instead of the way the count scales, so it works on any
snapshot and is far less sensitive to how deeply the box was jammed.

For a point set, the correlation integral is

    C(r) = (# unordered pairs closer than r) / (N (N-1) / 2)

and for a self-similar fractal C(r) ~ r^D2 over the scaling window between the
collision diameter CELL = 2/T and the box size. The slope d log C / d log r in
that window is the correlation dimension D2.

Two point sets are measured here, both built from a `--dump-params` file:

  * "turnaround" -- each worldline's 3D comoving position at z = pi/2,
    X = a*sin(b*pi/2) + a2.  This is the matter distribution at maximum
    expansion (the reviewer's structure-factor object); its D2 is the fractal
    dimension of that distribution.

  * "braid" -- each worldline sampled at many conformal times, giving points
    (z, X, Y, W) in comoving spacetime (z rescaled so a cell is cubic).  This is
    the same geometric object the packing exponent describes: a 1D curve swept
    by each worldline, so its spacetime D2 should land near D+1 and (D2 - 1) is
    the apples-to-apples analog of the box-counting packing D.

A box-counting cross-check on the same cloud (occupied cells vs cell size,
within one packing) is also reported; if it disagrees with C(r)'s slope the set
is multifractal.

Usage::

    python analyze_correlation_dim.py --params data/params/params_T120.csv \
        --T 120 --out corr_dim.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_params(path: str) -> tuple[list, list, list]:
    """Return (a, b, a2) as three lists of per-axis numpy arrays."""
    rows = list(csv.DictReader(open(path)))

    def col(k: str) -> np.ndarray:
        return np.array([float(r[k]) for r in rows])

    a = [col("ax"), col("ay"), col("aw")]
    b = [col("bx"), col("by"), col("bw")]
    a2 = [col("ax2"), col("ay2"), col("aw2")]
    return a, b, a2


def turnaround_cloud(a: list, b: list, a2: list) -> np.ndarray:
    """3D comoving positions at z = pi/2 -- shape (N, 3)."""
    half_pi = np.pi / 2.0
    axes = [a[i] * np.sin(b[i] * half_pi) + a2[i] for i in range(3)]
    return np.column_stack(axes)


def braid_cloud(a: list, b: list, a2: list, samples: int, n_lines: int) -> np.ndarray:
    """Spacetime points (z', X, Y, W) from worldlines sampled over z.

    z is rescaled to z' = z * (2/pi) so its span matches the comoving box (2),
    making a CELL cubic. A subsample of worldlines keeps the point count
    tractable.
    """
    n_total = len(a[0])
    take = min(n_lines, n_total)
    idx = np.linspace(0, n_total - 1, take).astype(int)
    zs = np.linspace(0.02, np.pi - 0.02, samples)
    sin_z = np.sin(zs)
    pts = []
    for z, sz in zip(zs, sin_z):
        z_scaled = z * (2.0 / np.pi)
        block = [np.full(take, z_scaled)]
        for i in range(3):
            x = a[i][idx] * np.sin(b[i][idx] * z) + a2[i][idx] * np.sin(z)
            block.append(x / sz)  # comoving X = x / sin(z)
        pts.append(np.column_stack(block))
    return np.vstack(pts)


def correlation_integral(
    pts: np.ndarray, radii: np.ndarray, n_centers: int = 4000, p: float = 2.0
) -> np.ndarray:
    """C(r) for each r: fraction of point pairs within distance r.

    Estimated by center-sampling -- count neighbours of a random subset of
    "center" points against the full cloud -- so the cost is O(n_centers * avg
    neighbours) instead of the O(N^2) of an all-pairs count. Unbiased because
    every center sees all N-1 partners; only the number of centers is reduced.

    ``p`` selects the metric / probe shape: p=2 counts within Euclidean spheres,
    p=inf within max-norm cubes (side 2r). The fractal exponent is the same for
    both; only the prefactor differs -- so comparing them tests probe-shape
    independence.
    """
    tree = cKDTree(pts)
    n = len(pts)
    take = min(n_centers, n)
    step = max(1, n // take)
    centers = cKDTree(pts[::step])
    n_c = centers.n
    counts = centers.count_neighbors(tree, radii, p=p)  # ordered, incl. self
    return (counts - n_c) / (n_c * (n - 1.0))


def box_count(pts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Number of occupied cells of each edge length (single-packing ruler)."""
    occupied = []
    for s in sizes:
        keys = np.floor(pts / s).astype(np.int64)
        occupied.append(len(np.unique(keys, axis=0)))
    return np.array(occupied, dtype=float)


def local_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Centered d log y / d log x on log-spaced samples."""
    lx, ly = np.log(x), np.log(y)
    return np.gradient(ly, lx)


def fit_window(
    x: np.ndarray, y: np.ndarray, lo: float, hi: float
) -> tuple[float, np.ndarray]:
    """Least-squares slope of log y vs log x over lo <= x <= hi."""
    mask = (x >= lo) & (x <= hi) & (y > 0)
    slope, _ = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return slope, mask


# Fixed *physical* scale window for every T, so the dimension is read over the
# same band of r regardless of resolution -- a like-for-like cross-T comparison.
FIT_LO, FIT_HI = 0.08, 0.5
R_MAX = 0.7


def turnaround_d2(
    params: str, T: int, p: float = 2.0
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (radii, C(r), correlation D2, box-counting D) for one packing.

    ``p`` is the correlation-integral metric: 2 = spheres, inf = cubes.
    """
    a, b, a2 = load_params(params)
    cell = 2.0 / T
    radii = np.logspace(np.log10(cell), np.log10(R_MAX), 40)
    cloud = turnaround_cloud(a, b, a2)
    c = correlation_integral(cloud, radii, p=p)
    d2, _ = fit_window(radii, c, FIT_LO, FIT_HI)
    box_sizes = np.logspace(np.log10(cell), np.log10(R_MAX), 18)
    db, _ = fit_window(box_sizes, box_count(cloud, box_sizes), FIT_LO, FIT_HI)
    return radii, c, d2, -db


def shapes_compare(specs: list[tuple[str, int]], out: Path) -> None:
    """Correlation dimension with spheres (L2) vs cubes (Linf), across T.

    Same expanding-neighbourhood count, two probe shapes. Agreement is the
    test that the dimension is a property of the cloud, not of the ruler.
    """
    import matplotlib.pyplot as plt

    specs = sorted(specs, key=lambda s: s[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ts, d2_sphere, d2_cube = [], [], []
    print("Correlation dimension, spheres vs cubes:")
    for params, T in specs:
        radii, c_s, ds, _ = turnaround_d2(params, T, p=2.0)
        _, c_c, dc, _ = turnaround_d2(params, T, p=np.inf)
        ts.append(T)
        d2_sphere.append(ds)
        d2_cube.append(dc)
        print(f"  T={T:>3}:  sphere D2 = {ds:.3f}   cube D2 = {dc:.3f}")
        if T == specs[-1][1]:  # slope overlay for the highest, cleanest T
            ax1.semilogx(radii, local_slope(radii, c_s), "o-",
                         color="tab:blue", ms=4, label=f"sphere (L2), T={T}")
            ax1.semilogx(radii, local_slope(radii, c_c), "s--",
                         color="tab:red", ms=4, label=f"cube (Linf), T={T}")
    ax1.axhline(2.8, color="black", ls=":", lw=1, label="converged D ~ 2.8")
    ax1.axvspan(FIT_LO, FIT_HI, color="gray", alpha=0.12, label="fit window")
    ax1.set_xlabel("r")
    ax1.set_ylabel("local slope = D2(r)")
    ax1.set_ylim(0, 5)
    ax1.set_title("Scale-resolved D2: sphere vs cube probe")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(ts, d2_sphere, "o-", color="tab:blue", label="sphere (L2) D2")
    ax2.plot(ts, d2_cube, "s-", color="tab:red", label="cube (Linf) D2")
    ax2.axhline(2.80, color="black", ls="--", lw=1.2, label="converged D ~ 2.80")
    t_clean = min(T for _, T in specs if 2.0 / T < FIT_LO / 2.0)
    ax2.axvspan(min(ts) - 1, t_clean, color="red", alpha=0.08,
                label="CELL intrudes on fit window")
    ax2.set_xlabel("T (resolution)")
    ax2.set_ylabel("correlation dimension D2")
    ax2.set_title("Both probe shapes converge to the same D")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def converge_plot(
    ts: list, d2s: list, dbs: list, out: Path, metric: str = "spheres"
) -> None:
    """Plot corr-D2 and box-D vs T -- the two estimators closing on one value.

    Below T ~ 2/FIT_LO/2 the cell CELL = 2/T intrudes into the fixed fit
    window and inflates D2, so those points are shaded as contaminated.
    ``metric`` names the correlation-integral probe shape (spheres / cubes).
    """
    import matplotlib.pyplot as plt

    ts = np.array(ts)
    clean = 2.0 / ts < FIT_LO / 2.0  # CELL comfortably below the window
    converged = float(np.mean(np.array(d2s)[clean][-3:])) if clean.any() else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, d2s, "o-", color="tab:blue",
            label=f"correlation D2, {metric} (from above)")
    ax.plot(ts, dbs, "s-", color="tab:orange",
            label="box-counting D, cubes (from below)")

    # Label every point with its value: D2 above the blue points, box D below
    # the orange ones, so the two never collide even where the series are close.
    for t, d2, db in zip(ts, d2s, dbs):
        ax.annotate(f"{d2:.2f}", (t, d2), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="tab:blue")
        ax.annotate(f"{db:.2f}", (t, db), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color="tab:orange")
    ax.axhline(converged, color="black", ls="--", lw=1.2,
               label=f"converged D ~ {converged:.2f}")
    t_clean = ts[clean].min() if clean.any() else ts.max()
    ax.axvspan(ts.min() - 1, t_clean, color="red", alpha=0.08,
               label="CELL intrudes on fit window")
    ax.set_xlabel("T (resolution)")
    ax.set_ylabel("dimension")
    ax.set_title(f"Two dimension estimators converge as resolution rises"
                 f"  ({metric} + cubes)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"converged D2 (clean high-T mean) = {converged:.3f}")
    print(f"wrote {out}")


def compare(specs: list[tuple[str, int]], out: Path, p: float = 2.0) -> None:
    """Overlay turnaround D2(r) for several T to test resolution-independence.

    ``p`` selects the correlation probe shape: 2 = spheres, inf = cubes.
    """
    import matplotlib.pyplot as plt

    metric = "cubes" if np.isinf(p) else "spheres"
    specs = sorted(specs, key=lambda s: s[1])  # by T, low -> high
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.colormaps["viridis"]
    n = len(specs)
    ts, d2s, dbs = [], [], []
    print(f"Turnaround correlation dimension across resolution ({metric}):")
    for i, (params, T) in enumerate(specs):
        color = cmap(i / max(1, n - 1))
        radii, c, d2, db = turnaround_d2(params, T, p=p)
        print(f"  T={T:>3}:  corr D2 = {d2:.3f}   box D = {db:.3f}")
        ts.append(T)
        d2s.append(d2)
        dbs.append(db)
        ax1.loglog(radii, c, "-", color=color, lw=1.5, label=f"T={T} (D2={d2:.2f})")
        ax2.semilogx(radii, local_slope(radii, c), "-", color=color, lw=1.5,
                     label=f"T={T}")
    converge_plot(ts, d2s, dbs, out.with_name(out.stem + "_converge.png"), metric)
    for ax in (ax1, ax2):
        ax.axvspan(FIT_LO, FIT_HI, color="gray", alpha=0.12, label="fit window")
        ax.set_xlabel("r")
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("C(r)")
    ax1.set_title("Turnaround correlation integral vs resolution")
    ax1.legend(fontsize=8, loc="lower right")
    ax2.axhline(2.5, color="black", ls="--", lw=1, label="box-counting D ~ 2.5")
    ax2.set_ylabel("local slope = D2(r)")
    ax2.set_ylim(0, 5)
    ax2.set_title("Scale-resolved dimension (plateau = jamming-free D)")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="data/params/params_T120.csv")
    parser.add_argument("--T", type=int, default=120)
    parser.add_argument("--samples", type=int, default=240,
                        help="z-samples per worldline for the braid cloud")
    parser.add_argument("--lines", type=int, default=4000,
                        help="worldline subsample for the braid cloud")
    parser.add_argument("--compare", default="",
                        help="comma list T:path,... to overlay turnaround D2(r)")
    parser.add_argument("--shapes", default="",
                        help="comma list T:path,... sphere-vs-cube probe compare")
    parser.add_argument("--cubes", action="store_true",
                        help="use cubic (L-inf) neighbourhoods in --compare")
    parser.add_argument("--out", type=Path, default=Path("corr_dim.png"))
    args = parser.parse_args()

    def parse_specs(text: str) -> list[tuple[str, int]]:
        out = []
        for item in text.split(","):
            t_str, path = item.split(":")
            out.append((path, int(t_str)))
        return out

    if args.compare:
        compare(parse_specs(args.compare), args.out,
                p=np.inf if args.cubes else 2.0)
        return
    if args.shapes:
        shapes_compare(parse_specs(args.shapes), args.out)
        return
    import matplotlib.pyplot as plt

    a, b, a2 = load_params(args.params)
    cell = 2.0 / args.T
    print(f"loaded {len(a[0])} worldlines from {args.params}  (CELL = {cell:.4g})")

    # Scaling window: a few cells up to a fraction of the box. The radius is
    # capped just past the window -- near r ~ box size almost every pair is
    # inside r, which is both uninformative and very slow to count.
    fit_lo, fit_hi = 3.0 * cell, 0.5
    r_max = 0.7
    radii = np.logspace(np.log10(cell), np.log10(r_max), 40)

    # --- turnaround matter distribution (3D) ---
    tcloud = turnaround_cloud(a, b, a2)
    c_turn = correlation_integral(tcloud, radii)
    d2_turn, mask_t = fit_window(radii, c_turn, fit_lo, fit_hi)

    box_sizes = np.logspace(np.log10(cell), np.log10(r_max), 18)
    nbox_turn = box_count(tcloud, box_sizes)
    db_turn, _ = fit_window(box_sizes, nbox_turn, fit_lo, fit_hi)
    db_turn = -db_turn  # N_box ~ size^(-D)

    # --- full braid in comoving spacetime (4D) ---
    bcloud = braid_cloud(a, b, a2, args.samples, args.lines)
    c_braid = correlation_integral(bcloud, radii)
    d2_braid, mask_b = fit_window(radii, c_braid, fit_lo, fit_hi)

    print(f"\nturnaround cloud:  N = {len(tcloud)} points in 3D")
    print(f"  correlation dimension D2 = {d2_turn:.3f}"
          f"  (window {fit_lo:.3g}..{fit_hi:.3g})")
    print(f"  box-counting    D  = {db_turn:.3f}  (same window)")
    print(f"\nbraid spacetime cloud:  N = {len(bcloud)} points in 4D")
    print(f"  correlation dimension D2 = {d2_braid:.3f}")
    print(f"  packing-D analog (D2 - 1 for the curve direction) = {d2_braid - 1:.3f}")

    # --- plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.loglog(radii, c_turn, "o-", color="tab:blue", ms=4,
               label=f"turnaround C(r), D2={d2_turn:.2f}")
    ax1.loglog(radii, c_braid, "s-", color="tab:orange", ms=4,
               label=f"braid spacetime C(r), D2={d2_braid:.2f}")
    ax1.axvline(cell, color="gray", ls=":", lw=1, label="CELL = 2/T")
    ax1.axvspan(fit_lo, fit_hi, color="gray", alpha=0.12, label="fit window")
    ax1.set_xlabel("r")
    ax1.set_ylabel("C(r) = pair fraction within r")
    ax1.set_title(f"Correlation integral (T={args.T})")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, which="both", alpha=0.3)

    # Local slope: the dimension as a function of scale.
    ax2.semilogx(radii, local_slope(radii, c_turn), "o-", color="tab:blue",
                 ms=4, label="turnaround D2(r)")
    ax2.semilogx(radii, local_slope(radii, c_braid), "s-", color="tab:orange",
                 ms=4, label="braid spacetime D2(r)")
    ax2.axhline(2.5, color="tab:green", ls="--", lw=1,
                label="box-counting D ~ 2.5")
    ax2.axvline(cell, color="gray", ls=":", lw=1, label="CELL")
    ax2.axvspan(fit_lo, fit_hi, color="gray", alpha=0.12)
    ax2.set_xlabel("r")
    ax2.set_ylabel("local slope  d log C / d log r")
    ax2.set_title("Scale-resolved dimension")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
