"""Baked observer frames: reader, observer-view remap, and measurements.

`cuda/frame_bake.cu` bakes past-light-cone frame maps from parameter dumps
(the wrapped viewer's construction, GPU-scaled to campaign dumps) into a
binary ``.frames`` file. This module reads those files, applies the
observer-view remap (static, or moving with aberration + Doppler — the
viewer's draw-time transform), and provides the in-universe observational
measurements: redshift-binned number counts, ghost-image multiplicity, and
sky statistics.

It also carries :func:`bake_reference`, a pure-Python port of the same
crossing-finder, used as the validation oracle for the CUDA baker (small
inputs only — it is deliberately simple and slow).

File layout (little-endian; see frame_bake.cu):
  header    : magic b"BRF1", u32 n_paths, u32 n_observers, u32 n_instants,
              u32 front (0 speed, 1 fit, 2 budget), u32 cheb, f64 speed,
              f64 zmin, i32 wrapK
  observers : (u32 type, i32 path_idx, f64 px, f64 py) * n_observers
  instants  : f64 z_obs, f64 chi_max, i32 K, then per observer
              (f64 ox, f64 oy, f64 betax, f64 betay, u64 n_hits,
               hits (u32 src, f32 dx, f32 dy, f32 chi, f32 z_emit) * n_hits)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from braidlab.corrdim import AxisTerms

#: Observer boost cap (worldlines are not enforced subluminal) — the viewer's
#: FRAME_BETA_CAP.
BETA_CAP = 0.995
#: Grid / refinement constants, matching the viewer and the CUDA baker.
FRAME_GRID = 420
FRAME_BISECT = 24
BUDGET_GRID_CAP = 4000

HIT_DTYPE = np.dtype(
    [
        ("src", "<u4"),
        ("dx", "<f4"),
        ("dy", "<f4"),
        ("chi", "<f4"),
        ("z_emit", "<f4"),
    ]
)


@dataclass(frozen=True)
class ObserverFrame:
    """One (instant, observer) frame: pulse centre, boost, and its hits."""

    z_obs: float
    chi_max: float
    wraps: int
    ox: float
    oy: float
    beta: np.ndarray  # (2,) proper peculiar velocity / front speed, uncapped
    hits: np.ndarray  # HIT_DTYPE record array, sorted by chi

    @property
    def redshift(self) -> np.ndarray:
        """Cosmological 1+Z per hit: sin(z_obs)/sin(z_emit)."""
        return np.sin(self.z_obs) / np.sin(self.hits["z_emit"].astype(np.float64))


@dataclass(frozen=True)
class FrameSet:
    """A baked .frames file: header + frames[instant][observer]."""

    n_paths: int
    front: str  # "speed" | "fit" | "budget"
    cheb: bool
    speed: float
    z_min: float
    observers: list[tuple[int, int, float, float]]  # (type, path_idx, px, py)
    frames: list[list[ObserverFrame]]


_FRONTS = ("speed", "fit", "budget")


def load_frames(path: str | Path) -> FrameSet:
    """Read a .frames file into numpy hit tables (hits sorted by chi)."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"BRF1":
        raise ValueError(f"{path}: not a BRF1 frames file")
    off = 4
    n_paths, n_obs, n_inst, front_code, cheb = struct.unpack_from("<5I", raw, off)
    off += 20
    speed, z_min = struct.unpack_from("<2d", raw, off)
    off += 16
    (_wrap_k,) = struct.unpack_from("<i", raw, off)
    off += 4
    observers = []
    for _ in range(n_obs):
        typ, path_idx = struct.unpack_from("<Ii", raw, off)
        off += 8
        px, py = struct.unpack_from("<2d", raw, off)
        off += 16
        observers.append((typ, path_idx, px, py))
    frames: list[list[ObserverFrame]] = []
    for _ in range(n_inst):
        z_obs, chi_max = struct.unpack_from("<2d", raw, off)
        off += 16
        (k,) = struct.unpack_from("<i", raw, off)
        off += 4
        row = []
        for _o in range(n_obs):
            ox, oy, bx, by = struct.unpack_from("<4d", raw, off)
            off += 32
            (n_hits,) = struct.unpack_from("<Q", raw, off)
            off += 8
            hits = np.frombuffer(raw, dtype=HIT_DTYPE, count=n_hits, offset=off)
            off += n_hits * HIT_DTYPE.itemsize
            # The GPU appends hits in nondeterministic atomic order; sort by
            # chi (the conformal reveal order) for reproducible analysis.
            hits = np.sort(hits, order=["chi", "src"])
            row.append(
                ObserverFrame(
                    z_obs=z_obs,
                    chi_max=chi_max,
                    wraps=k,
                    ox=ox,
                    oy=oy,
                    beta=np.array([bx, by]),
                    hits=hits,
                )
            )
        frames.append(row)
    return FrameSet(
        n_paths=n_paths,
        front=_FRONTS[front_code],
        cheb=bool(cheb),
        speed=speed,
        z_min=z_min,
        observers=observers,
        frames=frames,
    )


def apparent_positions(
    frame: ObserverFrame, mode: str = "static", cheb: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apparent sky offsets and Doppler log-factor per hit.

    ``static`` returns the stored offsets unchanged (ln_dopp = 0). ``moving``
    boosts into the observer's instantaneous rest frame: the aberrated
    direction n' = [n + gamma*beta + (gamma-1)(u.n)u] / [gamma(1 + beta.n)]
    and Doppler 1+Z_dopp = gamma(1 + beta.n), acting on the Euclidean unit
    direction; square-metric hits are re-projected onto their own Chebyshev
    shell afterwards — the viewer's apparentFrom, vectorized.
    """
    dx = frame.hits["dx"].astype(np.float64)
    dy = frame.hits["dy"].astype(np.float64)
    if mode == "static":
        return dx, dy, np.zeros(len(dx))
    if mode != "moving":
        raise ValueError(f"mode must be static|moving, got {mode!r}")
    beta = frame.beta.copy()
    beta_mag = float(np.hypot(*beta))
    if beta_mag > BETA_CAP:
        beta *= BETA_CAP / beta_mag
        beta_mag = BETA_CAP
    if beta_mag <= 1e-6:
        return dx, dy, np.zeros(len(dx))
    gamma = 1.0 / np.sqrt(1.0 - beta_mag**2)
    ubx, uby = beta[0] / beta_mag, beta[1] / beta_mag
    eucl = np.hypot(dx, dy)
    ok = eucl > 0
    nx = np.where(ok, dx / np.where(ok, eucl, 1.0), 0.0)
    ny = np.where(ok, dy / np.where(ok, eucl, 1.0), 0.0)
    bdotn = beta[0] * nx + beta[1] * ny
    udotn = ubx * nx + uby * ny
    dfac = gamma * (1.0 + bdotn)
    apx = (nx + gamma * beta[0] + (gamma - 1.0) * udotn * ubx) / dfac
    apy = (ny + gamma * beta[1] + (gamma - 1.0) * udotn * uby) / dfac
    chi = frame.hits["chi"].astype(np.float64)
    norm = np.maximum(np.abs(apx), np.abs(apy)) if cheb else 1.0
    ax = np.where(ok, apx * chi / norm, dx)
    ay = np.where(ok, apy * chi / norm, dy)
    ln_dopp = np.where(ok, np.log(dfac), 0.0)
    return ax, ay, ln_dopp


def counts_vs_redshift(
    frame: ObserverFrame, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(histogram, cumulative N(<Z)) of images binned by cosmological 1+Z."""
    oneplusz = frame.redshift
    hist, _ = np.histogram(oneplusz, bins=edges)
    return hist, np.cumsum(hist)


def ghost_multiplicity(frame: ObserverFrame) -> np.ndarray:
    """Images per source worldline (ghosts: torus wraps + earlier epochs)."""
    return np.bincount(frame.hits["src"])


# ---------------------------------------------------------------------------
# Pure-Python reference baker (the validation oracle for frame_bake.cu).
# ---------------------------------------------------------------------------


def _comov_xy(axes: list[AxisTerms], p: int, z: float) -> tuple[float, float]:
    s = np.sin(z)
    out = []
    for ax in axes:
        v = ax.a2[p] * s + float(
            (ax.a[p] * (np.sin(ax.b[p] * z + ax.f[p]) - np.sin(ax.f[p]))).sum()
        )
        out.append(v / s)
    return out[0], out[1]


def _reach(z: float, z_obs: float, b_max: int) -> float:
    s, s_obs = np.sin(z), np.sin(z_obs)
    b = np.arange(2, b_max + 1)
    da = (np.cos(b * z) - 1.0) / (b * s) - (np.cos(b * z_obs) - 1.0) / (b * s_obs)
    db = np.sin(b * z) / (b * s) - np.sin(b * z_obs) / (b * s_obs)
    d = np.where(b % 2 == 0, np.hypot(da, db), np.abs(db))
    return float(d.max())


def bake_reference(
    axes: list[AxisTerms],
    z_obs: float,
    z_min: float,
    *,
    observer_path: int | None = None,
    observer_point: tuple[float, float] | None = None,
    front: str = "fit",
    cheb: bool = True,
    speed: float = 0.33,
    wraps: int | None = None,
) -> list[tuple[int, float, float, float, float]]:
    """Slow reference implementation of the frame bake for small inputs.

    Returns hits as (src, dx, dy, chi, z_emit) tuples, unordered. The
    algorithm is a line-for-line port of the viewer's frameSetup +
    framePathHits (which the CUDA baker also mirrors); it exists so the two
    independent implementations can be diffed in tests.
    """
    n = len(axes[0].a2)
    b_max = max(2, int(max(ax.b.max() for ax in axes)))
    eta_obs = float(np.log(np.tan(z_obs / 2.0)))
    budget = front == "budget"
    cheb = True if budget else cheb
    spd = (
        max(1e-3, speed)
        if front == "speed"
        else -1.0 / float(np.log(np.tan(z_min / 2.0)))
    )

    def dist(dx: float, dy: float) -> float:
        return max(abs(dx), abs(dy)) if cheb else float(np.hypot(dx, dy))

    def z_at_chi(chi: float) -> float:
        return 2.0 * float(np.arctan(np.exp(eta_obs - chi / spd)))

    m = min(BUDGET_GRID_CAP, max(FRAME_GRID, 2 * b_max)) if budget else FRAME_GRID
    if budget:
        zg = z_obs - (z_obs - z_min) * np.arange(m + 1) / m
        chig = np.array([_reach(z, z_obs, b_max) for z in zg])
        chi_max = float(chig.max())
    else:
        chi_max = spd * (eta_obs - float(np.log(np.tan(z_min / 2.0))))
        chig = chi_max * np.arange(m + 1) / m
        zg = np.array([z_at_chi(c) for c in chig])

    if observer_path is not None:
        ox, oy = _comov_xy(axes, observer_path, z_obs)
        obs_idx = observer_path
    else:
        assert observer_point is not None
        ox, oy = observer_point
        obs_idx = -1
    k = int(np.ceil((chi_max + 4.0) / 2.0)) if wraps is None else wraps

    px = np.empty(m + 1)
    py = np.empty(m + 1)
    hits = []
    for p in range(n):
        for i in range(m + 1):
            px[i], py[i] = _comov_xy(axes, p, float(zg[i]))
        for nx in range(-k, k + 1):
            for ny in range(-k, k + 1):
                bx, by = 2.0 * nx - ox, 2.0 * ny - oy
                dmin = max(0.0, 2.0 * max(abs(nx), abs(ny)) - 4.0)
                if dmin >= chi_max:
                    continue
                i0 = 1 if budget else max(1, int(np.ceil(dmin * m / chi_max)))
                if i0 > m:
                    continue
                prev = dist(px[i0 - 1] + bx, py[i0 - 1] + by) - chig[i0 - 1]
                # Observer self-contact: epsilon band, not exact zero — the
                # CUDA baker evaluates the pulse centre on the host and the
                # grid on the device, so a one-ulp libm difference must not
                # decide the branch (see frame_bake.cu).
                if (
                    p == obs_idx
                    and nx == 0
                    and ny == 0
                    and i0 == 1
                    and abs(prev) < 1e-9
                ):
                    prev = -1e-12
                elif prev == 0.0:
                    prev = 1e-12
                for i in range(i0, m + 1):
                    g = dist(px[i] + bx, py[i] + by) - chig[i]
                    if (prev < 0.0) != (g < 0.0):
                        inside_prev = prev < 0.0
                        if budget:
                            lo, hi = float(zg[i - 1]), float(zg[i])
                            for _ in range(FRAME_BISECT):
                                mid = (lo + hi) / 2.0
                                qx, qy = _comov_xy(axes, p, mid)
                                gm = dist(qx + bx, qy + by) - _reach(mid, z_obs, b_max)
                                if (gm < 0.0) == inside_prev:
                                    lo = mid
                                else:
                                    hi = mid
                            z_emit = (lo + hi) / 2.0
                            chi = _reach(z_emit, z_obs, b_max)
                        else:
                            lo, hi = float(chig[i - 1]), float(chig[i])
                            for _ in range(FRAME_BISECT):
                                mid = (lo + hi) / 2.0
                                qx, qy = _comov_xy(axes, p, z_at_chi(mid))
                                gm = dist(qx + bx, qy + by) - mid
                                if (gm < 0.0) == inside_prev:
                                    lo = mid
                                else:
                                    hi = mid
                            chi = (lo + hi) / 2.0
                            z_emit = z_at_chi(chi)
                        qx, qy = _comov_xy(axes, p, z_emit)
                        hits.append((p, qx + bx, qy + by, chi, z_emit))
                    prev = g
    return hits
