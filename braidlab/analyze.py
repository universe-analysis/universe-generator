"""Analysis: the large-scale exponent D and the kinetic cost law.

Under the fixed-convergence protocol, the seed-averaged count per timestep is
N(T) = theta * C * T**D with theta held (approximately) constant, so the slope
of log<N> vs log T is an unbiased estimate of D. Error bars come from
bootstrapping over the seed ensemble. We also expose the power-law constancy
check (local slopes should agree) and the cost exponent k (attempts ~ T**k).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from braidlab.store import RunResult


def formula_ceiling(dim: int, t: int) -> float:
    """Chris's grid-packing reference ceiling (T/2)**d."""
    return (t / 2) ** dim


def load_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an engine kinetic curve CSV into (attempts, n) arrays."""
    rows = Path(path).read_text().splitlines()[1:]
    data = np.array([[float(x) for x in r.split(",")] for r in rows])
    if data.size == 0:
        return np.array([]), np.array([])
    return data[:, 0], data[:, 1]


@dataclass(frozen=True)
class DResult:
    """Outcome of a D measurement for one (dim, band)."""

    dim: int
    band: str
    t_values: list[int]
    n_mean: list[float]
    n_sem: list[float]
    n_seeds: list[int]
    d: float
    d_err: float
    intercept: float
    local_slopes: list[float]
    local_slope_std: float


def _group_by_t(results: list[RunResult]) -> dict[int, list[float]]:
    by_t: dict[int, list[float]] = {}
    for r in results:
        if r.n_final is not None:
            by_t.setdefault(r.t, []).append(float(r.n_final))
    return by_t


def _wls_slope(
    logt: np.ndarray, logn: np.ndarray, w: np.ndarray
) -> tuple[float, float]:
    """Weighted least-squares slope and intercept of logn ~ slope*logt + b."""
    sw = w.sum()
    mx = (w * logt).sum() / sw
    my = (w * logn).sum() / sw
    cov = (w * (logt - mx) * (logn - my)).sum()
    var = (w * (logt - mx) ** 2).sum()
    slope = cov / var
    return slope, my - slope * mx


def measure_d(
    results: list[RunResult], dim: int, band: str, *, n_boot: int = 2000
) -> DResult:
    """Estimate D from a seed ensemble via weighted log-log slope + bootstrap.

    Args:
        results: Completed runs for one (dim, band).
        dim, band: Identifiers carried into the result.
        n_boot: Bootstrap resamples (over seeds within each T) for the error bar.
    """
    by_t = _group_by_t(results)
    t_values = sorted(by_t)
    if len(t_values) < 2:
        raise ValueError("need >= 2 timesteps with data to fit a slope")

    n_mean = np.array([np.mean(by_t[t]) for t in t_values])
    n_seeds = np.array([len(by_t[t]) for t in t_values])
    n_std = np.array(
        [np.std(by_t[t], ddof=1) if len(by_t[t]) > 1 else 0.0 for t in t_values]
    )
    n_sem = n_std / np.sqrt(np.maximum(n_seeds, 1))

    logt = np.log(np.array(t_values, dtype=float))
    logn = np.log(n_mean)
    # weight by inverse variance in log space; floor avoids div-by-zero for k=1
    sigma_log = np.where(n_mean > 0, n_sem / n_mean, 0.0)
    w = 1.0 / np.maximum(sigma_log, 1e-3) ** 2
    d, intercept = _wls_slope(logt, logn, w)

    rng = np.random.default_rng(12345)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        means = np.array(
            [
                np.mean(rng.choice(by_t[t], size=len(by_t[t]), replace=True))
                for t in t_values
            ]
        )
        boot[b], _ = _wls_slope(logt, np.log(means), np.ones_like(logt))
    d_err = float(np.std(boot))

    local = [
        float((logn[i + 1] - logn[i]) / (logt[i + 1] - logt[i]))
        for i in range(len(t_values) - 1)
    ]
    return DResult(
        dim=dim,
        band=band,
        t_values=t_values,
        n_mean=[float(x) for x in n_mean],
        n_sem=[float(x) for x in n_sem],
        n_seeds=[int(x) for x in n_seeds],
        d=float(d),
        d_err=d_err,
        intercept=float(intercept),
        local_slopes=local,
        local_slope_std=float(np.std(local)) if local else 0.0,
    )


def cost_exponent(
    curves_by_t: dict[int, tuple[np.ndarray, np.ndarray]],
    fractions: tuple[float, ...] = (0.5, 0.7, 0.9),
    *,
    ref: dict[int, float] | None = None,
) -> dict[float, float]:
    """Cost exponent k(f): attempts to reach fraction f of a reference ~ T**k.

    Args:
        curves_by_t: Per-T kinetic curves (attempts, n).
        fractions: Coverage fractions of the reference to time.
        ref: Per-T reference count; defaults to each curve's final N.
    """
    t_values = sorted(curves_by_t)
    out: dict[float, float] = {}
    for f in fractions:
        ts: list[float] = []
        att: list[float] = []
        for t in t_values:
            a, n = curves_by_t[t]
            if a.size == 0:
                continue
            target = f * (ref[t] if ref else n[-1])
            hit = np.searchsorted(n, target)
            if hit < len(a):
                ts.append(t)
                att.append(a[hit])
        if len(ts) >= 3:
            k, _ = _wls_slope(np.log(ts), np.log(att), np.ones(len(ts)))
            out[f] = float(k)
    return out
