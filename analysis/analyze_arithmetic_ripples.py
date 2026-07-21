"""Arithmetic ripples in N(T) -- does the Z/TZ lattice's number theory leak out?

The jam count follows N ~ T^D. If the packing were sensitive to the arithmetic
of the comoving lattice Z/TZ (resonant closed orbits need frequencies dividing
T), the residuals around the smooth power law should correlate with arithmetic
functions of T: divisor count d(T), distinct prime factors omega(T), largest
prime factor, abundance sigma(T)/T. If the residuals are pure seed noise, the
model is purely archimedean at current precision and the profinite structure is
irrelevant.

Method, per ensemble (one sampler era x cutoff x dim, stores pooled only when
verified consistent):
  1. Fit log<N> = poly(log T) (degree 2 by default -- absorbs the known low-T
     finite-size bend; degree 1 as robustness) with 1/SEM^2 weights.
  2. Excess-ripple chi^2: are per-T residuals larger than seed noise implies?
     If not, there is no per-T structure of ANY kind to attribute to arithmetic.
  3. Regress residuals on each arithmetic covariate; significance via
     permutation of the covariate across T values (exact null: "is the
     alignment with T's arithmetic stronger than a random assignment?").

Caveat printed in the report: every stored T is a multiple of 20, so the sample
contains no prime T -- the prime-vs-composite contrast is untestable on
existing data (primality of m = T/20 is the nearest available analog).

Usage::

    uv run python -m analysis.analyze_arithmetic_ripples \
        --out figures/arithmetic_ripples.png
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- arithmetic


def factorize(n: int) -> dict[int, int]:
    """Prime factorization of n >= 1 as {prime: exponent}."""
    if n < 1:
        raise ValueError("n must be >= 1")
    factors: dict[int, int] = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def divisor_count(n: int) -> int:
    """d(n): number of divisors."""
    out = 1
    for e in factorize(n).values():
        out *= e + 1
    return out


def omega(n: int) -> int:
    """omega(n): number of distinct prime factors."""
    return len(factorize(n))


def big_omega(n: int) -> int:
    """Omega(n): number of prime factors with multiplicity."""
    return sum(factorize(n).values())


def largest_prime_factor(n: int) -> int:
    """Largest prime factor of n >= 2."""
    return max(factorize(n))


def abundance(n: int) -> float:
    """sigma(n)/n: sum of divisors over n (highly composite -> large)."""
    out = 1.0
    for p, e in factorize(n).items():
        out *= (p ** (e + 1) - 1) / (p**e * (p - 1))
    return out


def is_prime(n: int) -> bool:
    """Primality of n."""
    return n >= 2 and factorize(n) == {n: 1}


#: covariate name -> function of T evaluated on each ensemble's T ladder
COVARIATES = {
    "log d(T)": lambda t: float(np.log(divisor_count(t))),
    "omega(T)": lambda t: float(omega(t)),
    "Omega(T)": lambda t: float(big_omega(t)),
    "log lpf(T)": lambda t: float(np.log(largest_prime_factor(t))),
    "sigma(T)/T": abundance,
    "T/20 prime": lambda t: float(t % 20 == 0 and is_prime(t // 20)),
}

# ------------------------------------------------------------------ datasets


@dataclass(frozen=True)
class Ensemble:
    """One homogeneous ensemble: pooled stores sharing sampler era + cutoff."""

    label: str
    dbs: tuple[str, ...]
    dim: int
    terms: int = 2


#: Pooling verified 2026-07-20: pack3d_e6 and frequni3d_e6 agree seed-for-seed
#: at every shared T (uniform sampler); freq3d/freq2d are the removed smart
#: sampler (N ~30-40% higher) and are kept as separate ensembles, never pooled.
ENSEMBLES = [
    Ensemble(
        "3+1 uniform 1e-6 (pack+frequni+converge)",
        (
            "data/pack/pack3d_e6.db",
            "data/freq/frequni3d_e6.db",
            "data/converge/converge3d_e6.db",
            "data/converge/converge3d_e6ext.db",
        ),
        dim=3,
    ),
    Ensemble(
        "3+1 uniform 1e-7 (pack+converge)",
        ("data/pack/pack3d_e7.db", "data/converge/converge3d_e7.db"),
        dim=3,
    ),
    Ensemble("2+1 uniform 1e-6 (pack)", ("data/pack/pack2d_e6.db",), dim=2),
    Ensemble("2+1 uniform 1e-7 (pack)", ("data/pack/pack2d_e7.db",), dim=2),
    Ensemble("2+1 uniform 1e-8 (converge)", ("data/converge/converge2d_e8.db",), dim=2),
    Ensemble("3+1 smart 1e-6 (freq)", ("data/freq/freq3d_e6.db",), dim=3),
    Ensemble("2+1 smart 1e-6 (freq)", ("data/freq/freq2d_e6.db",), dim=2),
]


def load_counts(root: Path, ens: Ensemble) -> dict[int, list[float]]:
    """Per-T jam counts pooled across the ensemble's stores."""
    by_t: dict[int, list[float]] = {}
    for db in ens.dbs:
        conn = sqlite3.connect(root / db)
        rows = conn.execute(
            "SELECT t, n_final FROM runs WHERE status='done' AND dim=? AND "
            "terms=? AND n_final IS NOT NULL",
            (ens.dim, ens.terms),
        ).fetchall()
        conn.close()
        for t, n in rows:
            by_t.setdefault(int(t), []).append(float(n))
    return by_t


# ---------------------------------------------------------------- statistics


@dataclass(frozen=True)
class Ripples:
    """Trend-fit residuals for one ensemble."""

    t_values: np.ndarray  # (nT,) int
    resid: np.ndarray  # (nT,) per-T mean residual in log N
    sigma: np.ndarray  # (nT,) SEM of the per-T mean residual
    slope: float  # power-law exponent D (linear coefficient at center)
    chi2: float  # excess-ripple statistic sum (resid/sigma)^2
    chi2_dof: int
    chi2_p: float  # P(chi2 >= observed | residuals are pure seed noise)
    basis: np.ndarray  # (nT, degree+1) trend design matrix (for FWL projection)
    weights: np.ndarray  # (nT,) WLS weights used for the trend fit


def fit_ripples(by_t: dict[int, list[float]], degree: int) -> Ripples:
    """Weighted poly(log T) trend fit; residuals + excess-variance test."""
    t_values = np.array(sorted(t for t, v in by_t.items() if len(v) >= 2))
    means = np.array([np.mean(by_t[t]) for t in t_values])
    sems = np.array([np.std(by_t[t], ddof=1) / np.sqrt(len(by_t[t])) for t in t_values])
    logt = np.log(t_values.astype(float))
    logn = np.log(means)
    sigma = sems / means  # SEM in log space
    w = 1.0 / np.maximum(sigma, 1e-6) ** 2

    x = logt - logt.mean()
    coeffs = np.polyfit(x, logn, deg=degree, w=np.sqrt(w))
    resid = logn - np.polyval(coeffs, x)
    slope = float(coeffs[-2])  # d(logN)/d(logT) at the ladder center

    dof = len(t_values) - (degree + 1)
    chi2 = float(np.sum((resid / sigma) ** 2))
    chi2_p = float(stats.chi2.sf(chi2, dof))
    basis = np.vander(x, degree + 1, increasing=True)
    return Ripples(t_values, resid, sigma, slope, chi2, dof, chi2_p, basis, w)


def lag1_autocorr(rip: Ripples, *, n_perm: int = 20000) -> tuple[float, float]:
    """Lag-1 autocorrelation of the residual ladder + permutation p-value.

    Discriminates the two ways residuals can beat seed noise: leftover smooth
    trend curvature makes adjacent-T residuals alike (positive lag-1), while
    arithmetic ripples are spiky in T (adjacent T have unrelated divisor
    structure) and give lag-1 near zero.
    """
    r = rip.resid - rip.resid.mean()
    obs = float(np.sum(r[:-1] * r[1:]) / np.sum(r * r))
    rng = np.random.default_rng(20260720)
    perms = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.permutation(r)
        perms[i] = np.sum(s[:-1] * s[1:]) / np.sum(s * s)
    p = float((np.sum(perms >= obs) + 1) / (n_perm + 1))
    return obs, p


@dataclass(frozen=True)
class CovariateTest:
    """Residual-vs-covariate partial regression with a permutation p-value."""

    name: str
    pearson_r: float  # correlation of residuals with the detrended covariate
    perm_p: float  # two-sided permutation p for the partial slope
    wls_slope: float  # partial (FWL) slope: unbiased for a planted amplitude
    spearman_rho: float
    surviving: float  # fraction of covariate variance orthogonal to the trend


def regress_covariate(
    rip: Ripples,
    name: str,
    values: np.ndarray,
    *,
    n_perm: int = 20000,
    rng: np.random.Generator | None = None,
) -> CovariateTest | None:
    """Partial regression of residuals on one covariate.

    The trend fit absorbs any component of the covariate that is smooth in
    log T, which would attenuate a naive residual-on-covariate slope. Following
    Frisch-Waugh-Lovell, the covariate is residualized against the same trend
    basis (same WLS weights) first; the slope on the detrended covariate is
    then unbiased for a planted arithmetic amplitude. `surviving` reports how
    much covariate variance the detrending leaves -- the test only has power
    over that spiky component. Returns None if the covariate is constant.
    """
    if np.ptp(values) == 0:
        return None
    rng = rng or np.random.default_rng(20260720)
    w = rip.weights
    bw = rip.basis * w[:, None]
    gram_inv = np.linalg.inv(rip.basis.T @ bw)

    def detrend(v: np.ndarray) -> np.ndarray:
        return v - rip.basis @ (gram_inv @ (bw.T @ v))

    def partial_slope(v: np.ndarray) -> tuple[float, float]:
        """(slope on detrended v, surviving variance fraction)."""
        vp = detrend(v)
        vc = v - np.average(v, weights=w)
        denom = float(np.sum(w * vp**2))
        surviving = denom / float(np.sum(w * vc**2))
        return float(np.sum(w * vp * rip.resid) / denom), surviving

    obs, surviving = partial_slope(values)
    perms = np.array([partial_slope(rng.permutation(values))[0] for _ in range(n_perm)])
    perm_p = float((np.sum(np.abs(perms) >= abs(obs)) + 1) / (n_perm + 1))
    vp = detrend(values)
    pearson = float(np.corrcoef(vp, rip.resid)[0, 1])
    spearman = float(
        np.corrcoef(stats.rankdata(values), stats.rankdata(rip.resid))[0, 1]
    )
    return CovariateTest(name, pearson, perm_p, obs, spearman, surviving)


# --------------------------------------------------------------------- main


def degree_scan(
    root: Path, ens: Ensemble, out: Path, degrees: range = range(1, 5)
) -> None:
    """The stability plot: log d(T) partial slope vs trend degree.

    A genuine arithmetic ripple is orthogonal to smooth trends, so its slope
    holds steady as the polynomial degree rises; smooth-trend leakage collapses
    toward zero instead. The shaded band is the central 95% of the permutation
    null at each degree.
    """
    import matplotlib.pyplot as plt

    by_t = load_counts(root, ens)
    slopes: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    chi2_note: list[str] = []
    for deg in degrees:
        rip = fit_ripples(by_t, deg)
        vals = np.array([np.log(divisor_count(int(t))) for t in rip.t_values])
        res = regress_covariate(rip, "log d(T)", vals)
        assert res is not None
        rng = np.random.default_rng(1)
        w = rip.weights
        bw = rip.basis * w[:, None]
        gram_inv = np.linalg.inv(rip.basis.T @ bw)
        null = []
        for _ in range(4000):
            v = rng.permutation(vals)
            vp = v - rip.basis @ (gram_inv @ (bw.T @ v))
            null.append(float(np.sum(w * vp * rip.resid) / np.sum(w * vp**2)))
        lo.append(float(np.percentile(null, 2.5)))
        hi.append(float(np.percentile(null, 97.5)))
        slopes.append(res.wls_slope)
        chi2_note.append(f"deg {deg}: chi2/dof={rip.chi2:.0f}/{rip.chi2_dof}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    degs = list(degrees)
    ax.fill_between(degs, lo, hi, color="lightgray", label="permutation null 95%")
    ax.plot(degs, slopes, "o-", color="tab:blue", label="log d(T) partial slope")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(degs)
    ax.set_xlabel("trend polynomial degree in log T")
    ax.set_ylabel("d ln N / d ln d(T)")
    ax.set_title(
        f"Divisor-count slope vs trend flexibility -- {ens.label}\n"
        + "; ".join(chi2_note),
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repo root holding data/")
    parser.add_argument(
        "--out", type=Path, default=Path("figures/arithmetic_ripples.png")
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="poly(log T) trend degree (2 absorbs the low-T bend; 1 = pure law)",
    )
    parser.add_argument(
        "--scan-out",
        type=Path,
        default=None,
        help="also write the slope-vs-trend-degree stability plot for the "
        "first (primary) ensemble",
    )
    args = parser.parse_args()
    root = Path(args.root)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [e for e in ENSEMBLES if all((root / db).exists() for db in e.dbs)]
    fig, axes = plt.subplots(
        len(available), 2, figsize=(13, 3.0 * len(available)), squeeze=False
    )

    print(
        "NOTE: every stored T is a multiple of 20 -- no prime T exists in the\n"
        "sample, so prime-vs-composite is untestable here; 'T/20 prime' is the\n"
        "nearest available analog.\n"
    )

    for row, ens in enumerate(available):
        by_t = load_counts(root, ens)
        rip = fit_ripples(by_t, args.degree)
        print(f"== {ens.label}")
        print(
            f"   {len(rip.t_values)} T values {rip.t_values.min()}-"
            f"{rip.t_values.max()}, center slope D = {rip.slope:.3f} "
            f"(degree-{args.degree} trend)"
        )
        print(
            f"   excess ripple: chi2/dof = {rip.chi2:.1f}/{rip.chi2_dof} "
            f"(p = {rip.chi2_p:.3g}) -- residual rms "
            f"{np.std(rip.resid) * 100:.2f}% vs seed-noise floor "
            f"{np.mean(rip.sigma) * 100:.2f}%"
        )
        ac, ac_p = lag1_autocorr(rip)
        print(
            f"   residual lag-1 autocorr = {ac:+.2f} (perm p = {ac_p:.3f}) "
            "-- positive = smooth trend leftover, ~0 = spiky"
        )
        tests: list[CovariateTest] = []
        for name, fn in COVARIATES.items():
            vals = np.array([fn(int(t)) for t in rip.t_values])
            res = regress_covariate(rip, name, vals)
            if res is not None:
                tests.append(res)
        for tst in tests:
            print(
                f"   {tst.name:>11}: r = {tst.pearson_r:+.2f}  "
                f"rho = {tst.spearman_rho:+.2f}  slope = {tst.wls_slope:+.4f}  "
                f"perm p = {tst.perm_p:.3f}  "
                f"(covariate surviving detrend: {tst.surviving:.0%})"
            )
        print()

        dvals = np.array([np.log(divisor_count(int(t))) for t in rip.t_values])
        bw = rip.basis * rip.weights[:, None]
        gram_inv = np.linalg.inv(rip.basis.T @ bw)
        dvals_perp = dvals - rip.basis @ (gram_inv @ (bw.T @ dvals))
        ax_t, ax_d = axes[row]
        color = "tab:blue" if ens.dim == 3 else "tab:green"
        ax_t.errorbar(
            rip.t_values,
            rip.resid * 100,
            yerr=rip.sigma * 100,
            fmt="o",
            color=color,
            ms=4,
        )
        ax_t.axhline(0, color="gray", lw=0.8)
        ax_t.set_xlabel("T")
        ax_t.set_ylabel("residual in log N (%)")
        ax_t.set_title(ens.label, fontsize=10)
        ax_d.errorbar(
            dvals_perp,
            rip.resid * 100,
            yerr=rip.sigma * 100,
            fmt="o",
            color=color,
            ms=4,
        )
        dtest = next(t for t in tests if t.name == "log d(T)")
        xx = np.linspace(dvals_perp.min(), dvals_perp.max(), 2)
        ax_d.plot(xx, dtest.wls_slope * xx * 100, "--", color="gray")
        ax_d.axhline(0, color="gray", lw=0.5)
        ax_d.set_xlabel("log d(T), detrended")
        ax_d.set_title(
            f"r = {dtest.pearson_r:+.2f}, perm p = {dtest.perm_p:.3f}",
            fontsize=10,
        )

    fig.suptitle(
        "Arithmetic ripples in N(T): residuals around the smooth "
        f"degree-{args.degree} log-log trend vs divisor count",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    if args.scan_out is not None and available:
        degree_scan(root, available[0], args.scan_out)


if __name__ == "__main__":
    main()
