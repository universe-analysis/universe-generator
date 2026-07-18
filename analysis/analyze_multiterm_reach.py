"""Adversarial check: can multi-term worldlines out-reach the single-term envelope?

The reach chart (analyze_wiggle_reach) claims that with the unit rapidity
budget |a*b| summed across terms, no split of the budget over several
frequencies -- at any phase assignment -- can beat the best single frequency.
The argument is linearity: each unit-budget term displaces by at most

    r_b(z) = hypot(alpha_b, beta_b - 1)   (even b, free phase)
             |beta_b - 1|                 (odd b, f in {0, pi})

so a k-term config with budget shares w_i reaches at most sum(w_i * r_i)
<= max_i r_i.  This module stress-tests that claim with data instead of
algebra, two ways:

1. a large random sweep over the actual sampling ensemble (distinct integer
   frequencies, Dirichlet budget splits, random amplitude signs, phases
   f ~ U[0, 2pi) on even frequencies and f in {0, pi} on odd ones), and
2. an adversarial scipy optimizer at fixed times, free to pick any budget
   split (signed shares) and any admissible phases over frequency sets
   seeded with the top single-term performers.

Both report their best displacement as a ratio to the phased single-term
envelope (the blue line); the claim survives iff no ratio exceeds 1.

Usage::

    python -m analysis.analyze_multiterm_reach --out figures/multiterm_reach.png
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize

from analysis.analyze_wiggle_reach import alpha_beta, reach_from_bang

PI = np.pi


def bang_gains(freqs: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-term Bang-displacement gains (alpha, beta - 1) at time z.

    A term with budget share w, sign s and phase f displaces the comoving
    coordinate from the Bang by  w * s * (alpha * sin f + (beta - 1) * cos f).
    """
    a, be = alpha_beta(freqs.astype(float), z)
    return a, be - 1.0


def random_sweep(
    z_grid: np.ndarray,
    n_terms: int,
    n_configs: int,
    bmax: int,
    rng: np.random.Generator,
    chunk: int = 20_000,
) -> np.ndarray:
    """Max Bang displacement over random n_terms-configs, per z-grid point.

    Draws the actual ensemble: distinct integer frequencies in [2, bmax],
    Dirichlet(1) budget split, random amplitude signs, f ~ U[0, 2pi) on even
    frequencies, f in {0, pi} on odd ones.
    """
    best = np.zeros(len(z_grid))
    done = 0
    while done < n_configs:
        n = min(chunk, n_configs - done)
        freqs = np.empty((n, n_terms), dtype=np.int64)
        for row in range(0, n):
            freqs[row] = rng.choice(np.arange(2, bmax + 1), n_terms, replace=False)
        weights = rng.dirichlet(np.ones(n_terms), size=n)
        signs = rng.choice([-1.0, 1.0], size=(n, n_terms))
        phases = rng.uniform(0.0, 2.0 * PI, size=(n, n_terms))
        odd = freqs % 2 == 1
        phases[odd] = rng.choice([0.0, PI], size=int(odd.sum()))

        coef = weights * signs
        sin_f, cos_f = np.sin(phases), np.cos(phases)
        for i, z in enumerate(z_grid):
            a, bm1 = bang_gains(freqs, float(z))
            disp = np.abs(np.sum(coef * (a * sin_f + bm1 * cos_f), axis=1))
            m = float(disp.max())
            if m > best[i]:
                best[i] = m
        done += n
    return best


def optimize_at(
    z: float,
    n_terms: int,
    bmax: int,
    rng: np.random.Generator,
    n_freq_sets: int = 12,
    n_starts: int = 6,
) -> float:
    """Adversarial max Bang displacement of an n_terms config at time z.

    Optimizes signed budget shares t (w = |t| / sum|t|, sign = amplitude
    sign) and free phases on even frequencies; odd frequencies keep f = 0
    with the sign carrying their two admissible phases. Frequency sets are
    seeded with the top single-term performers plus random draws.
    """
    band = np.arange(2, bmax + 1)
    a_all, bm1_all = bang_gains(band, z)
    r_single = np.where(band % 2 == 0, np.hypot(a_all, bm1_all), np.abs(bm1_all))
    ranked = band[np.argsort(r_single)[::-1]]

    freq_sets = [ranked[:n_terms]]
    for k in range(1, n_freq_sets):
        if k < n_terms:
            # best frequency plus random companions
            rest = rng.choice(band[band != ranked[0]], n_terms - 1, replace=False)
            freq_sets.append(np.concatenate([[ranked[0]], rest]))
        else:
            freq_sets.append(rng.choice(band, n_terms, replace=False))

    best = 0.0
    for freqs in freq_sets:
        freqs = np.asarray(freqs)
        a, bm1 = bang_gains(freqs, z)
        even = freqs % 2 == 0

        def neg_disp(p: np.ndarray) -> float:
            t, f = p[:n_terms], p[n_terms:]
            norm = np.sum(np.abs(t))
            if norm < 1e-12:
                return 0.0
            f_eff = np.where(even, f, 0.0)
            gains = a * np.sin(f_eff) + bm1 * np.cos(f_eff)
            return -abs(float(np.dot(t, gains))) / float(norm)

        for _ in range(n_starts):
            p0 = np.concatenate(
                [rng.normal(size=n_terms), rng.uniform(0, 2 * PI, n_terms)]
            )
            res = minimize(neg_disp, p0, method="Nelder-Mead")
            if -res.fun > best:
                best = -res.fun
    return best


def make_figure(
    out: str,
    z_grid: np.ndarray,
    envelope: np.ndarray,
    sweeps: dict[int, np.ndarray],
    z_opt: np.ndarray,
    opts: dict[int, np.ndarray],
) -> None:
    """Two panels: reach curves, and every ratio to the single-term envelope."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.2))
    axa.plot(
        z_grid / PI,
        envelope,
        lw=2.4,
        color="tab:blue",
        label="single-term envelope (blue line)",
    )
    colors = {2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    env_opt = np.array([reach_from_bang(float(z), 400)[0] for z in z_opt])
    for k, curve in sweeps.items():
        axa.plot(
            z_grid / PI,
            curve,
            lw=1.2,
            color=colors[k],
            label=f"best random {k}-term config",
        )
    for k, vals in opts.items():
        axa.plot(
            z_opt / PI,
            vals,
            "x",
            ms=5,
            color=colors[k],
            label=f"optimizer, {k} terms",
        )
    axa.axhline(1.0, color="k", lw=0.8, ls=":", label="antipode $\\chi=1$")
    axa.set_xlabel("z / $\\pi$")
    axa.set_ylabel("max comoving displacement from Bang")
    axa.set_title("Multi-term reach vs the single-term envelope")
    axa.legend(fontsize=8, loc="lower right")
    axa.grid(alpha=0.25)

    for k, curve in sweeps.items():
        axb.plot(z_grid / PI, curve / envelope, lw=1.2, color=colors[k])
    for k, vals in opts.items():
        axb.plot(z_opt / PI, vals / env_opt, "x", ms=5, color=colors[k])
    axb.axhline(1.0, color="tab:blue", lw=2, alpha=0.6)
    axb.set_xlabel("z / $\\pi$")
    axb.set_ylabel("ratio to single-term envelope")
    axb.set_title("Every config lands at or below ratio 1")
    axb.set_ylim(0, 1.05)
    axb.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", default="figures/multiterm_reach.png")
    ap.add_argument("--bmax", type=int, default=400, help="frequency band ceiling")
    ap.add_argument("--configs", type=int, default=200_000, help="random configs per k")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    z_grid = np.linspace(0.005 * PI, 0.995 * PI, 160)
    envelope = np.array([reach_from_bang(float(z), args.bmax)[0] for z in z_grid])
    z_opt = np.linspace(0.02 * PI, 0.98 * PI, 25)
    env_opt = np.array([reach_from_bang(float(z), args.bmax)[0] for z in z_opt])

    sweeps: dict[int, np.ndarray] = {}
    opts: dict[int, np.ndarray] = {}
    for k in (2, 3, 4):
        sweeps[k] = random_sweep(z_grid, k, args.configs, args.bmax, rng)
        opts[k] = np.array([optimize_at(float(z), k, args.bmax, rng) for z in z_opt])
        r_sweep = float(np.max(sweeps[k] / envelope))
        r_opt = float(np.max(opts[k] / env_opt))
        print(
            f"k={k}: best random-sweep ratio {r_sweep:.6f} "
            f"(excess {r_sweep - 1.0:+.2e})  "
            f"best optimizer ratio {r_opt:.9f} (excess {r_opt - 1.0:+.2e})"
        )
        if r_sweep > 1.0 + 1e-9 or r_opt > 1.0 + 1e-6:
            print(f"  *** k={k} BEAT THE ENVELOPE — the linearity claim fails ***")

    make_figure(args.out, z_grid, envelope, sweeps, z_opt, opts)


if __name__ == "__main__":
    main()
