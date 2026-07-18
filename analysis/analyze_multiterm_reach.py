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
from dataclasses import dataclass

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
        freqs, weights, signs, phases = draw_ensemble(n, n_terms, bmax, rng)
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


def draw_ensemble(
    n: int, n_terms: int, bmax: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One random ensemble draw: (freqs, weights, signs, phases), each (n, k).

    Distinct integer frequencies in [2, bmax], Dirichlet(1) budget split,
    random amplitude signs, f ~ U[0, 2pi) on even frequencies, f in {0, pi}
    on odd ones.
    """
    band = np.arange(2, bmax + 1)
    freqs = np.empty((n, n_terms), dtype=np.int64)
    for row in range(0, n):
        freqs[row] = rng.choice(band, n_terms, replace=False)
    weights = rng.dirichlet(np.ones(n_terms), size=n)
    signs = rng.choice([-1.0, 1.0], size=(n, n_terms))
    phases = rng.uniform(0.0, 2.0 * PI, size=(n, n_terms))
    odd = freqs % 2 == 1
    phases[odd] = rng.choice([0.0, PI], size=int(odd.sum()))
    return freqs, weights, signs, phases


@dataclass
class ExampleConfig:
    """One concrete multi-term worldline from the ensemble.

    The physical worldline (per axis) is  x(z) = sum_i a_i * [sin(b_i z + f_i)
    - sin f_i]  with amplitudes a_i = sign_i * w_i / b_i, so the rapidity
    budget sum_i |a_i * b_i| = sum_i w_i = 1 exactly.
    """

    freqs: np.ndarray
    weights: np.ndarray
    signs: np.ndarray
    phases: np.ndarray
    disp: float

    @property
    def amps(self) -> np.ndarray:
        """Sinusoid amplitudes a_i = sign_i * w_i / b_i."""
        return self.signs * self.weights / self.freqs

    def budget(self) -> float:
        """The spent rapidity budget sum |a_i * b_i| (must be 1)."""
        return float(np.sum(np.abs(self.amps) * self.freqs))

    def formula(self) -> str:
        """The worldline as an explicit x(z) expression."""
        parts = []
        for a, b, f in zip(self.amps, self.freqs, self.phases):
            sgn = "-" if a < 0 else ("+" if parts else "")
            if abs(f) < 1e-12:
                parts.append(f"{sgn} {abs(a):.3g} sin({b}z)")
            else:
                parts.append(f"{sgn} {abs(a):.3g} [sin({b}z + {f:.2f}) - sin {f:.2f}]")
        return " ".join(parts).lstrip()

    def trajectory(self, z_grid: np.ndarray) -> np.ndarray:
        """Signed comoving displacement from the Bang, X(z) - X(0), on a grid."""
        out = np.empty(len(z_grid))
        coef = self.weights * self.signs
        sin_f, cos_f = np.sin(self.phases), np.cos(self.phases)
        for i, z in enumerate(z_grid):
            a, bm1 = bang_gains(self.freqs, float(z))
            out[i] = float(np.sum(coef * (a * sin_f + bm1 * cos_f)))
        return out


def top_examples(
    z_star: float,
    n_terms: int,
    n_configs: int,
    bmax: int,
    rng: np.random.Generator,
    n_top: int = 3,
    min_share: float = 0.0,
) -> list[ExampleConfig]:
    """The n_top farthest-displaced random configs at time z_star.

    With min_share > 0 only genuinely split configs qualify: every term must
    hold at least that fraction of the budget.
    """
    freqs, weights, signs, phases = draw_ensemble(n_configs, n_terms, bmax, rng)
    a, bm1 = bang_gains(freqs, z_star)
    disp = np.abs(
        np.sum(weights * signs * (a * np.sin(phases) + bm1 * np.cos(phases)), axis=1)
    )
    if min_share > 0.0:
        disp = np.where(weights.min(axis=1) >= min_share, disp, -1.0)
    order = np.argsort(disp)[::-1][:n_top]
    return [
        ExampleConfig(freqs[i], weights[i], signs[i], phases[i], float(disp[i]))
        for i in order
    ]


def split_scan(
    z_star: float, b_pair: tuple[int, int], n_weights: int = 21, n_phase: int = 720
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reach of a 2-term config vs its budget split, maximized over phases.

    For weight w on the first frequency (1 - w on the second), numerically
    maximizes |X(z_star) - X(0)| over both phases (grid on even frequencies,
    {0, pi} on odd) and both signs. Returns (w_grid, numeric_max, predicted)
    where predicted = w * r_1 + (1 - w) * r_2 is the linearity line.
    """
    w_grid = np.linspace(0.0, 1.0, n_weights)
    r = np.empty(2)
    phase_sets = []
    for j, b in enumerate(b_pair):
        a, bm1 = bang_gains(np.array([b]), z_star)
        if b % 2 == 0:
            f = np.linspace(0.0, 2.0 * PI, n_phase, endpoint=False)
            r[j] = float(np.hypot(a[0], bm1[0]))
        else:
            f = np.array([0.0, PI])
            r[j] = float(abs(bm1[0]))
        phase_sets.append(a[0] * np.sin(f) + bm1[0] * np.cos(f))
    g1 = phase_sets[0][:, None]
    g2 = phase_sets[1][None, :]
    numeric = np.array([float(np.max(np.abs(w * g1 + (1.0 - w) * g2))) for w in w_grid])
    predicted = w_grid * r[0] + (1.0 - w_grid) * r[1]
    return w_grid, numeric, predicted


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


def make_examples_figure(
    out: str,
    z_star: float,
    examples: list[ExampleConfig],
    bmax: int,
    pairs: tuple[tuple[int, int], ...] = ((2, 3), (2, 4), (2, 6)),
) -> None:
    """Concrete 2-term functions: their trajectories, and the split linearity.

    Left: the displacement trajectories |X(z) - X(0)| of the given example
    configs under the envelope, each labelled with its explicit x(z) formula.
    Right: reach at z_star vs the budget split w for fixed frequency pairs,
    numerically maximized over phases and signs at every w — the dots land on
    the straight interpolation line, which is the whole reason splitting can
    never beat concentrating.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.4))

    z_grid = np.linspace(0.005 * PI, 0.995 * PI, 500)
    envelope = np.array([reach_from_bang(float(z), bmax)[0] for z in z_grid])
    axa.plot(
        z_grid / PI,
        envelope,
        lw=2.4,
        color="tab:blue",
        label="single-term envelope",
    )
    ex_colors = ("tab:orange", "tab:green", "tab:red", "tab:purple")
    for cfg, col in zip(examples, ex_colors):
        traj = np.abs(cfg.trajectory(z_grid))
        axa.plot(z_grid / PI, traj, lw=1.3, color=col, label=f"x(z) = {cfg.formula()}")
        axa.plot([z_star / PI], [cfg.disp], "o", ms=5, color=col)
    axa.axvline(z_star / PI, color="k", lw=0.7, ls=":", alpha=0.6)
    axa.set_xlabel("z / $\\pi$")
    axa.set_ylabel("comoving displacement from Bang  |X(z) - X(0)|")
    axa.set_title(
        f"Example 2-term worldlines (best of the random draw at z = {z_star / PI:.2f}"
        "$\\pi$)"
    )
    axa.legend(fontsize=7, loc="upper left")
    axa.grid(alpha=0.25)

    pair_colors = ("tab:green", "tab:orange", "tab:red")
    for (b1, b2), col in zip(pairs, pair_colors):
        w_grid, numeric, predicted = split_scan(z_star, (b1, b2))
        axb.plot(w_grid, predicted, lw=1.2, color=col, alpha=0.7)
        axb.plot(
            w_grid,
            numeric,
            "o",
            ms=4,
            color=col,
            label=f"b = {b1} + {b2}: numeric max over phases",
        )
    axb.axhline(
        np.sqrt(2), color="tab:blue", lw=1.4, ls="--", label="$\\sqrt{2}$ envelope"
    )
    axb.set_xlabel("budget share w on b = 2  (1 - w on the partner)")
    axb.set_ylabel(f"max reach at z = {z_star / PI:.2f}$\\pi$")
    axb.set_title("Splitting interpolates linearly — concentration always wins")
    axb.legend(fontsize=8, loc="lower right")
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
    ap.add_argument(
        "--examples-out",
        default=None,
        help="write the 2-term example figure here and skip the big sweep",
    )
    args = ap.parse_args()

    if args.examples_out is not None:
        rng = np.random.default_rng(args.seed)
        z_star = PI / 2
        # the two best free draws, plus the two best genuinely split
        # (every term holding >= 30% of the budget)
        examples = top_examples(z_star, 2, args.configs, args.bmax, rng, n_top=2)
        examples += top_examples(
            z_star, 2, args.configs, args.bmax, rng, n_top=2, min_share=0.3
        )
        env = reach_from_bang(z_star, args.bmax)[0]
        print(f"envelope at z = pi/2: {env:.6f} (sqrt(2) = {np.sqrt(2):.6f})")
        for cfg in examples:
            print(
                f"  x(z) = {cfg.formula()}   "
                f"disp {cfg.disp:.4f} ({cfg.disp / env:.4f} of envelope)  "
                f"budget {cfg.budget():.12f}"
            )
        make_examples_figure(args.examples_out, z_star, examples, args.bmax)
        return

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
