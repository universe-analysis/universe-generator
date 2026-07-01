"""Knot-complexity vs frequency: Chris's spitball, made concrete.

Knot proxy. A curve with winding numbers (p, q) is the (p, q) torus knot when
gcd(p, q)=1 and degrades to a gcd-component link otherwise. "The biggest knot you
can make from those frequencies" is then the size of the *reduced* knot:

    K(p, q) = (p/g - 1) * (q/g - 1),   g = gcd(p, q)

Coprime -> full (p-1)(q-1) ~ pq; a shared factor g shrinks it by ~1/g^2. For a
worldline with d axis-frequencies we average K over the axis pairs.

(A) Abstract growth law -- Chris's idea: sample random frequency pairs from high
    value ranges [N, 2N), average K, and fit the growth in N. Plus the explicit
    "N +/- 1" construction he suggested.

(B) Dump-grounded selection test -- the packings already chose their frequencies.
    Compare the *realized* mean K of packed worldlines vs T against two baselines
    at the same band: the engine's smart sampler (coprime + high-freq bias) and a
    naive uniform draw. Does jamming further select for knottiness?

Usage::

    python analyze_knots.py --out figures/knots.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from math import gcd
from pathlib import Path

import numpy as np

_NAME_RE = re.compile(r"_T(?P<t>\d+)_s(?P<seed>\d+)")


def knot_K(freqs: list[int]) -> float:
    """Mean reduced torus-knot size over the axis-frequency pairs."""
    vals = []
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            p, q = freqs[i], freqs[j]
            g = gcd(p, q)
            vals.append((p // g - 1) * (q // g - 1))
    return float(np.mean(vals))


def smart_draw(rng: np.random.Generator, modmax: int, dim: int) -> list[int]:
    """Reproduce the engine's smart sampler: high-freq bias + pairwise coprime."""
    for _ in range(24):
        b = [int(max(rng.integers(0, modmax), rng.integers(0, modmax))) + 2
             for _ in range(dim)]
        if all(gcd(b[i], b[j]) == 1
               for i in range(dim) for j in range(i + 1, dim)):
            return b
    return b  # guard hit -- accept whatever we have (matches the engine)


def uniform_draw(rng: np.random.Generator, modmax: int, dim: int) -> list[int]:
    """Naive baseline: each frequency uniform in [2, modmax+1]."""
    return [int(rng.integers(0, modmax)) + 2 for _ in range(dim)]


def growth_law(out_ax) -> None:
    """(A) mean/max K for random pairs across high value ranges."""
    rng = np.random.default_rng(0)
    Ns = np.array([30, 100, 300, 1000, 3000, 10000, 30000])
    mean_k, max_k, coprime_frac = [], [], []
    for n in Ns:
        p = rng.integers(n, 2 * n, size=40000)
        q = rng.integers(n, 2 * n, size=40000)
        g = np.gcd(p, q)
        k = (p // g - 1) * (q // g - 1)
        k_full = (p - 1) * (q - 1)  # the coprime ("biggest") value
        mean_k.append(k.mean())
        max_k.append(k_full.mean())
        coprime_frac.append(np.mean(g == 1))
    mean_k, max_k = np.array(mean_k), np.array(max_k)
    slope = np.polyfit(np.log(Ns), np.log(mean_k), 1)[0]

    print("=== (A) abstract growth law (pairs in [N, 2N)) ===")
    for n, mk, xk, cf in zip(Ns, mean_k, max_k, coprime_frac):
        print(f"  N={n:>6}  meanK={mk:.3e}  coprimeK={xk:.3e}  "
              f"coprime%={cf:.3f}  meanK/N^2={mk / n**2:.3f}")
    print(f"  fitted growth exponent of mean K vs N = {slope:.3f}")

    # Chris's explicit N+/-1 construction: gcd(N-1, N+1) is 2 (N odd) else 1.
    odd_N = np.arange(1001, 2000, 2)
    even_N = np.arange(1000, 2000, 2)
    cop_odd = np.mean([gcd(n - 1, n + 1) == 1 for n in odd_N])
    cop_even = np.mean([gcd(n - 1, n + 1) == 1 for n in even_N])
    print(f"  N+/-1: coprime frac  odd N={cop_odd:.2f}  even N={cop_even:.2f} "
          "(odd N -> both even -> always share 2)")

    out_ax.loglog(Ns, mean_k, "o-", color="tab:blue", label="mean K (random pairs)")
    out_ax.loglog(Ns, max_k, "s--", color="tab:green", label="coprime K (biggest)")
    ref = mean_k[0] * (Ns / Ns[0]) ** 2
    out_ax.loglog(Ns, ref, ":", color="gray", label="slope 2 (N^2)")
    out_ax.set_xlabel("N (frequency scale)")
    out_ax.set_ylabel("knot size K")
    out_ax.set_title(f"(A) knot growth: mean K ~ N^{slope:.2f}")
    out_ax.legend(fontsize=8)
    out_ax.grid(True, which="both", alpha=0.3)


def load_freqs(path: str, dim: int) -> np.ndarray:
    """Read the integer axis frequencies of every worldline in a dump."""
    keys = ["bx", "by", "bw"][:dim]
    rows = list(csv.DictReader(open(path)))
    return np.array([[int(round(float(r[k]))) for k in keys] for r in rows])


def selection_test(out_ax, dim: int, dumps_dir: str, label: str) -> None:
    """(B) realized vs smart/uniform-baseline mean K of the packing vs T."""
    rng = np.random.default_rng(1)
    by_t: dict[int, list[str]] = {}
    for path in sorted(glob.glob(f"{dumps_dir}/d{dim}_nyq_T*_s*.csv")):
        m = _NAME_RE.search(Path(path).name)
        if m:
            by_t.setdefault(int(m["t"]), []).append(path)

    ts, realized, smart, uniform = [], [], [], []
    print(f"\n=== (B) packing selection, {label} ===")
    print(f"{'T':>5} {'realized':>9} {'smart':>9} {'uniform':>9} {'real/smart':>10}")
    for t in sorted(by_t):
        freqs = np.vstack([load_freqs(p, dim) for p in by_t[t][:2]])
        if len(freqs) > 8000:
            freqs = freqs[rng.choice(len(freqs), 8000, replace=False)]
        r_k = np.mean([knot_K(list(f)) for f in freqs])
        # baselines at the same band (nyq: modmax = T-1, freq in [2, T])
        s_k = np.mean([knot_K(smart_draw(rng, t - 1, dim)) for _ in range(4000)])
        u_k = np.mean([knot_K(uniform_draw(rng, t - 1, dim)) for _ in range(4000)])
        ts.append(t)
        realized.append(r_k)
        smart.append(s_k)
        uniform.append(u_k)
        print(f"{t:>5} {r_k:>9.1f} {s_k:>9.1f} {u_k:>9.1f} {r_k / s_k:>10.3f}")

    out_ax.plot(ts, realized, "o-", color="tab:blue", label="realized (packed)")
    out_ax.plot(ts, smart, "s--", color="tab:orange", label="smart-sampler baseline")
    out_ax.plot(ts, uniform, "^:", color="tab:green", label="uniform baseline")
    out_ax.set_xlabel("T (band = max frequency)")
    out_ax.set_ylabel("mean knot size K")
    out_ax.set_title(f"(B) {label}: does jamming select for knottiness?")
    out_ax.legend(fontsize=8)
    out_ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/knots.png"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    growth_law(axes[0])
    selection_test(axes[1], 3, "data/corrdim/dumps", "3+1")
    selection_test(axes[2], 2, "data/corrdim2d/dumps", "2+1")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
