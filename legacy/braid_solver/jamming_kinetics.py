"""Approach-to-jamming kinetics at FIXED time-steps.

At a fixed T we keep proposing paths and record, at every successful accept,
the cumulative number of attempts so far. That gives N(attempts): the growth
curve that asymptotes to the saturation ceiling and slows without bound.

We characterise the slowdown two ways:
  - N reached at a sequence of attempt budgets (diminishing returns, concretely)
  - the acceptance rate dN/d(attempts) vs attempts, fit as a power law in the
    tail.  For random sequential adsorption the rate decays algebraically
    (Feder): rate ~ attempts^(-p).  p is the growth-rate exponent you want.
"""

from __future__ import annotations

import math
import random
import sys

from freq_strategy_test import d_coprime_high, d_uniform


def run_budget(T, budget, rng, draw):
    """Run up to `budget` attempts; return list of cumulative-attempts-at-accept."""
    z0, z1 = 0.01, math.pi - 0.01
    step = (z1 - z0) / (T - 1)
    z = [z0 + i * step for i in range(T)]
    sz = [math.sin(v) for v in z]
    iv = [1.0 / v for v in sz]
    cell = 2.0 / T
    mod = max(2, T // 2)
    grids = [{} for _ in range(T)]
    accept_at = []  # cumulative attempts when each path was accepted
    att = 0

    def coll(X, Y):
        for i in range(T):
            xi, yi = X[i], Y[i]
            cx, cy = math.floor(xi / cell), math.floor(yi / cell)
            g = grids[i]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    a = g.get((cx + dx, cy + dy))
                    if a:
                        for px, py in a:
                            if abs(xi - px) <= cell and abs(yi - py) <= cell:
                                return True
        return False

    def ins(X, Y):
        for i in range(T):
            xi, yi = X[i], Y[i]
            grids[i].setdefault((math.floor(xi / cell), math.floor(yi / cell)), []).append((xi, yi))

    while att < budget:
        att += 1
        xs, ys = rng.random(), rng.random()
        bx, by = draw(rng, mod)
        ax2, ay2 = xs, ys
        ax, ay = (1 - xs) / bx, (1 - ys) / by
        if rng.random() > 0.5:
            ax = -ax
        if rng.random() > 0.5:
            ay = -ay
        if rng.random() > 0.5:
            ax2 = -ax2
        if rng.random() > 0.5:
            ay2 = -ay2
        X = [(ax * math.sin(bx * z[i]) + ax2 * sz[i]) * iv[i] for i in range(T)]
        Y = [(ay * math.sin(by * z[i]) + ay2 * sz[i]) * iv[i] for i in range(T)]
        if not coll(X, Y):
            ins(X, Y)
            accept_at.append(att)
    return accept_at


def analyse(accept_at, budget):
    N = len(accept_at)
    print(f"  total accepted: {N} in {budget:,} attempts")
    print("  N reached at attempt budgets:")
    for b in (1_000, 10_000, 100_000, 1_000_000, budget):
        if b > budget:
            continue
        n = sum(1 for a in accept_at if a <= b)
        print(f"    {b:>10,} attempts -> N = {n}")
    # cost to add the next group at several N levels
    print("  attempts to add the *next* group at N =")
    for frac in (0.5, 0.8, 0.9, 0.95, 0.99):
        k = int(N * frac)
        if 1 <= k < N:
            cost = accept_at[k] - accept_at[k - 1]
            print(f"    N={k:4d} ({int(frac*100):2d}% of reached): ~{cost:,} attempts for the next one")
    # power-law fit of acceptance rate vs cumulative attempts, tail half
    pts = []
    lo = N // 2
    for k in range(lo, N - 1):
        a = accept_at[k]
        da = accept_at[k + 1] - accept_at[k]
        if da > 0:
            pts.append((math.log(a), math.log(1.0 / da)))
    if len(pts) > 5:
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        print(f"  tail acceptance-rate ~ attempts^({slope:.2f})   (more negative = faster slowdown)")


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 3_000_000
    print(f"T={T}, Smart freqs ON, budget={budget:,} attempts:")
    aa = run_budget(T, budget, random.Random(11), d_coprime_high)
    analyse(aa, budget)
