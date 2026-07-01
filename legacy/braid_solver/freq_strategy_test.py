"""Does smarter frequency selection improve the 2-term torus generator?

Compares frequency-draw strategies in the faithful model, measuring:
  - N_jam: paths packed at saturation (higher = denser universe)
  - eff:   accepted / attempted  (higher = less wasted generation work)
"""

from __future__ import annotations

import math
import random

MAX_ATTEMPTS = 50
SAT_FAILS = 500


def bake_z(T):
    z0, z1 = 0.01, math.pi - 0.01
    step = (z1 - z0) / (T - 1)
    return [z0 + i * step for i in range(T)]


def run(T, rng, draw):
    """draw(rng, mod_max) -> (bx, by). Returns (N_jam, attempts)."""
    z_cache = bake_z(T)
    cell = 2.0 / T
    mod_max = max(2, T // 2)
    sin_z = [math.sin(z) for z in z_cache]
    inv_z = [1.0 / s for s in sin_z]
    grids = [{} for _ in range(T)]
    attempts = 0

    def collides(X, Y):
        for i in range(T):
            xi, yi = X[i], Y[i]
            cx, cy = math.floor(xi / cell), math.floor(yi / cell)
            g = grids[i]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    arr = g.get((cx + dx, cy + dy))
                    if not arr:
                        continue
                    for (px, py) in arr:
                        if abs(xi - px) <= cell and abs(yi - py) <= cell:
                            return True
        return False

    def insert(X, Y):
        for i in range(T):
            xi, yi = X[i], Y[i]
            k = (math.floor(xi / cell), math.floor(yi / cell))
            grids[i].setdefault(k, []).append((xi, yi))

    def make_one():
        nonlocal attempts
        for _ in range(MAX_ATTEMPTS):
            attempts += 1
            xs, ys = rng.random(), rng.random()
            bx, by = draw(rng, mod_max)
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
            X = [(ax * math.sin(bx * z) + ax2 * sin_z[i]) * inv_z[i] for i, z in enumerate(z_cache)]
            Y = [(ay * math.sin(by * z) + ay2 * sin_z[i]) * inv_z[i] for i, z in enumerate(z_cache)]
            if not collides(X, Y):
                insert(X, Y)
                return True
        return False

    n = 0
    consec = 0
    while consec <= SAT_FAILS:
        if make_one():
            n += 1
            consec = 0
        else:
            consec += 1
    return n, attempts


def d_uniform(rng, mod_max):
    return rng.randrange(mod_max) + 2, rng.randrange(mod_max) + 2


def d_distinct(rng, mod_max):
    bx = rng.randrange(mod_max) + 2
    by = rng.randrange(mod_max) + 2
    while by == bx:
        by = rng.randrange(mod_max) + 2
    return bx, by


def d_coprime(rng, mod_max):
    while True:
        bx = rng.randrange(mod_max) + 2
        by = rng.randrange(mod_max) + 2
        if math.gcd(bx, by) == 1:
            return bx, by


def d_highbias(rng, mod_max):
    # sample weighted toward higher frequencies (triangular)
    def hi():
        a, b = rng.randrange(mod_max), rng.randrange(mod_max)
        return max(a, b) + 2
    return hi(), hi()


def d_coprime_high(rng, mod_max):
    while True:
        bx, by = d_highbias(rng, mod_max)
        if math.gcd(bx, by) == 1:
            return bx, by


STRATS = {
    "uniform (current)": d_uniform,
    "distinct bx!=by": d_distinct,
    "coprime(bx,by)": d_coprime,
    "high-frequency bias": d_highbias,
    "coprime + high-bias": d_coprime_high,
}


if __name__ == "__main__":
    import sys

    T = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    seeds = 3
    print(f"2-term torus, T={T}, {seeds} seeds each:")
    print(f"{'strategy':22s} {'N_jam':>8s} {'eff(acc/att)':>13s}")
    for name, draw in STRATS.items():
        Ns, accs, atts = [], [], []
        for s in range(seeds):
            n, att = run(T, random.Random(500 + s), draw)
            Ns.append(n)
            accs.append(n)
            atts.append(att)
        N = sum(Ns) / seeds
        eff = sum(accs) / sum(atts)
        print(f"{name:22s} {N:8.0f} {eff:13.3f}")
