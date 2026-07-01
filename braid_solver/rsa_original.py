"""Complexity-growth measurement on the ORIGINAL collision model.

Faithful Python port of the generator in ``twoplusone_2torus.html``:

  path:   x(z) = ax*sin(bx*z) + ax2*sin(z),   y(z) likewise
  slope:  ax*bx + ax2 = 1   (the speed limit / amplitude*frequency = 1)
  comoving:  X = x/sin(z),  Y = y/sin(z)        (removes the Bang/Crunch funnel)
  collide:   per-timestep Chebyshev distance <= CELL = 2/T  (square hash)
  stack:     random sequential adsorption until 500 consecutive misfits

The scale knob is ``T`` (time_steps): it sets the threshold (2/T), the max
frequency (T/2) and the time sampling all at once -- raising T is "zooming in".
We measure the jamming count N_jam(T) and fit N_jam ~ T^D.
"""

from __future__ import annotations

import math
import random

MAX_ATTEMPTS = 50  # maxAttemptsPerFunction in the original
SAT_FAILS = 500  # consecutive misfits => saturated


def bake_z(T: int) -> list[float]:
    z0, z1 = 0.01, math.pi - 0.01
    step = (z1 - z0) / (T - 1)
    return [z0 + i * step for i in range(T)]


def comoving(z_cache: list[float], ax, ay, bx, by, ax2, ay2):
    X = [0.0] * len(z_cache)
    Y = [0.0] * len(z_cache)
    for i, z in enumerate(z_cache):
        inv = 1.0 / math.sin(z)
        X[i] = (ax * math.sin(bx * z) + ax2 * math.sin(z)) * inv
        Y[i] = (ay * math.sin(by * z) + ay2 * math.sin(z)) * inv
    return X, Y


def run_to_saturation(T: int, rng: random.Random) -> int:
    """Stack paths via RSA until saturated; return the jamming count."""
    z_cache = bake_z(T)
    cell = 2.0 / T
    mod_max = max(2, T // 2)
    grids: list[dict[tuple[int, int], list[tuple[float, float]]]] = [
        {} for _ in range(T)
    ]

    def collides(X, Y) -> bool:
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

    def insert(X, Y) -> None:
        for i in range(T):
            xi, yi = X[i], Y[i]
            key = (math.floor(xi / cell), math.floor(yi / cell))
            grids[i].setdefault(key, []).append((xi, yi))

    def make_one():
        for _ in range(MAX_ATTEMPTS):
            xs, ysp = rng.random(), rng.random()
            bx = rng.randrange(mod_max) + 2
            by = rng.randrange(mod_max) + 2
            ax2, ay2 = xs, ysp
            ax, ay = (1 - xs) / bx, (1 - ysp) / by
            if rng.random() > 0.5:
                ax = -ax
            if rng.random() > 0.5:
                ay = -ay
            if rng.random() > 0.5:
                ax2 = -ax2
            if rng.random() > 0.5:
                ay2 = -ay2
            X, Y = comoving(z_cache, ax, ay, bx, by, ax2, ay2)
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
    return n


def measure(T_values: list[int], seeds: int = 3) -> list[tuple[int, float]]:
    out = []
    for T in T_values:
        vals = [run_to_saturation(T, random.Random(1000 + s)) for s in range(seeds)]
        out.append((T, sum(vals) / len(vals)))
    return out


if __name__ == "__main__":
    import sys

    Ts = [int(x) for x in sys.argv[1:]] or [40, 80, 160, 320]
    data = measure(Ts)
    print("T (zoom) -> N_jam (mean saturation count):")
    for T, n in data:
        print(f"  T={T:5d}  CELL={2/T:.5f}  freq<= {max(2, T // 2):4d}  ->  N_jam={n:.1f}")
    print()
    print("log-log growth exponent  N_jam ~ T^D :")
    for i in range(1, len(data)):
        (t0, n0), (t1, n1) = data[i - 1], data[i]
        if n0 > 0 and n1 > 0:
            D = math.log(n1 / n0) / math.log(t1 / t0)
            print(f"  T {t0}->{t1}:  D = {D:.2f}")
