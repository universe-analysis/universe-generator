"""Complexity-growth measurement on the 3+1 model (threeplusone_3sphere.html).

Faithful port: three spatial axes, each two-term with the slope-1 constraint
(a*b + a2 = 1), comoving by sin(tau), 3D Chebyshev collision at CELL = 2/T.
Raising T zooms in. We measure N_jam(T) and fit N_jam ~ T^D.
"""

from __future__ import annotations

import math
import random
import sys

MAX_ATTEMPTS = 50
SAT_FAILS = 500


def bake_z(T: int) -> list[float]:
    z0, z1 = 0.01, math.pi - 0.01
    step = (z1 - z0) / (T - 1)
    return [z0 + i * step for i in range(T)]


def run_to_saturation(T: int, rng: random.Random) -> int:
    z_cache = bake_z(T)
    cell = 2.0 / T
    mod_max = max(2, T // 2)
    grids: list[dict[tuple[int, int, int], list[tuple[float, float, float]]]] = [
        {} for _ in range(T)
    ]
    sin_z = [math.sin(z) for z in z_cache]
    inv_z = [1.0 / s for s in sin_z]
    neigh = [(dx, dy, dw) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dw in (-1, 0, 1)]

    def comoving(ax, ay, aw, bx, by, bw, ax2, ay2, aw2):
        X = [0.0] * T
        Y = [0.0] * T
        W = [0.0] * T
        for i in range(T):
            z = z_cache[i]
            inv = inv_z[i]
            sz = sin_z[i]
            X[i] = (ax * math.sin(bx * z) + ax2 * sz) * inv
            Y[i] = (ay * math.sin(by * z) + ay2 * sz) * inv
            W[i] = (aw * math.sin(bw * z) + aw2 * sz) * inv
        return X, Y, W

    def collides(X, Y, W) -> bool:
        for i in range(T):
            xi, yi, wi = X[i], Y[i], W[i]
            cx, cy, cw = math.floor(xi / cell), math.floor(yi / cell), math.floor(wi / cell)
            g = grids[i]
            for dx, dy, dw in neigh:
                arr = g.get((cx + dx, cy + dy, cw + dw))
                if not arr:
                    continue
                for (px, py, pw) in arr:
                    if abs(xi - px) <= cell and abs(yi - py) <= cell and abs(wi - pw) <= cell:
                        return True
        return False

    def insert(X, Y, W) -> None:
        for i in range(T):
            xi, yi, wi = X[i], Y[i], W[i]
            key = (math.floor(xi / cell), math.floor(yi / cell), math.floor(wi / cell))
            grids[i].setdefault(key, []).append((xi, yi, wi))

    def make_one() -> bool:
        for _ in range(MAX_ATTEMPTS):
            xs, ys, ws = rng.random(), rng.random(), rng.random()
            bx = rng.randrange(mod_max) + 2
            by = rng.randrange(mod_max) + 2
            bw = rng.randrange(mod_max) + 2
            ax2, ay2, aw2 = xs, ys, ws
            ax, ay, aw = (1 - xs) / bx, (1 - ys) / by, (1 - ws) / bw
            if rng.random() > 0.5:
                ax = -ax
            if rng.random() > 0.5:
                ay = -ay
            if rng.random() > 0.5:
                aw = -aw
            if rng.random() > 0.5:
                ax2 = -ax2
            if rng.random() > 0.5:
                ay2 = -ay2
            if rng.random() > 0.5:
                aw2 = -aw2
            X, Y, W = comoving(ax, ay, aw, bx, by, bw, ax2, ay2, aw2)
            if not collides(X, Y, W):
                insert(X, Y, W)
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


if __name__ == "__main__":
    Ts = [int(x) for x in sys.argv[1:]] or [40, 80, 160]
    data = []
    for T in Ts:
        vals = [run_to_saturation(T, random.Random(1000 + s)) for s in range(2)]
        m = sum(vals) / len(vals)
        data.append((T, m))
        print(f"  T={T:5d}  CELL={2/T:.5f}  freq<={max(2, T//2):4d}  ->  N_jam={m:.1f}", flush=True)
    print("\nlog-log growth exponent  N_jam ~ T^D :", flush=True)
    for i in range(1, len(data)):
        (t0, n0), (t1, n1) = data[i - 1], data[i]
        if n0 > 0 and n1 > 0:
            print(f"  T {t0}->{t1}:  D = {math.log(n1/n0)/math.log(t1/t0):.2f}", flush=True)
