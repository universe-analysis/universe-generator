"""Clean D extraction with error bars + power-law residual check.

For each T: average ceiling over seeds (N_final when the tail is essentially
saturated, else a guarded N_sat extrapolation). D = log-log slope; error from
per-seed refits; residuals flag whether N_sat(T) is a clean power law.
"""

from __future__ import annotations

import glob
import math
import re
import sys
from collections import defaultdict


def load(p):
    return [(float(a), float(n)) for a, n in (l.split(",") for l in open(p).read().splitlines()[1:])]


def lsfit(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-15:
        return float("nan"), float("nan")
    s = (n * sxy - sx * sy) / den
    return s, (sy - s * sx) / n


def ceiling(pts):
    """Best ceiling estimate for one curve, with the headroom %."""
    fa = pts[-1][0]
    lo = fa / 100
    nfin = pts[-1][1]
    rate = [
        (math.log10((a0 + a1) / 2), math.log10((n1 - n0) / (a1 - a0)))
        for (a0, n0), (a1, n1) in zip(pts, pts[1:])
        if (a0 + a1) / 2 >= lo and a1 > a0 and n1 > n0
    ]
    if len(rate) < 5:
        return nfin, 0.0  # tail flat -> saturated
    p, _ = lsfit([x for x, _ in rate], [y for _, y in rate])
    if p != p or p >= -1.05:
        return nfin, 0.0
    q = p + 1
    X = [a**q for a, _ in pts if a >= lo]
    Y = [n for a, n in pts if a >= lo]
    _, icept = lsfit(X, Y)  # N -> intercept as a^q -> 0
    if icept != icept or not (nfin <= icept < 1.6 * nfin):
        return nfin, 0.0
    return icept, (icept - nfin) / icept * 100


def analyze(indir, dim, label):
    g = defaultdict(dict)  # T -> {seed: pts}
    for f in glob.glob(f"{indir}/*.csv"):
        T = int(re.search(r"T(\d+)", f).group(1))
        s = int(m.group(1)) if (m := re.search(r"[sS](\d+)", f)) else 0
        g[T][s] = load(f)
    Ts = sorted(g)
    print(f"\n=== {label}  (dim d={dim}) ===")
    per_seed_ns = defaultdict(dict)  # seed -> {T: nsat}
    avg = {}
    for T in Ts:
        vals, hds = [], []
        for s, pts in g[T].items():
            ns, hd = ceiling(pts)
            vals.append(ns)
            hds.append(hd)
            per_seed_ns[s][T] = ns
        avg[T] = sum(vals) / len(vals)
        print(f"  T={T:4d}: N_sat={avg[T]:8.0f}  headroom~{sum(hds)/len(hds):4.1f}%  seeds={len(vals)}")
    D, b = lsfit([math.log10(t) for t in Ts], [math.log10(avg[t]) for t in Ts])
    res = [math.log10(avg[t]) - (D * math.log10(t) + b) for t in Ts]
    # per-seed D spread as an error estimate
    seedD = []
    for s, d in per_seed_ns.items():
        ts = [t for t in Ts if t in d]
        if len(ts) >= 2:
            sd, _ = lsfit([math.log10(t) for t in ts], [math.log10(d[t]) for t in ts])
            seedD.append(sd)
    spread = (max(seedD) - min(seedD)) / 2 if len(seedD) > 1 else float("nan")
    print(f"  => D = {D:.3f} ± {spread:.3f} (seed spread),  D/d = {D/dim:.3f}")
    print(f"     max |residual| = {max(abs(r) for r in res):.4f} dex  "
          f"({'clean power law' if max(abs(r) for r in res) < 0.02 else 'some curvature'})")
    return D, spread


if __name__ == "__main__":
    d2, e2 = analyze("/tmp/p2d", 2, "2+1 precision (6 T, 3 seeds, 5e11)")
    d3, e3 = analyze("/tmp/dc3d", 3, "3+1 D-campaign (4 T, 2 seeds, 1e12)")
    print(f"\n=== SUMMARY ===")
    print(f"  2+1: D = {d2:.2f}±{e2:.2f}, D/d = {d2/2:.3f}")
    print(f"  3+1: D = {d3:.2f}±{e3:.2f}, D/d = {d3/3:.3f}")
    print(f"  codimension ratio rises with dimension: {d2/2:.2f} -> {d3/3:.2f}" if d3/3 > d2/2 else "  ratio ~constant")
