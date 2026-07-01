"""Analyse the growth-rate campaign curves in /tmp/campaign.

For each (T, mode, seed) curve N(attempts) we extract:
  - N at decade milestones (1e6, 1e7, 1e8, 1e9) and the per-decade increment
  - B  = paths per decade  (slope of N vs log10(attempts), tail fit)
  - p  = tail exponent of the acceptance rate  (rate ~ attempts^p);
         p = -1 means N ~ log(attempts) exactly (logarithmic, never saturates)
results are averaged over seeds and tabulated by (T, mode).
"""

from __future__ import annotations

import glob
import math
import os
import re
from collections import defaultdict


def load(path: str) -> list[tuple[float, float]]:
    pts = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            a, n = line.split(",")
            pts.append((float(a), float(n)))
    return pts


def n_at(pts: list[tuple[float, float]], a: float) -> float | None:
    """N reached by attempt budget a (last point with attempts <= a)."""
    val = None
    for att, n in pts:
        if att <= a:
            val = n
        else:
            break
    return val


def linfit(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def metrics(pts: list[tuple[float, float]]) -> dict:
    final_a, final_n = pts[-1]
    # B: paths per decade over the last ~2 decades
    lo = final_a / 100.0
    tail = [(a, n) for a, n in pts if a >= lo]
    b = linfit([math.log10(a) for a, n in tail], [n for a, n in tail])
    # p: acceptance-rate exponent (rate ~ a^p) over the same tail
    rate_pts = []
    for (a0, n0), (a1, n1) in zip(tail, tail[1:]):
        da, dn = a1 - a0, n1 - n0
        if da > 0 and dn > 0:
            rate_pts.append((math.log10((a0 + a1) / 2), math.log10(dn / da)))
    p = linfit([x for x, _ in rate_pts], [y for _, y in rate_pts]) if len(rate_pts) > 4 else float("nan")
    return {"final_a": final_a, "final_n": final_n, "B": b, "p": p}


def main() -> None:
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    decades = [1e6, 1e7, 1e8, 1e9]
    for path in sorted(glob.glob("/tmp/campaign/T*_*.csv")):
        m = re.match(r"T(\d+)_(\w+?)_s(\d+)\.csv", os.path.basename(path))
        if not m:
            continue
        T, mode, seed = int(m.group(1)), m.group(2), int(m.group(3))
        pts = load(path)
        if len(pts) < 6:
            continue
        d = metrics(pts)
        d["decades"] = [n_at(pts, a) for a in decades]
        groups[(T, mode)].append(d)

    def avg(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return sum(xs) / len(xs) if xs else float("nan")

    def std(xs):
        xs = [x for x in xs if x is not None]
        m = avg(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0

    print(f"{'T':>4} {'mode':>8} {'seeds':>5} {'N@1e9':>10} {'B/decade':>10} {'p(rate)':>9}  N at 1e6/1e7/1e8/1e9")
    for (T, mode) in sorted(groups):
        rs = groups[(T, mode)]
        bn = avg([r["final_n"] for r in rs])
        bB = avg([r["B"] for r in rs])
        bBs = std([r["B"] for r in rs])
        bp = avg([r["p"] for r in rs])
        dvals = [avg([r["decades"][i] for r in rs]) for i in range(4)]
        dstr = " / ".join(f"{v:.0f}" if v == v else "—" for v in dvals)
        print(f"{T:>4} {mode:>8} {len(rs):>5} {bn:>10.0f} {bB:>6.0f}±{bBs:<3.0f} {bp:>9.2f}  {dstr}")
    print("\np ≈ -1 → N ~ log(attempts) (logarithmic, never truly saturates).")
    print("B/decade = paths gained per 10× more attempts.")


if __name__ == "__main__":
    main()
