"""Independent numerical cross-check of the exact solver.

Densely samples t in (0, pi) and measures the closest approach between every
pair of paths.  If the exact solver says two paths never intersect, the
numerical minimum gap should stay well above 0; if it says they intersect, the
gap should dip to ~0 at some interior time.  Any disagreement is a bug.
"""

from __future__ import annotations

import math

from braid import Path, generate_candidates, max_non_intersecting, paths_intersect


def closest_approach(
    p: Path, q: Path, samples: int = 200_000, margin: float = 0.01
) -> float:
    """Min L-inf distance between p(t) and q(t) on t/pi in [margin, 1-margin].

    The endpoints t=0 and t=pi are excluded: every path collapses to the origin
    there (the Bang/Crunch), so all gaps vanish at the endpoints by definition.
    """
    best = math.inf
    lo = int(samples * margin)
    hi = int(samples * (1 - margin))
    for k in range(lo, hi + 1):
        t = math.pi * k / samples
        gap = 0.0
        for (a, f), (b, g) in zip(p, q):
            d = abs(a * math.sin(f * t) - b * math.sin(g * t))
            if d > gap:
                gap = d
        if gap < best:
            best = gap
    return best


def check(n: int, freqs: list[int], permutation: bool) -> None:
    cands = generate_candidates(n, freqs, permutation=permutation)
    res = max_non_intersecting(cands)
    # 1) The reported maximum set must be pairwise non-intersecting numerically.
    bad = []
    for i in range(len(res.paths)):
        for j in range(i + 1, len(res.paths)):
            gap = closest_approach(res.paths[i], res.paths[j])
            if gap < 1e-2:
                bad.append((i, j, gap))
    tag = "permutation" if permutation else "free"
    print(f"dims={n} freqs={freqs} ({tag}): max={res.count}")
    if bad:
        print(f"  !! numeric collision in 'non-intersecting' set: {bad}")
    else:
        print("  OK: selected set is numerically collision-free")

    # 2) Spot-check that exact verdict matches numeric for ALL candidate pairs.
    mism = 0
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            exact = paths_intersect(cands[i], cands[j])
            gap = closest_approach(cands[i], cands[j], samples=60_000)
            numeric = gap < 1e-2
            if exact != numeric:
                mism += 1
                if mism <= 5:
                    print(
                        f"  mismatch {cands[i]} vs {cands[j]}: "
                        f"exact={exact} numeric_gap={gap:.2e}"
                    )
    print(f"  exact-vs-numeric mismatches over all pairs: {mism}")


if __name__ == "__main__":
    check(2, [3, 10], permutation=True)
    check(3, [3, 10], permutation=False)
