"""Max non-intersecting braid count for 4+1 and 5+1 (permutation model).

Tests the n·2^(n-1) conjecture (1, 4, 12, 32, 80). Uses a fast Tomita-style
max-clique (greedy-colouring bound) since the candidate graphs get large
(4D: 384 nodes, 5D: 3840). The max depends on the frequency set (as in 3D),
so we try several pairwise-coprime sets and report the best.
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from braid import generate_candidates, paths_intersect  # noqa: E402


def build_adj(cands) -> list[set[int]]:
    m = len(cands)
    adj: list[set[int]] = [set() for _ in range(m)]
    for i in range(m):
        ci = cands[i]
        for j in range(i + 1, m):
            if not paths_intersect(ci, cands[j]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def max_clique(adj: list[set[int]]) -> int:
    """Exact maximum clique, Tomita-style with greedy-colouring bound."""
    best = [0]

    def color_sort(P: set[int]) -> list[tuple[int, int]]:
        order = sorted(P, key=lambda v: len(adj[v] & P), reverse=True)
        classes: list[set[int]] = []
        colored: list[tuple[int, int]] = []
        for v in order:
            for c, cls in enumerate(classes):
                if not (adj[v] & cls):
                    cls.add(v)
                    colored.append((v, c + 1))
                    break
            else:
                classes.append({v})
                colored.append((v, len(classes)))
        colored.sort(key=lambda x: x[1])
        return colored

    def expand(r: int, P: set[int]) -> None:
        if not P:
            if r > best[0]:
                best[0] = r
            return
        for v, c in reversed(color_sort(P)):
            if r + c <= best[0]:
                return
            expand(r + 1, P & adj[v])
            P = P - {v}

    expand(0, set(range(len(adj))))
    return best[0]


def coprime(fs) -> bool:
    return all(math.gcd(a, b) == 1 for a, b in combinations(fs, 2))


def best_over_sets(n: int, freq_sets) -> tuple[int, tuple]:
    best_n, best_set = 0, None
    for fs in freq_sets:
        if not coprime(fs):
            continue
        cands = generate_candidates(n, list(fs), permutation=True)
        m = max_clique(build_adj(cands))
        print(f"    freqs {fs}: max = {m}  ({len(cands)} candidates)", flush=True)
        if m > best_n:
            best_n, best_set = m, fs
    return best_n, best_set


if __name__ == "__main__":
    print("n·2^(n-1) conjecture: 1, 4, 12, 32, 80")
    print("\n4+1 (try several good 4-frequency sets):")
    b4, s4 = best_over_sets(
        4,
        [(3, 5, 7, 8), (3, 4, 5, 7), (3, 5, 8, 11), (3, 7, 10, 13), (2, 3, 5, 7), (3, 4, 7, 11)],
    )
    print(f"  4+1 best: {b4}  with {s4}   (conjecture: 32)")
