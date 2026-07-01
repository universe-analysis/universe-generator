# Braid Solver — exact non-intersection counts for the constrained model

Answers Chris's question: under the constrained single-frequency-per-dimension
rules, **how many worldlines can you braid around a local origin so that no two
ever occupy the same point at the same time?** The answer is computed *exactly*
(no floating point), so "they can NEVER overlap" is a proof, not a measurement.

## The model

- Time `t` runs over the open loop `(0, π)`. The endpoints are the Bang/Crunch:
  every path sits at the origin there, so they are excluded by convention.
- In `n` dimensions you are given `n` integer frequencies (frequency `1` is
  disallowed). Each dimension of each path is a single component `±sin(f·t)`
  using one of those frequencies.
- Two paths **intersect** if some interior `t` makes *every* coordinate match
  simultaneously.

## Why it is exact

For integer frequencies, `a·sin(f t) = b·sin(g t)` holds only at times that are
**rational multiples of π**. Each coordinate therefore contributes a finite set
of candidate times `r = t/π ∈ (0,1)` (as `fractions.Fraction`). Two paths
collide iff the intersection of these sets across all coordinates is non-empty.
All arithmetic is exact rationals — there is no rounding, so a "non-intersecting"
verdict is mathematically certain over all time.

## Usage

```bash
# Canonical 1D / 2D / 3D sweep:
python solve_braid.py --report

# A specific case (permutation model: n frequencies placed one-per-axis):
python solve_braid.py --dims 2 --freqs 3 10

# Free model: every axis may reuse any of the frequencies:
python solve_braid.py --dims 3 --freqs 3 10 --free
```

`--report` prints the maximum count and an explicit example strand set for each
case. Each path renders like `(x=+sin3t, y=-sin10t)` — paste straight into
Desmos `(x(t), y(t), t)` or the 3+1 simulator.

## Results (validated)

| dims | frequencies | model       | max strands | min interior gap |
|------|-------------|-------------|-------------|------------------|
| 1    | {100}       | —           | 1           | —                |
| 2    | {3, 10}     | permutation | **4**       | 0.20             |
| 3    | {3, 10}     | free        | **8**       | 0.11             |
| 3    | {3, 7, 10}  | permutation | **12**      | 0.10             |

"Min interior gap" is the smallest L∞ separation between any two selected
strands over `(0, π)` (amplitude is 1), confirming these are robust braids, not
near-misses. The permutation-model pattern is `n · 2^(n-1)` (1, 4, 12, …).

Note: the manual Desmos counts (2 in 2D, 6 in 3D) are undercounts — e.g. in 2D
the four strands form two antipodal pairs, so counting diameters gives 2.

## Verification

- `python verify_numeric.py` — independent dense-sampling cross-check: confirms
  every exact verdict matches a numerical closest-approach scan, and that the
  selected maximum sets are interior-collision-free.
- `python -m pytest tests/` — unit tests.

No third-party dependencies (standard library only).
