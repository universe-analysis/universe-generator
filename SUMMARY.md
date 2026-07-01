# Braided-Universe Packing: Summary of Approach and Findings

*A working summary of an exploratory numerical experiment, written to solicit
outside opinions. We are not claiming a breakthrough; we are trying to describe
clearly what we're doing and measuring so others can judge what, if anything,
is of value.*

## 1. The model

We study a toy geometric model of a "braided universe." Time runs over a closed
conformal loop `z ∈ (0, π)` — a universe that expands to `z = π/2` and contracts
back, with `sin(z)` playing the role of the scale factor. Each *worldline* is a
parametric curve built from sinusoids, one per spatial dimension:

```
x(z) = ax · sin(bx · z) + ax2 · sin(z)
```

with integer frequency `bx`, and a "speed-of-light" constraint
`|ax · bx| + |ax2| = 1` that bounds the maximum comoving velocity (a slope-1
limit). The physically meaningful coordinate is the *comoving* position
`X = x / sin(z) ∈ [-1, 1]`. We work in 2+1 (two spatial axes) and 3+1 (three).

Two worldlines "collide" if, at any sampled time step, their comoving positions
fall within a cell size `CELL = 2/T` of each other (Chebyshev distance, per
axis), where `T` is the number of time steps — i.e. the resolution. The comoving
boundary `|X| = 1` is a hard wall (a worldline whose body would cross it is
rejected).

We then ask: **how many mutually non-colliding worldlines can be packed?** This
is a random sequential adsorption (RSA) jamming process — propose random
worldlines, keep the ones that don't collide with any already accepted, until no
more fit.

## 2. The quantity we measure

The jammed count `N` scales with resolution as a power law:

```
N(T) ~ T^D
```

`D` is a packing / box-counting (fractal) dimension: it measures how the number
of distinguishable, non-overlapping worldlines grows as we resolve finer detail
(smaller cells). Informally, it's "how tightly the universe braids detail around
the central worldline." We measure:

- **3+1: D ≈ 2.5–2.6** (between a surface, D=2, and a solid, D=3)
- **2+1: D ≈ 1.6**

A space-filling weave would give D = 3 (or 2 in 2+1); the deficit (`3 − 2.6`)
means the packing is fractal — densely braided but never filling the volume, in
a self-similar way at every scale.

## 3. Method: GPU-accelerated batch-reject RSA

The engine is a CUDA program. Each GPU thread proposes one candidate worldline
and tests it against the current frozen packing using a per-time-step spatial
grid (a hash of accepted points), so collision queries are local. Survivors are
re-checked and committed on the host; the grid is rebuilt as the packing grows.

This matters because the acceptance rate decays roughly as `attempts^(-1)` (see
§4), so approaching jamming requires enormous sampling — we routinely run
**10^11 to 10^14 candidate attempts** per configuration. On the GPUs we use
(consumer RTX 3080s) this is hours; the same depth on a CPU would be impractical
to infeasible. The deep-sampling capability is what makes the jamming question
answerable at all.

## 4. Findings

**The packing is fractal, far below the cell capacity.** The geometric cell cap
is `T^d`. The packing jams at only ~6–20% of that, and the fraction *decreases*
with `T` (as ~`T^(D−d)`), confirming D < d. Example (3+1, fully jammed at low
T): T=5 → N=26, T=6 → 43, T=10 → 195; the 1e-7-cutoff campaign continues
T=18 → 850 … T=60 → 17,615, T=100 → 57,934.

**Three distinct exponents — easy to conflate.** A recurring source of confusion
we had to untangle:
- `D ≈ 2.6` — the packing dimension (N vs resolution); needs multiple `T`.
- `p ≈ −1` — the acceptance-rate decay slope (acceptance vs attempts *within* one
  run). It drifts from −1.13 at T=18 toward −1.0 at T=60. This reflects the
  dimension of the *parameter space* being sampled, not the geometric D.
- A growth-over-attempts exponent. None of these are the same number.

**`p ≈ −1` is the marginal case, and it dominates feasibility.** Because the
acceptance decay sits right at `attempts^(-1)`, the approach to true jamming is
near-logarithmic at high `T`. Time-to-jamming explodes: T=18 jams in hours,
T=60 in days, T=100 in months. This is why higher-T runs use a fixed
acceptance-rate cutoff rather than running to completion — and it's a property
of the process, not the hardware.

**D is robust to how we sample candidates.** We tested three proposal
distributions (uniform, edge-weighted, center-weighted, differing in how the
amplitude split is drawn). They move *where* worldlines pack but leave D
unchanged (2.52–2.65, overlapping within seed error). This argues D is an
intrinsic property of the geometry, not an artifact of the sampler.

**Finite-size effects at small T.** Below T≈18 the count falls below the power
law (T=5, 6 dip; T=10 is on it). The fractal scaling needs enough scales to
develop, so very small universes under-pack.

## 5. Methodological notes (the part we think is worth scrutiny)

- **Analytic reference models.** We bracket D between two exactly solvable
  references — random fill of `T^2` vs `T^3` cells (coupon-collector decay,
  closed form) — so the measured exponent has known rails.
- **Jamming verified, not assumed.** A flat plateau in N is necessary but not
  sufficient. We bound the residual insertion probability from the number of
  consecutive failed attempts (e.g. < 10^-12 after ~10^12 dry attempts), and we
  lean on RSA's monotonicity (available volume only shrinks, so a plateau can't
  reactivate) to distinguish true jamming from a slow crawl.
- **Direct hypothesis testing via instrumentation.** When a collaborator
  proposed that the sampler was over-filling the center and starving the edges,
  we added engine diagnostics (occupancy histograms; a probe that fires fresh
  candidates at the frozen jammed state) and showed the occupancy was actually
  flat-to-edge-peaked — refuting the hypothesis with data rather than argument.
- **Band discipline.** The frequency cap is set at the Nyquist limit
  (`maxfreq = T`); we caught and corrected a run that had silently used the
  half-Nyquist default, which changed low-T counts by ~40%.

## 6. Relationship to known mathematics

Essentially all the primitives are textbook, and we want to be explicit about
that:
- Random sequential adsorption / jamming, Feder's law for the approach to
  jamming, jamming fractions below close packing — standard statistical physics.
- Box-counting / fractal dimension — standard.
- The coupon-collector cell-fill reference — standard.
- Nyquist sampling / aliasing — standard signal processing.

We are not introducing new mathematics. We are applying known tools to a
specific, somewhat unusual object.

## 7. What might be of value (stated cautiously)

We are genuinely unsure how much of this is interesting, which is why we're
asking. Candidates, in decreasing confidence:

1. **The model itself.** The braided-universe construction — sinusoidal
   worldlines on a closed conformal-time loop with a slope-1 constraint, and the
   "how much detail braids around the centerline" question — is original to this
   project (a collaborator's idea). Its physical meaning, if any, is an open
   question; we are treating it purely as a well-defined geometric packing
   problem and measuring its fractal dimension.

2. **GPU acceleration making the measurement tractable.** The technique
   (batch-reject RSA on a GPU with a spatial grid) is not a new algorithm. But
   applying it to push a continuous, parameter-constrained, geometrically
   non-trivial RSA process out to 10^12–10^14 attempts — deep enough to actually
   reach or tightly bound jamming for a process with marginal (`~t^-1`) decay —
   is the kind of thing that would have been impractical historically. If
   there's anything "uncharted" here, we think it's this: using cheap GPU compute
   to make deep-jamming measurements of awkward constrained objects routine,
   rather than heroic.

3. **The empirical discipline.** The bracketing, residual-bound jamming
   verification, sampler-invariance check, and direct-instrumentation hypothesis
   tests are a reasonable template for "measure a fractal dimension you can't
   derive, and defend the number." Not novel individually; possibly a useful
   worked example.

## 8. Honest limitations

- The model's physical relevance is unestablished — it's a toy.
- True jamming is only reached at low `T`; mid/high `T` rely on an
  acceptance-rate cutoff, so the global D fit is a slight underestimate of the
  jammed value and carries a known high-T sag.
- Most runs use a single GPU vendor / consumer FP64, single or few seeds.
- We have not done a formal finite-size-scaling extrapolation of D, nor error
  analysis beyond seed scatter.

---

*Questions we'd most value outside opinion on: (a) Is the measured D meaningful
or just "fractal packings give fractal dimensions"? (b) Is the deep-jamming-via-
GPU angle actually opening anything, or is it routine? (c) Is the model itself
worth taking seriously as anything beyond a curiosity?*
