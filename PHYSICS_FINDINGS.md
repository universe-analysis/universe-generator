# Braided Universe — Physics Findings

The current state of knowledge, stated as of 2026-07-10. This file describes
what we know about the model as it is now defined and measured; the
day-by-day evidence trail (including superseded models, estimators, and
samplers) lives in the lab notebook (`docs/lab-notes/`) and git history. The
distilled narrative is the paper (`paper/main.tex`, "A packing model of a
closed universe", Bentley & Bentley — **note: the paper predates the uniform
re-anchor below and needs revision**).

## 1. The model

Per spatial axis, `x(z) = a·[sin(bz + f) − sin f] + a1·sin z` over the
conformal loop z ∈ (0, π), with:

- **Rapidity budget on the wiggle alone:** `|a·b| = 1` (generally
  `Σ_{k≥2} |k·a_k| = 1` with K terms per axis; K = 2 — one wiggle plus the
  comoving term — unless stated) — a built-in speed of light.
- **Free comoving coordinate:** the frequency-1 amplitude `a1 ~ U(−1,1)` is
  exempt from the budget; in comoving coordinates `X = x/sin z` it is a static
  position. Homogeneity is a property of the *measure*, not an assumption.
- **Toroidal comoving space:** X lives on a circle of circumference 2
  (minimum-image exclusion); no wall, no statable center.
- **Phases:** even frequencies carry a free phase `f ~ U[0,π)`; odd
  frequencies provably already have their full phase freedom (sign flip = π
  shift), so they carry none. This parity asymmetry threads through everything.
- **Uniform proposals:** wiggle frequencies are drawn uniformly on the
  Nyquist band 2 ≤ b ≤ T, with no correlations between axes and no
  arithmetic constraints. The proposal measure is the maximum-entropy one on
  the parameter space; every selection effect reported below is produced by
  jamming alone.
- Exclusion radius 1/T (Chebyshev, minimum-image) at each of T sampled times;
  RSA to an acceptance-rate cutoff (1e-6 unless stated).

## 2. Headline: the jam-count exponent

The jammed count is a clean power law `N_sat ~ T^D` with **no geometric
carrier** (section 4) — a *packing-number* exponent measuring the dynamical
capacity of the joint all-times exclusion, strictly below the spatial
dimension:

- **3+1: D ≈ 2.2** at the 1e-6 cutoff. Full-ladder fits over T = 20–160 give
  2.20–2.23 across independent campaigns (`pack3d_e6` 2.201, the terms-sweep
  arms 2.226–2.240); the local slope steepens with T (T = 80→160 two-point:
  2.27), so a quoted D needs its T window. D/d ≈ 0.73–0.76, consistent with
  3/4.
- **2+1: D ≈ 1.36–1.40, cutoff-conditional.** Full ladder T = 20–300: 1.363
  at 1e-6, 1.397 at 1e-7; the 2+1 exponent drifts upward ~0.03–0.05 per
  cutoff decade with no sign of saturating, so 2+1 values must always be
  quoted with their cutoff, and a depth extrapolation is needed before any
  2+1 headline. D/d ≈ 0.68–0.70.
- **The exponent is terms-invariant.** Enriching the dictionary from 2 to 10
  terms per axis leaves D flat in 3+1 (2.231(3), 2.240(2), 2.226(2) for
  terms = 2, 3, 10 — `frequni3d_e6`); the packing exponent is a property of
  the budget and the torus, not of the dictionary size.
- Ladder tops (1e-6): ⟨N_sat⟩ ≈ 7.0×10⁴ (3+1, T=160), 3.2×10³ (2+1, T=300).

## 3. The packing number is knob-robust (PACK campaigns)

Systematic invariance tests of N(T) under the model's free knobs
(`pack{3,2}d_e{6,7}`, `packterms{3,2}d_e6`, 2026-07-09):

- **Cutoff depth moves the prefactor, not the law.** A decade of extra depth
  (1e-6 → 1e-7) lifts every N by ×1.150 (3+1) / ×1.203 (2+1), uniformly in T,
  while the fitted exponent moves ≤ 0.02 (3+1) / 0.034 (2+1).
- **There is no jamming plateau in reach.** Decay ladders to 1e-8 show N
  still growing 10–16% per cutoff decade with no curvature: every absolute N
  is a cutoff-conditional number; only exponents, ratios, and distribution
  shapes are portable.
- **Dictionary size barely moves N.** At fixed T, going from 2 to 10 terms
  per axis changes N by at most +10% (3+1, T=160) and stays within ±5% in
  2+1, with a small (≲1%) dip at terms = 3. The per-T enhancement curves do
  not collapse when replotted against the pool fraction (terms−1)/(T−1), so
  what little effect exists tracks the raw term count.

## 4. Exact homogeneity

The matter distribution of the jam is *exactly* uniform, in two senses:

- **Across space:** wrapped (minimum-image) correlation dimension of the
  turnaround slice converges onto the space-filling value with the
  probe-shape gap collapsed — **D2 = 3.019 (sphere) / 3.010 (cube)** at the
  3+1 ladder top and **2.025 / 2.019** at the 2+1 ladder top (5 seeds,
  `pack{3,2}d_e6` dumps), approaching 3 / 2 from above as T grows.
- **Across time:** fixing one universe and scanning every conformal time, the
  rms comoving spread sits at **√⅓ ≈ 0.5774 — the exact uniform-torus value —
  flat over the entire loop** (range 0.0004). The bang/crunch collapse lives
  entirely in physical coordinates (x = X·sin z → 0); the comoving arrangement
  never knows it is happening.

Homogeneity of the proposal measure is built in; the finding is that **jamming
preserves it exactly** while thinning the population by orders of magnitude
and selecting hard on everything else (parity, phase). All structure lives in
the joint, whole-loop character of the exclusion — none in where anything sits.

Estimator caution: a naive open-boundary C(r) estimator applied to the same
periodic clouds reads ~2.90 (3+1) / ~1.96 (2+1) with a spurious sphere/cube
split — edge depletion drags the slope down. On a periodic domain the wrapped
estimator is the only self-consistent one; a uniform D = 3 control cloud
returns exactly 3.000 under it.

## 5. The exponent has no geometric carrier

Pooling all timesteps of a jam into one cloud, the local box-counting
dimension sweeps from ~1.3 (finest scales — smooth 1D curves) through the
count exponent only in passing, up to the space-filling value at coarse
scales, with **no plateau at D**. Treating time as a fourth axis gives the
same sweep (~0 → ~3.7). Verified on three model variants in a row; D is not
the fractal dimension of any static point set the packing produces.

For point packings, packing number and box dimension coincide; for extended
worldlines under a joint across-time constraint they separate, and this model
separates them cleanly. One connection survives: *within* one universe, box
dimension is geometric (1 → d); *across* universes, with the box tied to the
exclusion cell 2/T, the count grows as T^D.

Torus-specific detail: the finest-scale 4D local dimension is ~0 ("dust") —
every torus worldline carries the full unit swing and the comoving frame
magnifies motion by 1/sin z near the endpoints, so consecutive timestep
samples land in distinct cells. Real model physics, not artifact.

## 6. Parity structure of the survivors

The comoving wiggle `sin(bz+f)/sin z` is symmetric about the turnaround for
odd b (at rest there) and antisymmetric for even b (swings through), so a
worldline's turnaround kinematics are set by the parities of its frequencies:

- With parity-neutral proposals, **jamming selects against even
  frequencies**: worldlines with one even axis are accepted at ~0.8× their
  proposal rate and with two-plus even axes at ~0.7×, but none are evicted
  outright.
- The packed population splits **23% cold** (all-odd, exactly at rest at the
  turnaround) / **77% movers** (at least one even axis); 39% of packed
  worldlines carry two or more even axes, so **multi-axis motion at maximum
  expansion is common**.
- **Parity selection is phase-blind**: a phase near quadrature lets an even
  worldline sit almost at rest at the turnaround, yet the odd preference is
  unchanged with phases on. The preference is set by whole-loop threading
  geometry, not turnaround kinematics.
- There is no forbidden zone in frequency: low-b evens are admitted while the
  universe is sparse and are simply the first casualties of jamming.

## 7. The arcsine speed law — and jamming selects on phase

At the turnaround, each even axis contributes speed `v = |cos f|`. With
proposal phases uniform, per-axis mover speeds follow the **arcsine law**
`P(v) = 2/(π√(1−v²))` — a parameter-free prediction with an integrable
divergence at the per-axis speed cap v = 1 (a built-in relativistic
pile-up), ⟨v⟩ = 2/π.

The packed state keeps the arcsine shape *including the pile-up at the cap*,
tilted toward slow movers: per-axis ⟨v⟩ = 0.603 vs the proposal 0.637.
Nothing in the acceptance rule references velocity, so **jamming itself
selects on phase**: fast movers sweep more comoving ground near the
turnaround and are harder to thread, so survivors' phases concentrate toward
quadrature — selection acting on a degree of freedom invisible to any
single-time snapshot.

Causal structure: the model's exclusion and budget are per-axis, so its light
cone is a **Chebyshev (per-axis) cone**, not a Euclidean one — 27% of movers
exceed Euclidean speed 1 while respecting the per-axis cap. Any Euclidean
kinematic quantity inherits a metric convention; both are reported below.

## 8. Equation of state at the turnaround

`w = P/ρ = ⅓·⟨E v²⟩/⟨E⟩` at maximum expansion gives **w = 0.145 (Chebyshev
metric) / 0.193 (Euclidean metric)** on the packed ensemble. The law-like
quantity is not the absolute value but the suppression: for a parity-blind
dictionary the arcsine law predicts w = 0.185 / 0.252 from the proposal
measure, and **jamming cools w by 20–24% below that prediction in both
metrics** — the phase-selection tilt of section 7 expressed
thermodynamically. The universe at maximum expansion is matter-like
(w ≪ 1/3) with pressure carried by the mover majority.

## 9. Phases are a prefactor knob, nothing else

The one-knob comparison (same grid/seeds/cutoff, phases on/off): phases add
**+6.9% to the jam count in 3+1 and +2.8% in 2+1** at the ladder top, and
reshape turnaround kinematics (section 7) — but the count exponents, wrapped
D2, uniformity-over-time, parity composition, and the spacetime box-count
sweeps are all **phase-blind**. How much packs changes; how packing *scales*
does not.

## 10. Robustness checks

- **Cutoff:** exponents are depth-robust in 3+1 (≤ 0.02/decade) and
  cutoff-conditional in 2+1 (~0.03–0.05/decade); the cutoff otherwise shifts
  only prefactors (section 3).
- **Terms:** exponent and N both dictionary-insensitive (sections 2–3).
- **Collision shape:** re-packing with spherical (L2) instead of cubic
  (Chebyshev) exclusion left the dimension estimators unchanged (Δ ≤ 0.004)
  in the predecessor model; not yet repeated on the torus.
- **Engine A/Bs:** the sparse-grid engine (`--sparse`, VRAM ~ N·T vs T⁴)
  validates against stored dense runs (`analysis/compare_jamming.py`).

## Caveats that bind the current results

- Cutoff states, not literal jamming (section 3: no plateau through 1e-8);
  absolute N values are cutoff-conditional everywhere. The extrapolation to
  true jamming (a Feder-law analog for curve packings) is open.
- Everything measured is the **singleton-group sector, subpaths off**; the
  terms knob is explored for N and D (invariant), but kinematics and
  homogeneity have only been measured at terms = 2.
- The physical mappings assume a mass dictionary (E ∝ b or E ∝ arc length),
  and Euclidean-metric quantities carry the Chebyshev-vs-Euclidean convention
  choice (section 7).
- CELL = 2/T is a resolution, not a physical scale; only dimensionless /
  universal quantities (exponents, w suppression, distribution shapes,
  fractions) are defensible predictions.
- On the periodic domain, only wrapped (minimum-image) estimators are valid;
  open-boundary estimators read low with probe-shape artifacts (section 4).

## Open questions / next steps

1. **2+1 depth extrapolation** — the 2+1 exponent is cutoff-limited;
   extrapolate the decay ladders to quote a jamming-limit value.
2. **Constant-fraction terms ladder** — terms tied to T at fixed pool
   fraction (needs a kMaxWiggle bump and a terms-per-T campaign knob), to
   close the pool-fraction question of section 3.
3. **w(z) history** of the current model — does a stiff → matter bathtub
   shape appear away from the turnaround? Needs careful treatment of
   peculiar velocities away from sin z = 1.
4. **Is D/d → 3/4 exact in 3+1?** Extend the ladder past T = 200 with the
   sparse engine (no VRAM ceiling); an analytic 1+1 reduction might settle
   the mechanism.
5. **Subpaths / unique groups** — post-jam filling by paths that join
   existing groups (engine `--subpaths`). How much subpath capacity does a
   jam retain, and what is a multi-place group, physically?
6. **Fixed-point ensemble** — iterate the jam's parameter measure back in as
   the proposal measure; does RSA have a self-consistent fixed point?
7. **Strange-attractor reading** — dimension spectrum of the accepted
   (b, f, a1) population in parameter space; seed-divergence; a map whose
   iteration RSA approximates.
8. **Knot selection and g(r)** — re-measure on the current model (the old
   measurements predate the torus + uniform-proposal definition).
9. **Kinematics at terms > 2** — the arcsine/phase-selection story is only
   measured in the 2-term sector.
