# Braided Universe — Physics Findings

The current state of knowledge, stated as of 2026-07-10. This file describes
what we know about the model as it is now defined and measured; the
day-by-day evidence trail (including superseded models, estimators, and
samplers) lives in the lab notebook (`docs/lab-notes/`) and git history. The
distilled narrative is the paper (`paper/main.tex`, "A packing model of a
closed universe", Bentley & Bentley — **note: the paper predates the uniform
re-anchor below and needs revision**).

## 0. Orientation — what this universe is

*For someone arriving from scratch. Everything here is restated precisely in
the numbered sections.*

**The universe.** A closed toy cosmos with one cycle: conformal time z runs
from a big bang at z = 0 to a crunch at z = π, and the scale factor is
sin z — space expands, turns around at z = π/2 (maximum expansion, "the
turnaround"), and recollapses. Space itself is a comoving torus: each of the
d spatial axes (we study d = 3 and d = 2, written 3+1 and 2+1) is a circle of
circumference 2, so there is no boundary and no statable center. Physical
positions are x = X·sin z: every history begins and ends at zero physical
size, but the comoving coordinate X — where things sit on the torus — never
collapses.

**Matter.** A particle is not a point with an equation of motion; it is an
entire worldline, chosen once, whole. Each is a closed-form sinusoidal thread
through the full history: a static home position on the torus (reached via
the `a1·sin z` term), plus a peculiar-motion wiggle `a·sin(bz + f)` with an
integer frequency b and, on even frequencies, a free phase f. A hard budget
`|a·b| = 1` caps every axis's peculiar velocity — a built-in speed of light
(per axis, so the causal cone is a box, not a ball). A worldline has no
forces acting on it and no choices to make; everything about it is fixed at
birth by the tuple (b, f, a1) per axis.

**The one law.** Matter is created by random sequential adsorption (RSA):
propose a worldline by drawing its parameters from the uniform (maximum-
entropy) measure, and accept it iff it stays at least one exclusion radius
(1/T, in the wrap-around Chebyshev sense) away from every already-accepted
worldline at every one of T sampled moments of the history. Accepted means
permanent; rejected means gone. Repeat until essentially nothing more fits —
the universe is *jammed*. There is no energy, no interaction, no dynamics,
no fine-tuned initial condition: the only physics is that two things cannot
be in the same place at the same time, enforced across an entire cosmic
history at once. A worldline is not admitted because of where it is now; it
is admitted because it can *coexist with everything else, forever*.

**The resolution knob.** T is the single control parameter: it is
simultaneously the number of time samples, the particle diameter (2/T), and
the highest peculiar frequency allowed (the Nyquist band b ≤ T). Making T
larger is refining the same universe — smaller particles, finer time
sampling, richer motion — which is why every result is stated as a scaling
in T. One RSA run at one seed is one universe; campaigns generate ensembles
of them (the GPU engines in `cuda/`, orchestrated across a small fleet by
`braidlab/`).

**The experiments.** Four questions organize everything measured so far.
*How much fits?* — the jam count N grows as a clean power T^D whose exponent
D is not the dimension of space or of any fractal the matter traces out; it
is a new, dynamical capacity exponent of the whole-history packing problem
(sections 2–3, 5). *Where does it sit?* — the jammed matter is exactly
uniform across space and across time, indistinguishable from ideal
homogeneity (section 4). *What survives?* — jamming is a selection: it
prunes even frequencies, and it prunes phases that make particles fast at
the turnaround, sculpting a relativistic arcsine speed distribution it was
never told about (sections 6–7). *What does it feel like?* — read
thermodynamically, the turnaround universe is matter-like, with a pressure
suppression below the proposal ensemble that is the model's most law-like
kinematic number (section 8).

**Why this is interesting.** Cosmology usually gets homogeneity, matter
content, and velocity distributions from dynamics plus initial conditions.
Here a single constraint — mutual exclusion over a complete closed history —
generates statistical regularities of all three kinds on its own, and they
are robust to every knob we have turned (cutoff depth, dictionary size,
phases). The project is an attempt to map which features of a universe can
be *packing effects*.

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

- **3+1: D = 2.321 ± 0.001 (stat), converged** at the 1e-6 cutoff. The local
  slope rises from ~2.15 at the ladder bottom (low-T curvature) and sits flat
  across five consecutive rungs from T = 160 through 360 (CONVERGE sparse
  ladder, 5 seeds; 2.316–2.328 with ±0.01 errors). D/d = 0.774: the plateau
  rules out D/d = 3/4 (2.25) by ~9σ and lands just below 7/3 ≈ 2.333 — with
  the +0.016/decade cutoff shift, the jamming-limit value plausibly reaches
  7/3 (open question 4).
- **2+1: jamming-limit D∞ = 1.434 ± 0.021** (Feder-law extrapolation of the
  kinetic curves: shared approach exponent p = 0.120 voted by the
  depth-constrained small-T tails; the p systematic moves D∞ by < 0.001).
  Finite-cutoff ladders approach it from below — full-ladder fits 1.393 /
  1.407 / 1.440 at 1e-6/1e-7/1e-8 — so quote cutoff values as
  approximations to D∞. D/d = 0.717, distinctly below 3+1's 0.774: the
  ratio is not dimension-universal.
- **The exponent is terms-invariant.** Enriching the dictionary from 2 to 10
  terms per axis leaves D flat in 3+1 (2.231(3), 2.240(2), 2.226(2) for
  terms = 2, 3, 10 — `frequni3d_e6`); the packing exponent is a property of
  the budget and the torus, not of the dictionary size.
- Ladder tops (1e-6): ⟨N_sat⟩ ≈ 4.6×10⁵ (3+1, T=360), 3.2×10³ (2+1, T=300).

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

1. **3+1 jamming-limit exponent via the approach law** — the 2+1 version is
   done (D∞ = 1.434 ± 0.021, section 2); in 3+1 every stored curve through
   1e-8 rails (no detectable ceiling curvature), so the p vote needs deep
   small-T 3+1 runs (T = 20–60 at 1e-9, cheap). Does the extrapolation
   carry 2.321 to 7/3?
2. **Constant-fraction terms ladder** — terms tied to T at fixed pool
   fraction (needs a kMaxWiggle bump and a terms-per-T campaign knob), to
   close the pool-fraction question of section 3.
3. **w(z) history** of the current model — does a stiff → matter bathtub
   shape appear away from the turnaround? Needs careful treatment of
   peculiar velocities away from sin z = 1.
4. **Is the jamming-limit 3+1 exponent exactly 7/3?** The converged 1e-6
   plateau (2.321) sits one cutoff-shift below 7/3; a converged-window
   ladder at 1e-7 (and T = 400+ rungs, cheap on the sparse engine) would
   test whether the limit lands on 2.333. An analytic 1+1 reduction might
   settle the mechanism.
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
