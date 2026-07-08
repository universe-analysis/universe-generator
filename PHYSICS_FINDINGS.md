# Braided Universe — Physics Findings

The running findings log. The model went through one major revision — the
**torus + phase model** (2026-07-02/05) replaced the original hard-wall model —
and one major methodological correction (the boundary-artifact reckoning of
2026-07-01). **Part I is the current picture**; Part II records how the old
headline fell; Part III preserves the hard-wall-era results with their present
status. The distilled version of Part I is the paper (`paper/main.tex`,
"A packing model of a closed universe", Bentley & Bentley, 2026-07-06); the
day-by-day evidence is the lab notebook (`docs/lab-notes/`).

## Part I — The current picture (torus + phase model)

### I.0 The model, as now defined

Per spatial axis, `x(z) = a·[sin(bz + f) − sin f] + a1·sin z` over the
conformal loop z ∈ (0, π), with:

- **Rapidity budget on the wiggle alone:** `|a·b| = 1` (generally
  `Σ_{k≥2} |k·a_k| = 1`) — a built-in speed of light.
- **Free comoving coordinate:** the frequency-1 amplitude `a1 ~ U(−1,1)` is
  exempt from the budget; in comoving coordinates `X = x/sin z` it is a static
  position. Homogeneity is a property of the *measure*, not an assumption.
- **Toroidal comoving space:** X lives on a circle of circumference 2
  (minimum-image exclusion); no wall, no statable center.
- **Phases:** even frequencies carry a free phase `f ~ U[0,π)`; odd
  frequencies provably already have their full phase freedom (sign flip = π
  shift), so they carry none. This parity asymmetry threads through everything.
- Exclusion radius 1/T (Chebyshev, minimum-image) at each of T sampled times;
  Nyquist band k ≤ T; RSA to a 1e-6 acceptance-rate cutoff.

Differences from the old hard-wall model: the budget no longer couples wiggle
to position (`|a·b| + |a2| = 1` is gone, and with it the slow "wall-hugger"
population), the wall at |X| = 1 is gone, and phases exist.

### I.1 Headline: the jam-count exponent

The jammed count is a clean power law `N_sat ~ T^D` with **no geometric
carrier** (see I.3) — a *packing-number* exponent measuring the dynamical
capacity of the joint all-times exclusion, strictly below the spatial
dimension:

```
                         full ladder        T ≥ 100 (converged)
3+1 (T=20–200, 5 seeds)  2.273 ± 0.003      2.331 ± 0.003
2+1 (T=40–400, 8 seeds)  1.396 ± 0.002      1.375 ± 0.004
```

- **D ≈ 2.33 (3+1) and ≈ 1.38 (2+1)** at the 1e-6 cutoff. Local pairwise
  slopes are flat across both ladders (no crossover); the phased 3+1 ladder is
  the cleanest (scatter 0.066) and firmly pins 2.33.
- Ladder tops: ⟨N_sat⟩ ≈ 1.6×10⁵ (3+1, T=200), 6.1×10³ (2+1, T=400).
- **D/d ≈ 0.78 (3+1), 0.69–0.70 (2+1).** The 3+1 value is numerically close
  to 7/3; whether that is exact is open.
- Model lineage of the exponent: hard-wall 2.46 → torus 2.33 (3+1);
  1.58 → 1.38–1.42 (2+1). The ~0.15 drop in both dimensions is the loss of the
  small-footprint wall-huggers the joint budget used to admit.
- **Cutoff caveat:** in 3+1 a cutoff decade moves D < 0.02, but the 2+1 torus
  exponent moves ~0.04 per decade (1.382 @ 1e-6 vs 1.422 @ 1e-7) — always
  quote the cutoff with 2+1 values.

### I.2 Exact homogeneity

The matter distribution of the jam is *exactly* uniform, in two senses:

- **Across space:** wrapped (minimum-image) correlation dimension of the
  turnaround slice converges onto the space-filling value with the probe-shape
  gap collapsed — **D2 = 3.019 (sphere) / 3.010 (cube)** in 3+1 and
  **2.032 / 2.025** in 2+1 (seed-averaged, T ≥ 100).
- **Across time:** fixing one universe and scanning every conformal time, the
  rms comoving spread sits at **√⅓ ≈ 0.5774 — the exact uniform-torus value —
  flat over the entire loop** (range 0.0004). The bang/crunch collapse lives
  entirely in physical coordinates (x = X·sin z → 0); the comoving arrangement
  never knows it is happening.

Homogeneity of the proposal measure is built in; the finding is that **jamming
preserves it exactly** while thinning the population by orders of magnitude
and selecting hard on everything else (parity, phase). All structure lives in
the joint, whole-loop character of the exclusion — none in where anything sits.

Methodological control: the naive open-boundary estimator applied to the same
periodic clouds reads ~2.89 (3+1) / ~1.93 (2+1) with a spurious sphere/cube
split — the artifact of Part II, reproduced on demand.

### I.3 The exponent has no geometric carrier

Pooling all timesteps of a jam into one cloud, the local box-counting
dimension sweeps 1.31 (finest scales — smooth 1D curves) → 2.76 (mid) → 2.89
(coarse — space-filling), crossing the count exponent 2.33 only in passing,
with **no plateau at D**. Treating time as a fourth axis gives the same sweep
(~0 → ~3.7; 0.75% of 4D cells occupied at the cell scale). Verified on three
model variants in a row (hard-wall, torus, torus+phase).

For point packings, packing number and box dimension coincide; for extended
worldlines under a joint across-time constraint they separate, and this model
separates them cleanly. One connection survives: *within* one universe, box
dimension is geometric (1 → d); *across* universes, with the box tied to the
exclusion cell 2/T, the count grows as T^D.

Torus-specific detail: the finest-scale 4D local dimension is ~0 ("dust") —
every torus worldline carries the full unit swing and the comoving frame
magnifies motion by 1/sin z near the endpoints, so consecutive timestep
samples land in distinct cells. Real model physics, not artifact.

### I.4 Parity structure of the survivors

The comoving wiggle `sin(bz+f)/sin z` is symmetric about the turnaround for
odd b (at rest there) and antisymmetric for even b (swings through). The jam's
composition along this axis is definite and resolution-stable (flat above
T ≈ 50):

- **48% of packed worldlines are entirely odd-frequency; 82.7% of all accepted
  frequencies are odd.**
- Coprimality permits at most one even frequency per worldline, so the
  population splits into a **cold** component (all-odd, exactly at rest at the
  turnaround) and a **mobile** one (one even axis) — motion at maximum
  expansion is always single-axis.
- **The selection is dynamical:** stage-resolved acceptance (ordered dumps,
  T=200) shows evens accepted at 20–27% across the band while the universe is
  sparse, squeezed to 3–13% among the final pre-jam quartile. What fits into
  the last gaps holds still when the universe is largest.
- **Parity is phase-blind** — the surprise. A phase near quadrature lets an
  even worldline sit almost at rest at the turnaround, yet the odd preference
  is unchanged with phases on (48.0% vs 47.3% all-odd). So the preference is
  set by whole-loop threading geometry, not turnaround kinematics.
- The hard-wall era's "no evens below b ≈ 34" forbidden zone is **absent on
  the torus** (it was a joint-budget artifact); low-b evens appear when
  sparse and are simply the first casualties of jamming.

### I.5 The arcsine speed law — and jamming selects on phase

At the turnaround the mover speed collapses to `v = |cos f|` on the (single)
even axis. With proposal phases uniform, mover speeds should follow the
**arcsine law** `P(v) = 2/(π√(1−v²))` — a parameter-free prediction with an
integrable divergence at the speed cap v = 1 (a built-in relativistic
pile-up), ⟨v⟩ = 2/π, ⟨v²⟩ = ½.

Measured (T=200, 1.6×10⁵ movers, 5 seeds): the packed state keeps the arcsine
shape *including the pile-up at the cap*, but tilted toward slow movers:

```
⟨v⟩  = 0.539   (proposal 0.637)
⟨v²⟩ = 0.404   (proposal 0.500)
```

Nothing in the acceptance rule references velocity, so **jamming itself
selects on phase**: fast movers sweep more comoving ground near the turnaround
and are harder to thread, so survivors' phases concentrate toward quadrature.
The suppression factor is the same at every T — selection acting on a degree
of freedom invisible to any single-time snapshot.

### I.6 Equation of state at the turnaround

`w = P/ρ = ⅓·⟨E v²⟩/⟨E⟩` gives **w = 0.070, flat across the entire ladder
(T = 20–200)** and dictionary-robust: E ∝ Σb (wave) and E ∝ physical arc
length (string) agree to 1% (0.0692 vs 0.0700 at T=200). For any parity-blind
dictionary the arcsine law predicts w = ⅙ × (mobile fraction) ≈ 0.087; the
measured value sits **20% below — the phase-selection tilt of I.5 expressed
thermodynamically**. The universe at maximum expansion is matter-like
(w ≪ 1/3), its pressure carried by a single-axis-moving minority.

### I.7 Phases are a prefactor knob, nothing else

The one-knob comparison (same grid/seeds/cutoff, phases on/off): phases add
**+6.9% to the jam count in 3+1 and +2.8% in 2+1** at the ladder top, and
reshape turnaround kinematics (I.5) — but the count exponents, wrapped D2,
uniformity-over-time, parity composition, and the spacetime box-count sweeps
are all **phase-blind**. How much packs changes; how packing *scales* does not.

### I.8 Robustness checks

- **Collision shape does not matter:** re-packing with spherical (L2)
  exclusion instead of cubic (Chebyshev) leaves the dimension estimators
  unchanged (correlation 2.791 vs 2.795, box 2.674 vs 2.677 on the hard-wall
  ladder) — the `corrdim3d_euclid` campaign.
- **Cutoff:** a full 1e-6 vs 1e-7 re-run (hard-wall, 95 jobs each) moved
  correlation D2 by 0.002 and box D by 0.011 while N dropped 18% — the N ratio
  is flat in T, so it shifts intercepts, not slopes. 1e-6/1e-7 are both in the
  scaling regime; 1e-8 is not worth a decade of compute. (2+1 caveat in I.1.)
- **Engine A/Bs:** the sparse-grid engine (`--sparse`, VRAM ~ N·T vs T⁴)
  validated against stored dense runs (`analysis/compare_jamming.py`).

## Part II — How the old headline fell (the estimator reckoning, 2026-07-01)

The project's long-standing headline — "packing dimension D ≈ 2.79 (3+1) /
1.82 (2+1), mildly multifractal" — was the **correlation dimension of the
turnaround slice measured with an open-boundary estimator on a bounded
model**, and it fell in two steps:

1. **The boundary artifact.** The naive `C(r)` estimator on a *uniform*
   control cloud (true D = 3) returns 2.87 (sphere) / 2.82 (cube): points near
   the wall miss outside-the-box neighbours, dragging the slope down and
   splitting the probes. Interior-only or wrapped measurement restores 3.00
   for both probes. The sphere/cube gap (2.79 vs 2.73) and the sub-3 value
   were finite-window boundary effects, not structure. (The earlier
   gap-prediction work — ~85% of the gap explained by the local-slope ramp —
   was a correct decomposition of an artifact.)
2. **The direct measurement.** The packing dimension proper is the count
   exponent `N_sat ~ T^D` (Chris's framing): hard-wall 2.46, torus 2.33 —
   and I.3 shows no static set carries it geometrically.

Numeric lineage for the record (3+1): 2.44 (pre edge-fix count exponent) →
2.69 (CELL/2 edge-collision fix) → "2.79" (correlation, boundary artifact) →
**2.46 (hard-wall count exponent) → 2.33 (torus, current)**. On the torus the
slice dimension is exactly 3 (I.2), so there is no fractal matter cloud at
all; the sub-dimensional number was always the *count*, not the geometry.

Consequences for downstream hard-wall results:

- **Dq multifractal spectrum (D1..D5 = 2.80..2.72) and the Menger-sponge
  hypothesis** (`INVESTIGATION_menger.md`): built on the artifact-bearing
  open-boundary estimator; superseded. The torus slices are exactly uniform,
  so there is no spatial multifractal to match to log20/log3.
- **Sphere-vs-cube probe universality, box/ball-cover comparisons:** artifact
  diagnostics in hindsight; kept in Part III for the methods record.

## Part III — Hard-wall era results (historical; status-flagged)

Results below are from the old model (joint budget `|a·b| + |a2| = 1`, hard
wall at |X| = 1, no phases), mostly at the 1e-7 cutoff. Some describe genuine
physics of that variant; none should be quoted as the current model without
re-verification.

- **Warm-fraction "geometric phase transition"** (0% movers below T ≈ 40,
  switching on to ~42% at T=120, first surviving evens b ≥ 34): specific to
  the joint budget, which forced even worldlines to buy small swings with
  high frequency. **Gone on the torus** (I.4) — evens face no frequency floor;
  the mobile fraction is instead set by the smooth parity selection. The
  odd/even rest/moving mechanism itself (symmetry about the turnaround)
  carries over and underlies I.4–I.5.
- **Equation-of-state history w(z)** (stiff w ~ 0.5–0.65 near bang/crunch,
  radiation crossings at z ≈ 0.8/2.4, matter-like cusp at the turnaround;
  dictionary-robust): measured on the hard-wall model only. The turnaround
  value is superseded by w = 0.070 (I.6); **the full w(z) history of the
  torus+phase model is open** — it needs careful treatment of peculiar
  velocities away from sin z = 1.
- **Knot selection** (realized knot proxy K of packed worldlines ~28–35% above
  the sampler input, enrichment growing with T; coprime fraction pinned at
  6/π² with no number-theoretic growth exponent): hard-wall measurement;
  the mechanism (high frequency → small footprint → easier fit) should
  survive on the torus but has not been re-measured.
- **Pair correlation g(r)** at the turnaround (soft correlation hole ~CELL,
  roughly collapsing in CELL units across T): hard-wall; not yet repeated.
- **"Movers are not heavier"** (cold vs warm mean total frequency 272.9 vs
  274.1 at T=120): hard-wall; the parity-blind-dictionary logic reappears in
  I.6.
- **Correlation-dimension campaign numbers** (seeded D2 = 2.79 ± 0.002 sphere
  / 2.73 ± 0.002 cube; Dq spectrum; box-counting convergence studies;
  structure-factor attempt): see Part II — estimator artifact on the bounded
  model; retained in git history and `TODO_corrdim.md` for the methods record.

## Caveats that bind the current results

- 1e-6 acceptance-rate cutoff states, not literal jamming; the flat local
  slopes say we are well inside the scaling regime, but the extrapolation to
  true jamming (a Feder-law analog for curve packings) is open.
- Everything measured is the **single-budget-frequency sector with singleton
  groups**. Multi-frequency worldlines, unique groups, and subpaths are
  defined by the model but unexplored (engine support has just landed — see
  below).
- The physical mappings assume a mass dictionary (E ∝ b or E ∝ arc length);
  robustness is shown for those two, both parity-blind.
- CELL = 2/T is a resolution, not a physical scale; only dimensionless /
  universal quantities (exponents, w, distribution shapes, fractions) are
  defensible predictions.

## Open questions / next steps

1. **Multi-frequency worldlines** — the budget split across many components
   (the 2+1 engine now supports K frequency terms per axis). Is D a property
   of the sector or of the budget? Sharpest next measurement.
2. **Subpaths / unique groups** — post-jam filling by paths that join existing
   groups (engine `--subpaths` just landed, matching the viewer). How much
   subpath capacity does a jam retain, and what is a multi-place group,
   physically?
3. **w(z) history of the torus+phase model** — does the stiff→matter bathtub
   of the hard-wall model survive?
4. **Convergence of D**: is 2.33 exact (7/3?); does D/d tend to a constant?
   T > 200 sparse-grid runs (no VRAM ceiling) can extend the 3+1 ladder;
   an analytic 1+1 reduction might settle the mechanism.
5. **Strange-attractor reading** — make it precise in *parameter* space:
   dimension spectrum of the accepted (b, f, a1) population, seed-divergence
   (sensitive dependence), a map whose iteration RSA approximates.
6. **Jamming extrapolation** — cutoff → literal jam bias correction.
7. Re-verify the transferable hard-wall results (knot selection, g(r)) on the
   torus+phase model.
