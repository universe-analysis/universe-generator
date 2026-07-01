# Braided Universe — Physics Findings (since the reviewer thread)

What we learned once we stopped measuring only the packing dimension D and
started extracting physical observables from the jammed/cutoff configurations.
Everything here is from the 3+1 model, edge-weighted sampler, full Nyquist band
(maxfreq = T), at the 1e-7 acceptance-rate cutoff unless noted.

## 0. What we built to get here

- **Engine parameter dump** (`--dump-params`): writes each accepted worldline's
  parameters (a, b, a2 per axis), so the full phase space can be reconstructed
  analytically at any conformal time z -- in particular the turnaround z=pi/2.
- Offline analysis at z=pi/2: positions `X = a*sin(b*pi/2)+a2`, velocities
  `dX/dz = a*b*cos(b*pi/2)`, pair correlation g(r), velocity distributions, and
  the equation of state.
- A param ladder T = 18,24,32,40,48,60,80,100,120 (single seed each).

## 1. A geometric phase transition (the headline)

The "warm" (moving at the turnaround) fraction vs resolution:

```
T:     18   24   32    40     48     60     80    100    120
warm:  0%   0%   0%   0.4%  14.5%  22.8%  32.7%  37.8%  42.0%
```

- **Below T ~= 40 the universe is exactly 100% cold** at maximum expansion --
  not approximately, literally zero movers.
- A near-discontinuous switch-on around T = 40-48, then **decelerating growth**
  toward an apparent asymptote (increments +8, +10, +5, +4 -> looks like it
  converges, plausibly ~50%; not yet pinned).

**Mechanism (confirmed by the data).** A worldline's comoving trajectory
`sin(b z)/sin(z)` is *symmetric* about the turnaround for **odd** b (sits at a
velocity extremum -> at rest) and *anti-symmetric* for **even** b (swings
through -> moving). An even worldline's swing amplitude ~ `(1-|a2|)/b`, so only
*high-frequency* (small-swing) even worldlines can thread the RSA packing. Below
T~=40 the band (maxfreq=T) contains no even integer large enough -- the geometry
*forbids* motion. Above it, high even frequencies unlock and the symmetry breaks.
The first even frequencies to survive were b >= 34.

## 2. Velocity structure at the turnaround

- **Cold component:** all-odd-frequency worldlines, exactly at rest at z=pi/2.
- **Warm component:** worldlines with an even frequency. The coprimality
  constraint allows *at most one* even among (bx,by,bw), so movers move along a
  **single axis**.
- **Speed distribution** (T=120, 38,661 movers): peaked at low speed (~0.1),
  declining to a **hard cutoff exactly at 1.0** -- the slope-1 constraint acts as
  a clean relativistic cap (c=1). So the warm component is mostly "lukewarm" with
  a genuine relativistic tail.

## 3. Pair correlation g(r) at the turnaround (Path B)

- A short-range **correlation hole** near r ~ CELL that **roughly collapses
  across T in CELL units** -- the short-range structure is approximately
  resolution-independent (the test that separates real prediction from CELL
  artifact).
- It is **softer than a hard hole**: exclusion is enforced over *all* timesteps,
  not this single slice, so two worldlines may be moderately close at the
  turnaround if they separate elsewhere.

## 4. Corrections to intuition (where the data overruled a guess)

- The "even movers are heavy and dominate rho" guess is **false**. Cold and warm
  worldlines have **nearly identical mean total frequency** (272.9 vs 274.1 at
  T=120) and mean proper length (1.487 vs 1.481). Because each worldline carries
  three frequencies, the single forced-high even frequency barely moves the
  total. Movers contribute to rho strictly in proportion to their *number*
  (~42%), not more.

## 5. Equation of state at the turnaround -- dictionary-robust

Using `w = P/rho = (1/3) * sum(E v^2)/sum(E)` with the slope-1-capped velocity:

```
w (E ~ b,      Quantum Wave)  = 0.0419
w (E ~ length, String)        = 0.0425     (agree to 1.6%)
reference: dust 0 | radiation 1/3 | stiff 1
```

The turnaround equation of state is **matter-like (w ~= 0.04)** and is the same
under both natural mass dictionaries. The robustness comes from both dictionaries
being cold/warm-blind (both scale with frequency-like quantities), so this holds
for any frequency/length-based weighting -- a velocity-discriminating dictionary
could differ.

## 6. Equation-of-state history w(z) -- the strongest result

Evaluating the kinetic pressure of the fixed T=120 packing at every conformal
time z gives a symmetric "bathtub":

```
near bang/crunch (z -> 0, pi):  w ~ 0.5-0.65   (stiff, above radiation)
crosses radiation w=1/3:        z ~ 0.8 and 2.4
at the turnaround (z=pi/2):     w ~ 0.04 (sharp cusp), ~0.13 just off it
```

- **Stiff/kinetic near the singularities, matter-dominated at maximum
  expansion**, crossing radiation on the way in and out -- a coherent closed-
  universe history.
- **Dictionary-robust across the *entire* cycle**: E~b and E~length overlay
  everywhere, not just at the turnaround.
- The turnaround minimum is a **razor-thin cusp**: cold worldlines are at rest
  only exactly at pi/2 and (being high-frequency) wake up almost instantly on
  either side.

## 7. The scaling ladder (context)

The 1e-7-cutoff packing dimension over T = 18..200 (edge, full Nyquist) fits
`N ~ T^2.47`, stable in the high-T window. This is a *suppressed* (cutoff) value;
the fully-jammed low-T anchors (T=5,6,10) give local slopes 2.8-3.0, so the true
jammed D is higher (~2.6+) and the cutoff fit is a lower bound. (The bias is the
high-T under-jamming; correcting it raises D, it does not point down.)

## 8. Correlation dimension -- a jamming-free measure of D

The box-counting exponent `N_sat ~ T^D` needs the *jammed count*, which is
permanently cutoff-limited at high T (true jamming is logarithmically
unreachable). The **correlation dimension** is a second, independent definition
of fractal dimension that reads the *spatial arrangement* of a single packing
instead of the count scaling, so it never touches the jamming wall. For the
turnaround comoving cloud (each worldline's 3D position at z=pi/2 -- the matter
distribution at maximum expansion), the correlation integral
`C(r) = pairs within r / total pairs ~ r^D2` is measured over a fixed physical
window r in [0.08, 0.5] (`analyze_correlation_dim.py`).

### Seeded fine-grid campaign (definitive)

A dedicated fleet campaign -- fine grid **T = 20..200 step 10, 5 seeds each**
(95 runs, `braidlab corrdim3d`, see CAMPAIGN/ORCHESTRATOR docs) -- pins the
result with real error bars:

- **Correlation D2 (spheres) = 2.79 ± 0.002**, dead flat from T~=80 through 200
  (clean-window mean 2.789); per-T SEM ~0.001-0.003 in the converged window.
- **Correlation D2 (cubes) = 2.73 ± 0.002** -- the probe-shape drift, confirmed.
- **Box-counting (within-cloud) rises from below to ~2.70-2.74**, meeting the
  cube estimate at the high-T end (T=150-190 touch ~2.74).
- Low-T (T<=40) is inflated by CELL intrusion as before, and the only region
  with visible error bars (SEM up to 0.025 at T=20).

This supersedes the single-seed numbers below as the definitive measurement:
**the 3+1 turnaround packing dimension is D ≈ 2.79 (spheres) / 2.73 (cubes)**,
resolution-independent and seed-stable. The single-seed ladder is kept below for
per-T detail and the mechanism narrative.

Across the full (older, single-seed) param ladder T = 18..160:

```
T:        18    24    32    40    48    60    80   100   120   140   160
corr D2  3.83  3.75  3.23  3.00  2.89  2.86  2.81  2.81  2.80  2.80  2.80
box  D   1.21  1.70  1.93  2.26  2.43  2.52  2.52  2.60  2.67  2.71  2.71
```

- **Converges to D = 2.80.** The correlation dimension settles from above and is
  flat at 2.80 from T=80 through T=160 (the clean-window mean is 2.796). The
  box-counting estimate on the same cloud rises from below (still 2.71 at T=160,
  climbing). Two independent estimators bracketing 2.80 from opposite sides.
- **Resolution-independent.** In the clean window the C(r) curves for high T
  collapse onto one master curve, despite T=160 having ~10x the worldlines and
  2.6x the max frequency of T=60. Box-counting D drifts with the cutoff; this
  does not.
- **Probe-shape independent.** Repeating the correlation integral with cubic
  (max-norm L-inf) neighbourhoods instead of Euclidean (L2) spheres gives D2 =
  2.74 vs 2.80 at T=160 -- agreement to ~0.06. The dimension is a property of the
  point distribution, not of the ruler shape. (The small residual is the same
  mild non-monofractality as the ramp, not an artifact.) So three estimators --
  box-counting cubes, correlation spheres, correlation cubes -- all land in
  2.71-2.80.
- **All-cube self-consistent version.** Running the convergence with *both*
  estimators on cube geometry (correlation L-inf + box-counting cubes) is the
  cleanest read: the residual gap nearly closes -- correlation 2.74 and
  box-counting 2.71 at T=160, only 0.03 apart -- and both converge to **D ~=
  2.73** (vs 2.80 sphere-based; the 0.07 is the probe-shape / non-monofractal
  drift). Same-geometry estimators meet rather than merely bracket.
- **Sphere vs cube is universal across estimator families.** Giving box-counting
  a sphere analog (a greedy *ball cover* vs cubic cells; `analyze_box_geometry.py`)
  shows the same ordering as correlation: ball 2.371 > cube-cells 2.338 (gap
  0.033), mirroring correlation sphere 2.796 > cube 2.729 (gap 0.067). The
  isotropic probe always reads higher -- the corner-reach + mild anisotropy is a
  general probe effect, not a correlation-method quirk. (Both box-counting
  variants are the least-converged estimators here, ~2.34-2.37; the ball cover is
  the *smoother* of the two -- the cube-cell count is grid-offset/saturation
  jumpy -- so it is the cleaner D0 estimator if one is wanted.)
- **The low-T points are CELL-contaminated, not physical.** Below T~=60, CELL =
  2/T intrudes into the fixed fit window [0.08, 0.5] and inflates D2 (and
  deflates box D). Those are excluded, not fitted -- the converged value uses
  only T>=80.
- **Resolves the box-counting ambiguity.** The N-vs-T cutoff fit `2.47` (section
  7) was flagged as a suppressed lower bound. This plot is that argument made
  visual: box-counting needs huge T to converge (0.1 short even at T=160), while
  the jamming-free correlation dimension reaches the answer by T~=80. The true
  packing dimension is **2.80**; 2.47 was simply unconverged.
- **The scaling band is CELL (~0.017) -> ~1.** The matter cloud fills the comoving
  box [-1,1]^3 (a hard wall: |X|<=1 by the slope-1 constraint), so r past ~1 just
  saturates C(r)->1 and the slope falls to 0 at the box diagonal ~3.43. The fit
  window sits well below that cliff, so box saturation does not bias the 2.80.
- **Caveats.** Inside the band the local slope still drifts gently (~3.0 small r
  -> ~2.4 large r), so 2.80 carries real ~+/-0.1 scale-dependence -- the cloud is
  *roughly* monofractal, not exactly. Single seed each; turnaround slice (the
  full 4D spacetime-braid version is noisier and not yet quoted).
- **No new runs.** Derived entirely from the existing `--dump-params` ladder
  (T=18..160), the same 1e-7-cutoff packings used for the physics above.

## 9. The 2+1 comparison -- same suppression, slower convergence

A deep 2+1 campaign (`corrdim2d`: T=40..400 step 20, 8 seeds, 152 runs, ~25 min
-- the 2D grid is ~T^3 so it is cheap) measured all three estimates on one set
of packings:

```
                          2+1            3+1
count-scaling box D    1.577 ± 0.002   2.47 (cutoff)
correlation D2 (sphere)  ~1.82          2.79
correlation D2 (cube)    ~1.80          2.73
within-cloud box D    rises to ~1.82  rises to ~2.7
D/d (jamming-free)       ~0.91          ~0.93
converged by             T ~= 320       T ~= 80
```

- **Same suppression story, confirmed.** The count-scaling `N ~ T^1.58` sits well
  below the jamming-free correlation ~1.82 (gap ~0.24, mirroring 3+1's 2.47 vs
  2.79). So `N_sat ~ T^D` is cutoff-suppressed in *both* dimensions; the true
  2+1 packing dimension is **~1.82**, not 1.58.
- **2+1 converges far more slowly.** Where 3+1 was flat by T~=80, the 2+1
  correlation D2 keeps drifting down and only plateaus around **T~=320** (a
  short T<=200 grid would have over-read it at ~1.85). This is the payoff of the
  long, cheap 2D window: 2+1 is *less cleanly self-similar* -- a longer transient
  / more scale-dependence than 3+1.
- **Sphere-cube gap is smaller in 2D** (~0.025 vs 3+1's ~0.06). Consistent with
  the corner-reach explanation of the gap: the cube reaches sqrt(2)*r into 2D
  corners vs sqrt(3)*r in 3D, so less reach -> smaller gap. A cross-dimensional
  check that supports "the gap is probe geometry x the ramp," not an artifact.
- **D/d is near-constant** at ~0.91-0.93 across both dimensions (jamming-free) --
  the pack fills a roughly fixed fraction of its embedding dimension.
- **Caveat.** The within-cloud box-counting is the noisiest 2D estimator
  (oscillates ~1.68-1.82 over T, small SEM so it is real window sensitivity, not
  scatter); it trends up to meet the correlation value (~1.82) at T=400.

## 10. Knot complexity -- the packing selects for knottiness

Knot proxy `K(p,q) = (p/g-1)(q/g-1)`, g=gcd (the reduced torus-knot size: full
when coprime, shrunk ~1/g^2 by a shared factor), averaged over a worldline's
axis-frequency pairs (`analyze_knots.py`).

- **(A) Abstract growth is plain geometry.** Random frequency pairs in [N, 2N)
  give mean `K ~ N^2.01`, with the coprime fraction pinned at `6/pi^2 ~ 0.61`
  and `meanK/N^2` constant at ~1.48 across N=30..30000. So coprimality is
  scale-invariant -- there is **no number-theory-driven growth exponent**; the
  prime/coprime structure is a constant prefactor, not a trend. (Chris's `N+/-1`
  construction is deterministic: odd N -> both neighbours even -> never coprime;
  even N -> both odd -> always coprime.)
- **(B) The packing actively selects for knottier worldlines.** The *realized*
  mean K of packed worldlines runs **~28-35% above the smart-sampler input** and
  **~3.5x above a uniform draw** at the same band, and the enrichment **grows with
  resolution** (realized/smart rises 1.06 -> ~1.28 over T=20..200 in 3+1, and to
  ~1.33 in 2+1) before saturating. Jamming preferentially keeps high-frequency,
  coprime, *knottier* worldlines -- because high frequency means a small wiggle
  (~1/b), a thinner footprint, and an easier fit. This is the
  "frequency drives the dimension" story seen as a **knot-selection effect**, and
  it is the genuine extractable pattern -- it lives in the packing, not in the
  pure number theory.

## 11. Generalized dimensions Dq -- the unifying picture

Computed via the generalized correlation integral (`analyze_spectral_dims.py`),
seed-averaged, over the same window:

```
q:    1      2      3      4      5
3+1  2.800  2.778  2.758  2.740  2.723   (T=200, +/-0.005-0.008)
2+1  1.836  1.821  1.805  1.788  1.770   (T=400, +/-0.002-0.003)
```

- **q=2 reproduces the correlation dimension** (3+1: 2.778 vs the pair-counting
  2.79) -- an independent estimator agreeing, the cross-check we wanted.
- **Mildly multifractal, correctly ordered.** Dq decreases monotonically with q
  (D1 > D2 > ... > D5), a gentle slope of ~0.08 over q=1..5 with tiny error bars.
- **This unifies every small discrepancy we have seen.** Box-counting,
  sphere-correlation (2.79), cube-correlation (2.73), and the 3.0->2.4 local-slope
  ramp are not inconsistencies -- they are different q-moments / probe-weightings
  of one **mildly multifractal** object whose spectrum runs ~2.80 (low q) down to
  ~2.72 (high q). The information dimension D1 ~= 2.80 is the natural headline.
- **Consensus jamming-free dimension: D ~= 2.78 +/- 0.04 (3+1), 1.82 +/- 0.03
  (2+1)**, mildly multifractal.

The structure factor S(k) was also attempted (reciprocal space) but did **not**
yield a clean dimension: the textbook `S(k) ~ k^-D` assumes a *dilute* fractal
cluster, whereas the turnaround cloud is a *dense, box-filling* distribution, so
the box form factor and near-uniform density dominate and the single-realization
S(k) is speckle-noisy. Parked; would need grid-FFT radial averaging plus
form-factor handling to be usable. The real-space Dq is the reliable angle.

## 12. Cutoff convergence -- D is pinned, only N drifts

A full corrdim3d campaign re-run at the **1e-6** acceptance cutoff (vs the 1e-7
baseline; `corrdim3d_e6`, 95 jobs, edge, dumps) settles whether deeper cutoffs
matter for the dimension:

```
                       1e-6        1e-7       difference
correlation D2        2.787       2.789      -0.0024  (< seed SEM)
box-counting D        2.456       2.445      +0.011
N at T=160          152,919     186,423      -18%
N ratio 1e-6/1e-7    ~0.82 flat across T = 60..200
```

- **Both dimensions are cutoff-stable -- and the N ratio explains why.** The
  1e-6 packing has ~82% of the 1e-7 worldlines, and that factor is essentially
  *constant* across T. A constant multiplicative shift in N moves the
  *intercept* of log N vs log T, not the *slope* -- and the slope is the
  box-counting D. So box-counting moves only 0.011 despite N dropping 18%, and
  the jamming-free correlation D2 moves just 0.002.
- **1e-8 is not worth it for either estimator.** Each cutoff decade costs ~a
  decade more attempts for a sub-0.01 change in D. **1e-7 is the right operating
  point.**
- **The box-counting (2.45) vs correlation (2.79) gap is NOT a cutoff problem.**
  It barely moves with cutoff (0.011). It is a *T-convergence-rate* problem:
  box-counting is still climbing at T=200 (needs far higher T), while the
  correlation dimension reaches ~2.79 by T~80. Only higher T or the jamming-free
  methods close that gap -- deeper cutoffs do not.

## Caveats that bind all of the above

- These are **1e-7 cutoff** states, not fully jammed; single seed (~1% engine
  non-determinism from atomic survivor ordering).
- The physical mappings assume a **mass dictionary** (E ~ b or E ~ length). We
  have shown robustness to those *two*, not to all conceivable choices.
- "Proper length" = physical arc length of x(z) over [0, pi/2] -- a defensible
  but specific interpretation.
- CELL = 2/T is a *resolution*, not a physical scale -- absolute scales remain
  resolution-dependent; only dimensionless/universal quantities (g(r) shape,
  w(z), exponents, the warm-fraction transition) are defensible predictions.

## Open questions / next steps

1. **Does the warm fraction converge?** Param dumps at T=140, 160 would pin the
   asymptote (~50%?). If finite, it is a dimensionless constant of the geometry.
2. **Confirm at true jamming / multiple seeds** -- do the transition, w, and w(z)
   survive? (Cheap at low/mid T, hard at high T.)
3. **A velocity-discriminating dictionary** -- the one case that could break the
   w robustness; worth testing the boundary of the claim.
4. The cutoff -> jamming **bias-correction extrapolation** for a clean D.
