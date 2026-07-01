# Experiment Queue

Parked experiments, with enough detail to run later. Both came out of the
sphere-vs-cube dimension gap in the 3+1 correlation-dimension result
(spheres D2 = 2.79, cubes = 2.73, box-counting ~2.70-2.74 — see
PHYSICS_FINDINGS section 8).

---

## 1. Cube-gap prediction from the ramp  [analysis only, no GPU]  DONE

**Result (`analyze_gap_prediction.py`, figures/gap_prediction.png).** The
isotropic prediction `C_cube(r) = <C_sphere(r/m)>` (zero free parameters)
recovers **~85% of the sphere-cube gap** in both dimensions:

```
        observed gap   predicted (ramp)   residual (anisotropy)
3+1        ~0.067          ~0.058              ~0.009  (flat in T)
2+1        ~0.025          ~0.021              ~0.004  (flat in T)
```

So most of the gap is isotropic scale-dependence (the ramp seen through cube
corner-reach), confirming the corner-reach story quantitatively. The small,
T-stable residual (~15% of the gap, scaling with dimension like the gap) is a
**genuine mild anisotropy** -- the observed cube dimension is a touch *lower*
than isotropy predicts, i.e. diagonal directions are slightly sparser than
isotropic (mild axis-alignment). Leading suspect: the cubic collision rule ->
**Experiment 2.**

---

## 1. Cube-gap prediction from the ramp  [analysis only, no GPU]  (original spec)

**Question.** Is the 0.06 sphere-vs-cube gap *entirely* the local-slope ramp
(the dimension drifting ~3.0 -> 2.4 with scale) seen through probe geometry —
i.e. pure isotropic scale-dependence — or is there a residual that is genuine
axis-vs-diagonal **anisotropy** in the packing?

**Why it matters.** If the gap is fully predicted by the ramp, the braid is an
isotropic mild-multifractal and the probe-dependence is just bookkeeping. A
residual would be real anisotropy — plausible, since warm worldlines move along
*single* axes at the turnaround, and the comoving box is a cube.

**Method (existing corrdim3d dumps, `data/corrdim/dumps`).**
- For each packing, take the measured sphere local-slope curve `D2_sphere(r)`.
- Model the cube (L-inf) probe's *effective* L2 sampling radius: a cube of
  half-side `r` reaches `r` along axes and `sqrt(3)*r` into corners; compute the
  direction-averaged effective radius `c*r` (c between 1 and sqrt(3), ~1.3).
- Predict `D2_cube(r) ~= D2_sphere(c*r)`, integrate over the fit window, and
  compare the predicted (sphere - cube) gap to the observed gap per T.
- **Residual gap = anisotropy signal.** Plot predicted vs observed vs T.

**Deliverable.** A chart: observed gap, ramp-predicted gap, residual. If residual
~ 0, isotropic; if not, quantify the anisotropy.

---

## 2. Euclidean (spherical) collision variant  [engine change + GPU campaign]

**Question.** Is some of the cube-vs-sphere gap an *artifact of the cubic
collision rule itself*? The packing is currently built with a Chebyshev (L-inf)
exclusion — two worldlines collide if within CELL on *all* axes (a cube of
half-side CELL) — which lets threads sit closer along diagonals than along axes,
baking axis-structure into the pack. A physical thread cross-section should be
isotropic (a sphere / spacetime cylinder).

**Method.**
- Add an engine flag `--euclid-collision` to `braid_cuda3d.cu` (and `braid_cuda`):
  replace the per-axis test
  `fabs(dX)<=cell && fabs(dY)<=cell && fabs(dW)<=cell`
  with the Euclidean ball test `dX*dX + dY*dY + dW*dW <= cell*cell`.
  The 3x3x3 neighbour scan still covers it (sphere of radius cell ⊂ the 3-cell
  cube), and the edge-collision wall test stays per-axis (the *domain* is
  genuinely cubic from the per-axis slope-1 constraint — only the *exclusion*
  changes).
- Thread it through `config.Job` / a campaign flag like `angle`, build_command
  emits `--euclid-collision`.
- Run a `corrdim3d`-style campaign with it (can be a coarser/cheaper grid first,
  e.g. T=60..200 step 20, 5 seeds).

**What to look for.**
- Does the **cube/sphere measurement gap shrink** under an isotropic exclusion?
  (If yes, part of the original gap was our own cubic ruler matching a cubic
  exclusion.)
- Does the converged D move, and toward which value (sphere ~2.79, cube ~2.73)?
- Sanity: N changes (different exclusion volume — sphere of radius cell has
  smaller volume than the cube of half-side cell, so expect a denser pack /
  higher N).

**Caveat.** This changes the *model* (the packing), not just the analysis, so it
is a genuine variant universe, not a re-measurement. Compare like-for-like
(same T grid, seeds, band).

**Optional follow-on.** The fully faithful exclusion is a 4D spacetime *cylinder*
(spherical spatial cross-section swept along z), and time may warrant a different
thickness than space (an *elliptical* cylinder) — only worth it if the spherical
variant shows the exclusion shape genuinely moves the physics.

---

## 3. Motion-aware / worldtube collision  [engine change + GPU campaign]  BACKLOG

**Idea.** Replace the isotropic per-timestep ball/cube exclusion with a shape
that follows the particle's motion -- a swept capsule along the inter-sample
velocity (the physically faithful object is the continuous spacetime
*worldtube* intersection, which the discrete circle/square only approximates).

**Two-sided effect on density (the interesting part).** It is NOT a one-way
change:
- *Catches tunneling*: fast (high-frequency) particles that pass through each
  other between the T sampled timesteps are currently missed; a swept capsule
  catches them -> packs **fewer** there (and would lower D if high-freq content
  is over-packed).
- *Tighter interlock*: particles moving in parallel can nestle their capsules
  side-by-side closer than two isotropic disks (which reserve CELL in every
  direction, including along-motion where parallel neighbours don't need it) ->
  packs **more** there.
The net effect on N (and D) is genuinely unknown -- worth measuring.

**Spec.** Per timestep, estimate the particle's comoving velocity (finite
difference between consecutive sampled positions, or analytic dX/dz), and test
segment-to-segment (or point-to-capsule) distance instead of point-to-point.
Optional stretch (**Experiment 4**): full continuous worldtube intersection in
(d+1) spacetime between consecutive timesteps.

**Why backlog.** Bigger engine change than the euclid flag; do after the
isotropic (euclid) result is in, which calibrates how much the collision shape
moves things at all.
