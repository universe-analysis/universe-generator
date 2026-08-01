<p align="center">
    <img src="https://raw.githubusercontent.com/universe-analysis/universe-generator/da2259700386fe0c80774a02093e447eb85fdff0/docs/figures/banner12342424.webp" width="600px">
</p>

# Universe Generator
This project is centered around discrete generation and analysis of a model universe. Our model universe has n spatial dimensions, and one dimension for time, it is flat and contains a torus submanifold, therefore describing a slightly modified generalized Minkowski spacetime. Worldlines are defined within this spacetime, and a discrete analysis with a Random-Sequential-Adsorption (RSA) technique is used to pack the universe according to rules of intersection and non-intersection. This model therefore describes n spatial dimensions on an expanding and collapsing n-torus (usually displayed as a comoving n-torus).
[`2+1 Browser Demo`](https://universe-analysis.github.io/universe-generator/viewers/twoplusone_2torus_wrapped.html)
## Worldlines
A worldline in this model is represented per-axis, using parametric form, as expansion of sine terms. The generic `a * sin(b * T+f) - a * sin(f)` where f is allowed to be between 0 and π if b (the frequency) is even - if b is odd then f must be 0 (or π, but a is already allowed to be inverted which accomplishes the same thing.) The critical constraints on these worldlines are as follows:

### Closure
All worldlines must be closed conformal-time loops between time(t) `t ∈ (0, π)`

### Free sin1 amplitude (-1,1)
All worldlines have a random budget for their sin1 component, different from the other frequencies. Every sin1 component has an independent random amplitude from -1 to 1, independent of the rapidity constraints on the other frequency terms below. The sin1 component is the 'comoving worldline' frequency of this model, and thus freeing sin1 ensures an even distribution across space.

### Seam wrap
<p align="left">
    <img src="https://raw.githubusercontent.com/universe-analysis/universe-generator/db3417af9b244c3f8e8842baa55df233393e8e13/docs/figures/timewrapped.JPG" width="300px">
</p>

The bounds of our universe in any axis direction are denoted by `|sint|`, therefore any function exceeding sint has 2sint subtracted from it, and it emerges from -sint. Similar, if a function goes below -sint, it has 2sint added to it, making it emerge from sint. By adding this seam to the model along with the free sin1 component, any preferred center is erased.

### Rapidity / Wiggle Budget on rest of frequency terms
<p align="left">
    <img src="https://raw.githubusercontent.com/universe-analysis/universe-generator/4299786d06a8d9233f5e591f9b1bc0600458ba60/docs/figures/wigglebudget1234.JPG" width="300px">
</p>

The rest of a path's axis frequency terms must obey a shared budget, known as the wiggle budget. It functions like a rapidity budget. The sum of the absolute value of every amplitude * frequency term must add to exactly 1 - this ensures a constant maximum slope of 1. It is of note that as many frequency terms up to a frequency limit as high as one wishes, however a maximum must be chosen for discrete analysis. In each path, per axis, every frequency term must be unique, and an integer above zero.

### Every integer frequency used
In discrete simulation, we must pick a cap, and analyze as the cap approaches infinity, but every possible integer frequency up to that must be included in every axis of every path. Fractional frequencies cannot be used due to closure requirement. This integer frequency requirement builds the semicircle causal reach structure.

### Phase rules
As mentioned earlier, due to the closure requirement, odd frequencies can only be inverted, which is already satisfied due to amplitude being absolute value and thus being able to be negative. That means only even frequencies need an explicit phase component, and to ensure closure, a constant term needs to be subtracted. This allows a full range of possible paths, and phase is critical to many behaviors.

### Unique Path Group / Rules
In this model, a single path does not need to represent only one particle. By defining rules of intersection and non-intersection, we define a unique group to be a group of paths which intersect each other at least once from  `t ∈ (0, π)`, while not intersecting any other path in a different unique group. This allows for a single unique group to consist of many paths. These intersection rules are the building blocks of the generation methods used in this project.

### Per-axis rules [technical clarification for Euclidean vs Chebyshev]
This allows n spatial dimensions, however the rapidity constraint is axis-independent. This means that this space is not directly euclidean, this is accounted for using Chebyshev calculations when relevant, such as for intersection volume.

### Additional Notes
Due to the use of sine waves over the closed interval, a shared expansion-collapse cycle is built into the worldlines themselves. The expansion is nearly constant at the start but decreases until the midpoint when the expansion turns to contraction.

## Random-Sequential-Adsorption (RSA)
<p align="left">
    <img src="https://raw.githubusercontent.com/universe-analysis/universe-generator/db3417af9b244c3f8e8842baa55df233393e8e13/docs/figures/rsaimage2.JPG" width="400px">
</p>

Traditional RSA packs objects in a single space, whereas this packs 1D paths across all of time at once. This is done using a comoving volume, and resembles classical sequential growth (CSG) dynamics seen in causal sets. Discussed more in the implementation section.

## Implementation
### Discrete timestep / nyquist frequency
To generate a universe to analyze, we first choose the maximum frequency we will allow. This determines virtually every other parameter used in generation. The universe is divided into n timesteps, where n is equal to the maxFrequency chosen. This is a nyquist timestep/frequency. This then also determines the maximum number of terms in each path's parametric axis formula. This also determines collision rules below. Analysis is done in regard to the maximum frequency approaching infinity.

### Collision - intersection
This model uses a very robust (but not perfect) method to check for intersection between paths. This method is most fallible in one dimension, however extremely strong in 2 or more spatial dimensions. The method uses a comoving intersection check, where T is equal to the number of discrete timesteps, a comoving intersection distance of 2/T per-axis is used to check for intersection at every discrete timestep. While this can miss intersections, the missed intersections are extremely rare and self-intersecting paths are allowed in this model, meaning that any missed intersection would simply join two unique groups as a single unique group. We believe this missed intersection error rate approaches zero for two or more dimensions in discrete analysis. This algorithm is open for discussion. Personally, I (Chris) think this algorithm is perfect in whatever it does, but I understand it fairly poorly. I came up with it while experimenting with manual numbers trying to find a good collision detection.

### Unique generation -> Subpath Generation
The generation process follows two RSA stages, the first stage generates only unique paths (potential missed intersections aside.) Once the universe is jammed with unique paths, or earlier if one chooses, the next stage can follow through generation of subpaths, using the unique paths essentially as seeds. During the first stage, non-intersection is the only priority, and no self-intersections are explicitly allowed (although it is possible for some to slip in.)<br><br>
The second phase still has the previous non-intersection requirement, but only for paths of a different unique group. When a path intersects another, it inherits it's groupID, and if the path never intersects a path with a different groupID, it is a valid subpath. It isn't possible to jam the subpaths due to them being able to occupy the same space as previously existing subpaths, but the growth rate does decay.<br><br>
Note that this discrete method does not explicitly force that two paths ever perfectly intersect or non-intersect, however as frequency and timestep approach infinity, and the comoving intersection box shrinks, the intersections approach true intersections, therefore this analysis is an approximation that can be measured as the limit of the maxfreq (or the timestep resolution, same thing) approaches infinity.

### Comoving visualizer  / n-torus
Our interactive viewers include a 2+1 generator / visualizer, viewing as a true 2-torus in 3d space (with a causal frame map whose front is the closed-form wiggle-budget reach), and a braid viewer rendering worldlines as 3D strands. All are published from [`docs/`](docs/) via GitHub Pages. [`2+1 Browser Demo`](https://universe-analysis.github.io/universe-generator/viewers/twoplusone_2torus_wrapped.html)

## Analysis and measurements

The complete measured-results record (numbers, caveats, and superseded-method errata) lives in [`PHYSICS_FINDINGS.md`](PHYSICS_FINDINGS.md); the dated evidence trail is in [`docs/lab-notes/`](docs/lab-notes/).

### Causal Structure
<p align="left">
    <img src="https://github.com/universe-analysis/universe-generator/blob/5f7791651a646cbd0e2c397d9751c6d37be09803/docs/figures/Causalreach1.JPG" width="600px">
</p>

Here the Causal reach (blue line) is shown in comoving coordinates. The maximum distance between any two points is 1, so the universe begins with complete causal connection and no horizon problem. Time can be measured from 0 to pi, or from pi to zero. This graph is the same except mirrored in the case of pi to zero. The bouncing semicircles, to our current knowledge, are due to only using integer frequencies as mandated by the closure requirement. The causal structure is not globally hyperbolic in the Cauchy sense and has a whole-history structure. At least this is the current understanding.

### Frame of reference (static observer)
<p align="left">
    <img src="https://raw.githubusercontent.com/universe-analysis/universe-generator/6251e7056211e257e0ecf3ebe03d45e9a1c6fda5/docs/figures/framewithredshift.JPG" width="600px">
</p>

Frame of reference for an observer is built by emitting a pulse back to time zero, covering the maximum causal reach at the time it was emitted. Color is white to red indicating redshift.

### Equation of State
<p align="left">
    <img src="https://universe-analysis.github.io/universe-generator/lab-notes/2026-07-28/fullspec_eos_multiT_3d.png" width="600px">
</p>

The w value of universes generated, in 3 dimensions, begins at ~.34 and has a turnaround approaching ~0. This graph uses a mass-energy index of E ∝ b. Subpaths are not included in this graph, they follow a similar curve however the mass-energy dictionary is still being evaluated for subpaths.

### Uniformity over time
<p align="left">
    <img src="https://universe-analysis.github.io/universe-generator/lab-notes/2026-07-26/fullspec_uniformity_over_time_3d.png" width="600px">
</p>

The blue line here shows a constant comoving root mean squared spread - showing that the universe is isotropic spatially at any given slice in time. The purple line shows that the box counting dimension increase and decrease over time.

### Correlation Dimension is equal to spatial dimension
<p align="left">
    <img src="https://universe-analysis.github.io/universe-generator/lab-notes/2026-07-26/fullspec_periodic_corrdim_3d.png" width="600px">
</p>

Blue line shows improper non-periodic measurment showing lower than 3 measured in the 3+1 model, however the periodic cube measurement approaches 3 exactly.

### Quantum Behavior
<p align="left">
    <img src="https://universe-analysis.github.io/universe-generator/lab-notes/2026-07-22/group_partners.png" width="600px">
</p>

A single unique group consists of a potentially infinite number of subpaths - paths which only intersect each other while avoiding intersections with every other path from a different unique group across all of time. This allows for a single unique group (particle) to be represented by many different positions and velocities at once. Subpaths have been measured to start far apart from each other at the big bang and get closest to each other on average near the turnaround point. We believe these carry causal implications. The graph above shows some measurements claude made on some data sets that had subpaths.

*Status:* the classical groundwork is measured (2026-07-22 partner-correlation
analysis, 2+1): co-grouped strands emerge a mean 0.33 comoving units apart yet
share their anchor a₁ at r = +0.95; never-touching co-group siblings correlate
at +0.90, and pairs whose first contact comes after the turnaround still carry
+0.74 — correlation precedes contact, with the cold/mover class
*anti*-matching between partners and no phase or velocity aiming at the Bang.
These are selection (common-constraint) correlations — the all-at-once
phenomenology, not yet a Bell test, which would additionally need a
measurement-settings analog.

### Big bang / crunch / inflation / expansion
This model naturally embeds an expansion and collapse cycle. The measured equation of state at maximum expansion (t = π/2) is matter-like — w = 0.145 (Chebyshev metric) / 0.193 (Euclidean), cooled 20–24% below the proposal ensemble by jamming's phase selection — so the turnaround region is cold and matter-dominated. In the full-spectrum limit (terms = T) the turnaround goes all the way to dust: w = d/(6T) → 0, a parameter-free central-limit law, identical under both energy dictionaries, and the whole history follows w(z) = (d/3)(cos²z/3 + 1/T) — radiation at the bang/crunch cooling to dust at the turnaround. It is of note that this model is not entirely symmetric across pi/2 due to the inclusion of phase components for even frequency components.

### Small-scale knots / braids
<p align="left">
    <img src="https://raw.githubusercontent.com/universe-analysis/universe-generator/db3417af9b244c3f8e8842baa55df233393e8e13/docs/figures/simplebraid1.JPG" width="300px">
</p>

The ability to form a stable knot of n unique paths depends on the dimension, therefore a goal of this project is to analyze the unique path knots and attempt to relate them to standard model particles. As far as we understand, the most unique paths that can be knotted in one dimension is one, in two dimensions is four, and in three dimensions is 12, in four is 32 and five is 80. However, the exploration into this has only just begun and these are all tentative findings.
*Status (2026-08-01): the observed values are running at exactly double the conjecture — 1+1 → 2, 2+1 → 8 (braid-viewer closure mode), 3+1 → 24 — i.e. the sequence d·2^d, the directed-edge count of the d-hypercube (the original guesses were the undirected edge count d·2^(d−1)). All three remain unverified by an automated census; see the [2026-08-01 lab note](https://universe-analysis.github.io/universe-generator/lab-notes/2026-08-01/) for the record and caveats.*

## Predictions / Theory

### Gravity and it's role in spacetime
One core feature of this model is it's flat space, it does contain a torus submanifold, but the space itself does not warp due to any force such as gravity. Therefore, given this model does have a thermodynamic ~.33 to ~0 evolution and seems to have a flow of entropy associated, we would likely assign gravity as entropic in nature. Furthermore, if photons are modeled by these paths, and photons wiggle due to gravity over long distances, they would not be taking the fastest path as a curved spacetime would imply. Therefore we assume that paths which are straighter over longer distances and don't appear as effected by gravity are neutrinos. 

The consequence of this on photons are that photons end up redshifted more than would be expected over longer distances, since this model has a near-constant but slowly decreasing rate of expansion, the extra redshift from this effect may match an interpretation of accelerating expansion. 

The consequence of this on neutrinos are many fold: neutrinos would not exert gravity but instead exert pressure. If this model assumes cold dark matter, the w value curve does not match the expected time progression of our universe, but if hot dark matter is assumed, the w value curve aligns highly with our universe. Hot dark matter, as neutrinos, are assumed to be unable to cluster in galaxies in order to exert gravitational pull, but this model poses that dark matter actually exerts a pressure outwards, and dark matter is everywhere we would expect it not to be - instead of inside galaxies pulling them together with gravity, the hot dark matter neutrinos are clustered outside of the galaxy, exerting a pressure from outside holding the galaxy together. Additionally, the implications on black holes and information are significant.


### Distant observation
The causal connection of this universe is above the maximum distance, so the most distant observations should be previous in time copies of matter closer to the observer on the other side of the universe.

## Live site

Interactive viewers and a results gallery are published from [`docs/`](docs/) via
GitHub Pages. The viewers pack a universe live in the browser and render every
worldline across all of conformal time.

## Layout

| Path | What |
|---|---|
| `braidlab/` | Python orchestration suite: campaigns, resumable SSH fleet runner, SQLite store, correlation-dimension analysis. |
| `cuda/` | GPU packing engines (`braid_cuda3d.cu` for 3+1, `braid_cuda.cu` for 2+1) + `Makefile`. |
| `analysis/` | Current analysis tools (correlation dimension, Dq spectrum, equation of state, knots, box geometry, …). Run from the repo root as modules: `uv run python -m analysis.analyze_correlation_dim …`. |
| `plots/` | Charting / overlay scripts (packing count vs T, approach curves, coupon-collector cells, …), same `-m` invocation. |
| `docs/` | GitHub Pages site — `viewers/` (interactive HTML) + `figures/` (curated charts). |
| `legacy/` | Superseded pre-braidlab tooling (`analyze_campaign`, `plot_campaign`, `make_report`) and the early `braid_solver/` prototypes (pre-CUDA), kept for reference. |
| `*.md` | Findings and design notes (`PHYSICS_FINDINGS`, `EXPERIMENTS_QUEUE`, `INVESTIGATION_menger`, …). |

Engine operations (the heterogeneous CUDA fleet, GPU-memory T-ceilings, the
gotchas) are documented in [`braidlab/ORCHESTRATOR.md`](braidlab/ORCHESTRATOR.md).

## Quick start

```bash
# Python side (uv)
./setup_env.sh                 # create/activate the venv
uv run pytest                  # tests
uv run python -m braidlab plan freq3d_e6 --hosts host1,host2   # dry-run a campaign

# Analysis on collected dumps
uv run python -m braidlab corrdim --db data/freq/freq3d_e6.db --dim 3 --band nyq
```

The GPU engines build with `nvcc` (see `cuda/Makefile`); the fleet runner builds
them per-host automatically.

## Notes

- Raw packing data (`data/`) and regenerated `figures/` are gitignored — they are
  large and reproducible from the engines. Only a curated set of result charts
  under `docs/figures/` is committed, for the site.
