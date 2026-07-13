# Universe Generator
This project is centered around discrete generation and analysis of a model universe. Our model universe has n spatial dimensions, and one dimension for time, it is flat and contains a torus submanifold, therefore describing generalized Minkowski spacetime. Worldlines are defined within this spacetime, and a discrete analysis with a Random-Sequential-Adsorption (RSA) technique is uses to pack the universe according to rules of intersection and non-intersection. This model therefore describes n spatial dimensions on an expanding and collapsing n-torus, allowing a cat-map automorphism. The worldlines are 1D paths through time, raising many similarities to string theory as discussed later.

## Worldines
A worldline in this model is represented per-axis, using parametric form, as expansion of sine terms. The generic `a * sin(b * T+f) - a * sin(f)` where f is allowed to be between 0 and π if b (the frequency) is even - if b is odd then f must be 0 (or π, but a is already allowed to be inverted which accomplishes the same thing.) The critical constraints on these worldlines are as follows:

### Closure
All worldlines must be closed conformal-time loops between time(t) `t ∈ (0, π)`

### Free sin1
All worldlines have a random budget for their sin1 component, different from the other frequencies. Every sin1 component has an independent random amplitude from -1 to 1, independent of the rapidity constraints on the other frequency terms below. The sin1 component is the 'comoving worldline' frequency of this model, and thus freeing sin1 ensures an even distribution across space.

### Seam wrap
The bounds of our universe in any axis direction is denoted by sint, therefore any function exceeding sint has 2sint subtracted from it, and it emerges from -sint. Similar, if a function goes below -sint, it has 2sint added to it, making it emerge from sint. By adding this seam to the model along with the free sin1 component, any preferred center is erased.

### Rapidity on rest of frequency terms
The rest of a path's axis frequency terms must obey a shared rapidity budget. The sum of the absoltue value of every amplitude * frequency term must add to exactly 1 - this ensures a constant speed of light, satisfying relativity. It is of note that as many frequency terms up to a frequency limit as high as one wishes, however a maximum must be chosen for discrete analysis. In each path, per axis, every frequency term must be unique, and an integer above zero.

### Phase rules
As mentioned earlier, due to the closure requirement, odd frequencies can only be inverted, which is already satisfied due to amplitude being absolute value and thus being able to be negative. That means only even frequencies need an explicit phase component, and to ensure closure, a constant term needs to be subtracted. This allows a full range of possible paths, and phase is critical to many behaviors.

### Per-axis rules [technical clarification for euclidian vs Chebyshev]
This allows n spatial dimensions, however the rapidity constraint is axis-independent. This means that this space is not directly euclidean, however this is accounted for in square-intersection bounds, and Chebyshev calculations when relevant (Peculiar speed.) A cat-map transformation is the equivalent of 'rotating' this space while preserving it's properties.

### Unique Path Group / Rules
In this model, a single path does not need to represent only one particle. By defining rules of intersection and non-intersection, we define a unique group to be a group of paths which intersect each other at least once from  `t ∈ (0, π)`, while not intersecting any other path in a different unique group. This allows for a single unique group to consist of many paths. These intersection rules are the building blocks of the generation methods used in this project.


## Physics

### Minkowski spacetime
This model describes generic n+1 dimension universes, with n spatial dimensions. This is known as a generalized Minkowski spacetime.

### RSA
Traditional RSA packs objects in a single space, whereas this packs 1D paths across all of time at once.

### Unique generation -> Subpath Generation
The generation process follows two stages, the first stage generates only unique paths (potential missed interactions aside - discussed below)

### Quantum Behavior
A single unique group consists of a potentially infinite number of subpaths - paths which only intersect each other while avoiding intersections with every other path from a different unique group across all of time. This allows for a single unique group [particle] to be represented by many different positions and velocities at once. A goal of this project is to study these subpath behaviors to determine if they follow expected Bell inequalities / other tests for quantum behavior.

### Big bang / crunch / inflation / expansion
This model naturally embeds a expansion and collapse cycle, as well as a measured rapid cooling inflation period at the very start, following by a cold - matter dominated region at time t=π/2, leading to a hot collapse. It is of note that this model is not entirely symmetric across pi/2 due to the inclusion of phase components for even frequency components.

## Implementation
### Discrete timestep / nyquist frequency
To generate a universe to analyze, we first choose the maximum frequency we will allow. This determines virtually every other parameter used in analysis. The universe is divided into n timesteps, where n is equal to the maxFrequency chosen. This is a nyquist timestep/frequency. This then also determines the maximum number of terms in each path's parametric axis formula. This also determines collision rules below.

### Collision - intersection
This model uses a very robust (but not perfect) method to check for intersection between paths. This method is most fallible in one dimension, however extremely strong in 2 or more spatial dimensions. The method uses a comoving intersection check, where T is equal to the number of discrete timesteps, a comoving intersection distance of 2/T per-axis is used to check for intersection at every discrete timestep. While this can miss intersections, the missed intersections are extremely rare and self-intersecting paths are allowed in this model, meaning that any missed intersection would simply join two unique groups as a single unique group. We believe this missed intersection error rate approaches zero for two or more dimensions in discrete analysis. This algorithim is open for discussion.

### Comoving visualizer  / n-torus
Our interactive viewers include a 2+1 generator / visualizer, viewing as a true 2-torus in 3d space. Other viewers are in the works.

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
