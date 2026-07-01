# Campaign Runbook — 2+1 Box-Counting vs Correlation Dimension (deep)

A thorough comparison of the *three* ways we can measure the packing dimension
of a 2+1 braided universe, with enough T and seeds to settle whether they agree.
The code is ready (`corrdim2d` campaign + dimension-aware `corrdim.py`); this doc
is how to run it and what to look for. **Not yet run** — set up for later.

## Why 2+1, why deep

The dense per-timestep collision grid scales ~T³ in 2+1 (T timesteps x a T²
spatial lattice) vs ~T⁴ in 3+1, so 2+1 is *much* cheaper: even T=400 is ~0.5 GB
of grid (vs 13.6 GB for 3+1 at T=200). We can therefore push T far higher than
3D's ~215 ceiling and get a much longer scaling window (more decades between
CELL = 2/T and the box), which is exactly what sharpens a dimension measurement.
No GPU-memory caps are needed. See ORCHESTRATOR.md for the full VRAM breakdown.

## The three dimension estimates (the whole point)

One `corrdim2d` run yields **all three** from the same packings, because the
campaign writes both the kinetic curve (→ N_final per job) and the parameter
dump (→ turnaround cloud):

| Estimate | Source | Command | What it is |
|---|---|---|---|
| **Count-scaling box D** | N_final(T) curves | `braidlab analyze` | operational packing dim `N_sat ~ T^D`; needs jamming, cutoff-suppressed |
| **Within-cloud box D** | dumps | `braidlab corrdim --dim 2` | box-count the turnaround cloud across cell size; jamming-free |
| **Correlation D2** (sphere + cube) | dumps | `braidlab corrdim --dim 2` | `C(r) ~ r^D2`; jamming-free, probe-shape cross-check |

The open question this resolves: **do they agree?** Preliminary single-seed
numbers (from the existing `data/params2d` dumps) say:

```
correlation D2 -> ~1.84   (converging from above)
within-cloud box -> ~1.75 (converging from below)
count-scaling D  -> ~1.6  (from the older curve campaign)
```

That ~0.2 gap between the jamming-free ~1.8 and the count-scaling ~1.6 mirrors
3+1 (where cutoff 2.47 sat below the true 2.8). The hypothesis to test: the
count-scaling 1.6 is **cutoff-suppressed**, and the true 2+1 packing dimension is
~1.8. Deep data (high T, seed-averaged) should confirm it and pin the value.

## Run it

```bash
# 1. (Optional) preview the job split
uv run python -m braidlab plan corrdim2d --hosts mother,kitt,deep-thought

# 2. Build + run. No --host-max needed (2D is memory-cheap). Pick whatever
#    hosts are free -- do NOT contend with a running corrdim3d campaign.
uv run python -m braidlab run corrdim2d \
    --hosts mother,kitt,deep-thought \
    --db data/corrdim2d/run.db \
    --poll 120

# 3a. Count-scaling box D (operational packing dimension)
uv run python -m braidlab analyze --db data/corrdim2d/run.db

# 3b. Seed-averaged correlation + within-cloud box D, error-barred convergence
uv run python -m braidlab corrdim --db data/corrdim2d/run.db --dim 2 --band nyq \
    --out corrdim2d_convergence.png
```

Campaign grid (in `campaigns.py`): **T = 40, 60, …, 400** (19 values) ×
**8 seeds** = 152 jobs, full Nyquist band, 1e-7 acceptance cutoff. The 2D engine
(`braid_cuda`) has **no `--angle-sample`**, so this uses the default amplitude
sampler — fine, because D is sampler-invariant and the comparison is internal to
2+1.

## What to look for

1. **Do the three estimates converge to one value?** Plot count-scaling D,
   within-cloud box D, and correlation D2 vs T on one axis. In 3+1 the jamming-
   free pair bracketed and met at ~2.73 (cubes) / ~2.80 (spheres); see whether
   2+1 brackets and meets similarly (~1.8 expected).
2. **Is 2+1 mono- or multi-fractal?** With deep T the scaling window is long, so
   the local slope `D2(r)` is trustworthy over more decades. In 3+1 it ramped
   ~3.0 → 2.4 (mild multifractality). Does 2+1 show a flat plateau (cleaner
   monofractal) or its own ramp? This is the cleanest test the long 2D window
   buys us.
3. **Sphere vs cube probe independence** — should agree to <0.1, as in 3+1.
4. **Where does count-scaling D land at high T?** If it climbs from 1.6 toward
   ~1.8 as T grows (less under-jamming), that confirms the suppression story.

## Gotchas (see ORCHESTRATOR.md for the full list)

- **2D engine = `braid_cuda`**, not `braid_cuda3d`; `deploy()` builds it from
  `dim=2` jobs automatically. The robust multi-candidate nvcc build applies.
- **No `--host-max` needed** — 2D memory is negligible at every T here.
- **High T → many attempts** to reach the 1e-7 cutoff (the cost is attempts, not
  memory); 2D collision checks are cheap, so it still moves fast, but the
  highest-T seeds are the long pole. Resumable, so interruptions are free.
- Don't launch while `corrdim3d` is still using the fleet — wait for it or use a
  free host subset.

## Expected deliverable

A 2+1 analog of `corrdim_multiT_converge.png`: count-scaling vs within-cloud box
vs correlation D2, seed-averaged with error bars, over T=40–400 — settling (the
hypothesis) around **D ≈ 1.8**, with a verdict on mono- vs multi-fractality from
the long-window local slope.
