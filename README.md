# The Braided Universe

A packing model of the cosmos. Parametric sinusoidal worldlines
(`x(z) = a·sin(b·z) + a2·sin(z)` per spatial axis, integer frequency `b`, a
slope-1 "speed of light" constraint) are threaded into a closed conformal-time
loop `z ∈ (0, π)` and packed to jamming by random sequential addition. We then
measure the fractal/packing dimension of the result and the physics that emerges
from the geometry.

**Headline result:** the jamming-free packing dimension is **D ≈ 2.79 (3+1)** and
**≈ 1.82 (2+1)** — mildly multifractal, triangulated across correlation
dimension, the generalized-dimension (Dq) spectrum, and box-counting. See
[`PHYSICS_FINDINGS.md`](PHYSICS_FINDINGS.md) for the full story.

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
uv run python -m braidlab plan corrdim3d --hosts host1,host2   # dry-run a campaign

# Analysis on collected dumps
uv run python -m braidlab corrdim --db data/corrdim/run.db --dim 3 --band nyq
```

The GPU engines build with `nvcc` (see `cuda/Makefile`); the fleet runner builds
them per-host automatically.

## Notes

- Raw packing data (`data/`) and regenerated `figures/` are gitignored — they are
  large and reproducible from the engines. Only a curated set of result charts
  under `docs/figures/` is committed, for the site.
