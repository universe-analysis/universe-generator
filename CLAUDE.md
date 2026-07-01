# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

> **Infrastructure (hostnames, GPU cards, per-host toolchain quirks) lives in
> `CLAUDE.local.md`** — a gitignored file that is not published. Check it out for
> the real fleet; the docs here use generic `host1/host2/host3`.

## What this project is

A packing model of the cosmos. Parametric sinusoidal worldlines
(`x(z) = a·sin(b·z) + a2·sin(z)` per spatial axis, integer frequency `b`, a
slope-1 "speed of light" constraint `|a·b| + |a2| = 1`) are threaded into a
closed conformal-time loop `z ∈ (0, π)` and packed to jamming by random
sequential addition (RSA). We measure the fractal/packing dimension of the
result and the physics that emerges. Headline: **D ≈ 2.79 (3+1)**, **≈ 1.82
(2+1)**, mildly multifractal. Full story in `PHYSICS_FINDINGS.md`.

## Layout

- `braidlab/` — Python orchestration suite (campaigns, resumable SSH fleet
  runner, SQLite store, correlation-dimension analysis). See
  `braidlab/ORCHESTRATOR.md` for how the fleet runner works.
- `cuda/` — GPU packing engines (`braid_cuda3d.cu` 3+1, `braid_cuda.cu` 2+1) +
  `Makefile`.
- `analyze_*.py`, `plot_*.py` — standalone analysis / plotting tools.
- `docs/` — the GitHub Pages site (`viewers/` interactive HTML + curated
  `figures/`).
- `*.md` — findings and notes (`PHYSICS_FINDINGS`, `EXPERIMENTS_QUEUE`,
  `INVESTIGATION_menger`, …).

## Environment

- Python via **uv**: `./setup_env.sh`, then `uv run …`.
- Tests: `uv run pytest`. Lint: `uv run ruff check .` / `uv run ruff format .`.
  Types: `uv run pyright` (or `uvx --with numpy --with scipy --with matplotlib
  pyright <files>` if pyright isn't in the venv).

## Using the fleet

The orchestrator dispatches packing jobs across GPU hosts over SSH, collects
results into a resumable SQLite store, and survives host drops (re-running picks
up where it left off). **The real hostnames and caps are in `CLAUDE.local.md`;**
below uses placeholders.

```bash
# dry-run: see the job split
uv run python -m braidlab plan corrdim3d --hosts host1,host2,host3

# run (resumable; safe to Ctrl-C and re-run). --host-max caps a host's largest
# T (GPU memory); the biggest-T jobs then land only on the roomiest card.
uv run python -m braidlab run corrdim3d \
    --hosts host1,host2,host3 --db data/corrdim/run.db \
    --host-max host1=160,host2=160 --poll 120

# seed-averaged correlation-dimension report from collected dumps
uv run python -m braidlab corrdim --db data/corrdim/run.db --dim 3 --band nyq
```

Predefined campaigns (`braidlab/campaigns.py`): `corrdim3d` / `corrdim2d`
(the dimension studies), `corrdim3d_e6` (cutoff check), `corrdim3d_euclid`
(sphere-collision variant), plus the older `3plus1` / `2plus1`.

Key gotchas (full list in `braidlab/ORCHESTRATOR.md`):
- **Heterogeneous CUDA toolchains** — `deploy()` tries every candidate nvcc ×
  {default, `-ccbin g++-11`} until one links. Never copy a binary between hosts
  (glibc).
- **Variant collisions** — a variant run (e.g. euclid, a different cutoff) must
  carry a distinct job-name tag or it re-collects the baseline's leftovers from
  the shared remote workspace.
- **GPU memory ~ T⁴** in 3+1 (dense per-timestep collision grid): T=200 ≈ 13.6 GB,
  capping a 24 GB card near T≈215; 10 GB cards cap ~160. 2+1 is ~T³ (cheap).

## Analysis workflow

Campaigns with `dump=True` write per-worldline parameter dumps; the correlation
dimension, Dq spectrum, structure factor, knot analysis, etc. all read those.
The correlation dimension is the trustworthy estimator; box-counting is noisy and
point-count sensitive (measure it on the full cloud, not a subsample).

## Conventions

- **Python:** type hints on all code; docstrings on public APIs; 88-char lines;
  ruff + pyright clean; new features get tests, bug fixes get regression tests.
- **C/C++/CUDA:** format with clang-format using the committed `.clang-format`
  (`clang-format -i cuda/*.cu`). Favor readability over density — one statement
  per line; the engines are read by people who aren't CUDA specialists.
- **JS/HTML** (viewers): same readability rule — no dense one-liners or chained
  ternaries; these are read by non-JS-specialists.
- **Commits:** never mention co-authored-by or the tool used. Commit locally;
  push only when asked. Use `git commit --trailer "Reported-by:<name>"` for
  user-reported bug fixes.
- Keep raw `data/` and regenerated `figures/` out of git (gitignored); only
  curated `docs/figures/` charts are committed for the site.
