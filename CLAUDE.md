# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

> **Infrastructure (hostnames, GPU cards, per-host toolchain quirks) lives in
> `CLAUDE.local.md`** — a gitignored file that is not published. Check it out for
> the real fleet; the docs here use generic `host1/host2/host3`.

## What this project is

A packing model of the cosmos. Parametric sinusoidal worldlines
(`x(z) = a·[sin(b·z + f) − sin f] + a1·sin(z)` per spatial axis, integer
frequency `b`, unit-rapidity budget `|a·b| = 1`, free comoving coordinate
`a1`, free phase `f` on even frequencies) are threaded into a closed
conformal-time loop `z ∈ (0, π)` on a comoving torus and packed by random
sequential adsorption (RSA). Headline: the jam count scales as **N ~ T^2.32
(3+1, converged over T = 160–520)** / **T^1.4 (2+1, cutoff-conditional)** —
a packing-number exponent
with no geometric carrier, invariant under cutoff depth and dictionary size;
matter slices are *exactly* uniform (wrapped D2 = 3.01/2.02); mover speeds
follow a phase-selected arcsine law; turnaround equation of state **w = 0.145
(Chebyshev) / 0.193 (Euclidean)**, with jamming cooling w 20–24% below the
proposal ensemble in both metrics. Full story in `PHYSICS_FINDINGS.md`
(current knowledge; the paper `paper/main.tex` predates the uniform re-anchor
and needs revision).

## Layout

- `braidlab/` — Python orchestration suite (campaigns, resumable SSH fleet
  runner, SQLite store, correlation-dimension analysis). See
  `braidlab/ORCHESTRATOR.md` for how the fleet runner works.
- `cuda/` — GPU packing engines (`braid_cuda3d.cu` 3+1, `braid_cuda.cu` 2+1) +
  `Makefile`.
- `analysis/` — current analysis tools (`analyze_correlation_dim`,
  `analyze_spectral_dims`, `analyze_knots`, `analyze_box_geometry`,
  `analyze_eos_history`, …). Run from the repo root as modules so `braidlab`
  and `data/` resolve: `uv run python -m analysis.analyze_correlation_dim …`.
- `plots/` — charting / overlay scripts (`plot_nvt`, `plot_approach`,
  `coupon_cells`, …), same `-m` invocation.
- `legacy/` — superseded pre-braidlab tooling (`analyze_campaign`,
  `plot_campaign`, `make_report`) and the early `braid_solver/` Python
  prototypes (pre-CUDA), kept for reference.
- `docs/` — the GitHub Pages site (`viewers/` interactive HTML + curated
  `figures/`). Pages serves the dedicated `gh_pages` branch (docs-only
  history, `.nojekyll` static serving), not `main`: after pushing site
  changes on `main`, republish with
  `git subtree push --prefix docs origin gh_pages`.
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
uv run python -m braidlab plan freq3d_e6 --hosts host1,host2,host3

# run (resumable; safe to Ctrl-C and re-run). --host-max caps a host's largest
# T (GPU memory); the biggest-T jobs then land only on the roomiest card.
uv run python -m braidlab run freq3d_e6 \
    --hosts host1,host2,host3 --db data/freq/freq3d_e6.db \
    --host-max host1=160,host2=160 --poll 120

# seed-averaged correlation-dimension report from collected dumps
uv run python -m braidlab corrdim --db data/freq/freq3d_e6.db --dim 3 --band nyq
```

Predefined campaigns (`braidlab/campaigns.py`): `torus3d` / `torus2d`
(+ `_e6` cutoff variants), `torus3d_phase_e6` / `torus2d_phase_e6` (from the
era when the phase schema was opt-in; now identical to the `_e6` variants,
kept so their stores stay resumable), `freq3d_e6` / `freq2d_e6` (the FREQ
campaign: term-count sweep, engine flag `--terms`, terms ∈ {2,3,4,6,10})
with `freqdecay{3,2}d_e{6,7,8}` companions (2 T × terms {2,10} per cutoff, for
the approach power law). The engines are torus-only with the uniform sampler
and the phase schema ALWAYS ON since 2026-07-09 (`--phase` is an accepted
no-op that the orchestrator still passes to guard against stale opt-in
binaries); the wall-era campaigns (`corrdim3d`/`corrdim2d`, the euclid
variant, `3plus1`/`2plus1`) were removed with that cleanup — their stores and
dumps remain under `data/`, and the definitions live in git history.

### Discord notifications

If `BRAIDLAB_DISCORD_WEBHOOK` is set, `braidlab run` posts campaign lifecycle
events to Discord: a pre-flight summary, progress pings at 25/50/75%,
host-stall warnings, and a completion summary. Posting is best-effort — a down
Discord never affects the run — and stdlib-only (no extra deps). The webhook
URL is a secret; it lives in `CLAUDE.local.md`, never committed. Post an
ad-hoc message with `python -m braidlab notify --title … --message … --color
start|progress|done|fail|info` (also exposed as the `discord-update` skill).

Key gotchas (full list in `braidlab/ORCHESTRATOR.md`):
- **Heterogeneous CUDA toolchains** — `deploy()` tries every candidate nvcc ×
  {default, `-ccbin g++-11`} until one links. Never copy a binary between hosts
  (glibc).
- **Variant collisions** — a variant run (e.g. euclid, a different cutoff) must
  carry a distinct job-name tag or it re-collects the baseline's leftovers from
  the shared remote workspace.
- **GPU memory ~ T⁴** in 3+1 (dense per-timestep collision grid): T=200 ≈ 6.8 GB
  (sentinel CSR), capping a 24 GB card near T≈255; 10 GB cards cap ~190. 2+1 is
  ~T³ (cheap). For larger T use the sparse grid (`Campaign(sparse=True)` /
  engine `--sparse`, VRAM ~ N·T — T=240 ran in 3.0 GB on a 3080); validate
  engine A/Bs with `analysis/compare_jamming.py` against a stored run.db.

## Analysis workflow

Campaigns with `dump=True` write per-worldline parameter dumps; the correlation
dimension, Dq spectrum, structure factor, knot analysis, etc. all read those.
The correlation dimension is the trustworthy estimator; box-counting is noisy and
point-count sensitive (measure it on the full cloud, not a subsample).

## Lab notes (daily campaign log)

The site publishes a running lab notebook under `docs/lab-notes/` — a dated,
reverse-chronological log of every campaign run and the plots it produced, each
tied to the commit that generated it so any figure can be reproduced. When you
run a campaign or make a plot worth keeping, log it:

1. **Day directory.** That day's plots live in `docs/lab-notes/<YYYY-MM-DD>/`.
   Create it on the day's first plot.
2. **Commit the code first.** Commit the generating change (analysis script /
   engine tweak) on its own, then grab the short hash: `git rev-parse --short HEAD`.
3. **Generate the plot** into the day directory.
4. **Write the entry.** On the day's first entry, create
   `docs/lab-notes/<YYYY-MM-DD>/index.html` by copying the previous day's page as
   a template (it links `../notes.css` + `../notes.js` and carries the lightbox
   markup). Append an `<article class="entry">` saying what the plot is, what it
   shows, and the takeaway, with a `.meta` provenance line citing the code commit
   as a link to `https://github.com/universe-analysis/universe-generator/commit/<hash>`.
   Entries are chronological — newest appended at the bottom, above the
   `<!-- NEW ENTRIES APPEND ABOVE THIS LINE -->` marker.
5. **Index the day.** On the day's first entry, add a `.day` link at the top
   (newest first) of the list in `docs/lab-notes/index.html`.
6. **Commit the note + plot(s)** as a second commit, separate from the code.

Lab-note plots **are** committed (they're site content) — the exception to the
"keep figures out of git" rule below, which still applies to top-level `figures/`.

## Conventions

- **Python:** type hints on all code; docstrings on public APIs; 88-char lines;
  ruff + pyright clean; new features get tests, bug fixes get regression tests.
- **C/C++/CUDA:** format with clang-format using the committed `.clang-format`
  (`clang-format -i cuda/*.cu`). Favor readability over density — one statement
  per line; the engines are read by people who aren't CUDA specialists.
- **JS/HTML** (viewers): same readability rule — no dense one-liners or chained
  ternaries; these are read by non-JS-specialists.
- **Viewer versioning:** `docs/viewers/twoplusone_2torus_wrapped.html` shows a
  version stamp (the `VIEWER_VERSION` constant near the top of its script) in
  the lower right of the 2D slice HUD. Increment it (`v1` → `v2` → …) as part
  of every commit that touches that file, so a screenshot or a deployed page
  can be matched to the code that produced it.
- **Commits:** never mention co-authored-by or the tool used. Commit locally;
  push only when asked. Use `git commit --trailer "Reported-by:<name>"` for
  user-reported bug fixes.
- Keep raw `data/` and regenerated `figures/` out of git (gitignored); only
  curated `docs/figures/` charts are committed for the site.
