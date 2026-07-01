# Correlation-Dimension Campaign — Self-Contained Simulation + Report

Goal: regenerate the turnaround correlation-dimension data on a fine T grid with
multiple seeds, so the convergence plots are smooth and carry real error bars,
then produce a report. Extend the existing `braidlab` fleet orchestrator rather
than building from scratch.

Scope (agreed): fine grid **T = 20..200 step 10** (19 values) x **5 seeds**
(~95 runs), full Nyquist band, smart + edge (angle) sampler, 1e-7 acceptance
cutoff. Hosts: host3 (3090, high-T tail), host1 + host2 (3080s).

## Phase 1 — Param-dump generation in the fleet  [code, no GPU]  DONE
- [x] `config.py`: `angle` on Job/Campaign, `dump` on Campaign.
- [x] `engine.py`: `build_command` emits `--angle-sample` when angle.
- [x] `orchestrator.py`: `render_runner(dump=True)` writes + host-side
      subsamples the dump (`shuf -n 60000`); `Fleet.fetch_dump` collects it.
      Plus `plan_assignment(host_max_t=...)` for GPU-memory caps.
- [x] `store.py`: `dumps_dir` alongside `curves_dir`.
- [x] Curve emission intact (old N~T^D path untouched).

## Phase 2 — Fine campaign definition  [config]  DONE
- [x] `campaigns.py`: `corrdim3d` (T=20..200 step 10, 5 seeds, nyq,
      smart+angle, dump=True).

## Phase 3 — Correlation-dimension report  [code, no GPU]  DONE
- [x] `braidlab/corrdim.py`: core math + per (T, seed) sphere/cube/box, seed
      mean +/- SEM per T (`aggregate`), error-barred `plot_convergence`.
- [x] `braidlab corrdim` CLI command.

## Phase 4 — Local smoke test  [no GPU]  DONE
- [x] New path reproduces the standalone numbers exactly on existing dumps
      (T=120 sphere 2.795 / cube 2.724 / box 2.669; converged 2.799).
- [x] `uv run pytest` (22) green; ruff + pyright clean.

## Phase 5 — Launch the campaign  [GPU fleet]  DONE
- [x] Hosts reachable; engine builds on all three (robust multi-candidate nvcc).
- [x] Dispatched: 95 jobs, host3 T=170-200, host1/host2 T=20-160.
- [x] Runners alive, GPUs packing (host3 T=200 -> 17.6GB, fits 3090).
- [x] Ran 02:15 -> 13:20 (~11h); 95/95 done, 95 dumps collected, no stragglers.

## Phase 6 — Collect, report, commit  [local]  DONE
- [x] `braidlab corrdim` seeded report -> figures/corrdim3d_seeded.png.
- [x] D2 spheres 2.79+/-0.002, cubes 2.73+/-0.002; box rises to ~2.70-2.74.
- [x] PHYSICS_FINDINGS section 8 updated with the definitive seeded result.

## Phase 6 — Collect, report, commit  [local]
- [ ] Collect dumps, build the seeded convergence report.
- [ ] Update PHYSICS_FINDINGS with the error-barred numbers.
- [ ] Commit (local only).
