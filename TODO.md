# braidlab — TODO

Measurement suite for the braided-universe RSA packing problem. See `DESIGN.md`
for what we are measuring and why.

## Phase 1 — engine (done)
- [x] `--maxfreq` band control on both engines
- [x] `--until-accept-rate` fixed-convergence stop
- [x] CELL/2 edge collision (Chris's correction; D 2.44 → 2.69 at small T)

## Phase 2 — package (done)
- [x] `config` (Campaign/Job, band rule)
- [x] `store` (resumable SQLite, one record per dim/band/T/seed)
- [x] `engine` (command builder + output parser)
- [x] `analyze` (seed-averaged D + bootstrap error, cost exponent)
- [x] `orchestrator` (resumable fleet over SSH)
- [x] `report` (self-contained HTML)
- [x] `cli` (plan / run / analyze / report)
- [x] predefined `campaigns` (2+1, 3+1; nyq band, fixed-convergence)
- [x] infra: pyproject, requirements, setup_env, tests

## Phase 3 — measurement (pending launch)
- [ ] run `3plus1` (edge engine, maxfreq=T, 16 seeds) to error < ±0.02
- [ ] run `2plus1` likewise — does D move off 1.66? (settles phi)
- [ ] curve-collapse check (validates r* ≈ fixed theta; DESIGN §7.1)
- [ ] cost exponent k(f) and Feder d_eff from the kinetic curves
- [ ] report + write-up
