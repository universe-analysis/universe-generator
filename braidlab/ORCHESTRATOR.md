# braidlab Orchestrator — Operations Runbook

How the fleet orchestrator works, how to drive a campaign, and the gotchas that
have actually bitten us (CUDA toolchains, SSH detach, GPU memory). Read this
before launching or debugging a fleet run.

## What it is

`braidlab` runs RSA packing jobs on remote GPU hosts over SSH and collects the
results. It is **resumable**: the source of truth is a SQLite store, so a run
that is interrupted (driver killed, host rebooted, host reclaimed) is continued
simply by re-issuing the same command — only the missing jobs are dispatched.

Module map:

| Module | Role |
|---|---|
| `config.py` | `Campaign` (declarative experiment) expands to `Job`s, one per `(dim, band, T, seed)`. |
| `campaigns.py` | Named campaigns (`3plus1`, `corrdim3d`, …). |
| `engine.py` | `build_command` — argv for one packing run. |
| `orchestrator.py` | `plan_assignment` (pure), `Fleet` (SSH/SCP side effects), `run_campaign` (the driver loop). |
| `store.py` | SQLite store keyed by job identity; `curves/` and `dumps/` dirs alongside the db. |
| `analyze.py` / `report.py` | The old N~T^D measurement + HTML report. |
| `corrdim.py` | Turnaround correlation dimension: per-dump sphere/cube/box, seed-averaged. |

The split that matters: `plan_assignment` and `render_runner` are **pure** (unit
tested without hardware); `Fleet` is the only thing that touches SSH.

## Running a campaign

```bash
# Dry run — see the job split (no launch):
uv run python -m braidlab plan corrdim3d --hosts host3,host1,host2

# Launch (resumable; safe to Ctrl-C and re-run):
uv run python -m braidlab run corrdim3d \
    --hosts host3,host1,host2 \
    --db data/corrdim/run.db \
    --host-max host1=160,host2=160 \
    --poll 120

# If binaries are already built and verified, skip the rebuild:
#   add --no-deploy

# Seed-averaged correlation-dimension report from collected dumps:
uv run python -m braidlab corrdim --db data/corrdim/run.db --out corrdim_convergence.png
```

The `run` driver **blocks**, polling every `--poll` seconds until every job is
`done`. Run it in the background; it survives nothing itself, but the remote
runners are detached (see below) and the store makes re-running idempotent.

## CUDA build gotchas (the big one)

The fleet is heterogeneous and **you cannot assume `command -v nvcc` gives a
working compiler, nor that the newest CUDA works.** What we found:

| Host | GPU (arch) | Working nvcc | Notes |
|---|---|---|---|
| host1 | RTX 3080 (sm_86) | `/usr/bin/nvcc` | default host compiler links fine |
| host2 | RTX 3080 (sm_86) | `/usr/local/cuda/bin/nvcc` | **nvcc not on PATH**; lives under `/usr/local/cuda*/bin` |
| host3 | RTX 3090 (sm_86) | `/usr/local/cuda/bin/nvcc` (CUDA 12.4) | **`/usr/bin/nvcc` is too new and fails** |

host3's PATH `nvcc` fails to compile the engine with:

```
/usr/include/c++/11/bits/std_function.h:530: error: parameter packs not expanded with '...'
```

That signature = **CUDA front-end vs. host libstdc++ mismatch**. On host3
neither `/usr/bin/nvcc` (default) nor `/usr/bin/nvcc -ccbin g++-11` works; only
the older CUDA at `/usr/local/cuda` (12.4) links. So `-ccbin g++-11` alone is not
enough — sometimes you need a *different CUDA*.

**The robust strategy `deploy()` now uses:** try every candidate nvcc
(`command -v nvcc`, then `/usr/local/cuda*/bin/nvcc` newest-first) crossed with
`{default host compiler, -ccbin g++-11}`, stop at the first that links:

```bash
cd ~/braidlab_run && rm -f braid_cuda3d
for NVCC in $(command -v nvcc) $(ls /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -rV); do
  for CC in "" "-ccbin g++-11"; do
    if $NVCC -O3 -arch=sm_86 $CC -o braid_cuda3d braid_cuda3d.cu 2>build.err; then break 2; fi
  done
done
test -x braid_cuda3d
```

Notes:
- `rm -f` the binary **before** building and `test -x` after, so a failed build
  leaves *no* binary rather than a stale one (see "empty dumps" below).
- All three cards are **sm_86** (Ampere). `deploy()` derives arch from
  `nvidia-smi --query-gpu=compute_cap`, defaulting to `sm_86`.
- **Never copy a binary between hosts.** They have different glibc; a binary
  built on host3 fails on host1 with a `GLIBC_2.xx not found` error.
  Always build per host.

## SSH / launch gotchas

- **Detached runners.** `launch()` starts the runner with
  `setsid bash run_braidlab.sh >/dev/null 2>&1 </dev/null &` so it survives the
  SSH disconnect. The SSH channel often lingers anyway (server waits on the
  session), so `launch()` uses a short timeout and treats `TimeoutExpired` as
  success — the runner is already detached and running.
- **pgrep/pkill self-match.** Always bracket the first character:
  `pgrep -f '[r]un_braidlab.sh'`. Without the bracket, pgrep matches the SSH
  shell running the very command (its cmdline contains the pattern) and reports a
  false "alive"; `pkill` would kill its own SSH session (exit 255).
- **host1's forwarding noise.** SSH to host1 prints
  `bind [::1]:NNNNN: Address already in use` / `channel_setup_fwd_listener_tcpip`
  / `Could not request local forwarding`. This is a local-forward clash in the
  SSH config and is **harmless** — the command still runs. Filter it when
  scraping output: `grep -vE 'bind|channel|forwarding'`.
- **macOS has no `timeout`.** Don't rely on it from the driver host; use
  until-loops / background polling instead.
- Use `ssh -o BatchMode=yes -o ConnectTimeout=8` for non-interactive checks.
- **One campaign per host at a time.** The remote workspace is shared: every
  campaign writes the SAME `run_braidlab.sh`, and `runner_alive()` cannot tell
  whose runner is running. Launching campaign B on a host where campaign A's
  runner is still alive makes B silently skip its launch (it mistakes A's
  runner for its own) and poll forever collecting nothing; B's deploy also
  overwrites the runner script under A. Run campaigns on disjoint host sets,
  or sequence them (this bit the concurrent torus3d_e6/torus2d_e6 launch on
  2026-07-02).

## GPU memory caps (T ceilings)

VRAM is dominated by a dense per-timestep collision grid (breakdown below), so
the safe T ceiling is set by card VRAM:

| Card | VRAM | Safe T ceiling (3+1) |
|---|---|---|
| RTX 3080 | 10 GB | **160** (≈5.6 GB grid + ~0.7 GB points + context); T≥170 risks OOM |
| RTX 3090 | 24 GB | **~215** (grid is 19.8 GB at T=220, 27.9 GB at T=240) |

Pass caps with `--host-max host1=160,host2=160`. `plan_assignment` only assigns a
job to hosts whose cap it fits, and **raises** if no host can run some T (so a
typo that strands the high-T jobs fails loudly instead of silently dropping
them). With these caps the high-T tail (T=170–200) lands only on host3.

### VRAM breakdown (where it actually goes)

For a high-T 3+1 run the memory is dominated by the **dense per-timestep
collision grid**. Collisions are checked per timestep, so the engine allocates a
full spatial lattice at every one of the T timesteps:

```
grid cells = T × (T+4)³ ≈ T⁴
grid VRAM  = cells × 8 bytes      # cellStart + cellLen, two int32 arrays (4B each)
```

The `× 8` (not 16) is the int32-index optimization — without it the grid doubles
and T=200 would not fit a 3090. The grid is the wall; everything else is small:

| Term | Scaling | At T=200 |
|---|---|---|
| Collision grid (`dCellStart`+`dCellLen`) | ~T⁴ | **13.6 GB** |
| Accepted points (`dPtsX/Y/W`, 3×8B × N·T) | ~N·T | ~1.6 GB |
| Survivor buffer (survCap 2²⁰ × 72 B Path) | fixed | ~75 MB |
| `z/sinz/invz/order` per-timestep arrays | ~T | KB |
| CUDA context / runtime | fixed | ~0.7 GB |
| **Total** | | **~16 GB alloc → ~17.6 GB resident** |

Consequences:
- **3+1 caps at T≈215 on 24 GB** because the grid is T⁴; **2+1 is ~T³** (T × a T²
  lattice), so even T=400 is ~0.5 GB — that is why `corrdim2d` can push T high.
- The grid is **mostly empty at high T** (the pack fills a ~2.8-D fractal subset
  of a 4-D lattice): a speed-for-memory trade. A sparse/hashed grid would cut
  memory hugely at the cost of lookup speed.
- **Host RAM matters too**: the host mirrors `cellStart`/`cellLen` (another
  ~13.6 GB at T=200) plus per-timestep point maps for grid rebuilds. A big GPU
  alone is not enough — the driver host needs the RAM as well.

## Resumability & idempotency

- **Store** (`run.db`): one row per `(dim, band, T, seed, accept_rate)`,
  `pending → done`. `braidlab run` diffs the campaign's jobs against the store
  and dispatches only what's missing.
- **Runner-level**: `run()` skips a job if `results.log` has its name *and*
  `curves/$name.csv` exists, so a re-launched runner never redoes finished work.
- `runner_alive()` prevents double-launching a host that is already packing.

Recover from a host drop by simply re-running the same `braidlab run` command.

## Parameter-dump campaigns (correlation dimension)

`corrdim3d` sets `angle=True` (`--angle-sample`, edge sampler) and `dump=True`.
When `dump=True`, `render_runner` has each job also write `dumps/$name.csv`, then
**subsamples on the host**:

```bash
head -1 dumps/$name.csv  > dumps/$name.sub.csv
tail -n +2 dumps/$name.csv | shuf -n 60000 >> dumps/$name.sub.csv
rm -f dumps/$name.csv
```

`Fleet.fetch_dump` pulls the `.sub.csv` into `store.dumps_dir/$name.csv`. The
subsample (60k worldlines) bounds disk/SCP; the correlation integral is
statistical (it center-samples 4000), so accuracy is unaffected. `shuf -n` on a
file with fewer rows just returns all of them.

## Monitoring

```bash
# Per-host: runner alive, GPU util/mem, jobs finished
ssh HOST 'pgrep -f "[r]un_braidlab.sh" >/dev/null && echo ALIVE;
          nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader;
          grep -c " " ~/braidlab_run/results.log'

# Store progress
uv run python - <<'PY'
import sqlite3
c = sqlite3.connect('data/corrdim/run.db'); c.row_factory = sqlite3.Row
for r in c.execute("SELECT status,count(*) n FROM runs GROUP BY status"):
    print(r["status"], r["n"])
PY
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `parameter packs not expanded` at build | CUDA/host-compiler mismatch. The multi-candidate `deploy()` loop handles it; manually, try `/usr/local/cuda/bin/nvcc` and/or `-ccbin g++-11`. |
| **Empty / missing dumps** | The host ran an **old binary** that lacks `--dump-params` (a silent build failure left a stale binary on PATH). `deploy()` now `rm -f`s before building. Verify a binary: `./braid_cuda3d -t 8 --dump-params /tmp/d.csv ...` then check `/tmp/d.csv` has the 9-column header. |
| OOM at high T | Host VRAM exceeded. Lower its `--host-max` cap and re-run (resumes). |
| `GLIBC_2.xx not found` | A binary was copied from another host. Build per host instead. |
| Jobs never leave `pending` | Runner not alive, or build failed (no binary). Check `runner_alive`, `build.err` on the host. |
| **Variant campaign finishes instantly with the *other* variant's numbers** | Job-name collision in the shared `~/braidlab_run`. The runner's idempotency (`grep "^$name " results.log && [ -f curves/$name.csv ]`) skips a job whose name already has leftover files from a previous campaign of the same `(dim, band, T, seed)`. Encode the variant in `Job.name` (e.g. the `_eu` suffix for euclid) so names are distinct, or wipe the remote `results.log`/`curves`/`dumps` first. |
| Driver hangs at end | A host stalled on its last job; the driver blocks until all `done`. Inspect that host; the run is resumable, so it's safe to kill and re-run. |

## Engine flags the orchestrator uses

`-t <T>` · `--attempts <N>` · `--smart` · `--seed <s>` ·
`--until-accept-rate <r>` (the fixed-convergence stop) · `--maxfreq <m>` (band) ·
`--angle-sample` (edge sampler) · `--torus` (new-dogma model) · `--phase`
(even-frequency phases + symmetric z grid; adds fx/fy/fw dump columns) ·
`--curve <path>` · `--dump-params <path>`.
