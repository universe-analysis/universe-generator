# Example datasets

Small curated copies of real GPU packing runs, committed so the interactive
viewers can be tried straight from a checkout (the full raw `data/` tree stays
gitignored).

## `dumps/` — engine parameter dumps

Each CSV is the output of a CUDA packing engine run with `--dump-params`: one
row per **accepted worldline**, parameters only — positions are analytic, so
this is the entire dataset. Columns: `ax,ay` (wiggle amplitudes, `|a·b| = 1`),
`bx,by` (integer frequencies), `ax2,ay2` (the `sin(1·z)` coherence term), and
optionally `fx,fy` (phase shifts; absent in pre-phase dumps, which read as 0).
The time-step count `T` is encoded in the `_T###_` filename token.

| File | Schema | Paths | Source |
|---|---|---|---|
| `d2_nyq_T200_s1.csv` | 2+1 hard-wall, T=200 | 4390 | `corrdim2d` campaign |
| `d2_nyq_T100_s1.csv` | 2+1 hard-wall, T=100 | 1480 | `corrdim2d` campaign |
| `d2_nyq_T200_s1_tor.csv` | 2+1 torus, T=200 | 2735 | `torus2d` campaign |
| `d2_demo_phase_T100_s0.csv` | synthetic 8-column demo | 50 | first rows of a torus dump with zero `fx,fy` appended (exercises the phase-column parser) |

**To view:** open `docs/viewers/twoplusone_2torus_wrapped.html` in a browser
and press **Import CSV**. These GPU-packed runs sit much nearer to jamming
than anything the in-page generator can produce in reasonable time.
