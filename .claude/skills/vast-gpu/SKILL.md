---
name: vast-gpu
description: Rent, manage, and tear down vast.ai cloud GPU instances end-to-end — search offers by GPU model and price, provision instances, run a job on them over SSH, then destroy the instances and verify nothing is still billing. Use this whenever the user mentions vast.ai, renting/spinning up GPUs, cloud GPU instances, running a job on rented 3090s/4090s, checking what rentals are running or billing, or deleting/tearing down GPU rentals — even if they only say something like "get me 4 3090s and run this".
---

# vast-gpu: rent GPUs, run the job, leave nothing billing

Rented instances bill from the moment they are created until they are
destroyed — including while idle, stuck, or "stopped" (stopped still bills
storage). This account has a history of rentals left idling after a job
finished. So the contract of every provisioning task is a full loop:

**pick offers → confirm cost → provision → run job → collect results →
destroy → verify destroyed → report actual spend.**

Never end a task with an instance you created still alive unless the user
explicitly asked to keep it. If the job fails, the loop still ends in
destroy (tell the user what failed first, and ask before destroying if the
instance state would be needed to debug). If *you* fail or get interrupted,
`status` then `destroy --label <label>` is the recovery path.

## The helper script

All lifecycle mechanics live in `scripts/vast_ops.py` (stdlib-only; shells
out to the `vastai` CLI, falling back to `uvx vastai` automatically). Run it
from the directory that holds the `.env` (normally the repo root):

```bash
python .claude/skills/vast-gpu/scripts/vast_ops.py <subcommand> ...
```

| Subcommand | What it does |
|---|---|
| `check [--register-ssh-key]` | API key valid? SSH key registered with the account? Run this first in a fresh session. |
| `pick --gpu RTX_3090 --count 4 [--num-gpus 1] [--max-dph 0.25] [--query '...']` | Search offers, dedupe to one per physical machine, print the N cheapest with $/hr total. `--json` for machine-readable. |
| `provision <offer_ids...> --image IMG --label LBL [--disk 20] [--onstart-cmd '...']` | Rent the offers, wait until each is `running` and its SSH port answers, print ready-to-paste `ssh` commands and total burn rate. |
| `status [--label LBL]` | What exists, burn rate, estimated spend so far. |
| `destroy --label LBL` (or ids) | Destroy, poll until actually gone, report per-instance spend, and list anything else still billing on the account. |

## API key

Resolution order: `VAST_API_KEY` env var → `VAST_API_KEY=...` line in
`./.env` or `./.env.local` → the CLI's own `~/.config/vastai/vast_api_key`.
If `check` fails with an auth error, ask the user to put the key in `.env`
(and make sure `.env` is gitignored — never commit or print the key).

## Cost confirmation

Before `provision`, show the user the picked offers and the total $/hr and
get a go-ahead — unless their request already authorized the spend (named a
budget, a price cap, or said to proceed without asking). One confirmation
covers the whole task; don't re-ask before destroy (destroying is the
default they signed up for).

Rules of thumb for picking: `pick`'s defaults (verified machines,
reliability > 0.98, cheapest per distinct machine) are usually right.
Prefer several 1-GPU instances for embarrassingly parallel work (one worker
per instance; a dead host loses less) and a single `--num-gpus N` instance
when the job itself is multi-GPU or needs shared disk. Add
`--query 'inet_down>500'` when the job moves big files.

## Choosing an image

- Job compiles CUDA on the host (nvcc needed): `nvidia/cuda:12.4.1-devel-ubuntu22.04`
  and match the machine with `--query 'cuda_vers>=12.4'`.
- Job only runs prebuilt binaries / Python: the lighter
  `nvidia/cuda:12.4.1-runtime-ubuntu22.04` or `pytorch/pytorch` variants.
- `--disk 20` is fine for most jobs; size up for datasets.

## Running the job

`provision` prints one SSH command per instance
(`ssh -o StrictHostKeyChecking=accept-new -p PORT root@HOST`). From there
it's ordinary remote work:

```bash
# run something
ssh -o StrictHostKeyChecking=accept-new -p PORT root@HOST 'nvidia-smi'
# push inputs / pull results
scp -o StrictHostKeyChecking=accept-new -P PORT job.tar.gz root@HOST:/root/
scp -o StrictHostKeyChecking=accept-new -P PORT root@HOST:/root/out/results.tgz .
```

For long jobs, launch under `nohup ... &` or `tmux` and poll, so an SSH
drop doesn't kill the run. `--onstart-cmd` runs a script at boot — good for
fire-and-forget setup (apt installs, git clone), but still verify over SSH
that it did what you expected before starting the real job.

Treat instances as untrusted, disposable machines: put nothing on them
beyond what the job needs (never the vast.ai API key or other secrets), and
copy results off before destroy — **destroy deletes the disk irreversibly**.

## Gotchas

- **Offer IDs are single-use** and go stale fast: a `pick` → `provision`
  pair should be back-to-back. If `provision` reports an offer as taken, it
  skips it — re-run `pick` for replacements.
- An instance can sit in `loading` for a few minutes while the image pulls;
  `provision` waits up to `--timeout` (default 600 s). If it reports NOT
  READY, the machine may be a dud — destroy it and rent a replacement
  rather than waiting indefinitely.
- `--label` is the handle for everything after creation. Always pass a
  distinct label per task (e.g. `job-YYYYMMDD-topic`) so `destroy --label`
  can't touch unrelated rentals.
- Multi-GPU offers: `--num-gpus '>=4'` needs quoting. GPU names use
  underscores (`RTX_3090`).
- If the user has existing rentals unrelated to this task, `destroy` will
  list them as still billing — mention that to the user, don't destroy
  what you didn't create.
