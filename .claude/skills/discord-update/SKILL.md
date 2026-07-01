---
name: discord-update
description: Post a status update to the project's Discord channel — a campaign description before a run, in-progress/finished status, or an ad-hoc note. Use when the user asks to notify Discord, ping the channel, or share campaign status.
---

# Discord update

Post a message to the project's Discord channel via the configured webhook.

## Prerequisite

The webhook URL is a secret read from the `BRAIDLAB_DISCORD_WEBHOOK` environment
variable (see `CLAUDE.local.md`). If it is not set, the command exits non-zero
with a hint — export it first, e.g.:

```bash
export BRAIDLAB_DISCORD_WEBHOOK="…"   # value lives in CLAUDE.local.md, never committed
```

## Posting

Use the `braidlab notify` subcommand from the repo root:

```bash
# titled embed, color-coded by kind (start | progress | done | fail | info)
uv run python -m braidlab notify \
    --title "Campaign corrdim3d_e6 starting" \
    --message "3+1 packing, cutoff 1e-6, 95 jobs across the fleet" \
    --color start

# plain-text note (omit --title)
uv run python -m braidlab notify -m "Rebuilt the 3090 engine, resuming the run."
```

Color guide: `start` (blurple) before a run, `progress` (blue) mid-run,
`done` (green) on completion, `fail` (red) for problems, `info` (grey) default.

## Note: campaigns already auto-post

`braidlab run` posts its own lifecycle updates (pre-flight summary, progress
pings at 25/50/75%, host-stall warnings, completion) whenever the webhook is
set. Use this skill for **ad-hoc** messages — a heads-up before you launch, a
mid-run note, or sharing a result — not to duplicate what the orchestrator
already sends.
