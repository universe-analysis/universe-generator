#!/bin/bash
# Distributed 2+1 campaign: shard (T, seed) jobs round-robin across SSH hosts,
# run each on its host (8 threads), collect the curve CSVs back locally.
#
# Usage: ./dist_campaign.sh <budget> <seeds> "<T list>"
#   ./dist_campaign.sh 5e9 2 "40 60 80 100 140 180 240 300"
set -u
BUDGET="${1:-2e8}"
SEEDS="${2:-1}"
TLIST="${3:-60 120 200}"
HOSTS=(deep-thought-claude mother-claude kitt-claude wopr-claude)
BIN='~/braid_engine/target/release/braid_engine'
OUT=/tmp/dist_campaign
mkdir -p "$OUT"

# build the round-robin job list: "T seed"
JOBS=()
for s in $(seq 1 "$SEEDS"); do for T in $TLIST; do JOBS+=("$T $s"); done; done
echo "dispatching ${#JOBS[@]} jobs (budget=$BUDGET) across ${#HOSTS[@]} hosts"

# launch each host's queue in parallel (round-robin by job index modulo nhosts)
nh=${#HOSTS[@]}
pids=()
for hi in $(seq 0 $((nh - 1))); do
  h=${HOSTS[$hi]}
  (
    idx=0
    for job in "${JOBS[@]}"; do
      if [ $((idx % nh)) -eq "$hi" ]; then
        set -- $job; T=$1; S=$2
        name="T${T}_smart_s${S}"
        ssh -o BatchMode=yes "$h" "$BIN -t $T --attempts $BUDGET --threads 8 --seed $S --smart --curve ~/braid_engine/out_${name}.csv >/dev/null 2>&1"
        rsync -az "$h":"braid_engine/out_${name}.csv" "$OUT/${name}.csv" 2>/dev/null \
          && echo "[$(date +%H:%M:%S)] $h -> ${name}.csv" || echo "[$(date +%H:%M:%S)] $h FAILED ${name}"
      fi
      idx=$((idx + 1))
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"
echo "DIST CAMPAIGN DONE -> $OUT"
