#!/bin/bash
# Overnight deep-tail campaign: does packing truly saturate or grow forever?
# Low T (reachable ceiling) pushed deep, 2 seeds. Breadth-first (all T at seed 1,
# then seed 2) so one full set survives even if interrupted.
set -e
BIN=./target/release/braid_engine
OUT=/tmp/campaign_overnight
mkdir -p "$OUT"
THREADS=16
BUDGET=2e10

run() { # T seed
  local T=$1 S=$2
  local f="$OUT/T${T}_smart_s${S}.csv"
  echo "[$(date +%H:%M:%S)] T=$T seed=$S budget=$BUDGET"
  "$BIN" -t "$T" --attempts "$BUDGET" --threads "$THREADS" --seed "$S" --smart --curve "$f" 2>&1 | grep done
}

for S in 1 2; do
  for T in 60 80 120; do
    run "$T" "$S"
  done
done

echo "OVERNIGHT DONE $(date +%H:%M:%S)"
