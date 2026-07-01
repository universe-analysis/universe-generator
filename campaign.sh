#!/bin/bash
# Growth-rate measurement campaign. Writes one curve CSV per (T, mode, seed).
set -e
BIN=./target/release/braid_engine
OUT=/tmp/campaign
mkdir -p "$OUT"
THREADS=16

run() { # T budget mode seed
  local T=$1 B=$2 M=$3 S=$4
  local f="$OUT/T${T}_${M}_s${S}.csv"
  echo "[$(date +%H:%M:%S)] T=$T mode=$M seed=$S budget=$B"
  "$BIN" -t "$T" --attempts "$B" --threads "$THREADS" --seed "$S" --"$M" --curve "$f" 2>&1 | grep done
}

# Focus: T=200 smart, 3 seeds (error bars on the kinetics)
for S in 1 2 3; do run 200 1.5e9 smart "$S"; done
# Scaling with resolution: T=120 and T=300, 2 seeds
for S in 1 2; do run 120 2e9 smart "$S"; done
for S in 1 2; do run 300 1e9 smart "$S"; done
# Natural vs smart contrast at T=200, 2 seeds
for S in 1 2; do run 200 1.5e9 uniform "$S"; done

echo "CAMPAIGN DONE $(date +%H:%M:%S)"
