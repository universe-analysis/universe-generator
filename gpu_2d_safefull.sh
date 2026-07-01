#!/bin/bash
# 2+1 band experiment: default-cap (T/2) vs safe-full (round 0.85*T) at a matched
# T-grid, deep budget, 1 seed each. Mirrors the 3+1 cap-variant run so we get a
# clean dD_2D to compare against dD_3D=+0.157 and settle the phi=1.618 question.
#   T-grid : 80 120 170 230 320   (~4x span for log-log leverage)
#   safe   : maxfreq round(0.85*T) = 68 102 145 196 272
# Runs on the two 3080s; 2070S (wopr) left free for a 2nd seed if warranted.
set -u
BIN='~/braid_engine/cuda/braid_cuda'

launch() {  # host  jobs-blob
  local h="$1" jobs="$2"
  ssh -o BatchMode=yes "$h" "cat > ~/gpu_run.sh <<'RUNNER'
#!/bin/bash
cd ~/braid_engine
run() { local name=\$1; shift; echo \"[\$(date +%H:%M:%S)] \$name\"; $BIN \"\$@\" --curve ~/braid_engine/o_\${name}.csv 2>&1 | grep done; }
$jobs
echo \"HOST DONE \$(date +%H:%M:%S)\"
RUNNER
chmod +x ~/gpu_run.sh; nohup ~/gpu_run.sh > ~/gpu_run.log 2>&1 & echo \"$h launched pid \$!\""
}

launch mother-claude '
run 2dd_T80  -t 80  --attempts 3e12 --smart --seed 1
run 2dd_T170 -t 170 --attempts 3e12 --smart --seed 1
run 2dd_T320 -t 320 --attempts 3e12 --smart --seed 1
run 2ds_T120 -t 120 --attempts 3e12 --smart --seed 1 --maxfreq 102
run 2ds_T230 -t 230 --attempts 3e12 --smart --seed 1 --maxfreq 196'

launch kitt-claude '
run 2dd_T120 -t 120 --attempts 3e12 --smart --seed 1
run 2dd_T230 -t 230 --attempts 3e12 --smart --seed 1
run 2ds_T80  -t 80  --attempts 3e12 --smart --seed 1 --maxfreq 68
run 2ds_T170 -t 170 --attempts 3e12 --smart --seed 1 --maxfreq 145
run 2ds_T320 -t 320 --attempts 3e12 --smart --seed 1 --maxfreq 272'

echo "2+1 default+safe-full sweep queued on mother + kitt"
