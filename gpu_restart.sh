#!/bin/bash
# Restart after the power trip — 3090 left out (in use). 2 seeds to fit ~1 day
# on the two 3080s; cap-sweep on the 2070S with the fixed --maxfreq semantics.
set -u
BIN='~/braid_engine/cuda/braid_cuda3d'

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
run d_T28_s1 -t 28 --attempts 3e12 --smart --seed 1
run d_T44_s1 -t 44 --attempts 3e12 --smart --seed 1
run d_T60_s1 -t 60 --attempts 3e12 --smart --seed 1
run d_T36_s2 -t 36 --attempts 3e12 --smart --seed 2
run d_T52_s2 -t 52 --attempts 3e12 --smart --seed 2'

launch kitt-claude '
run d_T36_s1 -t 36 --attempts 3e12 --smart --seed 1
run d_T52_s1 -t 52 --attempts 3e12 --smart --seed 1
run d_T28_s2 -t 28 --attempts 3e12 --smart --seed 2
run d_T44_s2 -t 44 --attempts 3e12 --smart --seed 2
run d_T60_s2 -t 60 --attempts 3e12 --smart --seed 2'

launch wopr-claude '
run cap_T48_f12 -t 48 --attempts 1e12 --smart --seed 1 --maxfreq 12
run cap_T48_f24 -t 48 --attempts 1e12 --smart --seed 1 --maxfreq 24
run cap_T48_f36 -t 48 --attempts 1e12 --smart --seed 1 --maxfreq 36
run cap_T48_f48 -t 48 --attempts 1e12 --smart --seed 1 --maxfreq 48'

echo "restart launched on mother, kitt, wopr"
