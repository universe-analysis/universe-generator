#!/bin/bash
# Thorough 3+1 GPU campaign, fanned across the fleet. Each host runs its own
# queue via a nohup'd remote runner (survives the launcher disconnecting).
#   3 fast GPUs (3090 + 2x 3080): main D-sweep, 5 T-values x 3 seeds @ 3e12
#   slow GPU (2070S):             the lighter --maxfreq cap-sweep
set -u
BIN='~/braid_engine/cuda/braid_cuda3d'

# host -> the list of "name|engine-args" jobs it should run, one per line
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

# --- main D-sweep, round-robin over the 3 fast hosts (5 jobs each) ---
launch deep-thought-claude '
run d_T28_s1 -t 28 --attempts 3e12 --smart --seed 1
run d_T52_s1 -t 52 --attempts 3e12 --smart --seed 1
run d_T36_s2 -t 36 --attempts 3e12 --smart --seed 2
run d_T60_s2 -t 60 --attempts 3e12 --smart --seed 2
run d_T44_s3 -t 44 --attempts 3e12 --smart --seed 3'

launch mother-claude '
run d_T36_s1 -t 36 --attempts 3e12 --smart --seed 1
run d_T60_s1 -t 60 --attempts 3e12 --smart --seed 1
run d_T44_s2 -t 44 --attempts 3e12 --smart --seed 2
run d_T28_s3 -t 28 --attempts 3e12 --smart --seed 3
run d_T52_s3 -t 52 --attempts 3e12 --smart --seed 3'

launch kitt-claude '
run d_T44_s1 -t 44 --attempts 3e12 --smart --seed 1
run d_T28_s2 -t 28 --attempts 3e12 --smart --seed 2
run d_T52_s2 -t 52 --attempts 3e12 --smart --seed 2
run d_T36_s3 -t 36 --attempts 3e12 --smart --seed 3
run d_T60_s3 -t 60 --attempts 3e12 --smart --seed 3'

# --- cap-sweep on the slow 2070S (T=48, vary the frequency cap) ---
launch wopr-claude '
run cap_T48_m12 -t 48 --attempts 1e12 --smart --seed 1 --maxfreq 12
run cap_T48_m24 -t 48 --attempts 1e12 --smart --seed 1 --maxfreq 24
run cap_T48_m36 -t 48 --attempts 1e12 --smart --seed 1 --maxfreq 36
run cap_T48_m47 -t 48 --attempts 1e12 --smart --seed 1 --maxfreq 47'

echo "all hosts launched"
