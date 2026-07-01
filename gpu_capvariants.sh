#!/bin/bash
# Phase-2: two cap-variant D-sweeps to disentangle band-width from aliasing.
#   safe-full = max-freq round(0.85*T)  (nearly full band, safely below Nyquist)
#   nyquist   = max-freq T              (full band, at the aliasing edge)
# 1 seed each, same 5 T-values as the default-cap baseline (D~2.6, 2 seeds).
# Queued AFTER each host's current default-cap sweep (waits for "HOST DONE").
set -u
BIN='~/braid_engine/cuda/braid_cuda3d'

launch2() {  # host  jobs-blob
  local h="$1" jobs="$2"
  ssh -o BatchMode=yes "$h" "cat > ~/gpu_run2.sh <<'RUNNER'
#!/bin/bash
cd ~/braid_engine
while ! grep -q 'HOST DONE' ~/gpu_run.log; do sleep 120; done
run() { local name=\$1; shift; echo \"[\$(date +%H:%M:%S)] \$name\"; $BIN \"\$@\" --curve ~/braid_engine/o_\${name}.csv 2>&1 | grep done; }
$jobs
echo \"FLEET2 DONE \$(date +%H:%M:%S)\"
RUNNER
chmod +x ~/gpu_run2.sh; nohup ~/gpu_run2.sh > ~/gpu_run2.log 2>&1 & echo \"$h cap-variant phase queued pid \$!\""
}

# nyquist (max-freq=T) and safe-full (0.85T) split so each host has 5 jobs
launch2 mother-claude '
run nyq_T28  -t 28 --attempts 3e12 --smart --seed 1 --maxfreq 28
run nyq_T44  -t 44 --attempts 3e12 --smart --seed 1 --maxfreq 44
run nyq_T60  -t 60 --attempts 3e12 --smart --seed 1 --maxfreq 60
run safe_T36 -t 36 --attempts 3e12 --smart --seed 1 --maxfreq 31
run safe_T52 -t 52 --attempts 3e12 --smart --seed 1 --maxfreq 44'

launch2 kitt-claude '
run nyq_T36  -t 36 --attempts 3e12 --smart --seed 1 --maxfreq 36
run nyq_T52  -t 52 --attempts 3e12 --smart --seed 1 --maxfreq 52
run safe_T28 -t 28 --attempts 3e12 --smart --seed 1 --maxfreq 24
run safe_T44 -t 44 --attempts 3e12 --smart --seed 1 --maxfreq 37
run safe_T60 -t 60 --attempts 3e12 --smart --seed 1 --maxfreq 51'

echo "cap-variant D-sweeps (safe-full 0.85T + nyquist T) queued on mother + kitt"
