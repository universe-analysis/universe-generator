#!/bin/bash
# Nyquist-cap D-sweep: same T-sweep as the default run, but with --maxfreq = T
# at every T (the full Nyquist frequency band). Queued AFTER the current
# default-cap sweep on each host (waits for its "HOST DONE").
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
chmod +x ~/gpu_run2.sh; nohup ~/gpu_run2.sh > ~/gpu_run2.log 2>&1 & echo \"$h nyquist phase queued pid \$!\""
}

launch2 mother-claude '
run nyq_T28_s1 -t 28 --attempts 3e12 --smart --seed 1 --maxfreq 28
run nyq_T44_s1 -t 44 --attempts 3e12 --smart --seed 1 --maxfreq 44
run nyq_T60_s1 -t 60 --attempts 3e12 --smart --seed 1 --maxfreq 60
run nyq_T36_s2 -t 36 --attempts 3e12 --smart --seed 2 --maxfreq 36
run nyq_T52_s2 -t 52 --attempts 3e12 --smart --seed 2 --maxfreq 52'

launch2 kitt-claude '
run nyq_T36_s1 -t 36 --attempts 3e12 --smart --seed 1 --maxfreq 36
run nyq_T52_s1 -t 52 --attempts 3e12 --smart --seed 1 --maxfreq 52
run nyq_T28_s2 -t 28 --attempts 3e12 --smart --seed 2 --maxfreq 28
run nyq_T44_s2 -t 44 --attempts 3e12 --smart --seed 2 --maxfreq 44
run nyq_T60_s2 -t 60 --attempts 3e12 --smart --seed 2 --maxfreq 60'

echo "Nyquist-cap D-sweep queued (starts when the current default-cap sweep finishes)"
