#!/bin/bash
# Re-home the 3090's orphaned jobs onto the two 3080s, appended AFTER their
# current queues (waits for phase-1 "HOST DONE" so it never double-occupies a GPU).
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
chmod +x ~/gpu_run2.sh; nohup ~/gpu_run2.sh > ~/gpu_run2.log 2>&1 & echo \"$h phase2 launched pid \$!\""
}

launch2 mother-claude '
run d_T28_s1 -t 28 --attempts 3e12 --smart --seed 1
run d_T36_s2 -t 36 --attempts 3e12 --smart --seed 2
run d_T44_s3 -t 44 --attempts 3e12 --smart --seed 3'

launch2 kitt-claude '
run d_T52_s1 -t 52 --attempts 3e12 --smart --seed 1
run d_T60_s2 -t 60 --attempts 3e12 --smart --seed 2'

echo "phase-2 queued on both 3080s"
