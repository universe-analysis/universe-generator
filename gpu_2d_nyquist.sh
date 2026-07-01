#!/bin/bash
# 2+1 nyquist arm (maxfreq = T) REQUEUE. Originally on the 3090, but the user
# reclaimed it (their job's GPU memory crashed all but T80, which completed:
# N=1157). The 4 survivors (T120/170/230/320) are chained onto the two 3080s
# AFTER their default+safe sweep (waits for "HOST DONE"), never double-occupying.
set -u
BIN='~/braid_engine/cuda/braid_cuda'

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
chmod +x ~/gpu_run2.sh; nohup ~/gpu_run2.sh > ~/gpu_run2.log 2>&1 & echo \"$h nyquist requeue queued pid \$!\""
}

launch2 mother-claude '
run 2dn_T120 -t 120 --attempts 3e12 --smart --seed 1 --maxfreq 120
run 2dn_T320 -t 320 --attempts 3e12 --smart --seed 1 --maxfreq 320'

launch2 kitt-claude '
run 2dn_T170 -t 170 --attempts 3e12 --smart --seed 1 --maxfreq 170
run 2dn_T230 -t 230 --attempts 3e12 --smart --seed 1 --maxfreq 230'

echo "2+1 nyquist requeue (T120/170/230/320) chained onto mother + kitt"
