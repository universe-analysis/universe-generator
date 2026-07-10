"""Resumable fleet orchestration over SSH.

Expands a campaign, diffs against the store, and dispatches only the missing
runs across GPU hosts. Each host runs its queue under nohup writing a results
log + curve files; the driver polls, collects, and records into the store. The
whole thing is idempotent: re-running picks up exactly where it left off, which
matters because hosts drop, reboot, and get reclaimed mid-campaign.

This module isolates the pure planning logic (`plan_assignment`,
`render_runner`) from the side-effecting `Fleet` so the former can be unit
tested without hardware.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from braidlab.config import Job
from braidlab.engine import binary_name, build_command
from braidlab.notify import DiscordNotifier
from braidlab.store import Store

REMOTE_DIR = "~/braidlab_run"

#: Post a progress ping each time this fraction of jobs completes (25/50/75%).
PROGRESS_STEP = 0.25
#: Consecutive polls a host must look dead-with-work before we flag a stall.
STALL_POLLS = 2


def plan_assignment(
    jobs: list[Job],
    hosts: list[str],
    host_max_t: dict[str, int] | None = None,
) -> dict[str, list[Job]]:
    """Greedily balance jobs across hosts by estimated cost (~T**2).

    Largest jobs first, each to the currently least-loaded *eligible* host, so
    hosts tend to finish together. ``host_max_t`` caps the largest T a host can
    run (GPU memory): a job is only assigned to hosts whose cap it fits. Hosts
    absent from the dict have no cap.

    Raises:
        ValueError: if some job's T exceeds every host's cap.
    """
    caps = host_max_t or {}
    load = {h: 0.0 for h in hosts}
    out: dict[str, list[Job]] = {h: [] for h in hosts}
    for job in sorted(jobs, key=lambda j: -j.t):
        eligible = [h for h in hosts if job.t <= caps.get(h, 10**9)]
        if not eligible:
            raise ValueError(f"no host can run T={job.t} within caps {caps}")
        host = min(eligible, key=lambda h: load[h])
        out[host].append(job)
        load[host] += job.t**2
    return out


#: Worldlines kept per parameter dump (random subsample to bound disk / scp).
DUMP_SUBSAMPLE = 60000


def host_parts(host: str) -> tuple[str, int | None]:
    """Split a host token into (ssh alias, GPU index).

    A plain token (``mother``) is a whole-host worker on the default GPU. A
    ``alias:N`` token (``vast:1``) is one worker pinned to GPU N of a
    multi-GPU host — each such worker gets its own remote workspace and
    ``CUDA_VISIBLE_DEVICES``, so a box's GPUs can be listed as separate
    fleet entries.
    """
    if ":" in host:
        alias, gpu = host.rsplit(":", 1)
        return alias, int(gpu)
    return host, None


def render_runner(
    jobs: list[Job], remote_dir: str, dump: bool = False, gpu: int | None = None
) -> str:
    """Render the bash runner that executes a host's job queue idempotently.

    When ``dump`` is set, each run also writes a `--dump-params` file, which is
    immediately subsampled to ``DUMP_SUBSAMPLE`` random worldlines (header
    preserved) so the collected `.sub.csv` stays small; the full dump is then
    removed. The correlation dimension is statistical, so the subsample costs
    nothing in accuracy. ``gpu`` pins the whole queue to one device of a
    multi-GPU host.
    """
    engine_line = '  "$@" --curve curves/$name.csv'
    if dump:
        engine_line += " --dump-params dumps/$name.csv"
    engine_line += " 2>err_$name.txt"

    lines = [
        "#!/bin/bash",
        f"cd {remote_dir} || exit 1",
    ]
    if gpu is not None:
        lines.append(f"export CUDA_VISIBLE_DEVICES={gpu}")
    lines += [
        "mkdir -p curves",
    ]
    if dump:
        lines.append("mkdir -p dumps")
    lines += [
        "touch results.log",
        "run() {",
        "  local name=$1; shift",
        '  if grep -q "^$name " results.log 2>/dev/null && '
        "[ -f curves/$name.csv ]; then return; fi",
        engine_line,
    ]
    if dump:
        lines += [
            "  if [ -f dumps/$name.csv ]; then",
            "    head -1 dumps/$name.csv > dumps/$name.sub.csv",
            f"    tail -n +2 dumps/$name.csv | shuf -n {DUMP_SUBSAMPLE} "
            ">> dumps/$name.sub.csv",
            "    rm -f dumps/$name.csv",
            "  fi",
        ]
    lines += [
        '  local n=$(grep -oE "N=[0-9]+" err_$name.txt | head -1 | cut -d= -f2)',
        '  local a=$(grep -oE "in [0-9]+ attempts" err_$name.txt | grep -oE "[0-9]+")',
        '  echo "$name ${n:-NA} ${a:-NA}" >> results.log',
        "}",
    ]
    for job in jobs:
        binary = f"{remote_dir}/{binary_name(job.dim)}"
        argv = build_command(job, binary, f"curves/{job.name}.csv")
        # build_command appends --curve; drop it (run() supplies its own)
        argv = argv[: argv.index("--curve")]
        lines.append(f"run {job.name} " + " ".join(argv))
    lines.append('echo "ALLDONE $(date +%s)" >> results.log')
    return "\n".join(lines) + "\n"


class Fleet:
    """Side-effecting SSH/SCP operations against GPU hosts.

    Host tokens are either plain SSH aliases (``mother``) or ``alias:N`` for
    one GPU of a multi-GPU host (``vast:0``, ``vast:1``). Each GPU token gets
    its own remote workspace (``~/braidlab_run_g<N>``) and a runner pinned via
    ``CUDA_VISIBLE_DEVICES``, so the GPUs behave as independent fleet workers.
    Do not mix a plain token and GPU tokens for the same box in one campaign:
    the plain token's liveness check matches any runner on the box.
    """

    def __init__(self, source_dir: str | Path, remote_dir: str = REMOTE_DIR) -> None:
        self.source = Path(source_dir)
        self.remote = remote_dir

    def _alias(self, host: str) -> str:
        return host_parts(host)[0]

    def _dir(self, host: str) -> str:
        gpu = host_parts(host)[1]
        return self.remote if gpu is None else f"{self.remote}_g{gpu}"

    def _ssh(self, host: str, cmd: str, timeout: int = 120) -> str:
        out = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                self._alias(host),
                cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return out.stdout

    def _scp_to(self, host: str, local: Path, remote: str) -> None:
        subprocess.run(
            ["scp", "-q", str(local), f"{self._alias(host)}:{remote}"],
            check=True,
            timeout=120,
        )

    def _scp_from(self, host: str, remote: str, local: Path) -> bool:
        res = subprocess.run(
            ["scp", "-q", f"{self._alias(host)}:{remote}", str(local)], timeout=120
        )
        return res.returncode == 0

    def deploy(self, host: str, dims: set[int]) -> None:
        """Push engine sources and build the needed binaries on `host`."""
        rdir = self._dir(host)
        self._ssh(host, f"mkdir -p {rdir}")
        cap = self._ssh(
            host,
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1",
        ).strip()
        arch = "sm_" + cap.replace(".", "") if cap else "sm_86"
        for dim in dims:
            b = binary_name(dim)
            src = self.source / "cuda" / f"{b}.cu"
            self._scp_to(host, src, f"{rdir}/{b}.cu")
            # Heterogeneous fleet: nvcc may not be on PATH and the right CUDA /
            # host-compiler pairing differs per host (the 3090 needs an older
            # CUDA than its newest). Try every candidate nvcc x {default,
            # g++-11} until one links, newest CUDA first.
            self._ssh(
                host,
                f"cd {rdir} && rm -f {b} && "
                "for NVCC in $(command -v nvcc) "
                "$(ls /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -rV); do "
                'for CC in "" "-ccbin g++-11"; do '
                f"if $NVCC -O3 -arch={arch} -Xcompiler -pthread $CC -o {b} {b}.cu "
                f"2>{b}_build.err; "
                "then break 2; fi; "
                f"done; done; test -x {b}",
                timeout=400,
            )

    def launch(self, host: str, jobs: list[Job], dump: bool = False) -> None:
        """Write and nohup the host's runner queue."""
        rdir = self._dir(host)
        gpu = host_parts(host)[1]
        script = render_runner(jobs, rdir, dump=dump, gpu=gpu)
        tmp = self.source / f".runner_{host.replace(':', '_')}.sh"
        tmp.write_text(script)
        self._scp_to(host, tmp, f"{rdir}/run_braidlab.sh")
        tmp.unlink()
        # Double-fork daemonization: the subshell backgrounds the setsid runner
        # and exits immediately, reparenting the runner to PID 1 before the ssh
        # session can be torn down. Plain `setsid ... &` left the runner a child
        # of the remote shell for an instant, and some sshds (Vast.ai's wrapper)
        # reap the whole session tree when the client disconnects -- which is
        # exactly what happens when the short timeout kills our ssh. The sleep
        # gives the reparent a beat to complete; timeout is still treated as
        # success. GPU tokens launch by path, so each runner's cmdline carries
        # its workspace and the liveness check can tell same-box runners apart.
        try:
            self._ssh(
                host,
                f"cd {rdir} && (setsid bash {rdir}/run_braidlab.sh "
                ">/dev/null 2>&1 </dev/null &); sleep 1; echo started",
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            pass

    def runner_alive(self, host: str) -> bool:
        # Bracket the first char so pgrep does not match the ssh shell running
        # this very command (whose cmdline contains the pattern). GPU tokens
        # match their own workspace path; plain tokens keep the legacy pattern
        # (which matches any runner on the box).
        gpu = host_parts(host)[1]
        if gpu is None:
            return bool(self._ssh(host, "pgrep -f '[r]un_braidlab.sh'").strip())
        base = self._dir(host).rsplit("/", 1)[-1]  # e.g. braidlab_run_g1
        pat = f"[{base[0]}]{base[1:]}/run_braidlab.sh"
        return bool(self._ssh(host, f"pgrep -f '{pat}'").strip())

    def poll(self, host: str, dest_dir: Path) -> list[tuple[str, int, int]]:
        """Fetch results.log, return [(name, n, attempts)] for finished jobs."""
        local = dest_dir / f".results_{host.replace(':', '_')}.log"
        if not self._scp_from(host, f"{self._dir(host)}/results.log", local):
            return []
        done: list[tuple[str, int, int]] = []
        for line in local.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[1] != "NA":
                done.append((parts[0], int(parts[1]), int(parts[2])))
        return done

    def fetch_curve(self, host: str, name: str, dest: Path) -> bool:
        return self._scp_from(host, f"{self._dir(host)}/curves/{name}.csv", dest)

    def fetch_dump(self, host: str, name: str, dest: Path) -> bool:
        """Pull back the subsampled parameter dump for a finished job."""
        return self._scp_from(host, f"{self._dir(host)}/dumps/{name}.sub.csv", dest)


def run_campaign(
    jobs: list[Job],
    store: Store,
    fleet: Fleet,
    hosts: list[str],
    *,
    poll_seconds: int = 120,
    deploy: bool = True,
    dump: bool = False,
    host_max_t: dict[str, int] | None = None,
    notifier: DiscordNotifier | None = None,
    campaign_name: str = "campaign",
    start_description: str = "",
    start_fields: dict[str, object] | None = None,
) -> None:
    """Drive a campaign to completion, resuming from the store.

    Blocks, polling every `poll_seconds`, until every job is recorded done.
    Safe to interrupt and re-run. When ``dump`` is set, each finished job's
    subsampled parameter dump is collected into ``store.dumps_dir``.
    ``host_max_t`` caps each host's largest T (GPU memory).

    If a ``notifier`` is given (or ``BRAIDLAB_DISCORD_WEBHOOK`` is set) it posts
    Discord updates: a pre-flight summary, progress pings every
    ``PROGRESS_STEP`` of completion, host-stall warnings, and a completion
    summary. Notifications are best-effort and never affect the run.
    """
    notifier = notifier or DiscordNotifier()
    for job in jobs:
        store.register(job)
    pending = store.pending(jobs)
    if not pending:
        return
    by_name = {j.name: j for j in jobs}
    assignment = plan_assignment(pending, hosts, host_max_t)
    dims = {j.dim for j in pending}
    total = len(pending)

    fields = dict(start_fields or {})
    fields.setdefault("Pending jobs", total)
    notifier.campaign_start(campaign_name, start_description, fields)
    started = time.monotonic()

    for host, host_jobs in assignment.items():
        if not host_jobs:
            continue
        try:
            if deploy:
                fleet.deploy(host, dims)
            if not fleet.runner_alive(host):
                fleet.launch(host, host_jobs, dump=dump)
        except Exception as exc:  # surface to Discord, then abort as before
            notifier.campaign_failed(
                campaign_name, f"deploy/launch failed on {host}: {exc}"
            )
            raise

    remaining = {j.key for j in pending}
    host_keys = {host: {j.key for j in hjobs} for host, hjobs in assignment.items()}
    next_progress = PROGRESS_STEP  # completion fraction of the next ping
    stalled_polls = {h: 0 for h in hosts}
    warned: set[str] = set()

    while remaining:
        time.sleep(poll_seconds)
        for host in hosts:
            for name, n_final, attempts in fleet.poll(host, store.curves_dir):
                job = by_name.get(name)
                if job is None or job.key not in remaining:
                    continue
                dest = store.curves_dir / f"{name}.csv"
                fleet.fetch_curve(host, name, dest)
                if dump:
                    fleet.fetch_dump(host, name, store.dumps_dir / f"{name}.csv")
                store.mark(
                    job,
                    "done",
                    n_final=n_final,
                    attempts=attempts,
                    curve_path=str(dest),
                    host=host,
                )
                remaining.discard(job.key)

        done_count = total - len(remaining)
        if remaining and done_count / total >= next_progress:
            notifier.campaign_progress(
                campaign_name, done_count, total, time.monotonic() - started
            )
            while done_count / total >= next_progress:
                next_progress += PROGRESS_STEP

        # Host-stall detection: a host whose runner has died while it still owns
        # unfinished jobs is stuck. Require two consecutive polls to avoid a
        # false alarm in the gap between a job finishing and being collected.
        for host in hosts:
            host_remaining = host_keys.get(host, set()) & remaining
            if host_remaining and not fleet.runner_alive(host):
                stalled_polls[host] += 1
            else:
                stalled_polls[host] = 0
            if stalled_polls[host] >= STALL_POLLS and host not in warned:
                warned.add(host)
                notifier.campaign_failed(
                    campaign_name,
                    f"host **{host}** looks stalled — its runner is gone with "
                    f"{len(host_remaining)} job(s) unfinished. Re-run to resume.",
                )

    notifier.campaign_done(
        campaign_name,
        total,
        time.monotonic() - started,
        str(store.dumps_dir if dump else store.curves_dir),
    )
