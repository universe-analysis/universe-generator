"""Tests for the pure orchestration logic (no SSH)."""

from pathlib import Path

import pytest

from braidlab.config import Job
from braidlab.orchestrator import plan_assignment, render_runner


def _jobs(n: int) -> list[Job]:
    return [
        Job(
            dim=3, band="nyq", t=20 + 4 * i, seed=1, accept_rate=1e-7, max_attempts=3e12
        )
        for i in range(n)
    ]


def test_assignment_covers_all_jobs() -> None:
    jobs = _jobs(7)
    assign = plan_assignment(jobs, ["a", "b", "c"])
    flat = [j for js in assign.values() for j in js]
    assert sorted(j.t for j in flat) == sorted(j.t for j in jobs)


def test_assignment_balances_by_cost() -> None:
    jobs = _jobs(6)
    assign = plan_assignment(jobs, ["a", "b"])
    loads = [sum(j.t**2 for j in js) for js in assign.values()]
    # greedy least-loaded keeps the two hosts within one big job of each other
    assert abs(loads[0] - loads[1]) <= max(j.t**2 for j in jobs)


def test_render_runner_has_jobs_and_marker() -> None:
    jobs = _jobs(2)
    script = render_runner(jobs, "~/braidlab_run")
    assert "ALLDONE" in script
    assert "--until-accept-rate" in script
    assert "--maxfreq" in script  # nyq band -> maxfreq emitted
    for j in jobs:
        assert f"run {j.name} " in script


def test_render_runner_dump_mode_emits_dump_and_subsample() -> None:
    script = render_runner(_jobs(1), "~/braidlab_run", dump=True)
    assert "--dump-params dumps/$name.csv" in script
    assert "mkdir -p dumps" in script
    assert "shuf -n" in script  # host-side subsample to bound disk
    # default (no dump) stays clean
    assert "--dump-params" not in render_runner(_jobs(1), "~/braidlab_run")


def test_plan_assignment_respects_host_caps() -> None:
    jobs = [
        Job(dim=3, band="nyq", t=t, seed=1, accept_rate=1e-7, max_attempts=3e12)
        for t in (60, 200)
    ]
    assign = plan_assignment(jobs, ["big", "small"], host_max_t={"small": 160})
    assert all(j.t <= 160 for j in assign["small"])
    assert any(j.t == 200 for j in assign["big"])


def test_plan_assignment_raises_when_no_host_fits() -> None:
    jobs = [Job(dim=3, band="nyq", t=200, seed=1, accept_rate=1e-7, max_attempts=3e12)]
    with pytest.raises(ValueError):
        plan_assignment(jobs, ["small"], host_max_t={"small": 160})


def test_host_parts_plain_and_gpu_tokens() -> None:
    from braidlab.orchestrator import host_parts

    assert host_parts("mother") == ("mother", None)
    assert host_parts("vast:1") == ("vast", 1)
    assert host_parts("vast:0") == ("vast", 0)


def test_render_runner_gpu_pin() -> None:
    """A GPU token's runner pins the whole queue to its device."""
    script = render_runner(_jobs(1), "~/braidlab_run_g1", gpu=1)
    assert "export CUDA_VISIBLE_DEVICES=1" in script
    assert "cd ~/braidlab_run_g1" in script
    # a plain host's runner must not pin anything
    assert "CUDA_VISIBLE_DEVICES" not in render_runner(_jobs(1), "~/braidlab_run")


def test_fleet_gpu_token_workspace_and_alias() -> None:
    from braidlab.orchestrator import Fleet

    fleet = Fleet("/tmp/src")
    assert fleet._alias("vast:1") == "vast"
    assert fleet._dir("vast:1") == "~/braidlab_run_g1"
    assert fleet._alias("mother") == "mother"
    assert fleet._dir("mother") == "~/braidlab_run"


def test_scp_from_timeout_reads_as_failed_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hung transfer must not raise out of the poll loop (it killed a
    campaign mid-leg on 2026-07-09); it reads as a failed fetch and the
    caller retries on a later poll."""
    import subprocess

    from braidlab.orchestrator import Fleet

    def fake_run(cmd: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 120))  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", fake_run)
    fleet = Fleet("/tmp/src")
    assert fleet._scp_from("vast:1", "~/x.csv", tmp_path / "x.csv") is False


def test_stalled_host_queue_is_relaunched(tmp_path: Path, monkeypatch) -> None:
    """A dead runner with unfinished jobs gets its queue pushed again
    (self-heal), instead of only a warning — the 2026-07-10 deep-thought
    stall, where a previous campaign's live runner suppressed the launch and
    the host went idle with its queue never run."""
    import braidlab.orchestrator as orch
    from braidlab.store import Store

    monkeypatch.setattr(orch.time, "sleep", lambda s: None)

    jobs = _jobs(2)

    class StubFleet:
        def __init__(self) -> None:
            self.launches: list[int] = []

        def deploy(self, host: str, dims: set) -> None:
            pass

        def runner_alive(self, host: str) -> bool:
            return False  # dead from the start: launch, then stall-heal

        def launch(self, host: str, hjobs: list, dump: bool = False) -> None:
            self.launches.append(len(hjobs))

        def poll(self, host: str, dest_dir: Path) -> list:
            # Results appear only after the self-heal relaunch.
            if len(self.launches) >= 2:
                return [(j.name, 10, 1000) for j in jobs]
            return []

        def fetch_curve(self, host: str, name: str, dest: Path) -> bool:
            return True

        def fetch_dump(self, host: str, name: str, dest: Path) -> bool:
            return True

    fleet = StubFleet()
    orch.run_campaign(
        jobs,
        Store(tmp_path / "t.db"),
        fleet,  # type: ignore[arg-type]
        ["hostx"],
        poll_seconds=0,
        deploy=False,
    )
    assert len(fleet.launches) >= 2  # initial + self-heal
    assert fleet.launches[-1] == 2  # the heal re-pushed both remaining jobs
