"""Tests for the pure orchestration logic (no SSH)."""

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
