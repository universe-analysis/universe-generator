"""Tests for campaign/job definitions and the band rule."""

import pytest

from braidlab.config import Campaign, Job, maxfreq_for


def test_maxfreq_rules() -> None:
    assert maxfreq_for("default", 40) == 0
    assert maxfreq_for("nyq", 40) == 40
    assert maxfreq_for("safe", 40) == 34  # round(0.85 * 40)
    assert maxfreq_for("safe", 52) == 44  # round(44.2)


def test_maxfreq_unknown() -> None:
    with pytest.raises(ValueError):
        maxfreq_for("bogus", 40)


def test_job_identity() -> None:
    j = Job(dim=3, band="nyq", t=44, seed=2, accept_rate=1e-7, max_attempts=3e12)
    assert j.name == "d3_nyq_T44_s2"
    assert j.key == (3, "nyq", 44, 2, 1e-7)
    assert j.maxfreq == 44


def test_torus_job_name_and_command() -> None:
    """Torus jobs get the _tor suffix (workspace collision guard) and --torus."""
    from braidlab.engine import build_command

    j = Job(
        dim=3,
        band="nyq",
        t=44,
        seed=2,
        accept_rate=1e-7,
        max_attempts=3e12,
        torus=True,
    )
    assert j.name == "d3_nyq_T44_s2_tor"
    assert "--torus" in build_command(j, "bin", "curve.csv")


def test_torus_campaign_expands_torus_jobs() -> None:
    c = Campaign(
        name="t",
        dim=3,
        band="nyq",
        t_values=(20,),
        seeds=(1,),
        accept_rate=1e-7,
        torus=True,
    )
    assert all(j.torus for j in c.jobs())


def test_campaign_expands() -> None:
    c = Campaign(
        name="t",
        dim=2,
        band="safe",
        t_values=(20, 40),
        seeds=(1, 2, 3),
        accept_rate=1e-7,
    )
    jobs = c.jobs()
    assert len(jobs) == 6
    assert {j.t for j in jobs} == {20, 40}
    assert {j.seed for j in jobs} == {1, 2, 3}


def test_campaign_validation() -> None:
    with pytest.raises(ValueError):
        Campaign(
            name="t", dim=5, band="nyq", t_values=(20,), seeds=(1,), accept_rate=1e-7
        )
    with pytest.raises(ValueError):
        Campaign(
            name="t", dim=2, band="x", t_values=(20,), seeds=(1,), accept_rate=1e-7
        )
