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
    assert j.key == (3, "nyq", 44, 2, 1e-7, 2)
    assert j.maxfreq == 44


def test_terms_job_name_key_and_command() -> None:
    """Multi-term jobs get the _tmN suffix, terms in the key, and --terms."""
    from braidlab.engine import build_command

    j = Job(
        dim=3,
        band="nyq",
        t=44,
        seed=2,
        accept_rate=1e-6,
        max_attempts=3e12,
        phase=True,
        terms=10,
        tag="fqe6",
    )
    assert j.name == "d3_nyq_T44_s2_ph_tm10_fqe6"
    assert j.key == (3, "nyq", 44, 2, 1e-6, 10)
    cmd = build_command(j, "bin", "curve.csv")
    assert cmd[cmd.index("--terms") + 1] == "10"


def test_legacy_terms_omits_flag_and_suffix() -> None:
    """terms=2 keeps the legacy name and argv (bit-compatible runs)."""
    from braidlab.engine import build_command

    j = Job(dim=2, band="nyq", t=44, seed=2, accept_rate=1e-7, max_attempts=3e12)
    assert "_tm" not in j.name
    assert "--terms" not in build_command(j, "bin", "curve.csv")


def test_campaign_expands_terms_values() -> None:
    c = Campaign(
        name="t",
        dim=2,
        band="nyq",
        t_values=(20, 40),
        seeds=(1, 2),
        accept_rate=1e-6,
        terms_values=(2, 3, 10),
    )
    jobs = c.jobs()
    assert len(jobs) == 12
    assert {j.terms for j in jobs} == {2, 3, 10}


def test_campaign_rejects_bad_terms() -> None:
    with pytest.raises(ValueError):
        Campaign(
            name="t",
            dim=2,
            band="nyq",
            t_values=(20,),
            seeds=(1,),
            accept_rate=1e-6,
            terms_values=(1, 3),
        )


def test_command_is_always_torus() -> None:
    """Every job passes --torus: the torus is the engines' only geometry
    (2026-07-09); the explicit flag guards against stale wall-default binaries.
    """
    from braidlab.engine import build_command

    j = Job(dim=3, band="nyq", t=44, seed=2, accept_rate=1e-7, max_attempts=3e12)
    assert "_tor" not in j.name
    cmd = build_command(j, "bin", "curve.csv")
    assert "--torus" in cmd and "--angle-sample" not in cmd


def test_phase_job_name_and_command() -> None:
    """Phase jobs get the _ph suffix (workspace collision guard) and --phase."""
    from braidlab.engine import build_command

    j = Job(
        dim=3,
        band="nyq",
        t=44,
        seed=2,
        accept_rate=1e-6,
        max_attempts=3e12,
        phase=True,
        tag="e6",
    )
    assert j.name == "d3_nyq_T44_s2_ph_e6"
    assert "--phase" in build_command(j, "bin", "curve.csv")


def test_phase_campaign_expands_phase_jobs() -> None:
    from braidlab.campaigns import get

    c = get("torus3d_phase_e6")
    assert all(j.phase for j in c.jobs())


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


def test_command_is_always_uniform() -> None:
    """Every job passes --uniform and never --smart (smart removed 2026-07-09).

    The explicit flag guards against a stale smart-default binary on a host.
    """
    from braidlab.engine import build_command

    j = Job(
        dim=3,
        band="nyq",
        t=44,
        seed=2,
        accept_rate=1e-6,
        max_attempts=3e12,
        phase=True,
        terms=3,
        tag="une6",
    )
    assert j.name == "d3_nyq_T44_s2_ph_tm3_une6"
    cmd = build_command(j, "bin", "curve.csv")
    assert "--uniform" in cmd and "--smart" not in cmd
    plain = build_command(
        Job(dim=3, band="nyq", t=44, seed=2, accept_rate=1e-6, max_attempts=3e12),
        "bin",
        "curve.csv",
    )
    assert "--uniform" in plain and "--smart" not in plain
