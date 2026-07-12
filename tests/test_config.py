"""Tests for campaign/job definitions and the band rule."""

import pytest

from braidlab.config import Campaign, Job


def test_maxfreq_is_always_t() -> None:
    """maxfreq = T is hard-coded (2026-07-09): every command passes --maxfreq T
    (a guard against stale T/2-default binaries; the engines reject any other
    value), and the only accepted band label is 'nyq'.
    """
    from braidlab.engine import build_command

    j = Job(dim=3, t=44, seed=2, accept_rate=1e-7, max_attempts=3e12)
    assert j.maxfreq == 44
    cmd = build_command(j, "bin", "curve.csv")
    assert cmd[cmd.index("--maxfreq") + 1] == "44"
    with pytest.raises(ValueError):
        Job(dim=3, t=44, seed=2, accept_rate=1e-7, max_attempts=3e12, band="safe")


def test_job_identity() -> None:
    j = Job(dim=3, band="nyq", t=44, seed=2, accept_rate=1e-7, max_attempts=3e12)
    assert j.name == "d3_nyq_T44_s2_ph"
    assert j.key == (3, "nyq", 44, 2, 1e-7, 2)
    assert j.maxfreq == 44


def test_terms_job_name_key_and_command() -> None:
    """Multi-term jobs get the _tmN suffix, terms in the key, and --terms."""
    from braidlab.engine import build_command

    j = Job(
        dim=3,
        t=44,
        seed=2,
        accept_rate=1e-6,
        max_attempts=3e12,
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


def test_phase_is_always_on() -> None:
    """Every job carries the _ph suffix and passes --phase: the phase schema
    is the engines' only mode (2026-07-09). The suffix keeps names identical
    to the opt-in era's phase runs and distinct from stale phase-off remote
    files; the explicit flag guards against stale opt-in binaries.
    """
    from braidlab.engine import build_command

    j = Job(
        dim=3,
        band="nyq",
        t=44,
        seed=2,
        accept_rate=1e-6,
        max_attempts=3e12,
        tag="e6",
    )
    assert j.name == "d3_nyq_T44_s2_ph_e6"
    assert "--phase" in build_command(j, "bin", "curve.csv")


def test_phase_campaign_matches_its_opt_in_era_names() -> None:
    """torus3d_phase_e6 job names are unchanged from when phase was opt-in,
    so its store and remote workspace stay resumable."""
    from braidlab.campaigns import get

    c = get("torus3d_phase_e6")
    assert all("_ph_" in j.name for j in c.jobs())


def test_campaign_expands() -> None:
    c = Campaign(
        name="t",
        dim=2,
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
        Campaign(name="t", dim=5, t_values=(20,), seeds=(1,), accept_rate=1e-7)
    with pytest.raises(ValueError):
        Campaign(name="t", dim=2, t_values=(20,), seeds=(1,), accept_rate=0)


def test_subpaths_knob() -> None:
    """Subpaths is a 2+1-only campaign knob: _sub name suffix + --subpaths."""
    from braidlab.engine import build_command

    c = Campaign(
        name="t", dim=2, t_values=(20,), seeds=(1,), accept_rate=1e-6, subpaths=True
    )
    (j,) = c.jobs()
    assert j.name == "d2_nyq_T20_s1_ph_sub"
    argv = build_command(j, "bin", "curve.csv")
    assert "--subpaths" in argv
    # Phase 2 must carry its own stop, or it runs to the raw attempts budget
    # (found the hard way 2026-07-11: 8 h/job instead of minutes).
    assert argv[argv.index("--sub-until-accept-rate") + 1] == repr(j.accept_rate)
    # Off by default, and rejected outside 2+1.
    plain = Job(dim=2, t=20, seed=1, accept_rate=1e-6, max_attempts=3e12)
    assert "--subpaths" not in build_command(plain, "bin", "curve.csv")
    with pytest.raises(ValueError):
        Campaign(
            name="t", dim=3, t_values=(20,), seeds=(1,), accept_rate=1e-6, subpaths=True
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
