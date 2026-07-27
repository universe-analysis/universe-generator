"""Tests for the baked-frames reader, remap, and reference baker."""

import struct

import numpy as np
import pytest

from braidlab.corrdim import load_axis_terms
from braidlab.frames import (
    HIT_DTYPE,
    ObserverFrame,
    apparent_positions,
    bake_reference,
    counts_vs_redshift,
    ghost_multiplicity,
    load_frames,
)

DUMP = (
    "ax2,ay2,ax_1,bx_1,fx_1,ay_1,by_1,fy_1,ax_2,bx_2,fx_2,ay_2,by_2,fy_2\n"
    "0.31,-0.62,0.20,3,0,-0.13,2,0.8,0.10,4,1.1,0.24,5,0\n"
    "-0.55,0.12,-0.33,2,0.4,0.25,4,2.1,0.08,5,0,-0.15,3,0\n"
    "0.05,0.77,0.5,2,2.2,-0.5,2,0.3,0.0,3,0,0.0,4,0\n"
)


@pytest.fixture()
def axes(tmp_path):
    p = tmp_path / "dump.csv"
    p.write_text(DUMP)
    return load_axis_terms(p)


def test_reference_bake_basic(axes) -> None:
    """The oracle produces in-range hits for both front modes."""
    z_obs, z_min = 0.5 * np.pi, 0.02 * np.pi
    for front in ("fit", "budget"):
        hits = bake_reference(axes, z_obs, z_min, observer_path=0, front=front)
        assert hits, front
        for src, dx, dy, chi, z_emit in hits:
            assert 0 <= src < 3
            assert z_min <= z_emit <= z_obs
            assert chi >= 0
            # The image sits on the front: its offset norm equals chi.
            assert max(abs(dx), abs(dy)) == pytest.approx(chi, abs=1e-6)


def test_reference_bake_observer_point_matches_path(axes) -> None:
    """A point observer at a path's position reproduces that path's frame.

    The only permitted difference is the observer's own chi ~ 0
    self-contact, whose recording depends on the nudge direction (the two
    observer types nudge the exact-zero start to opposite sides)."""
    from braidlab.frames import _comov_xy

    z_obs, z_min = 0.5 * np.pi, 0.05 * np.pi
    ox, oy = _comov_xy(axes, 1, z_obs)
    a = bake_reference(axes, z_obs, z_min, observer_path=1)
    b = bake_reference(axes, z_obs, z_min, observer_point=(ox, oy))

    def keys(hits):
        return {
            (h[0], round(h[3], 8))
            for h in hits
            if not (h[0] == 1 and h[3] < 1e-6)  # drop the self-contact
        }

    assert keys(a) == keys(b)


def _frames_blob(hits: np.ndarray) -> bytes:
    """A minimal one-instant, one-observer BRF1 blob around `hits`."""
    head = b"BRF1" + struct.pack("<5I", 3, 1, 1, 1, 1)
    head += struct.pack("<2d", 0.33, 0.0628)
    head += struct.pack("<i", -1)
    head += struct.pack("<Ii2d", 1, -1, 0.1, 0.2)  # one point observer
    inst = struct.pack("<2d", 1.5707, 1.0) + struct.pack("<i", 3)
    obs = struct.pack("<4d", 0.1, 0.2, 0.3, 0.0) + struct.pack("<Q", len(hits))
    return head + inst + obs + hits.tobytes()


def test_load_frames_roundtrip(tmp_path) -> None:
    hits = np.zeros(2, dtype=HIT_DTYPE)
    hits["src"] = [2, 1]
    hits["dx"] = [0.5, -0.2]
    hits["dy"] = [0.1, 0.2]
    hits["chi"] = [0.5, 0.2]
    hits["z_emit"] = [1.0, 0.8]
    p = tmp_path / "t.frames"
    p.write_bytes(_frames_blob(hits))
    fs = load_frames(p)
    assert fs.n_paths == 3 and fs.front == "fit" and fs.cheb
    assert fs.observers == [(1, -1, 0.1, 0.2)]
    fr = fs.frames[0][0]
    assert fr.z_obs == pytest.approx(1.5707)
    assert fr.wraps == 3
    # Hits come back sorted by chi.
    assert list(fr.hits["src"]) == [1, 2]
    assert fr.redshift[0] == pytest.approx(np.sin(1.5707) / np.sin(0.8))
    counts, cum = counts_vs_redshift(fr, np.array([1.0, 1.2, 3.0]))
    assert cum[-1] == 2
    assert list(ghost_multiplicity(fr)) == [0, 1, 1]


def test_apparent_positions_static_and_moving() -> None:
    hits = np.zeros(1, dtype=HIT_DTYPE)
    hits["dx"], hits["dy"], hits["chi"] = 0.3, 0.1, 0.3
    frame = ObserverFrame(
        z_obs=1.5,
        chi_max=1.0,
        wraps=1,
        ox=0.0,
        oy=0.0,
        beta=np.array([0.4, 0.0]),
        hits=hits,
    )
    sx, sy, sd = apparent_positions(frame, "static")
    assert sx[0] == pytest.approx(0.3) and sd[0] == 0.0
    ax, ay, ad = apparent_positions(frame, "moving", cheb=True)
    # Aberration bends the apparent direction toward the velocity and the
    # square-metric re-projection keeps the image on its chi shell.
    assert max(abs(ax[0]), abs(ay[0])) == pytest.approx(0.3, abs=1e-6)
    assert abs(ay[0] / ax[0]) < abs(0.1 / 0.3)
    assert ad[0] > 0.0  # approaching-side Doppler factor > 1
