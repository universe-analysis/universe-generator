"""Predefined campaigns — the corrected measurement.

These encode the decisions reached during the RSA analysis:
  * edge-detection engine (committed separately),
  * full Nyquist band (``maxfreq = T``, i.e. band="nyq"),
  * fixed-convergence stop (constant acceptance rate across T),
  * many seeds per T for a bootstrap error bar.

Adjust the ladders / seed counts here; everything downstream reads from them.
"""

from __future__ import annotations

from braidlab.config import Campaign

#: Stop each run when accepts/attempts falls below this (same for all T).
ACCEPT_RATE = 1e-7
#: Seeds per timestep (variance ~ 1/sqrt(N_seeds)).
SEEDS = tuple(range(1, 17))
#: Safety cap; the acceptance-rate stop should trigger first.
MAX_ATTEMPTS = 3e12

#: Correlation-dimension campaign (3+1): fine T grid, a few seeds, param dumps.
CORRDIM_SEEDS = tuple(range(1, 6))  # 5 seeds for an error bar
CORRDIM_T = tuple(range(20, 201, 10))  # 20, 30, ..., 200

#: Correlation-dimension campaign (2+1): the 2D grid scales ~T^2, so it is cheap
#: -- push T high (long scaling window) with many seeds for a thorough
#: box-counting vs correlation comparison. The 2D engine has no --angle-sample,
#: so this uses the default amplitude sampler (D is sampler-invariant).
CORRDIM2D_SEEDS = tuple(range(1, 9))  # 8 seeds
CORRDIM2D_T = tuple(range(40, 401, 20))  # 40, 60, ..., 400

CAMPAIGNS: dict[str, Campaign] = {
    "2plus1": Campaign(
        name="2plus1",
        dim=2,
        band="nyq",
        t_values=(24, 34, 48, 68, 96),
        seeds=SEEDS,
        accept_rate=ACCEPT_RATE,
        max_attempts=MAX_ATTEMPTS,
    ),
    "3plus1": Campaign(
        name="3plus1",
        dim=3,
        band="nyq",
        t_values=(18, 24, 32, 44, 60),
        seeds=SEEDS,
        accept_rate=ACCEPT_RATE,
        max_attempts=MAX_ATTEMPTS,
    ),
    # Edge sampler + parameter dumps, for the turnaround correlation dimension.
    "corrdim3d": Campaign(
        name="corrdim3d",
        dim=3,
        band="nyq",
        t_values=CORRDIM_T,
        seeds=CORRDIM_SEEDS,
        accept_rate=ACCEPT_RATE,
        max_attempts=MAX_ATTEMPTS,
        angle=True,
        dump=True,
    ),
    # 3+1 packing GENERATED with the L2-ball (sphere) collision instead of the
    # Chebyshev cube -- a different universe, not a different measurement. Full T
    # grid so box-counting and correlation on the sphere-collision packing can be
    # put apples-to-apples beside the cube-collision corrdim3d.
    "corrdim3d_euclid": Campaign(
        name="corrdim3d_euclid",
        dim=3,
        band="nyq",
        t_values=CORRDIM_T,
        seeds=CORRDIM_SEEDS,
        accept_rate=ACCEPT_RATE,
        max_attempts=MAX_ATTEMPTS,
        angle=True,
        dump=True,
        euclid=True,
    ),
    # Cutoff-depth check: identical to corrdim3d but stopping at 1e-6 instead of
    # 1e-7, to see how much the shallower cutoff moves D (does 1e-8 matter?).
    "corrdim3d_e6": Campaign(
        name="corrdim3d_e6",
        dim=3,
        band="nyq",
        t_values=CORRDIM_T,
        seeds=CORRDIM_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        angle=True,
        dump=True,
        tag="e6",
    ),
    # Torus (new-dogma) model, 3+1: |a*b|=1 budget on the wiggle term, free sin1
    # comoving offset, periodic comoving domain (wrap + minimum image, no wall).
    # Uniform sin1 sampler -- on the torus uniform IS the homogeneous measure;
    # the edge-weighted sampler existed to fight the hard wall, which is gone.
    "torus3d": Campaign(
        name="torus3d",
        dim=3,
        band="nyq",
        t_values=CORRDIM_T,
        seeds=CORRDIM_SEEDS,
        accept_rate=ACCEPT_RATE,
        max_attempts=MAX_ATTEMPTS,
        angle=False,
        dump=True,
        torus=True,
    ),
    # Torus model at the 1e-6 cutoff: ~10x cheaper in the rejection-dominated
    # tail, and the cutoff study (PHYSICS_FINDINGS section 12) showed D moves
    # < 0.02 per cutoff decade. Pairs exactly with corrdim3d_e6 for a
    # same-cutoff old-vs-new comparison. Tagged so it cannot collide with a
    # 1e-7 torus run in the shared remote workspace.
    "torus3d_e6": Campaign(
        name="torus3d_e6",
        dim=3,
        band="nyq",
        t_values=CORRDIM_T,
        seeds=CORRDIM_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        angle=False,
        dump=True,
        torus=True,
        tag="e6",
    ),
    # Phase schema on the torus model: even-frequency phases on the wiggle term
    # plus the symmetric z grid (Chris's viewer schema, engine flag --phase).
    # Same T grid / seeds / 1e-6 cutoff as torus3d_e6, so the with/without-phase
    # comparison is one-knob. The _ph name suffix keeps remote files distinct.
    "torus3d_phase_e6": Campaign(
        name="torus3d_phase_e6",
        dim=3,
        band="nyq",
        t_values=CORRDIM_T,
        seeds=CORRDIM_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        angle=False,
        dump=True,
        torus=True,
        phase=True,
        tag="e6",
    ),
    "torus2d_phase_e6": Campaign(
        name="torus2d_phase_e6",
        dim=2,
        band="nyq",
        t_values=CORRDIM2D_T,
        seeds=CORRDIM2D_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        angle=False,
        dump=True,
        torus=True,
        phase=True,
        tag="e6",
    ),
    "torus2d_e6": Campaign(
        name="torus2d_e6",
        dim=2,
        band="nyq",
        t_values=CORRDIM2D_T,
        seeds=CORRDIM2D_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        angle=False,
        dump=True,
        torus=True,
        tag="e6",
    ),
    # Torus model, 2+1 (same long-T grid as corrdim2d -- the 2D grid is cheap).
    "torus2d": Campaign(
        name="torus2d",
        dim=2,
        band="nyq",
        t_values=CORRDIM2D_T,
        seeds=CORRDIM2D_SEEDS,
        accept_rate=ACCEPT_RATE,
        max_attempts=MAX_ATTEMPTS,
        angle=False,
        dump=True,
        torus=True,
    ),
    # Deep 2+1 box-counting vs correlation-dimension study (no edge sampler).
    "corrdim2d": Campaign(
        name="corrdim2d",
        dim=2,
        band="nyq",
        t_values=CORRDIM2D_T,
        seeds=CORRDIM2D_SEEDS,
        accept_rate=ACCEPT_RATE,
        max_attempts=MAX_ATTEMPTS,
        angle=False,
        dump=True,
    ),
}


def get(name: str) -> Campaign:
    """Look up a predefined campaign by name."""
    if name not in CAMPAIGNS:
        raise KeyError(f"unknown campaign {name!r}; have {sorted(CAMPAIGNS)}")
    return CAMPAIGNS[name]
