"""Predefined campaigns — the corrected measurement.

These encode the decisions reached during the RSA analysis:
  * torus model, uniform sampler, symmetric z grid, phase schema (the
    engines' only mode since the 2026-07-09 cleanup; the wall-era campaigns
    that predate it — 2plus1/3plus1, corrdim3d, corrdim3d_euclid/_e6,
    corrdim2d — were removed with it and live in git history; their
    stores/dumps remain on disk),
  * full Nyquist band (``maxfreq = T`` — hard-coded in the engines since
    2026-07-09; no longer a campaign knob),
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

#: Correlation-dimension grids (3+1): fine T ladder, a few seeds, param dumps.
CORRDIM_SEEDS = tuple(range(1, 6))  # 5 seeds for an error bar
CORRDIM_T = tuple(range(20, 201, 10))  # 20, 30, ..., 200

#: Correlation-dimension grids (2+1): the 2D grid scales ~T^2, so it is cheap
#: -- push T high (long scaling window) with many seeds for a thorough
#: box-counting vs correlation comparison.
CORRDIM2D_SEEDS = tuple(range(1, 9))  # 8 seeds
CORRDIM2D_T = tuple(range(40, 401, 20))  # 40, 60, ..., 400

#: FREQ campaign (terms sweep on the torus+phase model). Five term counts
#: (2 = legacy single wiggle, up to 10), a coarser T grid than the corrdim
#: ladders (5 term counts multiply the job count), and the 1e-6 cutoff so the
#: terms=2 arm lines up with torus3d_phase_e6 / torus2d_phase_e6 for a direct
#: cross-check. Term count is capped by the frequency pool (terms-1 unique
#: frequencies per axis); the smallest T here has pool T-1 = 19 >> 9, so every
#: cell is valid. 3+1 stops at T=160: two-3080 fleet, and the multi-term jam
#: count runs higher than legacy so T=200's dense grid + points is too tight.
FREQ_TERMS = (2, 3, 4, 6, 10)
FREQ_SEEDS = tuple(range(1, 6))  # 5 seeds
FREQ3D_T = tuple(range(20, 161, 20))  # 20, 40, ..., 160
FREQ2D_T = tuple(range(40, 401, 40))  # 40, 80, ..., 400

#: FREQ decay study: two T values x terms {2, 10} at cutoffs 1e-6/1e-7/1e-8,
#: for the approach-to-jamming power law with recent data. One campaign per
#: cutoff (a Campaign carries a single accept_rate); distinct tags keep the
#: three variants apart in the shared remote workspace.
FREQDECAY_SEEDS = (1, 2, 3)
FREQDECAY3D_T = (80, 160)
FREQDECAY2D_T = (120, 320)
FREQDECAY_TERMS = (2, 10)

#: PACK campaigns (2026-07-09): is the packing number N(T) invariant under the
#: model knobs? Baseline legacy 2-term arms on an 8-value T ladder per
#: dimension at the 1e-6 cutoff, the same ladders again at 1e-7 (decay-rate
#: dependence), and a term-count sweep (FREQ_TERMS) at two T per dimension
#: (term dependence). Subpaths stay off throughout. All run under the
#: always-on phase schema and the hard-coded maxfreq = T.
PACK_SEEDS = tuple(range(1, 6))  # 5 seeds
PACK3D_T = tuple(range(20, 161, 20))  # 20, 40, ..., 160
PACK2D_T = tuple(range(20, 301, 40))  # 20, 60, ..., 300
PACKTERMS3D_T = (80, 160)
PACKTERMS2D_T = (140, 300)

#: CONVERGE campaigns (2026-07-10): pin both headline exponents. 3+1 extends
#: the T ladder past the dense-grid VRAM ceiling with the sparse engine (does
#: the local slope converge on D/d = 3/4, i.e. D = 2.25?); 2+1 instead
#: extends in cutoff depth for a jamming-limit extrapolation of its
#: cutoff-drifting exponent (the 1e-6/1e-7 rungs are the PACK stores, same
#: ladder). T = 400 is deliberately held back from the 3+1 ladder; to add it
#: later, append it here and re-run -- the store diff dispatches only the
#: new cells.
CONVERGE_SEEDS = tuple(range(1, 6))  # 5 seeds
CONVERGE3D_T = (200, 240, 280, 320, 360)
CONVERGE2D_T = PACK2D_T  # 20, 60, ..., 300 -- matches pack2d_e6/e7
CONVERGE2D_E9_T = (140, 300)  # far anchor for the depth extrapolation


def _pack(dim: int, rate: float, tag: str) -> Campaign:
    """One cutoff arm of the PACK baseline (2-term, 8-T ladder)."""
    return Campaign(
        name=f"pack{dim}d_{tag}",
        dim=dim,
        t_values=PACK3D_T if dim == 3 else PACK2D_T,
        seeds=PACK_SEEDS,
        accept_rate=rate,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        tag=f"pk{tag}",
    )


def _pack_terms(dim: int) -> Campaign:
    """The PACK term sweep: FREQ_TERMS at two T, 1e-6 cutoff."""
    return Campaign(
        name=f"packterms{dim}d_e6",
        dim=dim,
        t_values=PACKTERMS3D_T if dim == 3 else PACKTERMS2D_T,
        seeds=PACK_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        terms_values=FREQ_TERMS,
        tag="pktme6",
    )


def _freq_decay(dim: int, rate: float, tag: str) -> Campaign:
    """One cutoff arm of the FREQ decay study."""
    return Campaign(
        name=f"freqdecay{dim}d_{tag}",
        dim=dim,
        t_values=FREQDECAY3D_T if dim == 3 else FREQDECAY2D_T,
        seeds=FREQDECAY_SEEDS,
        accept_rate=rate,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        terms_values=FREQDECAY_TERMS,
        tag=f"fqd{tag}",
    )


CAMPAIGNS: dict[str, Campaign] = {
    # Torus (new-dogma) model, 3+1: |a*b|=1 budget on the wiggle term, free sin1
    # comoving offset, periodic comoving domain (wrap + minimum image, no wall).
    # Uniform sin1 sampler -- on the torus uniform IS the homogeneous measure;
    # the edge-weighted sampler existed to fight the hard wall, which is gone.
    "torus3d": Campaign(
        name="torus3d",
        dim=3,
        t_values=CORRDIM_T,
        seeds=CORRDIM_SEEDS,
        accept_rate=ACCEPT_RATE,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
    ),
    # Torus model at the 1e-6 cutoff: ~10x cheaper in the rejection-dominated
    # tail, and the cutoff study (PHYSICS_FINDINGS section 12) showed D moves
    # < 0.02 per cutoff decade. Pairs exactly with corrdim3d_e6 for a
    # same-cutoff old-vs-new comparison. Tagged so it cannot collide with a
    # 1e-7 torus run in the shared remote workspace.
    "torus3d_e6": Campaign(
        name="torus3d_e6",
        dim=3,
        t_values=CORRDIM_T,
        seeds=CORRDIM_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        tag="e6",
    ),
    # Phase schema on the torus model (Chris's viewer schema: even-frequency
    # phases on the wiggle term). The phase schema is always on since
    # 2026-07-09, so these are now IDENTICAL to torus3d_e6 / torus2d_e6 —
    # kept as separate entries so their stores in data/ stay resumable.
    # When they ran, phase was opt-in (--phase, which then also selected the
    # symmetric z grid) and torus*_e6 were the phase-free control arms.
    "torus3d_phase_e6": Campaign(
        name="torus3d_phase_e6",
        dim=3,
        t_values=CORRDIM_T,
        seeds=CORRDIM_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        tag="e6",
    ),
    "torus2d_phase_e6": Campaign(
        name="torus2d_phase_e6",
        dim=2,
        t_values=CORRDIM2D_T,
        seeds=CORRDIM2D_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        tag="e6",
    ),
    "torus2d_e6": Campaign(
        name="torus2d_e6",
        dim=2,
        t_values=CORRDIM2D_T,
        seeds=CORRDIM2D_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        tag="e6",
    ),
    # Torus model, 2+1 (same long-T grid as corrdim2d -- the 2D grid is cheap).
    "torus2d": Campaign(
        name="torus2d",
        dim=2,
        t_values=CORRDIM2D_T,
        seeds=CORRDIM2D_SEEDS,
        accept_rate=ACCEPT_RATE,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
    ),
    # FREQ campaign: term-count sweep on the torus+phase model (see the
    # FREQ_* constants above for the design rationale).
    "freq3d_e6": Campaign(
        name="freq3d_e6",
        dim=3,
        t_values=FREQ3D_T,
        seeds=FREQ_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        terms_values=FREQ_TERMS,
        tag="fqe6",
    ),
    "freq2d_e6": Campaign(
        name="freq2d_e6",
        dim=2,
        t_values=FREQ2D_T,
        seeds=FREQ_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        terms_values=FREQ_TERMS,
        tag="fqe6",
    ),
    # Uniform-sampler mini-sweep (ran 2026-07-08 as the control against the
    # then-default smart sampler; its verdict -- D is terms-invariant under
    # uniform proposals -- is why the smart sampler was removed 2026-07-09).
    # Kept so its store in data/freq/ stays resumable; the engine is
    # uniform-only now, so the definition no longer needs a sampler knob.
    "frequni3d_e6": Campaign(
        name="frequni3d_e6",
        dim=3,
        t_values=FREQ3D_T,
        seeds=(1, 2, 3),
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        terms_values=(2, 3, 10),
        tag="une6",
    ),
    "freqdecay3d_e6": _freq_decay(3, 1e-6, "e6"),
    "freqdecay3d_e7": _freq_decay(3, 1e-7, "e7"),
    "freqdecay3d_e8": _freq_decay(3, 1e-8, "e8"),
    "freqdecay2d_e6": _freq_decay(2, 1e-6, "e6"),
    "freqdecay2d_e7": _freq_decay(2, 1e-7, "e7"),
    "freqdecay2d_e8": _freq_decay(2, 1e-8, "e8"),
    # PACK: packing-number invariance (see the PACK_* constants above).
    "pack3d_e6": _pack(3, 1e-6, "e6"),
    "pack2d_e6": _pack(2, 1e-6, "e6"),
    "pack3d_e7": _pack(3, 1e-7, "e7"),
    "pack2d_e7": _pack(2, 1e-7, "e7"),
    "packterms3d_e6": _pack_terms(3),
    "packterms2d_e6": _pack_terms(2),
    # SUBPATH: post-jam filling by paths that join existing groups (engine
    # --subpaths, 2+1 only). Mirrors pack2d_e6 exactly (same ladder, seeds,
    # cutoff) so the subpath capacity of a jam reads off as an A/B against
    # that store: how much extra threading does a jammed universe retain?
    "subpath2d_e6": Campaign(
        name="subpath2d_e6",
        dim=2,
        t_values=PACK2D_T,
        seeds=PACK_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        subpaths=True,
        tag="sub2e6",
    ),
    # CONVERGE: exponent convergence (see the CONVERGE_* constants above).
    "converge3d_e6": Campaign(
        name="converge3d_e6",
        dim=3,
        t_values=CONVERGE3D_T,
        seeds=CONVERGE_SEEDS,
        accept_rate=1e-6,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        sparse=True,
        tag="cv3e6",
    ),
    # The 7/3 chase: the converged window again at 1e-7 (stitch with
    # pack3d_e7 for T <= 160). If the exponent's cutoff shift carries the
    # 2.321 plateau to ~2.333, the jamming-limit exponent is plausibly
    # exactly 7/3.
    "converge3d_e7": Campaign(
        name="converge3d_e7",
        dim=3,
        t_values=CONVERGE3D_T,
        seeds=CONVERGE_SEEDS,
        accept_rate=1e-7,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        sparse=True,
        tag="cv3e7",
    ),
    "converge2d_e8": Campaign(
        name="converge2d_e8",
        dim=2,
        t_values=CONVERGE2D_T,
        seeds=CONVERGE_SEEDS,
        accept_rate=1e-8,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        tag="cv2e8",
    ),
    # Deep small-T 3+1 curves: the Feder p-vote electorate for the 3+1
    # jamming-limit extrapolation (all 3+1 curves through 1e-8 rail).
    "converge3d_e9small": Campaign(
        name="converge3d_e9small",
        dim=3,
        t_values=(20, 40, 60),
        seeds=(1, 2, 3),
        accept_rate=1e-9,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        tag="cv3e9s",
    ),
    "converge2d_e9": Campaign(
        name="converge2d_e9",
        dim=2,
        t_values=CONVERGE2D_E9_T,
        seeds=(1, 2, 3),
        accept_rate=1e-9,
        max_attempts=MAX_ATTEMPTS,
        dump=True,
        tag="cv2e9",
    ),
}


def get(name: str) -> Campaign:
    """Look up a predefined campaign by name."""
    if name not in CAMPAIGNS:
        raise KeyError(f"unknown campaign {name!r}; have {sorted(CAMPAIGNS)}")
    return CAMPAIGNS[name]
