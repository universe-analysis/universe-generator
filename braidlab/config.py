"""Campaign and job definitions.

A `Campaign` is a declarative description of an experiment: which spatial
dimension, timestep ladder, seeds, stop threshold (decay cutoff), term count,
and subpath mode. It expands into a flat list of `Job`s -- one packing run
each. The frequency band is not a knob: the full Nyquist band (max frequency
= T) is hard-coded in the engines since 2026-07-09.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    """A single packing run: one (dim, T, seed) at a stop threshold."""

    dim: int
    t: int
    seed: int
    accept_rate: float
    max_attempts: float
    #: Frequency-band label. The full Nyquist band (max frequency = T) is the
    #: model's only rule since 2026-07-09 -- the engines reject any other
    #: --maxfreq value -- so this is a fixed identity column kept for store
    #: and job-name compatibility, not a knob.
    band: str = "nyq"
    #: L2-ball exclusion (--euclid-collision) instead of Chebyshev cube.
    euclid: bool = False
    #: Sparse collision grid (--sparse, 3+1 engine): sorted-key lookup +
    #: float32 points; VRAM ~ N*T instead of ~T^4, for T beyond the dense cap.
    sparse: bool = False
    #: Sinusoid terms per axis incl. sin1 (--terms). 2 = the legacy
    #: single-wiggle model (flag omitted, RNG stream bit-identical).
    terms: int = 2
    #: Second-phase subpath packing (--subpaths, 2+1 engine only): after the
    #: unique packing jams, pack candidates that touch exactly one existing
    #: group. Off by default.
    subpaths: bool = False
    #: Fixed phase-2 attempt budget (--sub-attempts). 0 = rate-stop only.
    #: Needed where the subpath admission rate never decays below the cutoff
    #: (small T): subpaths do not jam, so a rate stop cannot terminate there.
    sub_attempts: float = 0
    #: Free-form variant tag (e.g. "e6" for a different cutoff). Not part of the
    #: key; appended to the name so variant runs do not collide in the shared
    #: remote workspace.
    tag: str = ""

    def __post_init__(self) -> None:
        if self.band != "nyq":
            raise ValueError(
                f"band must be 'nyq' (maxfreq = T is hard-coded since "
                f"2026-07-09), got {self.band!r}"
            )
        if self.subpaths and self.dim != 2:
            raise ValueError("subpaths is a 2+1 engine feature (dim must be 2)")

    @property
    def key(self) -> tuple[int, str, int, int, float, int]:
        """Stable identity used as the store's unique key."""
        return (self.dim, self.band, self.t, self.seed, self.accept_rate, self.terms)

    @property
    def name(self) -> str:
        """Short filesystem-safe label, e.g. ``d3_safe_T44_s2``.

        Variant suffixes (``_eu`` for euclid, ``_<tag>`` for anything else) keep a
        variant run from colliding with a baseline run of the same
        (dim, band, T, seed) in a shared remote workspace -- otherwise the
        runner's idempotency check skips it and re-collects the baseline files.
        """
        base = f"d{self.dim}_{self.band}_T{self.t}_s{self.seed}"
        if self.euclid:
            base += "_eu"
        # The phase schema is always on (2026-07-09), so every job carries the
        # _ph suffix. This keeps names identical to the opt-in era's phase runs
        # (in-flight campaigns resume cleanly) and distinct from any stale
        # phase-OFF leftovers of the same (dim, band, T, seed) in a shared
        # remote workspace.
        base += "_ph"
        if self.sparse:
            base += "_sp"
        if self.terms != 2:
            base += f"_tm{self.terms}"
        if self.subpaths:
            base += "_sub"
        if self.tag:
            base += f"_{self.tag}"
        return base

    @property
    def maxfreq(self) -> int:
        """Actual max frequency: always T (the full Nyquist band)."""
        return self.t


@dataclass(frozen=True)
class Campaign:
    """A declarative experiment that expands into many `Job`s."""

    name: str
    dim: int
    t_values: tuple[int, ...]
    seeds: tuple[int, ...]
    accept_rate: float
    max_attempts: float = 3e12
    #: Emit + collect per-worldline parameter dumps (for correlation dimension).
    dump: bool = False
    #: Use the L2-ball exclusion instead of the Chebyshev cube.
    euclid: bool = False
    #: Use the sparse collision grid (3+1): VRAM ~ N*T, for T beyond dense cap.
    sparse: bool = False
    #: Sinusoid term counts per axis to sweep (--terms; 2 = legacy model).
    terms_values: tuple[int, ...] = (2,)
    #: Second-phase subpath packing (2+1 only). Off by default.
    subpaths: bool = False
    #: Fixed phase-2 attempt budget (0 = rate-stop only); see Job.sub_attempts.
    sub_attempts: float = 0
    #: Variant tag appended to job names (e.g. "e6" for a different cutoff).
    tag: str = ""

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if self.accept_rate <= 0:
            raise ValueError("accept_rate must be > 0")
        if any(k < 2 for k in self.terms_values):
            raise ValueError(f"terms_values must all be >= 2, got {self.terms_values}")
        if self.subpaths and self.dim != 2:
            raise ValueError("subpaths is a 2+1 engine feature (dim must be 2)")

    def jobs(self) -> list[Job]:
        """Expand into the flat list of runs (T x seed x terms)."""
        return [
            Job(
                dim=self.dim,
                t=t,
                seed=s,
                accept_rate=self.accept_rate,
                max_attempts=self.max_attempts,
                euclid=self.euclid,
                sparse=self.sparse,
                terms=k,
                subpaths=self.subpaths,
                sub_attempts=self.sub_attempts,
                tag=self.tag,
            )
            for t in self.t_values
            for s in self.seeds
            for k in self.terms_values
        ]
