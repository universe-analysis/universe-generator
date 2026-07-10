"""Campaign and job definitions, plus the frequency-band rule.

A `Campaign` is a declarative description of an experiment: which spatial
dimension, frequency band, timestep ladder, seeds, and stop threshold. It
expands into a flat list of `Job`s -- one packing run each.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Frequency-band rules. Each maps to an actual max frequency at a given T.
BANDS = ("default", "safe", "nyq")


def maxfreq_for(band: str, t: int) -> int:
    """Return the actual max frequency for a band rule at timestep count `t`.

    Returns 0 for the engine's built-in default (T/2), which is passed through
    by omitting the --maxfreq flag.

    Args:
        band: One of "default" (T/2), "safe" (round 0.85*T), "nyq" (T).
        t: Number of timesteps.

    Raises:
        ValueError: If `band` is not a known rule.
    """
    if band == "default":
        return 0
    if band == "safe":
        return round(0.85 * t)
    if band == "nyq":
        return t
    raise ValueError(f"unknown band {band!r}; expected one of {BANDS}")


@dataclass(frozen=True)
class Job:
    """A single packing run: one (dim, band, T, seed) at a stop threshold."""

    dim: int
    band: str
    t: int
    seed: int
    accept_rate: float
    max_attempts: float
    #: L2-ball exclusion (--euclid-collision) instead of Chebyshev cube.
    euclid: bool = False
    #: Sparse collision grid (--sparse, 3+1 engine): sorted-key lookup +
    #: float32 points; VRAM ~ N*T instead of ~T^4, for T beyond the dense cap.
    sparse: bool = False
    #: Sinusoid terms per axis incl. sin1 (--terms). 2 = the legacy
    #: single-wiggle model (flag omitted, RNG stream bit-identical).
    terms: int = 2
    #: Free-form variant tag (e.g. "e6" for a different cutoff). Not part of the
    #: key; appended to the name so variant runs do not collide in the shared
    #: remote workspace.
    tag: str = ""

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
        if self.tag:
            base += f"_{self.tag}"
        return base

    @property
    def maxfreq(self) -> int:
        """Actual max frequency for this job's band (0 = engine default)."""
        return maxfreq_for(self.band, self.t)


@dataclass(frozen=True)
class Campaign:
    """A declarative experiment that expands into many `Job`s."""

    name: str
    dim: int
    band: str
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
    #: Variant tag appended to job names (e.g. "e6" for a different cutoff).
    tag: str = ""

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if self.band not in BANDS:
            raise ValueError(f"band must be one of {BANDS}, got {self.band!r}")
        if self.accept_rate <= 0:
            raise ValueError("accept_rate must be > 0")
        if any(k < 2 for k in self.terms_values):
            raise ValueError(f"terms_values must all be >= 2, got {self.terms_values}")

    def jobs(self) -> list[Job]:
        """Expand into the flat list of runs (T x seed x terms)."""
        return [
            Job(
                dim=self.dim,
                band=self.band,
                t=t,
                seed=s,
                accept_rate=self.accept_rate,
                max_attempts=self.max_attempts,
                euclid=self.euclid,
                sparse=self.sparse,
                terms=k,
                tag=self.tag,
            )
            for t in self.t_values
            for s in self.seeds
            for k in self.terms_values
        ]
