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
    #: Edge-weighted amplitude sampler (--angle-sample). Not part of the key.
    angle: bool = False
    #: L2-ball exclusion (--euclid-collision) instead of Chebyshev cube.
    euclid: bool = False
    #: Torus (new-dogma) model (--torus): |a*b|=1 budget on the wiggle term,
    #: free sin1 comoving offset, periodic comoving domain.
    torus: bool = False
    #: Free-form variant tag (e.g. "e6" for a different cutoff). Not part of the
    #: key; appended to the name so variant runs do not collide in the shared
    #: remote workspace.
    tag: str = ""

    @property
    def key(self) -> tuple[int, str, int, int, float]:
        """Stable identity used as the store's unique key."""
        return (self.dim, self.band, self.t, self.seed, self.accept_rate)

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
        if self.torus:
            base += "_tor"
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
    #: Use the edge-weighted amplitude sampler for every job.
    angle: bool = False
    #: Emit + collect per-worldline parameter dumps (for correlation dimension).
    dump: bool = False
    #: Use the L2-ball exclusion instead of the Chebyshev cube.
    euclid: bool = False
    #: Use the torus (new-dogma) model: |a*b|=1, free sin1, periodic domain.
    torus: bool = False
    #: Variant tag appended to job names (e.g. "e6" for a different cutoff).
    tag: str = ""

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if self.band not in BANDS:
            raise ValueError(f"band must be one of {BANDS}, got {self.band!r}")
        if self.accept_rate <= 0:
            raise ValueError("accept_rate must be > 0")

    def jobs(self) -> list[Job]:
        """Expand into the flat list of runs (T x seed)."""
        return [
            Job(
                dim=self.dim,
                band=self.band,
                t=t,
                seed=s,
                accept_rate=self.accept_rate,
                max_attempts=self.max_attempts,
                angle=self.angle,
                euclid=self.euclid,
                torus=self.torus,
                tag=self.tag,
            )
            for t in self.t_values
            for s in self.seeds
        ]
