"""braidlab — RSA jamming measurement suite for the braided-universe model.

Measures the large-scale packing exponent D (and the kinetic cost law) for
parametric sine-wave worldlines packed under a per-timestep collision rule.

Protocol (the reason this package exists):
  * Hold the *convergence level* constant across timesteps T by stopping each
    run at a fixed acceptance-rate threshold (not a fixed attempt budget). Then
    N(T) = theta * C * T**D, so the log-log slope is D for any theta -- bias from
    incomplete jamming cancels in the exponent.
  * Average many seeds per T to drive down variance.
  * Use the full Nyquist frequency band (maxfreq = T; hard-coded in the
    engines since 2026-07-09).
"""

from braidlab.config import Campaign, Job

__all__ = ["Campaign", "Job"]
__version__ = "0.1.0"
