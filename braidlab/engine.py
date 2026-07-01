"""Build engine commands and parse their output.

Wraps the two CUDA binaries (`braid_cuda` for 2+1, `braid_cuda3d` for 3+1).
Both share the same flags, including `--until-accept-rate` (the fixed-
convergence stop) and `--maxfreq` (the band rule).
"""

from __future__ import annotations

import re

from braidlab.config import Job

_DONE_RE = re.compile(r"done:\s*N=(\d+)\s+in\s+(\d+)\s+attempts")


def binary_name(dim: int) -> str:
    """Engine binary for a spatial dimension."""
    return "braid_cuda" if dim == 2 else "braid_cuda3d"


def build_command(job: Job, binary: str, curve_path: str) -> list[str]:
    """Construct the argv for a packing run.

    Args:
        job: The run to execute.
        binary: Path to the engine binary on the target host.
        curve_path: Where the engine should write its kinetic curve CSV.

    The dump flag is *not* added here: render_runner appends `--dump-params`
    with a host-relative path of its own, so it stays out of the shared argv.
    """
    cmd = [
        binary,
        "-t",
        str(job.t),
        "--attempts",
        repr(job.max_attempts),
        "--smart",
        "--seed",
        str(job.seed),
        "--until-accept-rate",
        repr(job.accept_rate),
    ]
    if job.angle:
        cmd += ["--angle-sample"]
    if job.euclid:
        cmd += ["--euclid-collision"]
    if job.maxfreq:
        cmd += ["--maxfreq", str(job.maxfreq)]
    # --curve stays last: render_runner truncates the argv here and supplies its own
    cmd += ["--curve", curve_path]
    return cmd


def parse_done(text: str) -> tuple[int, int] | None:
    """Parse ``done: N=<n> in <attempts> attempts`` from engine stderr.

    Returns (n_final, attempts), or None if no done line is present.
    """
    m = _DONE_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))
