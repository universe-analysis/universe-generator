"""Build engine commands and parse their output.

Wraps the two CUDA binaries (`braid_cuda` for 2+1, `braid_cuda3d` for 3+1).
Both share the same flags, including `--until-accept-rate` (the fixed-
convergence stop) and `--maxfreq` (always T -- validated by the engines).
"""

from __future__ import annotations

import re

from braidlab.config import Job

_DONE_RE = re.compile(r"done:\s*N=(\d+)\s+in\s+(\d+)\s+attempts")


#: Wiggle-term cap compiled into the default engine binaries (kMaxWiggle).
DEFAULT_MAX_WIGGLE = 9
#: Cap compiled into the wide (full-spectrum) variant -- terms up to 150.
WIDE_MAX_WIGGLE = 149
#: Filename suffix of the wide variant binaries (braid_cuda_w150, ...).
WIDE_SUFFIX = "_w150"


def binary_name(dim: int, terms: int = 2) -> str:
    """Engine binary for a spatial dimension.

    Jobs whose term count exceeds the default binaries' compiled Path cap
    (kMaxWiggle = 9) run the wide variant, built from the same source with
    ``-DKMAX_WIGGLE=149``.
    """
    base = "braid_cuda" if dim == 2 else "braid_cuda3d"
    if terms - 1 > DEFAULT_MAX_WIGGLE:
        return base + WIDE_SUFFIX
    return base


def binary_source(binary: str) -> str:
    """The .cu source file a binary (default or wide variant) is built from."""
    return binary.removesuffix(WIDE_SUFFIX) + ".cu"


def binary_build_flags(binary: str) -> str:
    """Extra nvcc flags for a binary variant (empty for the defaults)."""
    if binary.endswith(WIDE_SUFFIX):
        return f"-DKMAX_WIGGLE={WIDE_MAX_WIGGLE}"
    return ""


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
        # Uniform is the engine's only sampler, the torus its only geometry,
        # and the phase schema always on (all since 2026-07-09); passed
        # explicitly so a stale binary with the old defaults (smart sampler,
        # hard wall, opt-in phase) fails loudly or runs the right model
        # instead of silently running the wrong one.
        "--uniform",
        "--torus",
        "--phase",
        "--seed",
        str(job.seed),
        "--until-accept-rate",
        repr(job.accept_rate),
    ]
    if job.euclid:
        cmd += ["--euclid-collision"]
    if job.sparse:
        cmd += ["--sparse"]
    if job.terms != 2:
        cmd += ["--terms", str(job.terms)]
    if job.subpaths:
        # Phase 2 has its own windowed stop; without this it runs to the raw
        # --attempts budget (~3e12, hours per job). Same convergence
        # criterion as phase 1.
        cmd += ["--subpaths", "--sub-until-accept-rate", repr(job.accept_rate)]
        if job.sub_attempts:
            # Small-T subpath admission never decays below the cutoff
            # (subpaths do not jam), so those cells need a hard budget.
            cmd += ["--sub-attempts", repr(job.sub_attempts)]
    # maxfreq = T is hard-coded in the engines (they reject any other value);
    # passing it explicitly makes a stale T/2-default binary run the right
    # band instead of silently narrowing it.
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
