"""Render baked observer frames to a PNG stack and assemble a video.

Reads a ``.frames`` file from `cuda/frame_bake.cu` (via braidlab.frames),
draws one panel per selected observer per instant — images at their apparent
positions, tinted by total received redshift (cosmological x Doppler), with
the observer at the centre and the front outline at chi_max — and assembles
the stack into a video with ffmpeg. One observer gives a single pane; two
give the viewer's side-by-side layout.

Usage::

    uv run python -m plots.render_frames --frames bake.frames \
        --observers 0 1 --mode moving --fps 12 --out frame_anim.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from braidlab.frames import FrameSet, apparent_positions, load_frames

#: Golden-ratio conjugate: consecutive gids land far apart on the hue wheel.
_HUE_STEP = 0.61803398875


def render_pane(
    ax,
    fs: FrameSet,
    fi: int,
    oi: int,
    mode: str,
    vmax: float,
    lim: float,
    label: str | None = None,
    gids: np.ndarray | None = None,
) -> None:
    """Draw one (instant, observer) pane onto a matplotlib axis.

    ``gids`` switches the pane from redshift tinting to family coloring:
    every image is colored by its source's group id (a stable golden-ratio
    hue per gid), so a subpath group reads as one same-colored clump.
    """
    from matplotlib import cm

    frame = fs.frames[fi][oi]
    x, y, ln_dopp = apparent_positions(frame, mode, cheb=fs.cheb)
    order = np.argsort(frame.hits["chi"])[::-1]  # nearest images drawn last
    if gids is not None:
        hues = (gids[frame.hits["src"]] * _HUE_STEP) % 1.0
        colors = cm.hsv(hues[order])
        ax.scatter(x[order], y[order], c=colors, s=4, linewidths=0)
    else:
        ln_total = np.log(frame.redshift) + ln_dopp
        ax.scatter(
            x[order],
            y[order],
            c=ln_total[order],
            s=4,
            cmap="inferno_r",
            vmin=0.0,
            vmax=vmax,
            linewidths=0,
        )
    ax.plot(0, 0, "+", color="tab:green", ms=10, mew=1.5)
    r = frame.chi_max
    if fs.cheb:
        ax.plot([-r, r, r, -r, -r], [-r, -r, r, r, -r], "-", color="gray", lw=0.7)
    else:
        th = np.linspace(0, 2 * np.pi, 120)
        ax.plot(r * np.cos(th), r * np.sin(th), "-", color="gray", lw=0.7)
    typ, path_idx, _, _ = fs.observers[oi]
    who = f"path #{path_idx}" if typ == 0 else "comoving point"
    prefix = f"{label} — " if label else ""
    ax.set_title(
        f"{prefix}{who} ({mode})  t = {frame.z_obs / np.pi:.3f}π  "
        f"{len(frame.hits):,} images",
        fontsize=9,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        nargs="+",
        required=True,
        help="one .frames file per pane (a single file may be repeated by "
        "giving it once with several --observers)",
    )
    parser.add_argument(
        "--observers",
        type=int,
        nargs="+",
        default=[0],
        help="observer index per pane; pane i pairs frames[i] (or the sole "
        "frames file) with observers[i]",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="optional pane title prefixes (one per pane)",
    )
    parser.add_argument(
        "--gid-dumps",
        nargs="+",
        default=None,
        help="per-pane dump CSV with a gid column for family coloring "
        "(subpath groups share a color); 'none' keeps the redshift tint",
    )
    parser.add_argument("--mode", choices=("static", "moving"), default="static")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--out", type=Path, required=True, help=".mp4 or .webm")
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="keep the PNG stack here (default: temp dir, deleted)",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panes = len(args.observers)
    files = args.frames if len(args.frames) > 1 else args.frames * n_panes
    if len(files) != n_panes:
        raise SystemExit("--frames must be one file, or one per --observers entry")
    if args.labels and len(args.labels) != n_panes:
        raise SystemExit("--labels must give one label per pane")
    if args.gid_dumps and len(args.gid_dumps) != n_panes:
        raise SystemExit("--gid-dumps must give one entry per pane ('none' to skip)")
    from braidlab.frames import load_gids

    pane_gids = [
        load_gids(g) if g != "none" else None for g in (args.gid_dumps or [])
    ] or [None] * n_panes
    sets = [load_frames(f) for f in files]
    n_inst = min(len(fs.frames) for fs in sets)
    # One color scale and one zoom across every pane of the whole animation:
    # the 99th percentile of ln(1+Z_total) (the Bang-adjacent tail is
    # unbounded) and the largest front.
    ln_all = []
    for k, oi in enumerate(args.observers):
        fs = sets[k]
        for row in fs.frames:
            f = row[oi]
            _, _, ln_dopp = apparent_positions(f, args.mode, cheb=fs.cheb)
            if len(f.hits):
                ln_all.append(np.log(f.redshift) + ln_dopp)
    vmax = float(np.percentile(np.concatenate(ln_all), 99)) if ln_all else 1.0
    lim = 1.05 * max(f.chi_max for fs in sets for row in fs.frames for f in row)

    png_dir = args.png_dir or Path(tempfile.mkdtemp(prefix="frames_"))
    png_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(n_inst):
        fig, axes = plt.subplots(
            1, n_panes, figsize=(5.2 * n_panes, 5.4), squeeze=False
        )
        for k, oi in enumerate(args.observers):
            label = args.labels[k] if args.labels else None
            render_pane(
                axes[0][k],
                sets[k],
                fi,
                oi,
                args.mode,
                vmax,
                lim,
                label,
                pane_gids[k],
            )
        fig.suptitle(
            f"past-light-cone frame — {sets[0].front} front, "
            f"{'square' if sets[0].cheb else 'circle'} metric",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(png_dir / f"frame_{fi:05d}.png", dpi=args.dpi)
        plt.close(fig)
    print(f"rendered {n_inst} PNGs to {png_dir}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found -- PNG stack kept, no video assembled")
        return
    codec = (
        ["-c:v", "libvpx-vp9", "-b:v", "2M"]
        if args.out.suffix == ".webm"
        else ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(png_dir / "frame_%05d.png"),
            # even dimensions for yuv420p
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            *codec,
            str(args.out),
        ],
        check=True,
        capture_output=True,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
