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


def render_pane(ax, fs: FrameSet, fi: int, oi: int, mode: str, vmax: float) -> None:
    """Draw one (instant, observer) pane onto a matplotlib axis."""
    frame = fs.frames[fi][oi]
    x, y, ln_dopp = apparent_positions(frame, mode, cheb=fs.cheb)
    ln_total = np.log(frame.redshift) + ln_dopp
    order = np.argsort(frame.hits["chi"])[::-1]  # nearest images drawn last
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
    ax.set_title(
        f"{who} ({mode})  t = {frame.z_obs / np.pi:.3f}π  {len(frame.hits):,} images",
        fontsize=9,
    )
    lim = 1.05 * max(f.chi_max for row in fs.frames for f in row)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument(
        "--observers",
        type=int,
        nargs="+",
        default=[0],
        help="observer indices to render (1 pane each, side by side)",
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

    fs = load_frames(args.frames)
    n_inst = len(fs.frames)
    # One color scale across the whole animation: the 99th percentile of
    # ln(1+Z_total) over every rendered frame (the Bang-adjacent tail is
    # unbounded).
    ln_all = []
    for row in fs.frames:
        for oi in args.observers:
            f = row[oi]
            _, _, ln_dopp = apparent_positions(f, args.mode, cheb=fs.cheb)
            if len(f.hits):
                ln_all.append(np.log(f.redshift) + ln_dopp)
    vmax = float(np.percentile(np.concatenate(ln_all), 99)) if ln_all else 1.0

    png_dir = args.png_dir or Path(tempfile.mkdtemp(prefix="frames_"))
    png_dir.mkdir(parents=True, exist_ok=True)
    n_panes = len(args.observers)
    for fi in range(n_inst):
        fig, axes = plt.subplots(
            1, n_panes, figsize=(5.2 * n_panes, 5.4), squeeze=False
        )
        for k, oi in enumerate(args.observers):
            render_pane(axes[0][k], fs, fi, oi, args.mode, vmax)
        fig.suptitle(
            f"past-light-cone frame — {fs.front} front, "
            f"{'square' if fs.cheb else 'circle'} metric",
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
