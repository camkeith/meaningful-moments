"""Render qualitative-gallery composite PDFs from picks JSON.

Six output PDFs: paper_tables_20260514_140732/gallery/{ssv2,k400,d48}_{wins,losses}.pdf
Each PDF stacks 5 single-row thumbnail strips. Each strip = 10 evenly-spaced
frames along the video timeline, with green/red border per thumbnail indicating
whether the segment containing that timestamp was kept or dropped by MSS.

Reuses the frame-extraction pattern from scripts/teaser_figure.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import textwrap
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

SAGE = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[1]))
FFMPEG = "/usr/bin/ffmpeg"

N_THUMBS = 10            # thumbnails per strip
KEPT_COLOR = "#2E7D32"   # green
DROPPED_COLOR = "#C62828" # red
THUMB_HEIGHT = 0.7
THUMB_LW = 2.0


def extract_frame(video_path: Path, t_seconds: float, out_path: Path):
    """Extract a single frame at timestamp t."""
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{t_seconds:.4f}", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "3", "-vf", "scale=320:-2", str(out_path)],
        check=True, capture_output=True,
    )


def find_segment_at(timestamp: float, segment_times: list, kept_set: set):
    """Given a timestamp, find which segment index contains it and whether kept."""
    for i, (s, e) in enumerate(segment_times):
        if s <= timestamp <= e:
            return i, (i in kept_set)
    # Edge case: timestamp past end of last segment → use last
    if segment_times and timestamp > segment_times[-1][1]:
        i = len(segment_times) - 1
        return i, (i in kept_set)
    return 0, (0 in kept_set)


def render_pick_strip(ax, pick, tmpdir: Path, strip_idx: int):
    """Render one pick as a single row of N_THUMBS color-bordered thumbnails."""
    video_path = pick["video_path"]
    if not Path(video_path).is_absolute():
        video_path = SAGE / video_path
    video_path = Path(video_path)

    duration = float(pick.get("duration_s") or 0.0)
    if duration <= 0:
        # Fallback: probe with ffmpeg
        result = subprocess.run(
            [FFMPEG, "-i", str(video_path)],
            capture_output=True, text=True,
        )
        # Try to parse from stderr; if we can't, default to 4s (SSv2 short clip)
        duration = 4.0

    segment_times = pick.get("segment_times", [])
    kept_set = set(pick.get("kept_indices", []))

    # Pick 10 evenly-spaced timestamps across the video duration
    timestamps = [duration * (i + 0.5) / N_THUMBS for i in range(N_THUMBS)]

    # Extract frames
    frame_dir = tmpdir / f"strip_{strip_idx:02d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for j, t in enumerate(timestamps):
        out = frame_dir / f"f_{j:02d}.png"
        try:
            extract_frame(video_path, t, out)
            frames.append(Image.open(out).copy())
        except subprocess.CalledProcessError:
            frames.append(Image.new("RGB", (320, 180), (200, 200, 200)))

    # Decide border color per thumbnail (kept vs dropped at each timestamp)
    border_colors = []
    for t in timestamps:
        _, is_kept = find_segment_at(t, segment_times, kept_set)
        border_colors.append(KEPT_COLOR if is_kept else DROPPED_COLOR)

    # Draw
    ax.set_xlim(0, N_THUMBS)
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    for j, (img, color) in enumerate(zip(frames, border_colors)):
        cx = j + 0.5
        cy = 0.5
        w = 0.92
        extent = (cx - w / 2, cx + w / 2, cy - THUMB_HEIGHT / 2, cy + THUMB_HEIGHT / 2)
        ax.imshow(np.asarray(img), extent=extent, aspect="auto", zorder=2,
                  interpolation="bilinear")
        ax.add_patch(Rectangle(
            (extent[0], extent[2]), w, THUMB_HEIGHT,
            facecolor="none", edgecolor=color, linewidth=THUMB_LW, zorder=3,
        ))


def render_substrate_bucket(picks, out_pdf: Path, substrate_label: str, bucket_label: str):
    """Render a composite PDF: 5 strips stacked vertically."""
    n_picks = len(picks)
    if n_picks == 0:
        print(f"  WARN: no picks for {out_pdf.name} — skipping")
        return

    # Figure: 7" wide, ~1.1" per strip + 0.6" for caption row each
    strip_h = 1.0   # inches per strip (image)
    caption_h = 0.45  # inches per strip (caption above)
    fig_h = n_picks * (strip_h + caption_h) + 0.4
    fig_w = 7.5

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180, facecolor="white")
    # Row layout: top title, then for each pick: [caption row, strip row]
    gs = fig.add_gridspec(
        nrows=n_picks * 2 + 1, ncols=1,
        height_ratios=[caption_h] + [caption_h, strip_h] * n_picks,
        hspace=0.10, left=0.02, right=0.99, top=0.99, bottom=0.01,
    )

    # Top title row
    top_ax = fig.add_subplot(gs[0, 0])
    top_ax.axis("off")
    top_ax.text(0.5, 0.5,
                f"{substrate_label} — {bucket_label} (n={n_picks})",
                ha="center", va="center", fontsize=12, weight="bold",
                transform=top_ax.transAxes)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i, pick in enumerate(picks):
            cap_idx = 1 + i * 2          # caption row
            strip_idx_row = cap_idx + 1  # strip row (image)
            cap_ax = fig.add_subplot(gs[cap_idx, 0])
            cap_ax.axis("off")
            full_mark = "✓" if pick["full_top1_correct"] else "✗"
            mss_mark = "✓" if pick["mss_top1_correct"] else "✗"
            gt = textwrap.shorten(pick["gt_label"], 60, placeholder="…")
            pf = textwrap.shorten(str(pick["pred_full"]), 35, placeholder="…")
            pm = textwrap.shorten(str(pick["pred_mss"]), 35, placeholder="…")
            cap = (
                f"{pick['video_id']}   GT: \"{gt}\"\n"
                f"full {full_mark} → \"{pf}\"     |     MSS-cut {mss_mark} → \"{pm}\"     "
                f"({len(pick['kept_indices'])}/{pick['total_segments']} segs kept)"
            )
            cap_ax.text(0.02, 0.5, cap, ha="left", va="center", fontsize=7.5,
                        transform=cap_ax.transAxes, family="DejaVu Sans")
            # Strip row
            strip_ax = fig.add_subplot(gs[strip_idx_row, 0])
            render_pick_strip(strip_ax, pick, tdp, i)

        out_pdf.parent.mkdir(exist_ok=True)
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Wrote: {out_pdf} ({n_picks} picks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(SAGE / "paper_tables_20260514_140732/qualitative_gallery_picks.json"))
    ap.add_argument("--output-dir",
                    default=str(SAGE / "paper_tables_20260514_140732/gallery"))
    args = ap.parse_args()

    picks_data = json.loads(Path(args.config).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.family": ["DejaVu Sans"],
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    substrate_labels = {
        "ssv2": "Something-Something v2 (closed-set MSS-kept vs full classification)",
        "k400": "Kinetics-400 (vlm-selected vs full, VideoMAE-K400-large)",
        "d48":  "Diving-48 (vlm-selected vs full, V-JEPA 2 HF-port)",
    }

    for substrate, label in substrate_labels.items():
        if substrate not in picks_data:
            continue
        for bucket in ["wins", "losses"]:
            picks = picks_data[substrate].get(bucket, [])
            out_pdf = out_dir / f"{substrate}_{bucket}.pdf"
            print(f"\n[{substrate} / {bucket}] rendering {len(picks)} picks → {out_pdf}")
            render_substrate_bucket(picks, out_pdf, label, bucket.capitalize())


if __name__ == "__main__":
    main()
