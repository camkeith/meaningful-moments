"""Input adapter: ``(video_path, kept_segments) → pixel_values_videos`` under one of three protocols.

See the development design notes Decision 5 for the
sparse-vs-repack-vs-mask rationale and the documented Option A fallback.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import av
import numpy as np
import torch

LOG = logging.getLogger(__name__)

FFMPEG = os.environ.get("FFMPEG", "/usr/bin/ffmpeg")  # v6.1; PATH default is ancient (workstation note)

Protocol = str  # "repack" | "mask" | "sparse"
Segment = tuple[float, float]  # (start_s, end_s)


def _decode_uniform_frames(video_path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``num_frames`` uniformly sampled frames from a video.

    Returns ``(frames_uint8_HWC, frame_indices)`` where ``frame_indices`` carries
    the original-timeline frame index for each sampled frame.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        if total <= 0:
            decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            total = len(decoded)
            if total == 0:
                raise RuntimeError(f"No frames decoded from {video_path}")
            indices = np.linspace(0, total - 1, num_frames).round().astype(int)
            frames = np.stack([decoded[i] for i in indices], axis=0)
            return frames, indices

    indices = np.linspace(0, total - 1, num_frames).round().astype(int)
    indices_set = set(indices.tolist())
    frames_by_idx: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices_set:
                frames_by_idx[i] = frame.to_ndarray(format="rgb24")
            if len(frames_by_idx) == len(indices_set):
                break
    if len(frames_by_idx) < len(indices_set):
        for idx in indices_set - frames_by_idx.keys():
            nearest = min(frames_by_idx.keys(), key=lambda k: abs(k - idx))
            frames_by_idx[idx] = frames_by_idx[nearest]
    frames = np.stack([frames_by_idx[int(i)] for i in indices], axis=0)
    return frames, indices


def _ffmpeg_cut_concat(video_path: Path, segments: list[Segment], out_path: Path) -> None:
    """Concatenate the given (start, end) segments of ``video_path`` into ``out_path``.

    Legacy path — replaced by ``_decode_kept_frames_pyav`` for the repack protocol.
    Kept here only as a fallback if PyAV decode fails.
    """
    import shutil
    if not segments:
        raise ValueError("No segments to concat")
    tmpdir = Path(tempfile.mkdtemp(prefix="vjepa2_cut_"))
    parts: list[Path] = []
    try:
        for i, (s, e) in enumerate(segments):
            part = tmpdir / f"part_{i:03d}.mp4"
            cmd = [
                FFMPEG, "-loglevel", "error", "-y",
                "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                "-i", str(video_path),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-an",
                str(part),
            ]
            subprocess.run(cmd, check=True)
            parts.append(part)
        listfile = tmpdir / "list.txt"
        listfile.write_text("\n".join(f"file '{p}'" for p in parts) + "\n")
        cmd = [FFMPEG, "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out_path)]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _decode_weighted_kept_frames_pyav(
    video_path: Path, kept_segments: list[Segment], weights: list[float], num_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment weighted frame allocation: segment i gets ceil(num_frames * w_i / sum(w)) frames.

    Within each segment, frames are uniform-sampled from that segment's decoded frames.
    Concatenates per-segment frame lists in time order. Used for the ``vlm-weighted``
    condition; ordinary ``vlm-selected`` uses _decode_kept_frames_pyav (uniform within
    concatenated kept content, no per-segment allocation).
    """
    if not kept_segments or not weights or len(weights) != len(kept_segments):
        raise ValueError(f"weighted decode requires matched kept_segments + weights: {len(kept_segments)} vs {len(weights or [])}")

    # First, group decoded frames by which kept-segment they belong to (in input order).
    seg_frames: list[list[np.ndarray]] = [[] for _ in kept_segments]
    seg_orig_indices: list[list[int]] = [[] for _ in kept_segments]
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                t = frame_idx / float(stream.average_rate or 25.0)
            else:
                t = float(frame.pts * time_base)
            for k, (s, e) in enumerate(kept_segments):
                if s <= t <= e:
                    seg_frames[k].append(frame.to_ndarray(format="rgb24"))
                    seg_orig_indices[k].append(frame_idx)
                    break

    # Allocate frames per segment proportional to weight (largest-remainders to fill exactly num_frames).
    total_w = sum(weights) if sum(weights) > 0 else 1.0
    raw_alloc = [num_frames * w / total_w for w in weights]
    alloc = [int(round(x)) for x in raw_alloc]
    diff = num_frames - sum(alloc)
    if diff != 0:
        # Fix rounding: distribute the residual to segments with the largest fractional remainder
        # (or smallest if we need to subtract).
        residuals = [(raw - rounded, k) for k, (raw, rounded) in enumerate(zip(raw_alloc, alloc))]
        residuals.sort(key=lambda x: -x[0] if diff > 0 else x[0])
        for i in range(abs(diff)):
            k = residuals[i % len(residuals)][1]
            alloc[k] += 1 if diff > 0 else -1
    # Force at least 1 frame for any segment that has decoded content; rebalance if needed.
    for k in range(len(alloc)):
        if alloc[k] < 1 and seg_frames[k]:
            # Steal one from the largest allocation
            largest = max(range(len(alloc)), key=lambda i: alloc[i])
            if largest != k and alloc[largest] > 1:
                alloc[largest] -= 1
                alloc[k] = 1
    # If any seg has alloc>0 but no decoded frames (shouldn't happen for in-segment selection), give to neighbor
    for k in range(len(alloc)):
        if alloc[k] > 0 and not seg_frames[k]:
            target = max(range(len(alloc)), key=lambda i: alloc[i] if seg_frames[i] else -1)
            if target != k and seg_frames[target]:
                alloc[target] += alloc[k]
                alloc[k] = 0

    out_frames: list[np.ndarray] = []
    out_indices: list[int] = []
    for k, n_k in enumerate(alloc):
        if n_k <= 0 or not seg_frames[k]:
            continue
        n = len(seg_frames[k])
        if n_k >= n:
            sample_idx = np.concatenate([np.arange(n), np.full(n_k - n, n - 1)])
        else:
            sample_idx = np.linspace(0, n - 1, n_k).round().astype(int)
        for i in sample_idx:
            out_frames.append(seg_frames[k][int(i)])
            out_indices.append(seg_orig_indices[k][int(i)])

    if len(out_frames) < num_frames:
        # Edge case: total allocation came up short (e.g., empty segments). Pad with last frame.
        if not out_frames:
            raise RuntimeError(f"No frames decoded for any kept segment in {video_path}")
        while len(out_frames) < num_frames:
            out_frames.append(out_frames[-1])
            out_indices.append(out_indices[-1])
    elif len(out_frames) > num_frames:
        out_frames = out_frames[:num_frames]
        out_indices = out_indices[:num_frames]

    return np.stack(out_frames, axis=0), np.array(out_indices)


def _decode_kept_frames_pyav(
    video_path: Path, kept_segments: list[Segment], num_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """PyAV-only path for repack: decode all frames, keep those whose timestamp falls in any kept range, uniform-sample ``num_frames``.

    Replaces the old ``ffmpeg cut + concat → PyAV decode`` round trip.
    Skips encode/decode entirely; ~5-10× faster on short SSv2/K400 clips.
    """
    if not kept_segments:
        raise ValueError("No kept segments")
    kept_frames: list[np.ndarray] = []
    kept_orig_indices: list[int] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                t = frame_idx / float(stream.average_rate or 25.0)
            else:
                t = float(frame.pts * time_base)
            in_kept = any(s <= t <= e for (s, e) in kept_segments)
            if in_kept:
                kept_frames.append(frame.to_ndarray(format="rgb24"))
                kept_orig_indices.append(frame_idx)
    if not kept_frames:
        # No frames matched the kept ranges (very short or off-by-one timestamps).
        # Fallback: take the frame nearest each segment's midpoint.
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 25.0)
            decoded_pairs: list[tuple[int, np.ndarray]] = [
                (i, f.to_ndarray(format="rgb24")) for i, f in enumerate(container.decode(video=0))
            ]
        if not decoded_pairs:
            raise RuntimeError(f"No frames decoded from {video_path}")
        for s, e in kept_segments:
            mid = 0.5 * (s + e)
            target_idx = int(round(mid * fps))
            target_idx = max(0, min(len(decoded_pairs) - 1, target_idx))
            kept_frames.append(decoded_pairs[target_idx][1])
            kept_orig_indices.append(decoded_pairs[target_idx][0])
    n = len(kept_frames)
    if n >= num_frames:
        sample_idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    else:
        # Repeat last frame to fill (rare; tiny clips with very short kept span)
        sample_idx = np.concatenate([np.arange(n), np.full(num_frames - n, n - 1)])
    frames = np.stack([kept_frames[i] for i in sample_idx], axis=0)
    indices = np.array([kept_orig_indices[i] for i in sample_idx])
    return frames, indices


def _decode_kept_frames_ordered_pyav(
    video_path: Path, kept_segments: list[Segment], num_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Repack variant that concatenates frames in the GIVEN segment order.

    Walks the video once and buckets each in-range frame into the FIRST kept
    segment that contains it (preserving the source frame's content), then
    emits frames bucket-by-bucket in the order kept_segments was passed.

    Used by the shuffle-test eval where kept_segments is intentionally in a
    non-monotonic order. For monotonic input (segments sorted by start_s with
    no overlap), the output is byte-identical to ``_decode_kept_frames_pyav``,
    so this function is safe to use unconditionally if ever desired.
    """
    if not kept_segments:
        raise ValueError("No kept segments")

    bucket: list[list[np.ndarray]] = [[] for _ in kept_segments]
    bucket_idx: list[list[int]] = [[] for _ in kept_segments]
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                t = frame_idx / float(stream.average_rate or 25.0)
            else:
                t = float(frame.pts * time_base)
            for k, (s, e) in enumerate(kept_segments):
                if s <= t <= e:
                    bucket[k].append(frame.to_ndarray(format="rgb24"))
                    bucket_idx[k].append(frame_idx)
                    break

    ordered_frames: list[np.ndarray] = []
    ordered_orig: list[int] = []
    for k in range(len(kept_segments)):
        ordered_frames.extend(bucket[k])
        ordered_orig.extend(bucket_idx[k])

    if not ordered_frames:
        # Fallback: nearest-midpoint frame per segment, in segment order
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 25.0)
            decoded_pairs: list[tuple[int, np.ndarray]] = [
                (i, f.to_ndarray(format="rgb24")) for i, f in enumerate(container.decode(video=0))
            ]
        if not decoded_pairs:
            raise RuntimeError(f"No frames decoded from {video_path}")
        for s, e in kept_segments:
            mid = 0.5 * (s + e)
            target_idx = int(round(mid * fps))
            target_idx = max(0, min(len(decoded_pairs) - 1, target_idx))
            ordered_frames.append(decoded_pairs[target_idx][1])
            ordered_orig.append(decoded_pairs[target_idx][0])

    n = len(ordered_frames)
    if n >= num_frames:
        sample_idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    else:
        sample_idx = np.concatenate([np.arange(n), np.full(num_frames - n, n - 1)])
    frames = np.stack([ordered_frames[i] for i in sample_idx], axis=0)
    indices = np.array([ordered_orig[i] for i in sample_idx])
    return frames, indices


def _segments_to_frame_index_set(
    kept_segments: list[Segment], fps: float, total_frames: int
) -> set[int]:
    keep: set[int] = set()
    for s, e in kept_segments:
        i0 = max(0, int(round(s * fps)))
        i1 = min(total_frames - 1, int(round(e * fps)))
        for i in range(i0, i1 + 1):
            keep.add(i)
    return keep


def _video_fps_and_frames(video_path: Path) -> tuple[float, int]:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        total = stream.frames or 0
        if total <= 0:
            container.seek(0)
            total = sum(1 for _ in container.decode(video=0))
    return fps, total


def build_input(
    video_path: str | Path,
    kept_segments: list[Segment] | None,
    protocol: Protocol,
    frames_per_clip: int,
    image_size: int,
    video_processor,
    kept_segment_weights: list[float] | None = None,
    respect_segment_order: bool = False,
):
    """Build a ``pixel_values_videos`` tensor for a V-JEPA 2 forward pass.

    Args:
        video_path: path to a video file.
        kept_segments: list of ``(start_s, end_s)`` segments to keep, or ``None`` for full-video.
        protocol: ``"repack"``, ``"mask"``, or ``"sparse"``.
        frames_per_clip: number of frames the probe expects (16 SSv2, 32 Diving-48).
        image_size: spatial size the probe expects (256).
        video_processor: a ``VJEPA2VideoProcessor`` instance.

    Returns:
        ``pixel_values_videos`` tensor ready for ``model.forward()``. The leading batch
        dimension is included (shape ``(1, T, C, H, W)`` after the processor).
    """
    video_path = Path(video_path)

    if protocol == "sparse":
        LOG.warning(
            "Protocol 'sparse' (Option C) is not implemented in transformers 4.57.6 — "
            "VJEPA2ForVideoClassification.forward() takes only `pixel_values_videos`. "
            "Falling back to 'repack' per vjepa2-seven-condition-experiment/design.md Decision 8."
        )
        protocol = "repack"

    def _proc_to_tensor(proc_out):
        """V-JEPA returns 'pixel_values_videos'; VideoMAE returns 'pixel_values'."""
        for k in ("pixel_values_videos", "pixel_values"):
            if k in proc_out:
                return proc_out[k]
        raise KeyError(f"Processor output keys do not include pixel_values[_videos]: {list(proc_out.keys())}")

    if kept_segments is None or protocol == "repack" and not kept_segments:
        frames, _ = _decode_uniform_frames(video_path, frames_per_clip)
        return _proc_to_tensor(video_processor(list(frames), return_tensors="pt"))

    if protocol == "repack":
        # Weighted variant: per-segment frame allocation by importance weights.
        # Used by the vlm-weighted condition.
        if kept_segment_weights is not None and len(kept_segment_weights) == len(kept_segments):
            try:
                frames, _ = _decode_weighted_kept_frames_pyav(
                    video_path, kept_segments, kept_segment_weights, frames_per_clip,
                )
                return _proc_to_tensor(video_processor(list(frames), return_tensors="pt"))
            except Exception as e:
                LOG.warning("Weighted-repack PyAV failed for %s (%s); falling back to uniform-within-kept", video_path, e)
        try:
            if respect_segment_order:
                frames, _ = _decode_kept_frames_ordered_pyav(
                    video_path, kept_segments, frames_per_clip,
                )
            else:
                frames, _ = _decode_kept_frames_pyav(video_path, kept_segments, frames_per_clip)
        except Exception as e:
            LOG.warning("PyAV repack failed for %s (%s); falling back to ffmpeg cut+concat", video_path, e)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                _ffmpeg_cut_concat(video_path, kept_segments, tmp_path)
                frames, _ = _decode_uniform_frames(tmp_path, frames_per_clip)
            finally:
                tmp_path.unlink(missing_ok=True)
        return _proc_to_tensor(video_processor(list(frames), return_tensors="pt"))

    if protocol == "mask":
        fps, total = _video_fps_and_frames(video_path)
        frames, frame_indices = _decode_uniform_frames(video_path, frames_per_clip)
        keep_set = _segments_to_frame_index_set(kept_segments, fps, total)
        pixel_values = _proc_to_tensor(video_processor(list(frames), return_tensors="pt"))
        for t, idx in enumerate(frame_indices):
            if int(idx) not in keep_set:
                pixel_values[0, t] = 0.0
        return pixel_values

    raise ValueError(f"Unknown protocol: {protocol!r}")


__all__ = ["build_input", "Segment", "Protocol"]
