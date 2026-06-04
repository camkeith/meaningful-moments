"""V-JEPA2 official-checkpoint wrapper for the seven-condition runner.

Reproduces the paper's evaluation pipeline (encoder + 4-block AttentiveClassifier
with multi-segment, multi-view aggregation), parameterized to match the official
configs/eval/vitl/{ssv2,diving48}.yaml.

Two entry points:
    predict_full(video_path)
        -> Official multi-segment, multi-view sampling (e.g. 4×3 for d48, 2×3 for ssv2).
           Used for the `full` baseline. Reproduces published numbers.

    predict_kept(video_path, kept_segments)
        -> Single 32-frame clip sampled from inside the kept_segments union, fed
           through encoder + ensemble of probe heads. Used for vlm-selected,
           uniform, motion, random, etc. Single spatial view (no 3-crop TTA on
           selection conditions; relative comparisons stay consistent).

The wrapper imports from the cloned `external/vjepa2` repo. The repo's `src/`
and `evals/` modules must be on PYTHONPATH at construction time (the wrapper
prepends them automatically).
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import av
import numpy as np
import torch
import torch.nn.functional as F

LOG = logging.getLogger("vjepa2_official_wrapper")

MM_ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[3]))

VJEPA2_REPO = Path(os.environ.get("VJEPA2_REPO", MM_ROOT / "external" / "vjepa2"))
if str(VJEPA2_REPO) not in sys.path:
    sys.path.insert(0, str(VJEPA2_REPO))

DEFAULT_CKPT_DIR = Path(
    os.environ.get("VJEPA2_CKPT_DIR", MM_ROOT / "external" / "vjepa2_checkpoints")
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)


# Per-dataset pinned configurations matching configs/eval/vitl/{ssv2,diving48}.yaml.
PROBE_CONFIGS = {
    "vjepa2-official-diving48": {
        "encoder_ckpt": DEFAULT_CKPT_DIR / "vitl.pt",
        "probe_ckpt": DEFAULT_CKPT_DIR / "diving48-vitl-256.pt",
        "module_name": "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip_multilevel",
        "num_classes": 48,
        "frames_per_clip": 32,
        "frame_step": 2,
        "num_segments": 4,
        "num_views_per_segment": 3,
        "resolution": 256,
        "out_layers": [17, 19, 21, 23],
        "num_probe_blocks": 4,
        "num_heads": 16,
        "tubelet_size": 2,
        "patch_size": 16,
    },
    "vjepa2-official-ssv2": {
        "encoder_ckpt": DEFAULT_CKPT_DIR / "vitl.pt",
        "probe_ckpt": DEFAULT_CKPT_DIR / "ssv2-vitl-16x2x3.pt",
        "module_name": "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip",
        "num_classes": 174,
        "frames_per_clip": 16,
        "frame_step": 4,
        "num_segments": 2,
        "num_views_per_segment": 3,
        "resolution": 256,
        "out_layers": None,  # single-layer (final)
        "num_probe_blocks": 4,
        "num_heads": 16,
        "tubelet_size": 2,
        "patch_size": 16,
    },
}


# ----------------------------------------------------------------------------
# Frame decoding + preprocessing — replicates the official EvalVideoTransform
# (resize to short-side `resolution`, center crop, ImageNet normalize).
# ----------------------------------------------------------------------------

def _decode_full_pyav(path: str) -> list[np.ndarray]:
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def _resize_short_side(frames_thwc: np.ndarray, target: int) -> np.ndarray:
    """Resize so short side = target, preserving aspect ratio. (T, H, W, C) uint8."""
    import cv2
    T, H, W, _ = frames_thwc.shape
    if H <= W:
        new_H, new_W = target, int(round(W * target / H))
    else:
        new_H, new_W = int(round(H * target / W)), target
    out = np.empty((T, new_H, new_W, 3), dtype=frames_thwc.dtype)
    for i in range(T):
        out[i] = cv2.resize(frames_thwc[i], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    return out


def _center_crop(frames_thwc: np.ndarray, size: int) -> np.ndarray:
    T, H, W, _ = frames_thwc.shape
    top = (H - size) // 2
    left = (W - size) // 2
    return frames_thwc[:, top:top + size, left:left + size, :]


def _spatial_views(frames_thwc: np.ndarray, size: int, num_views: int) -> list[np.ndarray]:
    """Replicates EvalVideoTransform's multi-view spatial cropping (utils.py:140)."""
    T, H, W, _ = frames_thwc.shape
    if num_views == 1:
        return [_center_crop(frames_thwc, size)]
    spatial_step = (max(H, W) - size) // max(1, num_views - 1)
    views = []
    for i in range(num_views):
        start = i * spatial_step
        if H > W:
            view = frames_thwc[:, start:start + size, :, :]
        else:
            view = frames_thwc[:, :, start:start + size, :]
        # Pad / clip to exactly (T, size, size, C) if needed
        view = view[:, :size, :size, :]
        views.append(view)
    return views


def _to_tensor_normalize(view_thwc: np.ndarray, device: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """(T, H, W, C) uint8 -> (1, C, T, H, W) bf16 normalized for V-JEPA2 encoder."""
    t = torch.from_numpy(view_thwc).to(device).float().div_(255.0)
    t = t.permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)
    t = (t - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    t = t.unsqueeze(0).to(dtype=dtype)  # (1, C, T, H, W)
    return t


def _sample_segment_indices(n_src: int, frames_per_clip: int, frame_step: int, num_segments: int) -> list[list[int]]:
    """Mirror VideoDataset's eval-time multi-segment sampling.

    For each of `num_segments` temporal segments, pick a contiguous window of
    `frames_per_clip` frames at stride `frame_step`. Segments are uniformly
    spaced across the video; segments overlap if the video is too short.
    """
    span = frames_per_clip * frame_step
    if n_src <= span:
        # One segment covers everything; replicate it num_segments times
        idx = [min(i * frame_step, n_src - 1) for i in range(frames_per_clip)]
        return [idx for _ in range(num_segments)]
    if num_segments == 1:
        starts = [(n_src - span) // 2]
    else:
        max_start = n_src - span
        starts = [int(round(i * max_start / (num_segments - 1))) for i in range(num_segments)]
    out = []
    for s in starts:
        out.append([s + i * frame_step for i in range(frames_per_clip)])
    return out


def _gather_frames(all_frames: list[np.ndarray], indices: list[int]) -> np.ndarray:
    n = len(all_frames)
    safe = [min(i, n - 1) for i in indices]
    return np.stack([all_frames[i] for i in safe], axis=0)  # (F, H, W, C)


def _kept_pool(
    all_frames_count: int,
    fps: float,
    kept_segments: list[tuple[float, float]],
    frame_step: int,
) -> list[int]:
    """Build the synthesized MSS-kept frame pool — concatenated frame indices,
    preserving kept_segments order."""
    if not kept_segments:
        return list(range(0, all_frames_count, frame_step))
    bucket: list[list[int]] = []
    for s, e in kept_segments:
        i0 = max(0, int(round(s * fps)))
        i1 = min(all_frames_count - 1, int(round(e * fps)))
        b = list(range(i0, i1 + 1, frame_step))
        if not b:
            b = [i0]
        bucket.append(b)
    flat = [i for b in bucket for i in b]
    return flat if flat else [0]


def _multi_clip_kept_indices(
    all_frames_count: int,
    fps: float,
    kept_segments: list[tuple[float, float]],
    frames_per_clip: int,
    frame_step: int,
    num_segments: int,
) -> list[list[int]]:
    """Paper-style multi-segment sampling, but on the MSS-kept frame pool.

    Mirrors ``_sample_segment_indices`` (which assumes a contiguous video). Builds
    the kept-pool first, then picks ``num_segments`` evenly-spaced temporal
    starting positions within it. Falls back to repeating one segment when the
    pool is shorter than ``frames_per_clip``.
    """
    pool = _kept_pool(all_frames_count, fps, kept_segments, frame_step)
    n_pool = len(pool)
    if n_pool <= frames_per_clip:
        # pad by repeating the last index, replicate across all segments
        base = pool + [pool[-1]] * (frames_per_clip - n_pool)
        return [base for _ in range(num_segments)]
    if num_segments == 1:
        starts = [(n_pool - frames_per_clip) // 2]
    else:
        max_start = n_pool - frames_per_clip
        starts = [int(round(i * max_start / (num_segments - 1))) for i in range(num_segments)]
    return [[pool[s + i] for i in range(frames_per_clip)] for s in starts]


def _weighted_kept_pool(
    all_frames_count: int,
    fps: float,
    kept_segments: list[tuple[float, float]],
    kept_segment_weights: list[float],
    frame_step: int,
) -> list[int]:
    """Build a variable-density frame pool for fast-forward multi-clip sampling.

    For each kept segment k, sample ``max(1, round(n_k × density_k))`` indices
    uniformly within the segment, where ``n_k`` is the segment's frame count
    at ``frame_step`` and ``density_k = weight_k / duration_k`` is the per-second
    sampling density (the selector packs ``weight = duration × density``, so
    dividing back recovers density).

    Concatenate per-segment lists in temporal order → pool. Multi-clip starts
    are then evenly-spaced *within the pool*, naturally clustering clip placement
    in dense (important) regions.
    """
    if not kept_segments or not kept_segment_weights:
        return list(range(0, all_frames_count, frame_step))
    bucket: list[list[int]] = []
    for k, (s, e) in enumerate(kept_segments):
        duration = max(1e-6, e - s)
        density = kept_segment_weights[k] / duration  # → in [alpha, 1.0]
        density = max(0.0, min(1.0, density))
        i0 = max(0, int(round(s * fps)))
        i1 = min(all_frames_count - 1, int(round(e * fps)))
        if i1 < i0:
            continue
        seg_frame_range = list(range(i0, i1 + 1, frame_step)) or [i0]
        n_seg = len(seg_frame_range)
        m_k = max(1, int(round(n_seg * density)))
        if m_k >= n_seg:
            bucket.append(seg_frame_range)
        else:
            sub = np.linspace(0, n_seg - 1, m_k).round().astype(int).tolist()
            bucket.append([seg_frame_range[i] for i in sub])
    flat = [i for b in bucket for i in b]
    return flat if flat else [0]


def _multi_clip_weighted_kept_indices(
    all_frames_count: int,
    fps: float,
    kept_segments: list[tuple[float, float]],
    kept_segment_weights: list[float],
    frames_per_clip: int,
    frame_step: int,
    num_segments: int,
) -> list[list[int]]:
    """Paper-style multi-segment sampling on a variable-density (fast-forward) pool.

    Same shape as ``_multi_clip_kept_indices`` but the underlying pool has fewer
    frames in low-density (filler) regions. Evenly-spacing clip starts in such a
    pool means more clips land on high-importance regions while the whole
    timeline still receives coverage.
    """
    pool = _weighted_kept_pool(
        all_frames_count, fps, kept_segments, kept_segment_weights, frame_step
    )
    n_pool = len(pool)
    if n_pool <= frames_per_clip:
        base = pool + [pool[-1]] * (frames_per_clip - n_pool)
        return [base for _ in range(num_segments)]
    if num_segments == 1:
        starts = [(n_pool - frames_per_clip) // 2]
    else:
        max_start = n_pool - frames_per_clip
        starts = [int(round(i * max_start / (num_segments - 1))) for i in range(num_segments)]
    return [[pool[s + i] for i in range(frames_per_clip)] for s in starts]


def _mean_importance_for_indices(
    frame_indices: list[int],
    fps: float,
    sidecar_segments: list[dict],
) -> float:
    """Mean MSS per-frame importance for a list of frame indices.

    For each frame, looks up the MSS segment containing its timestamp and reads
    ``weight`` (= inclusion frequency). Frames with no covering segment get 0.
    Returns the average over all queried frames.
    """
    if not frame_indices or not sidecar_segments:
        return 0.0
    # Build a sorted list of (start_s, end_s, weight) for fast lookup.
    seg_lookup = sorted(
        (
            (float(s["start_s"]), float(s["end_s"]),
             float(s.get("weight", s.get("frequency", 0.0)) or 0.0))
            for s in sidecar_segments
        ),
        key=lambda r: r[0],
    )
    starts = [r[0] for r in seg_lookup]
    total = 0.0
    n = 0
    for idx in frame_indices:
        t = idx / max(fps, 1e-6)
        # Binary-search the segment whose start <= t
        lo, hi = 0, len(starts) - 1
        chosen = 0.0
        while lo <= hi:
            mid = (lo + hi) // 2
            s_s, s_e, s_w = seg_lookup[mid]
            if s_s <= t <= s_e:
                chosen = s_w
                break
            if t < s_s:
                hi = mid - 1
            else:
                lo = mid + 1
        total += chosen
        n += 1
    return total / max(1, n)


def _importance_anchored_hybrid_starts(
    all_frames_count: int,
    fps: float,
    sidecar_segments: list[dict],
    num_segments: int,
    frames_per_clip: int,
    frame_step: int,
    fraction_anchored: float = 0.5,
) -> list[int]:
    """Hybrid clip placement: ``ceil(num_segments × fraction_anchored)`` clip starts at MSS-importance
    peaks + the remainder evenly-spaced across the timeline.

    For num_segments=4 and fraction_anchored=0.5 → 2 anchored + 2 even. Anchored
    clips cluster at the high-importance peaks of the per-frame cumulative
    importance signal; even clips preserve broad timeline coverage. The mix
    keeps the model's input distribution close to its training distribution
    (contiguous clips at original scale) while shifting some compute toward
    informative moments.

    Implementation: pick anchored starts at quantiles [0.25, 0.75] of the
    importance CDF if 2 anchored; [0.5] if 1; [0.2, 0.5, 0.8] if 3, etc. Then
    even starts at uniform positions, skipping any that collide with anchored
    starts (within ``frames_per_clip × frame_step / 2`` of each other).
    """
    n_anchored = max(1, int(np.ceil(num_segments * fraction_anchored)))
    n_anchored = min(num_segments, n_anchored)
    n_even = num_segments - n_anchored

    span = frames_per_clip * frame_step
    max_start = max(0, all_frames_count - span)

    # ---- Anchored starts: importance quantiles ----
    importance = np.zeros(all_frames_count, dtype=np.float64) if all_frames_count > 0 else np.zeros(1)
    for seg in sidecar_segments:
        s_s = float(seg["start_s"])
        s_e = float(seg["end_s"])
        w = float(seg.get("weight", seg.get("frequency", 0.0)) or 0.0)
        i0 = max(0, int(round(s_s * fps)))
        i1 = min(all_frames_count - 1, int(round(s_e * fps)))
        if i1 < i0:
            continue
        importance[i0:i1 + 1] = w
    cum = np.cumsum(importance)
    total = float(cum[-1]) if cum.size else 0.0
    anchored: list[int] = []
    if total > 0.0 and n_anchored > 0:
        anchored_q = [(i + 0.5) / n_anchored for i in range(n_anchored)]
        for q in anchored_q:
            target_cum = q * total
            idx = int(np.searchsorted(cum, target_cum))
            anchored.append(min(max_start, max(0, idx)))

    # ---- Even starts: uniform spacing across the video, skipping overlaps with anchored ----
    if n_even <= 0:
        return anchored
    if n_even == 1:
        evens = [max_start // 2]
    else:
        evens = [int(round(i * max_start / (n_even - 1))) for i in range(n_even)]

    # If no anchored (total=0), use uniform spacing for all num_segments.
    if not anchored:
        if num_segments == 1:
            return [max_start // 2]
        return [int(round(i * max_start / (num_segments - 1))) for i in range(num_segments)]

    return sorted(anchored + evens)


def _importance_quantile_starts(
    all_frames_count: int,
    fps: float,
    sidecar_segments: list[dict],
    num_segments: int,
    frames_per_clip: int,
    frame_step: int,
) -> list[int]:
    """Pick num_segments contiguous-clip start frame indices at MSS-importance quantiles.

    Builds a per-frame importance signal by piecewise-constant interpolation of
    segment weights, then computes the cumulative-importance CDF and samples
    ``num_segments`` evenly-spaced quantiles in the CDF range
    [0.5/N, 1 - 0.5/N] (centered quantiles), mapping each quantile to the frame
    index whose cumulative-importance hits it. Result: clip starts cluster around
    high-importance regions while still spanning the full informativity range.

    Each clip is a contiguous frames_per_clip × frame_step window starting at the
    chosen index (clipped to keep the window in-video).
    """
    if not sidecar_segments or all_frames_count <= 0:
        # Fall back to evenly-spaced starts across the video.
        span = frames_per_clip * frame_step
        if num_segments == 1:
            return [max(0, (all_frames_count - span) // 2)]
        max_start = max(0, all_frames_count - span)
        return [int(round(i * max_start / (num_segments - 1))) for i in range(num_segments)]
    span = frames_per_clip * frame_step
    max_start = max(0, all_frames_count - span)
    importance = np.zeros(all_frames_count, dtype=np.float64)
    for seg in sidecar_segments:
        s_s = float(seg["start_s"])
        s_e = float(seg["end_s"])
        w = float(seg.get("weight", seg.get("frequency", 0.0)) or 0.0)
        i0 = max(0, int(round(s_s * fps)))
        i1 = min(all_frames_count - 1, int(round(s_e * fps)))
        if i1 < i0:
            continue
        importance[i0:i1 + 1] = w
    cum = np.cumsum(importance)
    total = float(cum[-1])
    if total <= 0.0:
        # All-zero importance — fall back to even spacing.
        if num_segments == 1:
            return [max_start // 2]
        return [int(round(i * max_start / (num_segments - 1))) for i in range(num_segments)]
    targets = [(i + 0.5) / num_segments for i in range(num_segments)]
    starts = []
    for q in targets:
        target_cum = q * total
        idx = int(np.searchsorted(cum, target_cum))
        idx = min(max_start, max(0, idx))
        starts.append(idx)
    return starts


def _weighted_kept_segment_indices(
    all_frames_count: int,
    fps: float,
    kept_segments: list[tuple[float, float]],
    kept_segment_weights: list[float],
    frames_per_clip: int,
) -> list[int]:
    """Per-segment weighted frame allocation (fast-forward sampling).

    Each kept segment k gets ``round(frames_per_clip × w_k / sum(w))`` frames,
    uniform-sampled within the segment's frame range. Largest-remainders fills
    any rounding residual to total = frames_per_clip exactly. Frames concatenate
    in segment temporal order.

    Used by the ``vlm-fastforward-a<NN>`` conditions where the selector stored
    ``weight = duration × density`` per segment, so the resulting allocation is
    proportional to ``duration × density`` — i.e. dense on important segments
    and density-``alpha``-floored on filler.

    No ``frame_step`` here: the weighted allocation already controls how the
    fixed budget is spread. ``frame_step`` is only meaningful for uniform-within-
    pool subsampling, which is the non-weighted path.
    """
    if not kept_segments or not kept_segment_weights:
        idx = np.linspace(0, all_frames_count - 1, frames_per_clip).round().astype(int)
        return idx.tolist()
    if len(kept_segment_weights) != len(kept_segments):
        raise ValueError(
            f"weights length {len(kept_segment_weights)} != segments {len(kept_segments)}"
        )

    total_w = sum(kept_segment_weights)
    if total_w <= 0:
        # All-zero weights — degenerate; fall back to uniform across kept span.
        idx = np.linspace(0, all_frames_count - 1, frames_per_clip).round().astype(int)
        return idx.tolist()
    raw = [frames_per_clip * w / total_w for w in kept_segment_weights]
    alloc = [int(round(x)) for x in raw]
    diff = frames_per_clip - sum(alloc)
    if diff != 0:
        residuals = [(raw[k] - alloc[k], k) for k in range(len(alloc))]
        residuals.sort(key=lambda x: -x[0] if diff > 0 else x[0])
        for i in range(abs(diff)):
            k = residuals[i % len(residuals)][1]
            alloc[k] += 1 if diff > 0 else -1
    # Don't force any segment to 0 → 1; with weights pre-scaled by duration × density,
    # tiny short-and-filler segments legitimately can get zero frames, which is the
    # fast-forward semantics (skip past them entirely).
    out: list[int] = []
    for k, (s, e) in enumerate(kept_segments):
        n_k = max(0, alloc[k])
        if n_k <= 0:
            continue
        i0 = max(0, int(round(s * fps)))
        i1 = min(all_frames_count - 1, int(round(e * fps)))
        if i1 <= i0:
            out.extend([i0] * n_k)
            continue
        seg_idx = np.linspace(i0, i1, n_k).round().astype(int).tolist()
        out.extend(seg_idx)
    if len(out) < frames_per_clip:
        # Edge case (all segments allocated 0 due to extreme rounding): pad last frame.
        if not out:
            out = [0]
        while len(out) < frames_per_clip:
            out.append(out[-1])
    elif len(out) > frames_per_clip:
        out = out[:frames_per_clip]
    return out


def _kept_segment_indices(
    all_frames_count: int,
    fps: float,
    kept_segments: list[tuple[float, float]],
    frames_per_clip: int,
    frame_step: int,
) -> list[int]:
    """Map kept_segments (in seconds) to a list of frame indices.

    Strategy: collect all frame indices that fall inside any kept segment
    (in temporal order, deduplicated), then uniform-sample frames_per_clip
    from that pool — preserving any inter-segment ordering implied by the
    sequence of kept_segments (used for shuffle test).
    """
    if not kept_segments:
        # Fall back to whole-video uniform 32 frames at frame_step
        idx = np.linspace(0, all_frames_count - 1, frames_per_clip).round().astype(int)
        return idx.tolist()
    # Bucket frames per segment (so the shuffle test's segment-order is honored)
    bucket: list[list[int]] = []
    for s, e in kept_segments:
        i0 = max(0, int(round(s * fps)))
        i1 = min(all_frames_count - 1, int(round(e * fps)))
        # frames at frame_step within this segment
        b = list(range(i0, i1 + 1, frame_step))
        if not b:
            b = [i0]
        bucket.append(b)
    flat = [i for b in bucket for i in b]
    if not flat:
        flat = list(range(0, all_frames_count, frame_step))[:frames_per_clip]
    # Uniform-subsample to frames_per_clip
    if len(flat) >= frames_per_clip:
        sub = np.linspace(0, len(flat) - 1, frames_per_clip).round().astype(int)
        return [flat[i] for i in sub]
    # Pad by repeating last
    return flat + [flat[-1]] * (frames_per_clip - len(flat))


# ----------------------------------------------------------------------------
# Wrapper class
# ----------------------------------------------------------------------------

class VJepa2OfficialWrapper:
    """Encoder + multi-head AttentiveClassifier from the official paper checkpoint."""

    def __init__(
        self,
        recognizer_key: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        head_checkpoint: str | Path | None = None,
        attentive_probe_checkpoint: str | Path | None = None,
        frame_step: int | None = None,
    ):
        if recognizer_key not in PROBE_CONFIGS:
            raise ValueError(f"Unknown recognizer key '{recognizer_key}'. Known: {list(PROBE_CONFIGS)}")
        cfg = PROBE_CONFIGS[recognizer_key]
        self.recognizer_key = recognizer_key
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        self.frames_per_clip = cfg["frames_per_clip"]
        # Allow runtime override of frame_step (default = paper protocol from config).
        # Useful for short-video datasets where the default span exceeds video length.
        self.frame_step = int(frame_step) if frame_step is not None else cfg["frame_step"]
        self.num_segments = cfg["num_segments"]
        self.num_views_per_segment = cfg["num_views_per_segment"]
        self.resolution = cfg["resolution"]
        self.num_classes = cfg["num_classes"]

        # ---- Load encoder via the official init_module ----
        m = importlib.import_module(cfg["module_name"])
        wrapper_kwargs = {
            "max_frames": 128,
            "use_pos_embed": False,
        }
        if cfg["out_layers"] is not None:
            wrapper_kwargs["out_layers"] = cfg["out_layers"]
        encoder = m.init_module(
            resolution=self.resolution,
            frames_per_clip=self.frames_per_clip,
            checkpoint=str(cfg["encoder_ckpt"]),
            model_kwargs={
                "encoder": {
                    "checkpoint_key": "target_encoder",
                    "model_name": "vit_large",
                    "patch_size": cfg["patch_size"],
                    "tubelet_size": cfg["tubelet_size"],
                    "uniform_power": True,
                    "use_rope": True,
                }
            },
            wrapper_kwargs=wrapper_kwargs,
        )
        encoder = encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim

        # ---- Load probe checkpoint(s) ----
        from src.models.attentive_pooler import AttentiveClassifier

        ckpt = torch.load(str(cfg["probe_ckpt"]), map_location="cpu", weights_only=False)
        sds = ckpt["classifiers"]  # list of state dicts (one per LR sweep head)
        self.classifiers: list[torch.nn.Module] = []
        for sd in sds:
            c = AttentiveClassifier(
                embed_dim=self.embed_dim,
                num_heads=cfg["num_heads"],
                depth=cfg["num_probe_blocks"],
                num_classes=self.num_classes,
                use_activation_checkpointing=False,
            ).to(device).eval()
            sd_clean = {k.replace("module.", ""): v for k, v in sd.items()}
            c.load_state_dict(sd_clean, strict=True)
            for p in c.parameters():
                p.requires_grad = False
            self.classifiers.append(c)
        LOG.info(
            f"VJepa2OfficialWrapper[{recognizer_key}] loaded: "
            f"encoder embed_dim={self.embed_dim}, "
            f"num_probes={len(self.classifiers)}, num_classes={self.num_classes}"
        )

        # ---- Optional: overlay a finetuned final-Linear head (MSS-cut probe) ----
        # When set, drops the multi-probe ensemble down to a single classifier whose
        # .linear is replaced with the trained Linear from disk. Eval uses just probe[0]
        # + the trained Linear from then on — matches the single-probe / single-view
        # / single-segment distribution the head was finetuned against.
        self.head_checkpoint = None
        if head_checkpoint is not None:
            from oracle.scripts.vjepa2.head import load_head
            head = load_head(head_checkpoint)
            if head.classifier.in_features != self.embed_dim:
                raise ValueError(
                    f"head hidden={head.classifier.in_features} ≠ wrapper embed_dim={self.embed_dim}"
                )
            if head.classifier.out_features != self.num_classes:
                raise ValueError(
                    f"head num_labels={head.classifier.out_features} ≠ wrapper num_classes={self.num_classes}"
                )
            # Replace probe[0]'s final Linear with the trained one (in the wrapper's dtype/device).
            target_dtype = next(self.classifiers[0].linear.parameters()).dtype
            new_linear = head.classifier.to(device=device, dtype=target_dtype)
            for p in new_linear.parameters():
                p.requires_grad = False
            self.classifiers[0].linear = new_linear
            self.classifiers = self.classifiers[:1]  # drop the LR-sweep ensemble
            self.head_checkpoint = str(head_checkpoint)
            LOG.info(
                "Overlaid finetuned head from %s — kept probe[0] only (was ensemble of %d).",
                head_checkpoint, len(sds),
            )

        # ---- Optional: overlay a fully retrained AttentiveClassifier (MSS-cut attentive probe) ----
        # Replaces probe[0] (the whole 4-block AttentiveClassifier) with our MSS-cut-finetuned one.
        # Only one of head_checkpoint / attentive_probe_checkpoint should be set.
        self.attentive_probe_checkpoint = None
        if attentive_probe_checkpoint is not None:
            if head_checkpoint is not None:
                raise ValueError(
                    "Pass either head_checkpoint (linear-only) or attentive_probe_checkpoint "
                    "(full classifier), not both."
                )
            blob = torch.load(str(attentive_probe_checkpoint), map_location="cpu", weights_only=False)
            sd = blob["classifier"]
            cfg_info = blob.get("config", {})
            # Build a fresh AttentiveClassifier matching the saved config (or fall back to wrapper cfg)
            ac_kwargs = {
                "embed_dim": cfg_info.get("embed_dim", self.embed_dim),
                "num_heads": cfg_info.get("num_heads", cfg["num_heads"]),
                "depth": cfg_info.get("num_probe_blocks", cfg["num_probe_blocks"]),
                "num_classes": cfg_info.get("num_classes", self.num_classes),
                "use_activation_checkpointing": False,
            }
            new_c = AttentiveClassifier(**ac_kwargs).to(device).eval()
            new_c.load_state_dict(sd, strict=True)
            for p in new_c.parameters():
                p.requires_grad = False
            self.classifiers = [new_c]  # collapse ensemble down to the trained probe
            self.attentive_probe_checkpoint = str(attentive_probe_checkpoint)
            LOG.info(
                "Overlaid trained AttentiveClassifier from %s — replaced ensemble of %d with the trained probe.",
                attentive_probe_checkpoint, len(sds),
            )

    # ---- core forward ----

    @torch.inference_mode()
    def _forward_clips(self, clips: list[list[torch.Tensor]]) -> torch.Tensor:
        """Run encoder + ensemble of classifiers + softmax average over views/probes.

        clips: nested list [num_segments][num_views] of (B=1, C, F, H, W) tensors.
        Returns: (B=1, num_classes) softmax probs aggregated across segments,
                 views, and the multi-head probe ensemble (matches the official
                 eval's `top1_meters` aggregation pattern).
        """
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            outputs = self.encoder(clips, clip_indices=None)  # [V] (B, T*S, D)
            # For each probe head, compute softmax per spatial view, average
            per_probe = []
            for c in self.classifiers:
                probs = [F.softmax(c(o), dim=1) for o in outputs]
                avg_view = sum(probs) / len(probs)
                per_probe.append(avg_view)
            # Ensemble across probe heads
            ensemble = sum(per_probe) / len(per_probe)
        return ensemble.float()  # (B, num_classes)

    # ---- public predict APIs ----

    @torch.inference_mode()
    def predict_full(self, video_path: str) -> dict:
        """Multi-segment × multi-view evaluation matching the published config.

        Reproduces the official eval-time sampling (e.g. 4 segments × 3 views for
        Diving-48). Used for the `full` baseline.
        """
        all_frames = _decode_full_pyav(video_path)
        n_src = len(all_frames)
        if n_src == 0:
            raise RuntimeError(f"empty video: {video_path}")

        # Sample num_segments × frames_per_clip indices
        seg_indices = _sample_segment_indices(
            n_src, self.frames_per_clip, self.frame_step, self.num_segments
        )

        # For each segment, decode frames + build all spatial views
        clips = []
        for indices in seg_indices:
            seg_frames = _gather_frames(all_frames, indices)  # (F, H, W, C)
            seg_resized = _resize_short_side(seg_frames, self.resolution)  # (F, h, w, C)
            views = _spatial_views(seg_resized, self.resolution, self.num_views_per_segment)
            view_tensors = [_to_tensor_normalize(v, self.device, self.dtype) for v in views]
            clips.append(view_tensors)

        ensemble_probs = self._forward_clips(clips)
        return self._top5_dict(ensemble_probs)

    @torch.inference_mode()
    def predict_kept_features(
        self, video_path: str, kept_segments: list[tuple[float, float]] | None
    ) -> torch.Tensor:
        """Phase-A feature extractor for MSS-cut linear-probe finetuning.

        Same frame-sampling path as ``predict_kept`` (single-segment, single-view),
        but stops before the per-probe Linear: returns ``probe[0].pooler(encoder_out)``
        as a (embed_dim,) tensor on CPU (float32).
        """
        with av.open(video_path) as c:
            stream = c.streams.video[0]
            fps = float(stream.average_rate or 25.0)
            all_frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        n_src = len(all_frames)
        if n_src == 0:
            raise RuntimeError(f"empty video: {video_path}")

        indices = _kept_segment_indices(
            n_src, fps, kept_segments or [],
            self.frames_per_clip, self.frame_step,
        )
        seg_frames = _gather_frames(all_frames, indices)
        seg_resized = _resize_short_side(seg_frames, self.resolution)
        view = _center_crop(seg_resized, self.resolution)
        view_tensor = _to_tensor_normalize(view, self.device, self.dtype)

        clips = [[view_tensor]]  # 1 segment, 1 view
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            outputs = self.encoder(clips, clip_indices=None)  # [V=1] (B=1, T*S, D)
            probe = self.classifiers[0]
            # probe.pooler(x).squeeze(1) → (B, embed_dim)
            features = probe.pooler(outputs[0]).squeeze(1)
        return features[0].float().cpu()

    @torch.inference_mode()
    def predict_kept(
        self,
        video_path: str,
        kept_segments: list[tuple[float, float]] | None,
        kept_segment_weights: list[float] | None = None,
        num_segments: int = 1,
        num_views_per_segment: int = 1,
    ) -> dict:
        """Evaluation on frames sampled from kept_segments.

        Used for the seven-condition selection variants (vlm-selected, uniform,
        motion, random, lowest-evidence, uniform-equal-segs, and shuffled
        variants).

        ``num_segments × num_views_per_segment`` controls the inference protocol:
          * Default (1, 1) → single-clip / single-view, matches the per-condition
            equal-compute regime used by the seven-condition runner.
          * (self.num_segments, self.num_views_per_segment) → paper-protocol
            multi-clip aggregation (e.g. 4×3 for Diving-48). Use this to make
            kept-segment evaluations apples-to-apples on compute against
            ``predict_full``'s 4×3 baseline.

        If ``kept_segment_weights`` is provided (e.g. by ``vlm-fastforward-a<NN>``),
        the underlying frame pool is variable-density (dense on important
        segments, sparse on filler). Otherwise the pool is uniform-within-kept.

        If kept_segments is None or empty, falls back to whole-video uniform
        sampling.
        """
        with av.open(video_path) as c:
            stream = c.streams.video[0]
            fps = float(stream.average_rate or 25.0)
            all_frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        n_src = len(all_frames)
        if n_src == 0:
            raise RuntimeError(f"empty video: {video_path}")

        # --- Build per-segment frame index lists ---
        if num_segments <= 1:
            # Single-clip path (default). Two sub-paths depending on weights.
            if kept_segment_weights is not None and kept_segments:
                indices = _weighted_kept_segment_indices(
                    n_src, fps, kept_segments, kept_segment_weights,
                    self.frames_per_clip,
                )
            else:
                indices = _kept_segment_indices(
                    n_src, fps, kept_segments or [],
                    self.frames_per_clip, self.frame_step,
                )
            seg_indices_list = [indices]
        else:
            # Multi-clip path: build num_segments frame-index lists.
            if kept_segment_weights is not None and kept_segments:
                seg_indices_list = _multi_clip_weighted_kept_indices(
                    n_src, fps, kept_segments, kept_segment_weights,
                    self.frames_per_clip, self.frame_step, num_segments,
                )
            else:
                seg_indices_list = _multi_clip_kept_indices(
                    n_src, fps, kept_segments or [],
                    self.frames_per_clip, self.frame_step, num_segments,
                )

        # --- For each clip, decode frames and build num_views_per_segment spatial crops ---
        clips: list[list[torch.Tensor]] = []
        for seg_indices in seg_indices_list:
            seg_frames = _gather_frames(all_frames, seg_indices)
            seg_resized = _resize_short_side(seg_frames, self.resolution)
            if num_views_per_segment <= 1:
                views_np = [_center_crop(seg_resized, self.resolution)]
            else:
                views_np = _spatial_views(seg_resized, self.resolution, num_views_per_segment)
            view_tensors = [_to_tensor_normalize(v, self.device, self.dtype) for v in views_np]
            clips.append(view_tensors)

        ensemble_probs = self._forward_clips(clips)
        return self._top5_dict(ensemble_probs)

    @torch.inference_mode()
    def predict_full_with_clip_weighting(
        self,
        video_path: str,
        sidecar: dict,
    ) -> dict:
        """Same frame sampling as predict_full (uniform 4×3 across full timeline)
        but the per-clip softmax-average is weighted by per-clip mean VLM importance.

        Implements the ``vlm-aggregate`` condition: never loses video coverage
        (always sees the full timeline at paper-protocol resolution), but
        down-weights the contribution of clips that land mostly on low-importance
        filler regions. At worst it's equal to predict_full (uniform weights);
        at best it filters out misleading low-quality clip predictions.
        """
        with av.open(video_path) as c:
            stream = c.streams.video[0]
            fps = float(stream.average_rate or 25.0)
            all_frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        n_src = len(all_frames)
        if n_src == 0:
            raise RuntimeError(f"empty video: {video_path}")

        # Paper-protocol sampling: num_segments × frames_per_clip × num_views
        seg_indices_list = _sample_segment_indices(
            n_src, self.frames_per_clip, self.frame_step, self.num_segments
        )
        clips = []
        per_clip_weights: list[float] = []  # one per (segment, view) clip
        # Build per-frame importance from sidecar
        sidecar_segs = sidecar.get("segments", []) or []
        for seg_indices in seg_indices_list:
            # Compute mean VLM importance for the frames in this temporal segment
            clip_imp = _mean_importance_for_indices(seg_indices, fps, sidecar_segs)
            seg_frames = _gather_frames(all_frames, seg_indices)
            seg_resized = _resize_short_side(seg_frames, self.resolution)
            views_np = _spatial_views(seg_resized, self.resolution, self.num_views_per_segment)
            view_tensors = [_to_tensor_normalize(v, self.device, self.dtype) for v in views_np]
            clips.append(view_tensors)
            # Same importance weight applies to all spatial views of this segment
            per_clip_weights.extend([clip_imp] * len(view_tensors))

        ensemble_probs = self._forward_clips_weighted(clips, per_clip_weights)
        return self._top5_dict(ensemble_probs)

    @torch.inference_mode()
    def predict_anchored_hybrid(
        self,
        video_path: str,
        sidecar: dict,
        fraction_anchored: float = 0.5,
    ) -> dict:
        """Hybrid clip placement: half clips anchored on importance peaks, half
        evenly-spaced across the timeline. Preserves broader temporal coverage
        than pure ``predict_anchored`` while still shifting compute toward
        informative moments.

        Implements the ``vlm-anchored-hybrid`` condition (or
        ``vlm-anchored-hybrid-h<NN>`` to override fraction_anchored = NN/100).
        """
        with av.open(video_path) as c:
            stream = c.streams.video[0]
            fps = float(stream.average_rate or 25.0)
            all_frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        n_src = len(all_frames)
        if n_src == 0:
            raise RuntimeError(f"empty video: {video_path}")

        sidecar_segs = sidecar.get("segments", []) or []
        starts = _importance_anchored_hybrid_starts(
            n_src, fps, sidecar_segs, self.num_segments,
            self.frames_per_clip, self.frame_step, fraction_anchored,
        )
        clips = []
        for s in starts:
            indices = [min(s + i * self.frame_step, n_src - 1) for i in range(self.frames_per_clip)]
            seg_frames = _gather_frames(all_frames, indices)
            seg_resized = _resize_short_side(seg_frames, self.resolution)
            views_np = _spatial_views(seg_resized, self.resolution, self.num_views_per_segment)
            view_tensors = [_to_tensor_normalize(v, self.device, self.dtype) for v in views_np]
            clips.append(view_tensors)

        ensemble_probs = self._forward_clips(clips)
        return self._top5_dict(ensemble_probs)

    @torch.inference_mode()
    def predict_anchored(
        self,
        video_path: str,
        sidecar: dict,
    ) -> dict:
        """Anchored multi-clip sampling: clip starts at MSS-importance quantiles
        instead of evenly-spaced across the timeline.

        Implements the ``vlm-anchored`` condition. Each clip is a standard
        contiguous frames_per_clip × frame_step window, but the starts are
        chosen via the cumulative-importance CDF (see _importance_quantile_starts)
        so that more clips land on informative regions. Spatial views still 3-crop.
        Aggregation is uniform softmax-mean (no per-clip weighting).
        """
        with av.open(video_path) as c:
            stream = c.streams.video[0]
            fps = float(stream.average_rate or 25.0)
            all_frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        n_src = len(all_frames)
        if n_src == 0:
            raise RuntimeError(f"empty video: {video_path}")

        sidecar_segs = sidecar.get("segments", []) or []
        starts = _importance_quantile_starts(
            n_src, fps, sidecar_segs, self.num_segments,
            self.frames_per_clip, self.frame_step,
        )
        clips = []
        for s in starts:
            indices = [min(s + i * self.frame_step, n_src - 1) for i in range(self.frames_per_clip)]
            seg_frames = _gather_frames(all_frames, indices)
            seg_resized = _resize_short_side(seg_frames, self.resolution)
            views_np = _spatial_views(seg_resized, self.resolution, self.num_views_per_segment)
            view_tensors = [_to_tensor_normalize(v, self.device, self.dtype) for v in views_np]
            clips.append(view_tensors)

        ensemble_probs = self._forward_clips(clips)
        return self._top5_dict(ensemble_probs)

    @torch.inference_mode()
    def _forward_clips_weighted(
        self, clips: list[list[torch.Tensor]], per_clip_weights: list[float]
    ) -> torch.Tensor:
        """Like _forward_clips but the final softmax-mean across (segment × view)
        clips is weighted by per_clip_weights (one weight per spatial view).

        per_clip_weights: flat list of length sum(len(views) for views in clips),
        ordered as [(seg0, view0), (seg0, view1), ..., (seg1, view0), ...].
        """
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            outputs = self.encoder(clips, clip_indices=None)
            # outputs is a flat list of length sum(len(v) for v in clips), one tensor per (seg, view).
            # Normalize weights → strictly positive, sum>0; if all-zero fall back to uniform.
            w = torch.tensor(per_clip_weights, dtype=torch.float32, device=outputs[0].device)
            if w.sum().item() <= 0.0:
                w = torch.ones_like(w)
            w = w / w.sum()
            per_probe = []
            for c in self.classifiers:
                probs_list = [F.softmax(c(o), dim=1) for o in outputs]
                stacked = torch.stack(probs_list, dim=0)  # (num_clips_flat, B, num_classes)
                w_reshaped = w.to(stacked.dtype).view(-1, 1, 1)
                weighted = (stacked * w_reshaped).sum(dim=0)  # (B, num_classes)
                per_probe.append(weighted)
            ensemble = sum(per_probe) / len(per_probe)
        return ensemble.float()

    # ---- output formatting ----

    @staticmethod
    def _top5_dict(probs: torch.Tensor) -> dict:
        topv, topi = torch.topk(probs[0], 5)
        return {
            "top5_label_ids": topi.tolist(),
            "top5_probs": topv.tolist(),
        }
