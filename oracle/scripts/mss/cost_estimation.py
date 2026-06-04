"""Cost estimation for token-budget bin-packed batching.

The per-video VRAM cost during a Qwen-VL forward pass is dominated by:

    visual_tokens = n_frames * tokens_per_frame
                  = (duration * source_fps) * (resized_h * resized_w) / 28**2

plus a fixed prompt overhead. The current MSS pipeline pre-decodes ALL native
frames in masking.read_video_frames() and passes them to Qwen as pil_frames —
qwen-vl-utils does not subsample further when given a frame list. So the right
frame count is `duration * source_fps`, NOT a target sampling rate.

Output (generation) cost is small relative to input for direct scoring and is
folded into the overhead constant.

Resolution is clamped by Qwen's smart-resize using min_pixels / max_pixels,
then snapped to multiples of the 28-pixel effective stride (14 patch * 2 merge).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import av


# Qwen-VL processor defaults — keep in sync with qwen_vl_utils.process_vision_info.
DEFAULT_PATCH_STRIDE = 28          # patch_size (14) * merge_size (2)
DEFAULT_MAX_PIXELS = 16384 * DEFAULT_PATCH_STRIDE * DEFAULT_PATCH_STRIDE  # 12,845,056
DEFAULT_MIN_PIXELS = 4 * DEFAULT_PATCH_STRIDE * DEFAULT_PATCH_STRIDE      #     3,136

# Rough fixed cost added per video for the system prompt + user turn + generation.
# Direct-scoring prompts (~5.7k chars / ~1.5k tokens) plus ~800 output tokens for
# a 20-segment JSON. Tune via --token-budget — this constant doesn't need to be
# exact since visual tokens dominate.
DEFAULT_PROMPT_TOKEN_OVERHEAD = 2300

# Hard cap on frames per video — guards against pathological inputs (e.g. a
# 60s video at 60 fps would otherwise estimate at 3600 frames).
DEFAULT_MAX_FRAMES = 768


def probe_video_dimensions(video_path: Path) -> Tuple[float, int, int, float]:
    """Return (duration_s, width, height, fps) from the container header.

    Does NOT decode frames — much cheaper than read_video_frames().
    """
    with av.open(str(video_path)) as container:
        stream = next(st for st in container.streams if st.type == "video")

        if container.duration is not None:
            duration = container.duration / 1e6
        elif stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            raise ValueError(f"Cannot determine duration: {video_path}")

        fps = float(stream.average_rate) if stream.average_rate else 30.0
        return duration, int(stream.width), int(stream.height), fps


def smart_resize(
    width: int,
    height: int,
    *,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    patch_stride: int = DEFAULT_PATCH_STRIDE,
) -> Tuple[int, int]:
    """Mirror qwen_vl_utils.smart_resize: clamp to [min, max] pixels, snap to stride."""
    pixels = width * height
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    else:
        scale = 1.0

    w = max(patch_stride, round(width * scale / patch_stride) * patch_stride)
    h = max(patch_stride, round(height * scale / patch_stride) * patch_stride)
    return w, h


def estimate_input_tokens(
    duration_s: float,
    width: int,
    height: int,
    fps: float,
    *,
    patch_stride: int = DEFAULT_PATCH_STRIDE,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    prompt_token_overhead: int = DEFAULT_PROMPT_TOKEN_OVERHEAD,
) -> int:
    """Estimate total input tokens for a single video forward pass.

    Uses the native source fps because the pipeline pre-decodes all frames
    (masking.read_video_frames) and passes them straight to Qwen with no
    further subsampling.
    """
    w_resized, h_resized = smart_resize(
        width, height,
        min_pixels=min_pixels, max_pixels=max_pixels,
        patch_stride=patch_stride,
    )
    tokens_per_frame = (w_resized * h_resized) // (patch_stride * patch_stride)
    n_frames = min(max_frames, max(1, round(duration_s * fps)))
    visual_tokens = n_frames * tokens_per_frame
    return visual_tokens + prompt_token_overhead
