"""
Video masking operators for counterfactual analysis.

Provides operators to mask (remove visual content from) specified segments
while preserving video structure (duration, FPS).
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set, Tuple

import av
import numpy as np
import torch

from .segments import Segment, segments_to_frame_ranges


class MaskOperator(Enum):
    """Masking operator types."""

    BLUR = "blur"
    FREEZE = "freeze"
    BLACK = "black"
    MEAN = "mean"
    CUT = "cut"  # Remove segments entirely (concatenate remaining)


@dataclass
class MaskConfig:
    """Configuration for masking operation.

    Attributes:
        operator: Type of masking to apply
        blur_kernel_size: Gaussian blur kernel size (for BLUR operator)
        blur_sigma: Gaussian blur sigma (for BLUR operator)
        audio_policy: "remove" to strip audio, "keep" to preserve
    """

    operator: MaskOperator = MaskOperator.BLUR
    blur_kernel_size: int = 51
    blur_sigma: float = 20.0
    audio_policy: str = "remove"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "operator": self.operator.value,
            "blur_kernel_size": self.blur_kernel_size,
            "blur_sigma": self.blur_sigma,
            "audio_policy": self.audio_policy,
        }


def _apply_gaussian_blur(frame: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to a frame using OpenCV-style convolution.

    Args:
        frame: HxWxC numpy array (uint8)
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation of Gaussian

    Returns:
        Blurred frame as numpy array
    """
    try:
        import cv2
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    except ImportError:
        # Fallback: simple box blur approximation
        from scipy.ndimage import uniform_filter
        return uniform_filter(frame, size=(kernel_size // 3, kernel_size // 3, 1)).astype(np.uint8)


def _compute_mean_frame(frames: List[np.ndarray]) -> np.ndarray:
    """Compute mean frame from a list of frames.

    Args:
        frames: List of HxWxC numpy arrays

    Returns:
        Mean frame as numpy array (uint8)
    """
    if not frames:
        raise ValueError("Cannot compute mean of empty frame list")
    stacked = np.stack(frames, axis=0).astype(np.float32)
    mean = np.mean(stacked, axis=0)
    return mean.astype(np.uint8)


def compute_mask_pattern_hash(
    removed_indices: Set[int],
    total_segments: int,
) -> str:
    """Compute a hash for a mask pattern (for caching).

    Args:
        removed_indices: Set of segment indices that are masked
        total_segments: Total number of segments

    Returns:
        Hex string hash of the pattern
    """
    # Create binary pattern string
    pattern = "".join(
        "1" if i in removed_indices else "0" for i in range(total_segments)
    )
    return hashlib.md5(pattern.encode()).hexdigest()[:16]


def mask_frames_in_memory(
    frames: List[np.ndarray],
    mask_frame_ranges: List[Tuple[int, int]],
    config: MaskConfig,
) -> List[np.ndarray]:
    """Apply masking to frames in memory.

    Args:
        frames: List of HxWxC numpy arrays
        mask_frame_ranges: List of (start, end) frame ranges to mask (end exclusive)
        config: Masking configuration

    Returns:
        List of frames with masking applied.
        For CUT operator, returns only non-masked frames (shorter video).
        For other operators, returns same-length video with masked frames replaced.
    """
    if not frames:
        return frames

    # Build set of frame indices to mask
    mask_indices = set()
    for start, end in mask_frame_ranges:
        mask_indices.update(range(start, end))

    # CUT operator: return only non-masked frames (concatenate kept segments)
    if config.operator == MaskOperator.CUT:
        return [frame for i, frame in enumerate(frames) if i not in mask_indices]

    # Compute replacement frame(s) based on operator
    if config.operator == MaskOperator.BLACK:
        h, w, c = frames[0].shape
        replacement = np.zeros((h, w, c), dtype=np.uint8)
    elif config.operator == MaskOperator.MEAN:
        # Use mean of all frames
        replacement = _compute_mean_frame(frames)
    elif config.operator == MaskOperator.BLUR:
        # Will blur each masked frame individually
        replacement = None
    elif config.operator == MaskOperator.FREEZE:
        # Will use last unmasked frame before each masked segment
        replacement = None
    else:
        raise ValueError(f"Unknown operator: {config.operator}")

    result = []
    last_unmasked = frames[0].copy() if frames else None

    for i, frame in enumerate(frames):
        if i in mask_indices:
            if config.operator == MaskOperator.BLUR:
                masked = _apply_gaussian_blur(frame, config.blur_kernel_size, config.blur_sigma)
            elif config.operator == MaskOperator.FREEZE:
                masked = last_unmasked.copy() if last_unmasked is not None else frame
            else:
                masked = replacement.copy()
            result.append(masked)
        else:
            result.append(frame)
            last_unmasked = frame.copy()

    return result


def read_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float, int, int]:
    """Read all frames from a video file.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (frames, fps, width, height)
    """
    frames = []
    fps = 30.0
    width, height = 0, 0

    with av.open(str(video_path)) as container:
        stream = next(st for st in container.streams if st.type == "video")
        if stream.average_rate:
            fps = float(stream.average_rate)
        width = stream.width
        height = stream.height

        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)

    return frames, fps, width, height


def write_video_frames(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float,
    codec: str = "libx264",
) -> None:
    """Write frames to a video file.

    Args:
        frames: List of HxWxC numpy arrays (RGB)
        output_path: Output video path
        fps: Frames per second
        codec: Video codec (default libx264)
    """
    if not frames:
        raise ValueError("Cannot write empty frame list")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape

    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"

        for img in frames:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)


def mask_video(
    video_path: Path,
    segments: List[Segment],
    removed_indices: Set[int],
    config: MaskConfig,
    output_path: Optional[Path] = None,
) -> Tuple[Optional[Path], List[np.ndarray]]:
    """Create a masked version of a video.

    Args:
        video_path: Input video path
        segments: All video segments
        removed_indices: Indices of segments to mask
        config: Masking configuration
        output_path: Optional output path (if None, returns frames only)

    Returns:
        Tuple of (output_path or None, masked_frames)
    """
    # Read video
    frames, fps, width, height = read_video_frames(video_path)
    total_frames = len(frames)

    # Get segments to mask
    segments_to_mask = [seg for seg in segments if seg.index in removed_indices]

    # Convert to frame ranges
    frame_ranges = segments_to_frame_ranges(segments_to_mask, fps, total_frames)

    # Apply masking
    masked_frames = mask_frames_in_memory(frames, frame_ranges, config)

    # Write if output path provided
    if output_path is not None:
        write_video_frames(masked_frames, output_path, fps)

    return output_path, masked_frames


def get_kept_segment_mask(
    all_segments: List[Segment],
    kept_indices: Set[int],
) -> Set[int]:
    """Get indices of segments to mask (complement of kept).

    Args:
        all_segments: All segments
        kept_indices: Indices of segments to keep

    Returns:
        Set of indices to mask (remove)
    """
    all_indices = {seg.index for seg in all_segments}
    return all_indices - kept_indices


@dataclass
class MaskingInfo:
    """Information about masking operation for oracle context.

    Attributes:
        total_segments: Total number of segments in original video
        kept_segments: Number of segments kept
        removed_segments: Number of segments removed
        operator: Masking operator used
        is_cut: Whether segments were physically removed (CUT operator)
    """

    total_segments: int
    kept_segments: int
    removed_segments: int
    operator: MaskOperator
    is_cut: bool

    @property
    def removal_note(self) -> Optional[str]:
        """Generate a note about removed segments for the oracle.

        Returns:
            Note string if segments were removed with CUT, None otherwise.
        """
        if not self.is_cut or self.removed_segments == 0:
            return None
        return (
            f"Note: {self.removed_segments} of {self.total_segments} video segments "
            f"have been removed. The video you see contains only {self.kept_segments} "
            f"concatenated segments from the original video."
        )


def compute_masking_info(
    all_segments: List[Segment],
    kept_indices: Set[int],
    config: MaskConfig,
) -> MaskingInfo:
    """Compute masking information for oracle context.

    Args:
        all_segments: All video segments
        kept_indices: Indices of segments being kept
        config: Masking configuration

    Returns:
        MaskingInfo with removal details
    """
    total = len(all_segments)
    kept = len(kept_indices)
    removed = total - kept

    return MaskingInfo(
        total_segments=total,
        kept_segments=kept,
        removed_segments=removed,
        operator=config.operator,
        is_cut=config.operator == MaskOperator.CUT,
    )
