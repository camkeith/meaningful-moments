"""
Video segmentation utilities for MSS extraction.

Divides videos into fixed-length temporal segments for counterfactual analysis.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import av


@dataclass
class Segment:
    """A temporal segment of a video.

    Attributes:
        index: 0-based segment index
        start_s: Start time in seconds
        end_s: End time in seconds
    """

    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Segment duration in seconds."""
        return self.end_s - self.start_s

    @property
    def mid_s(self) -> float:
        """Midpoint time in seconds."""
        return (self.start_s + self.end_s) / 2

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        """Create from dictionary."""
        return cls(index=d["index"], start_s=d["start_s"], end_s=d["end_s"])

    def __hash__(self) -> int:
        return hash((self.index, self.start_s, self.end_s))


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using PyAV.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds

    Raises:
        ValueError: If video cannot be opened or has no duration
    """
    try:
        with av.open(str(video_path)) as container:
            if container.duration is None:
                raise ValueError(f"Video has no duration: {video_path}")
            return container.duration / 1e6  # microseconds to seconds
    except (av.error.InvalidDataError, av.AVError, OSError) as e:
        raise ValueError(f"Cannot open video {video_path}: {e}") from e


def get_video_fps(video_path: Path) -> float:
    """Get video FPS using PyAV.

    Args:
        video_path: Path to video file

    Returns:
        Frames per second

    Raises:
        ValueError: If video cannot be opened
    """
    try:
        with av.open(str(video_path)) as container:
            stream = next(st for st in container.streams if st.type == "video")
            if stream.average_rate:
                return float(stream.average_rate)
            return 30.0  # fallback
    except (av.error.InvalidDataError, av.AVError, OSError) as e:
        raise ValueError(f"Cannot open video {video_path}: {e}") from e


def segment_video(
    video_path: Path,
    delta_t: float = 0.5,
    duration_s: Optional[float] = None,
) -> List[Segment]:
    """Divide video into fixed-length temporal segments.

    Args:
        video_path: Path to video file
        delta_t: Segment length in seconds (default 0.5s)
        duration_s: Video duration in seconds (computed if not provided)

    Returns:
        List of Segment objects covering the entire video

    Raises:
        ValueError: If delta_t <= 0 or video cannot be processed
    """
    if delta_t <= 0:
        raise ValueError(f"delta_t must be positive, got {delta_t}")

    if duration_s is None:
        duration_s = get_video_duration(video_path)

    n_segments = math.ceil(duration_s / delta_t)
    segments = []

    for i in range(n_segments):
        start_s = i * delta_t
        end_s = min((i + 1) * delta_t, duration_s)
        segments.append(Segment(index=i, start_s=start_s, end_s=end_s))

    return segments


def segments_to_frame_ranges(
    segments: List[Segment],
    fps: float,
    total_frames: Optional[int] = None,
) -> List[tuple]:
    """Convert segments to frame index ranges.

    Args:
        segments: List of Segment objects
        fps: Video frames per second
        total_frames: Total frame count (for clamping final segment)

    Returns:
        List of (start_frame, end_frame) tuples (end exclusive)
    """
    ranges = []
    for seg in segments:
        start_frame = int(seg.start_s * fps)
        end_frame = int(seg.end_s * fps)
        if total_frames is not None:
            end_frame = min(end_frame, total_frames)
        ranges.append((start_frame, end_frame))
    return ranges


def segment_indices_to_set(segments: List[Segment]) -> set:
    """Convert list of segments to set of indices."""
    return {seg.index for seg in segments}


def filter_segments_by_indices(
    all_segments: List[Segment], indices: set
) -> List[Segment]:
    """Filter segments by index set."""
    return [seg for seg in all_segments if seg.index in indices]
