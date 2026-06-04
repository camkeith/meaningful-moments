"""
Video -> frame-sequence adapters for the SSv2 classifier.

Two modes:
  - Full video:  read all frames from disk.
  - MSS kept:    read all frames, then keep only the frames that fall inside
                 the time ranges of the MSS-kept segments, in order (CUT + concat).
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

# Reuse existing MSS utilities (no duplication).
import sys
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # oracle/scripts/
from mss.masking import (  # noqa: E402
    MaskConfig,
    MaskOperator,
    mask_frames_in_memory,
    read_video_frames,
)
from mss.segments import Segment, segments_to_frame_ranges  # noqa: E402

from .mss_loader import MSSRecord


_CUT_CONFIG = MaskConfig(operator=MaskOperator.CUT)


def load_full_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    """Load all frames from a video file.

    Returns:
        (frames, fps)
    """
    frames, fps, _w, _h = read_video_frames(video_path)
    return frames, fps


def load_mss_kept_frames(record: MSSRecord) -> Tuple[List[np.ndarray], float]:
    """Build a CUT+concat frame sequence containing only the MSS-kept segments.

    If kept_indices == all segment indices, returns the full video unchanged.

    Returns:
        (kept_frames, fps)
    """
    frames, fps, _w, _h = read_video_frames(record.video_path)

    all_segs = [Segment(index=s["index"], start_s=s["start_s"], end_s=s["end_s"])
                for s in record.segments]
    all_indices = {s.index for s in all_segs}
    kept = record.kept_indices & all_indices
    removed_indices = all_indices - kept

    # Nothing to remove: pass through.
    if not removed_indices:
        return frames, fps

    removed_segs = [s for s in all_segs if s.index in removed_indices]
    frame_ranges = segments_to_frame_ranges(removed_segs, fps, len(frames))
    kept_frames = mask_frames_in_memory(frames, frame_ranges, _CUT_CONFIG)
    return kept_frames, fps
