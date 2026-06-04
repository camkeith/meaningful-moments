"""
Closed-set SSv2 classification with Qwen3-VL.

Evaluates whether MSS direct-scoring preserves class-discriminative signal
by running the same Qwen3-VL classifier on (a) full videos and (b) concatenated
MSS-kept segments, then comparing top-1 / top-5 accuracy.
"""

from .prompt import SSV2_LABELS, build_classification_messages, prompt_hash
from .mss_loader import MSSRecord, load_mss_record
from .video_io import load_full_frames, load_mss_kept_frames
from .parser import parse_top5, PARSE_FAILED

__all__ = [
    "SSV2_LABELS",
    "build_classification_messages",
    "prompt_hash",
    "MSSRecord",
    "load_mss_record",
    "load_full_frames",
    "load_mss_kept_frames",
    "parse_top5",
    "PARSE_FAILED",
]
