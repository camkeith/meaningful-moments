"""
Pseudo-label generation from MSS extraction results.

Converts MSS results to binary importance labels (Important/Unimportant).
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from .extraction import MSSResult
from .segments import Segment

logger = logging.getLogger(__name__)


class ImportanceLabel(Enum):
    """Segment importance labels (binary)."""

    IMPORTANT = "important"      # Segment kept in MSS (necessary for classification)
    UNIMPORTANT = "unimportant"  # Segment removed from MSS (not necessary)


@dataclass
class SegmentLabel:
    """Labeled segment with importance score.

    Attributes:
        segment: The video segment
        frequency: Inclusion frequency (1.0 if kept, 0.0 if removed)
        label: Importance label
        weight: Soft weight for training (equals frequency)
        removal_score: P(YES) when this segment was removed
        drop: Confidence drop from baseline when removed
    """

    segment: Segment
    frequency: float
    label: ImportanceLabel
    weight: float
    removal_score: Optional[float] = None
    drop: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "index": self.segment.index,
            "start_s": round(self.segment.start_s, 3),
            "end_s": round(self.segment.end_s, 3),
            "frequency": round(self.frequency, 4),
            "label": self.label.value,
            "weight": round(self.weight, 4),
        }
        if self.removal_score is not None:
            result["removal_score"] = round(self.removal_score, 4)
        if self.drop is not None:
            result["drop"] = round(self.drop, 4)
        return result


def assign_labels(
    segments: List[Segment],
    inclusion_frequencies: Dict[int, float],
    kept_indices: Optional[set] = None,
) -> List[SegmentLabel]:
    """Assign importance labels to segments.

    Binary labeling based on MSS membership:
    - Important: segment was kept in MSS
    - Unimportant: segment was removed from MSS

    When kept_indices is provided (direct scoring mode), it determines the
    binary label directly — the model's minimum_sufficient_set. When None
    (iterative MSS), segments with frequency > 0 are labeled important.

    The continuous weight (from inclusion_frequencies) is always stored
    regardless of which method determines the binary label.

    Args:
        segments: All video segments
        inclusion_frequencies: Per-segment inclusion frequency (or importance weight)
        kept_indices: If provided, segment indices to label as important
                     (overrides frequency-based logic)

    Returns:
        List of SegmentLabel objects
    """
    labels = []

    for seg in segments:
        freq = inclusion_frequencies.get(seg.index, 0.0)

        if kept_indices is not None:
            is_important = seg.index in kept_indices
        else:
            is_important = freq > 0

        label = ImportanceLabel.IMPORTANT if is_important else ImportanceLabel.UNIMPORTANT

        labels.append(
            SegmentLabel(
                segment=seg,
                frequency=freq,
                label=label,
                weight=freq,
            )
        )

    return labels


def summarize_labels(labels: List[SegmentLabel]) -> dict:
    """Summarize label distribution.

    Args:
        labels: List of segment labels

    Returns:
        Summary statistics
    """
    important = [l for l in labels if l.label == ImportanceLabel.IMPORTANT]
    unimportant = [l for l in labels if l.label == ImportanceLabel.UNIMPORTANT]

    return {
        "total_segments": len(labels),
        "important_count": len(important),
        "unimportant_count": len(unimportant),
        "important_ratio": round(len(important) / len(labels), 3) if labels else 0,
        "unimportant_ratio": round(len(unimportant) / len(labels), 3) if labels else 0,
    }


@dataclass
class VideoLabels:
    """Complete labeled output for a video.

    Attributes:
        video_id: Video identifier
        video_path: Path to video file
        action_label: Ground-truth action label
        segments: Labeled segments
        mss_result: Raw MSS extraction result
        timestamp: When labels were generated
        metadata: Additional dataset-specific metadata (e.g., SSv2 template, placeholders)
    """

    video_id: str
    video_path: str
    action_label: str
    segments: List[SegmentLabel]
    mss_result: MSSResult
    timestamp: str = ""
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        summary = summarize_labels(self.segments)

        result = {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "action_label": self.action_label,
            "timestamp": self.timestamp,
            "summary": summary,
            "segments": [s.to_dict() for s in self.segments],
            "mss_result": self.mss_result.to_dict(),
        }

        # Include metadata if present
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_mss_result(
        cls,
        mss_result: MSSResult,
        video_path: Path,
        action_label: str,
        metadata: Optional[Dict] = None,
    ) -> "VideoLabels":
        """Create VideoLabels from MSS extraction result.

        For direct scoring results, binary labels come from the model's
        minimum_sufficient_set (kept_indices). For iterative MSS, labels
        come from inclusion frequency > 0.

        Args:
            mss_result: MSS extraction result
            video_path: Path to video file
            action_label: Ground-truth action label
            metadata: Additional dataset-specific metadata (e.g., SSv2 template)

        Returns:
            VideoLabels instance
        """
        # For direct scoring, use the model's kept_indices for binary labels
        kept_indices = None
        if (mss_result.mss_runs
                and mss_result.mss_runs[0].terminated_reason == "direct_scoring"):
            kept_indices = mss_result.mss_runs[0].kept_indices

        segment_labels = assign_labels(
            mss_result.segments, mss_result.inclusion_frequencies,
            kept_indices=kept_indices,
        )

        return cls(
            video_id=mss_result.video_id,
            video_path=str(video_path),
            action_label=action_label,
            segments=segment_labels,
            mss_result=mss_result,
            metadata=metadata,
        )


def save_video_labels(labels: VideoLabels, output_path: Path) -> None:
    """Save video labels to JSON file.

    Args:
        labels: VideoLabels to save
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved labels to {output_path}")


def load_video_labels(input_path: Path) -> dict:
    """Load video labels from JSON file.

    Args:
        input_path: Input file path

    Returns:
        Dictionary representation of labels
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_important_indices(labels: List[SegmentLabel]) -> set:
    """Get indices of Important segments."""
    return {l.segment.index for l in labels if l.label == ImportanceLabel.IMPORTANT}


def get_unimportant_indices(labels: List[SegmentLabel]) -> set:
    """Get indices of Unimportant segments."""
    return {l.segment.index for l in labels if l.label == ImportanceLabel.UNIMPORTANT}
