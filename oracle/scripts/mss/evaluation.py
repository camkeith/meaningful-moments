"""
Evaluation utilities for MSS pseudo-labels.

Implements sufficiency tests, causal evidence curves, and robustness controls.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .extraction import MSSResult
from .labeling import (
    ImportanceLabel,
    SegmentLabel,
    get_important_indices,
    get_unimportant_indices,
)
from .masking import (
    MaskConfig,
    compute_mask_pattern_hash,
    get_kept_segment_mask,
    mask_frames_in_memory,
    read_video_frames,
    segments_to_frame_ranges,
)
from .oracle import OracleInterface, get_logit_confidence
from .segments import Segment

logger = logging.getLogger(__name__)


@dataclass
class SufficiencyTestResult:
    """Result from a sufficiency test.

    Attributes:
        test_name: Name of the test
        kept_indices: Indices of segments kept
        masked_indices: Indices of segments masked
        p_yes: P(YES) confidence from oracle (logit-based)
        passed: Whether test passed (interpretation depends on test type)
        oracle_calls: Number of oracle calls made
    """

    test_name: str
    kept_indices: Set[int]
    masked_indices: Set[int]
    p_yes: float
    passed: bool
    oracle_calls: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "kept_indices": sorted(self.kept_indices),
            "masked_indices": sorted(self.masked_indices),
            "p_yes": round(self.p_yes, 3),
            "passed": self.passed,
            "oracle_calls": self.oracle_calls,
        }


@dataclass
class CurvePoint:
    """Single point on a deletion/insertion curve."""

    step: int
    kept_count: int
    p_yes: float
    segment_idx: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "kept_count": self.kept_count,
            "p_yes": round(self.p_yes, 3),
            "segment_idx": self.segment_idx,
        }


@dataclass
class CurveResult:
    """Result from deletion or insertion curve computation."""

    curve_type: str  # "deletion" or "insertion"
    points: List[CurvePoint]
    auc: float
    random_auc: float
    auc_improvement: float

    def to_dict(self) -> dict:
        return {
            "curve_type": self.curve_type,
            "points": [p.to_dict() for p in self.points],
            "auc": round(self.auc, 4),
            "random_auc": round(self.random_auc, 4),
            "auc_improvement": round(self.auc_improvement, 4),
        }


class MSSEvaluator:
    """Evaluates MSS pseudo-label quality.

    Runs sufficiency tests and computes causal evidence curves.
    """

    def __init__(
        self,
        oracle: OracleInterface,
        mask_config: MaskConfig,
    ):
        """Initialize evaluator.

        Args:
            oracle: Oracle interface for verification queries
            mask_config: Masking configuration
        """
        self.oracle = oracle
        self.mask_config = mask_config

    def _create_masked_frames(
        self,
        all_frames: List[np.ndarray],
        segments: List[Segment],
        kept_indices: Set[int],
        fps: float,
    ) -> List[np.ndarray]:
        """Create masked video frames."""
        removed_indices = get_kept_segment_mask(segments, kept_indices)

        if not removed_indices:
            return all_frames

        segments_to_mask = [s for s in segments if s.index in removed_indices]
        frame_ranges = segments_to_frame_ranges(segments_to_mask, fps, len(all_frames))

        return mask_frames_in_memory(all_frames, frame_ranges, self.mask_config)

    def _query_final_for_mask(
        self,
        all_frames: List[np.ndarray],
        segments: List[Segment],
        kept_indices: Set[int],
        action_label: str,
        video_id: str,
        fps: float,
    ) -> Tuple[float, bool, int]:
        """Query oracle with final scoring for a mask pattern.

        Uses single logit-based query with P(YES) > 0.5 threshold.

        Returns:
            Tuple of (p_yes, ok_final, oracle_calls)
        """
        masked_frames = self._create_masked_frames(
            all_frames, segments, kept_indices, fps
        )

        removed_indices = get_kept_segment_mask(segments, kept_indices)
        mask_hash = compute_mask_pattern_hash(removed_indices, len(segments))

        response, p_yes, ok_final = self.oracle.query_final(
            masked_frames, action_label, video_id, mask_hash
        )

        return p_yes, ok_final, 1  # Single query

    def test_keep_important(
        self,
        video_path: Path,
        segments: List[Segment],
        labels: List[SegmentLabel],
        action_label: str,
        video_id: str,
    ) -> SufficiencyTestResult:
        """Test: Keep only Important segments, should still pass.

        Args:
            video_path: Path to video
            segments: All video segments
            labels: Segment labels
            action_label: Action label
            video_id: Video identifier

        Returns:
            SufficiencyTestResult
        """
        all_frames, fps, _, _ = read_video_frames(video_path)

        important_indices = get_important_indices(labels)
        masked_indices = get_kept_segment_mask(segments, important_indices)

        p_yes, passed, calls = self._query_final_for_mask(
            all_frames, segments, important_indices, action_label, video_id, fps
        )

        return SufficiencyTestResult(
            test_name="keep_important",
            kept_indices=important_indices,
            masked_indices=masked_indices,
            p_yes=p_yes,
            passed=passed,
            oracle_calls=calls,
        )

    def test_remove_unimportant(
        self,
        video_path: Path,
        segments: List[Segment],
        labels: List[SegmentLabel],
        action_label: str,
        video_id: str,
        full_p_yes: float,
    ) -> SufficiencyTestResult:
        """Test: Mask only Unimportant segments, should not hurt.

        Args:
            video_path: Path to video
            segments: All video segments
            labels: Segment labels
            action_label: Action label
            video_id: Video identifier
            full_p_yes: P(YES) for full video (for comparison)

        Returns:
            SufficiencyTestResult
        """
        all_frames, fps, _, _ = read_video_frames(video_path)

        unimportant_indices = get_unimportant_indices(labels)
        all_indices = {s.index for s in segments}
        kept_indices = all_indices - unimportant_indices

        p_yes, ok_final, calls = self._query_final_for_mask(
            all_frames, segments, kept_indices, action_label, video_id, fps
        )

        # Passed if p_yes didn't drop significantly
        delta = full_p_yes - p_yes
        passed = delta <= 0.2  # Allow small drop

        return SufficiencyTestResult(
            test_name="remove_unimportant",
            kept_indices=kept_indices,
            masked_indices=unimportant_indices,
            p_yes=p_yes,
            passed=passed,
            oracle_calls=calls,
        )

    def test_keep_unimportant(
        self,
        video_path: Path,
        segments: List[Segment],
        labels: List[SegmentLabel],
        action_label: str,
        video_id: str,
    ) -> SufficiencyTestResult:
        """Test: Keep only Unimportant segments, should fail.

        Args:
            video_path: Path to video
            segments: All video segments
            labels: Segment labels
            action_label: Action label
            video_id: Video identifier

        Returns:
            SufficiencyTestResult
        """
        all_frames, fps, _, _ = read_video_frames(video_path)

        unimportant_indices = get_unimportant_indices(labels)

        if not unimportant_indices:
            # No unimportant segments - test N/A
            return SufficiencyTestResult(
                test_name="keep_unimportant",
                kept_indices=set(),
                masked_indices=set(),
                p_yes=0.0,
                passed=True,  # Vacuously true
                oracle_calls=0,
            )

        masked_indices = get_kept_segment_mask(segments, unimportant_indices)

        p_yes, ok_final, calls = self._query_final_for_mask(
            all_frames, segments, unimportant_indices, action_label, video_id, fps
        )

        # Passed if oracle FAILS on unimportant-only (p_yes is low)
        passed = p_yes < 0.5

        return SufficiencyTestResult(
            test_name="keep_unimportant",
            kept_indices=unimportant_indices,
            masked_indices=masked_indices,
            p_yes=p_yes,
            passed=passed,
            oracle_calls=calls,
        )

    def compute_deletion_curve(
        self,
        video_path: Path,
        segments: List[Segment],
        frequencies: Dict[int, float],
        action_label: str,
        video_id: str,
        max_steps: int = 20,
    ) -> CurveResult:
        """Compute deletion curve (remove low→high importance).

        Args:
            video_path: Path to video
            segments: All video segments
            frequencies: Per-segment frequencies
            action_label: Action label
            video_id: Video identifier
            max_steps: Maximum number of curve points

        Returns:
            CurveResult with deletion curve
        """
        all_frames, fps, _, _ = read_video_frames(video_path)

        # Sort segments by frequency ascending (remove least important first)
        sorted_indices = sorted(frequencies.keys(), key=lambda i: frequencies[i])

        # Start with all segments
        kept = set(sorted_indices)
        points = []

        # Initial point (all segments)
        p_yes, _, _ = self._query_final_for_mask(
            all_frames, segments, kept, action_label, video_id, fps
        )
        points.append(CurvePoint(step=0, kept_count=len(kept), p_yes=p_yes))

        # Progressively remove segments
        step_size = max(1, len(sorted_indices) // max_steps)
        for step, idx in enumerate(sorted_indices[::step_size], 1):
            if len(kept) <= 1:
                break

            # Remove segment(s)
            to_remove = set(sorted_indices[(step - 1) * step_size : step * step_size])
            kept = kept - to_remove

            if not kept:
                break

            p_yes, _, _ = self._query_final_for_mask(
                all_frames, segments, kept, action_label, video_id, fps
            )
            points.append(
                CurvePoint(
                    step=step, kept_count=len(kept), p_yes=p_yes, segment_idx=idx
                )
            )

        # Compute AUC (normalized)
        auc = self._compute_auc(points, len(segments))
        random_auc = 0.5  # Random baseline expectation

        return CurveResult(
            curve_type="deletion",
            points=points,
            auc=auc,
            random_auc=random_auc,
            auc_improvement=auc - random_auc,
        )

    def compute_insertion_curve(
        self,
        video_path: Path,
        segments: List[Segment],
        frequencies: Dict[int, float],
        action_label: str,
        video_id: str,
        max_steps: int = 20,
    ) -> CurveResult:
        """Compute insertion curve (add high→low importance).

        Args:
            video_path: Path to video
            segments: All video segments
            frequencies: Per-segment frequencies
            action_label: Action label
            video_id: Video identifier
            max_steps: Maximum number of curve points

        Returns:
            CurveResult with insertion curve
        """
        all_frames, fps, _, _ = read_video_frames(video_path)

        # Sort segments by frequency descending (add most important first)
        sorted_indices = sorted(
            frequencies.keys(), key=lambda i: frequencies[i], reverse=True
        )

        # Start with no segments
        kept: Set[int] = set()
        points = []

        # Initial point (no segments - all masked)
        p_yes = 0.0  # Assume fails with nothing
        points.append(CurvePoint(step=0, kept_count=0, p_yes=p_yes))

        # Progressively add segments
        step_size = max(1, len(sorted_indices) // max_steps)
        for step, idx in enumerate(sorted_indices[::step_size], 1):
            # Add segment(s)
            to_add = set(sorted_indices[(step - 1) * step_size : step * step_size])
            kept = kept | to_add

            p_yes, _, _ = self._query_final_for_mask(
                all_frames, segments, kept, action_label, video_id, fps
            )
            points.append(
                CurvePoint(
                    step=step, kept_count=len(kept), p_yes=p_yes, segment_idx=idx
                )
            )

        # Compute AUC (normalized)
        auc = self._compute_auc(points, len(segments))
        random_auc = 0.5

        return CurveResult(
            curve_type="insertion",
            points=points,
            auc=auc,
            random_auc=random_auc,
            auc_improvement=auc - random_auc,
        )

    def _compute_auc(self, points: List[CurvePoint], total_segments: int) -> float:
        """Compute normalized AUC from curve points."""
        if len(points) < 2:
            return 0.0

        # Normalize x to [0, 1] based on kept_count / total
        # Use trapezoidal rule
        auc = 0.0
        for i in range(1, len(points)):
            x0 = points[i - 1].kept_count / total_segments
            x1 = points[i].kept_count / total_segments
            y0 = points[i - 1].p_yes
            y1 = points[i].p_yes
            auc += 0.5 * (y0 + y1) * abs(x1 - x0)

        return auc


def compute_label_iou(
    labels_a: List[SegmentLabel],
    labels_b: List[SegmentLabel],
    label_type: ImportanceLabel = ImportanceLabel.IMPORTANT,
) -> float:
    """Compute IoU between two label sets for a given label type.

    Args:
        labels_a: First label set
        labels_b: Second label set
        label_type: Label type to compare

    Returns:
        Intersection over Union
    """
    set_a = {l.segment.index for l in labels_a if l.label == label_type}
    set_b = {l.segment.index for l in labels_b if l.label == label_type}

    if not set_a and not set_b:
        return 1.0  # Both empty

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def compute_frequency_correlation(
    freq_a: Dict[int, float],
    freq_b: Dict[int, float],
) -> float:
    """Compute Spearman correlation between frequency dictionaries.

    Args:
        freq_a: First frequency dict
        freq_b: Second frequency dict

    Returns:
        Spearman correlation coefficient
    """
    from scipy.stats import spearmanr

    common_keys = set(freq_a.keys()) & set(freq_b.keys())
    if len(common_keys) < 2:
        return 0.0

    values_a = [freq_a[k] for k in sorted(common_keys)]
    values_b = [freq_b[k] for k in sorted(common_keys)]

    corr, _ = spearmanr(values_a, values_b)
    return corr if not np.isnan(corr) else 0.0


@dataclass
class WrongLabelControlResult:
    """Result from wrong-label control test.

    Attributes:
        correct_label: Original correct label
        wrong_label: Confusable wrong label used
        p_yes_correct: P(YES) with correct label
        p_yes_wrong: P(YES) with wrong label
        passed: True if wrong label gets low p_yes (< 0.5)
    """

    correct_label: str
    wrong_label: str
    p_yes_correct: float
    p_yes_wrong: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "correct_label": self.correct_label,
            "wrong_label": self.wrong_label,
            "p_yes_correct": round(self.p_yes_correct, 3),
            "p_yes_wrong": round(self.p_yes_wrong, 3),
            "passed": self.passed,
            "delta": round(self.p_yes_correct - self.p_yes_wrong, 3),
        }


def run_wrong_label_control(
    evaluator: "MSSEvaluator",
    video_path: Path,
    segments: List[Segment],
    correct_label: str,
    wrong_label: str,
    video_id: str,
) -> WrongLabelControlResult:
    """Run wrong-label control test.

    Tests that oracle gives low vote_yes when asked about a wrong label.

    Args:
        evaluator: MSSEvaluator instance
        video_path: Path to video
        segments: Video segments
        correct_label: Correct action label
        wrong_label: Confusable wrong label
        video_id: Video identifier

    Returns:
        WrongLabelControlResult
    """
    all_frames, fps, _, _ = read_video_frames(video_path)
    all_indices = {s.index for s in segments}

    # Query with correct label
    p_yes_correct, _, _ = evaluator._query_final_for_mask(
        all_frames, segments, all_indices, correct_label, video_id, fps
    )

    # Query with wrong label
    p_yes_wrong, _, _ = evaluator._query_final_for_mask(
        all_frames, segments, all_indices, wrong_label, f"{video_id}_wrong", fps
    )

    # Passed if wrong label gets low p_yes
    passed = p_yes_wrong < 0.5

    return WrongLabelControlResult(
        correct_label=correct_label,
        wrong_label=wrong_label,
        p_yes_correct=p_yes_correct,
        p_yes_wrong=p_yes_wrong,
        passed=passed,
    )
