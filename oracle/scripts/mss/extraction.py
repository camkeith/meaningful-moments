"""
MSS (Minimal Sufficient Subset) extraction algorithm.

Implements greedy segment removal with oracle verification to find
minimal sufficient subsets for action classification.
"""

import json
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .masking import (
    MaskConfig,
    MaskingInfo,
    MaskOperator,
    compute_mask_pattern_hash,
    compute_masking_info,
    get_kept_segment_mask,
    mask_frames_in_memory,
    read_video_frames,
    segments_to_frame_ranges,
)
from .oracle import OracleConfig, OracleInterface, OracleResponse, get_logit_confidence, is_skip_flagged
from .segments import Segment, get_video_fps, segment_video

logger = logging.getLogger(__name__)


# Progress callback type for external progress tracking
ProgressCallback = Callable[[str, int, int, Optional[dict]], None]


@dataclass
class SegmentScore:
    """Score information for a segment during score-based removal.

    Attributes:
        segment_index: Index of the segment
        removal_score: Confidence score when this segment is removed (0.0 if NO)
        drop: Confidence drop from baseline (baseline - removal_score)
        is_removable: Whether the segment can be removed
        response: The oracle response for this segment
    """

    segment_index: int
    removal_score: float
    drop: float
    is_removable: bool
    response: "OracleResponse"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "segment_index": self.segment_index,
            "removal_score": round(self.removal_score, 4),
            "drop": round(self.drop, 4),
            "is_removable": self.is_removable,
        }


@dataclass
class MSSRunResult:
    """Result from a single MSS extraction run.

    Attributes:
        kept_indices: Set of segment indices in the MSS
        removal_order: Order in which segments were removed
        oracle_calls: Number of oracle calls made
        terminated_reason: Why the search terminated
        search_responses: Oracle responses during search (segment_idx -> response)
        iteration_scores: Per-iteration score information (for score-based strategy)
    """

    kept_indices: Set[int]
    removal_order: List[int] = field(default_factory=list)
    oracle_calls: int = 0
    terminated_reason: str = ""
    search_responses: List[Dict] = field(default_factory=list)
    iteration_scores: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "kept_indices": sorted(self.kept_indices),
            "removal_order": self.removal_order,
            "oracle_calls": self.oracle_calls,
            "terminated_reason": self.terminated_reason,
            "search_responses": self.search_responses,
            "iteration_scores": self.iteration_scores,
        }


@dataclass
class MSSResult:
    """Aggregate result from multiple MSS runs.

    Attributes:
        video_id: Video identifier
        segments: All video segments
        mss_runs: Results from each MSS run
        inclusion_frequencies: Per-segment inclusion frequency
        total_oracle_calls: Total oracle calls across all runs
        precheck_passed: Whether pre-check was passed
        precheck_vote_yes: Vote_yes from pre-check
        precheck_responses: Oracle responses from precheck (for debugging/analysis)
    """

    video_id: str
    segments: List[Segment]
    mss_runs: List[MSSRunResult]
    inclusion_frequencies: Dict[int, float] = field(default_factory=dict)
    total_oracle_calls: int = 0
    precheck_passed: bool = True
    precheck_vote_yes: float = 1.0
    precheck_responses: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "video_id": self.video_id,
            "segments": [s.to_dict() for s in self.segments],
            "mss_runs": [r.to_dict() for r in self.mss_runs],
            "inclusion_frequencies": {
                str(k): round(v, 4) for k, v in self.inclusion_frequencies.items()
            },
            "total_oracle_calls": self.total_oracle_calls,
            "precheck_passed": self.precheck_passed,
            "precheck_vote_yes": round(self.precheck_vote_yes, 3),
            "precheck_responses": self.precheck_responses,
        }


@dataclass
class MSSConfig:
    """Configuration for MSS extraction.

    Attributes:
        delta_t: Segment length in seconds
        r_runs: Number of MSS runs
        shortlist_fraction: Fraction of K to sample for shortlist
        shortlist_min: Minimum shortlist size
        shortlist_max: Maximum shortlist size
        tie_breaker: "random" or "vote"
        seed: Random seed for reproducibility
        show_progress: Whether to show progress bars
        removal_strategy: "score" (deterministic) or "random" (legacy)
        use_thresholds: If True, apply tau_final/delta/tau_min thresholds
        tau_final: Min removal confidence for removability (only if use_thresholds=True)
        delta: Maximum confidence drop for removability (only if use_thresholds=True)
        tau_min: Minimum baseline confidence to continue (only if use_thresholds=True)
    """

    delta_t: float = 0.5
    r_runs: int = 10
    shortlist_fraction: float = 0.30
    shortlist_min: int = 5
    shortlist_max: int = 10
    tie_breaker: str = "random"
    seed: Optional[int] = None
    show_progress: bool = True
    removal_strategy: str = "score"  # "score" or "random"
    use_thresholds: bool = False  # If False, only P(YES) > 0.5 matters; if True, also check thresholds
    tau_final: float = 0.88  # Min removal confidence (only if use_thresholds=True)
    delta: float = 0.20  # Max confidence drop for removability (only if use_thresholds=True)
    tau_min: float = 0.80  # Min baseline confidence to continue (only if use_thresholds=True)
    direct_scoring: bool = False  # Use oracle's single-query importance scores instead of greedy removal
    max_fps: Optional[float] = None  # Subsample video frames to <= this rate before sending to oracle
    max_pixels: Optional[int] = None  # Cap frame resolution (Qwen smart_resize max_pixels)

    def compute_shortlist_size(self, k_size: int) -> int:
        """Compute shortlist size M(K).

        M(K) = min(max, max(min, ceil(fraction * |K|)))

        Args:
            k_size: Current size of K

        Returns:
            Shortlist size
        """
        computed = math.ceil(self.shortlist_fraction * k_size)
        return min(self.shortlist_max, max(self.shortlist_min, computed))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "delta_t": self.delta_t,
            "r_runs": self.r_runs,
            "shortlist_fraction": self.shortlist_fraction,
            "shortlist_min": self.shortlist_min,
            "shortlist_max": self.shortlist_max,
            "tie_breaker": self.tie_breaker,
            "seed": self.seed,
            "removal_strategy": self.removal_strategy,
            "use_thresholds": self.use_thresholds,
            "tau_final": self.tau_final,
            "delta": self.delta,
            "tau_min": self.tau_min,
            "direct_scoring": self.direct_scoring,
        }


class MSSExtractor:
    """Extracts Minimal Sufficient Subsets from videos.

    Uses greedy segment removal with oracle verification.
    """

    def __init__(
        self,
        oracle: OracleInterface,
        mss_config: MSSConfig,
        mask_config: MaskConfig,
        progress_callback: Optional[ProgressCallback] = None,
        batch_size: int = 4,
    ):
        """Initialize MSS extractor.

        Args:
            oracle: Oracle interface for verification queries
            mss_config: MSS extraction configuration
            mask_config: Masking configuration
            progress_callback: Optional callback for progress updates
            batch_size: Number of masked videos to process per forward pass
        """
        self.oracle = oracle
        self.mss_config = mss_config
        self.mask_config = mask_config
        self.progress_callback = progress_callback
        self.batch_size = batch_size
        self._current_video_info: Optional[dict] = None

        if mss_config.seed is not None:
            random.seed(mss_config.seed)

    def _report_progress(
        self, stage: str, current: int, total: int, extra: Optional[dict] = None
    ):
        """Report progress to callback if set."""
        if self.progress_callback:
            self.progress_callback(stage, current, total, extra)

    def _create_masked_frames(
        self,
        all_frames: List[np.ndarray],
        segments: List[Segment],
        kept_indices: Set[int],
        fps: float,
    ) -> Tuple[List[np.ndarray], MaskingInfo]:
        """Create masked video frames by keeping only specified segments.

        Args:
            all_frames: All video frames
            segments: All video segments
            kept_indices: Indices of segments to keep (others masked)
            fps: Video FPS

        Returns:
            Tuple of (masked_frames, masking_info)
        """
        # Compute masking info for oracle context
        masking_info = compute_masking_info(segments, kept_indices, self.mask_config)

        # Get segments to mask (complement of kept)
        removed_indices = get_kept_segment_mask(segments, kept_indices)

        if not removed_indices:
            return all_frames, masking_info  # Nothing to mask

        # Get segments to mask
        segments_to_mask = [s for s in segments if s.index in removed_indices]
        frame_ranges = segments_to_frame_ranges(segments_to_mask, fps, len(all_frames))

        masked_frames = mask_frames_in_memory(all_frames, frame_ranges, self.mask_config)
        return masked_frames, masking_info

    def _run_single_mss(
        self,
        all_frames: List[np.ndarray],
        segments: List[Segment],
        action_label: str,
        video_id: str,
        fps: float,
        run_idx: int,
        pbar_iteration: Optional["tqdm"] = None,
        prompt_context: Optional[Dict[str, str]] = None,
        baseline_confidence: Optional[float] = None,
    ) -> MSSRunResult:
        """Execute a single MSS extraction run.

        Dispatches to score-based or random strategy based on config.

        Args:
            all_frames: All video frames
            segments: All video segments
            action_label: Action label to verify
            video_id: Video identifier
            fps: Video FPS
            run_idx: Run index for logging
            pbar_iteration: Optional tqdm progress bar for iteration updates
            prompt_context: Optional dict with additional prompt variables
            baseline_confidence: Pre-computed baseline confidence (for score strategy)

        Returns:
            MSSRunResult with kept segments and metadata
        """
        if self.mss_config.removal_strategy == "score":
            return self._run_single_mss_score_based(
                all_frames, segments, action_label, video_id, fps, run_idx,
                pbar_iteration, prompt_context, baseline_confidence,
            )
        else:
            return self._run_single_mss_random(
                all_frames, segments, action_label, video_id, fps, run_idx,
                pbar_iteration, prompt_context,
            )

    def _run_single_mss_score_based(
        self,
        all_frames: List[np.ndarray],
        segments: List[Segment],
        action_label: str,
        video_id: str,
        fps: float,
        run_idx: int,
        pbar_iteration: Optional["tqdm"] = None,
        prompt_context: Optional[Dict[str, str]] = None,
        baseline_confidence: Optional[float] = None,
    ) -> MSSRunResult:
        """Execute score-based MSS extraction (deterministic removal).

        Uses logit-based confidence to deterministically select which segment
        to remove (smallest drop from baseline).

        Args:
            all_frames: All video frames
            segments: All video segments
            action_label: Action label to verify
            video_id: Video identifier
            fps: Video FPS
            run_idx: Run index for logging
            pbar_iteration: Optional tqdm progress bar for iteration updates
            prompt_context: Optional dict with additional prompt variables
            baseline_confidence: Pre-computed baseline confidence from precheck

        Returns:
            MSSRunResult with kept segments and metadata
        """
        start_time = time.time()

        # Initialize K = all segments
        k_indices = {seg.index for seg in segments}
        initial_k_size = len(k_indices)
        removal_order = []
        oracle_calls = 0
        iteration = 0
        search_responses = []
        iteration_scores = []

        # Use provided baseline or set to 1.0 (will be computed if not provided)
        baseline = baseline_confidence if baseline_confidence is not None else 1.0

        tau_final = self.mss_config.tau_final
        delta = self.mss_config.delta
        tau_min = self.mss_config.tau_min

        logger.debug(
            f"[Run {run_idx + 1}] Starting score-based removal with |K|={len(k_indices)} segments"
        )
        logger.debug(
            f"[Run {run_idx + 1}] Parameters: tau_final={tau_final}, delta={delta}, "
            f"tau_min={tau_min}, baseline={baseline:.3f}"
        )

        while len(k_indices) > 1:
            iteration += 1
            iter_start = time.time()

            # Check baseline threshold (only when use_thresholds=True)
            if self.mss_config.use_thresholds and baseline < tau_min:
                elapsed = time.time() - start_time
                logger.info(
                    f"[Run {run_idx + 1}] Terminated: baseline {baseline:.3f} < tau_min {tau_min}. "
                    f"|K|={len(k_indices)}, {oracle_calls} oracle calls, {elapsed:.1f}s"
                )
                return MSSRunResult(
                    kept_indices=k_indices,
                    removal_order=removal_order,
                    oracle_calls=oracle_calls,
                    terminated_reason="baseline_below_tau_min",
                    search_responses=search_responses,
                    iteration_scores=iteration_scores,
                )

            logger.debug(
                f"[Run {run_idx + 1}][Iter {iteration}] |K|={len(k_indices)}, baseline={baseline:.3f}"
            )

            # Score all candidates using batch processing
            candidate_scores: List[SegmentScore] = []
            sorted_candidates = sorted(k_indices)

            # Prepare batch queries in parallel (frame masking is CPU-bound)
            batch_items = []

            def prepare_candidate(candidate_idx: int) -> dict:
                """Prepare masked frames for a candidate removal."""
                kept_without = k_indices - {candidate_idx}
                masked_frames, masking_info = self._create_masked_frames(
                    all_frames, segments, kept_without, fps
                )
                removed_indices = get_kept_segment_mask(segments, kept_without)
                mask_hash = compute_mask_pattern_hash(removed_indices, len(segments))
                return {
                    "candidate_idx": candidate_idx,
                    "frames": masked_frames,
                    "action_label": action_label,
                    "video_id": video_id,
                    "mask_pattern_hash": mask_hash,
                    "removal_note": masking_info.removal_note,
                    "prompt_context": prompt_context,
                    "kept_without": kept_without,
                }

            # Parallel frame preparation (CPU-bound)
            with ThreadPoolExecutor(max_workers=min(4, len(sorted_candidates))) as executor:
                futures = {executor.submit(prepare_candidate, idx): idx for idx in sorted_candidates}
                for future in as_completed(futures):
                    batch_items.append(future.result())

            # Sort back to original order
            batch_items.sort(key=lambda x: x["candidate_idx"])

            # Batch query the oracle (processes multiple videos per forward pass)
            responses = self.oracle.query_batch(batch_items, batch_size=self.batch_size)
            oracle_calls += len(responses)

            # Process responses
            for item, response in zip(batch_items, responses):
                candidate_idx = item["candidate_idx"]
                kept_without = item["kept_without"]

                # Save response
                search_responses.append({
                    "iteration": iteration,
                    "candidate_segment": candidate_idx,
                    "kept_segments": sorted(kept_without),
                    "response": response.to_dict(),
                })

                # Compute removal score using binary P(YES|not SKIP)
                # SKIP is checked separately — if the model thinks the video
                # is corrupted after removal, treat segment as non-removable
                if is_skip_flagged(response):
                    # Model thinks remaining video is unreadable — keep segment
                    removal_score = 0.0
                elif response.logit_confidence is not None:
                    removal_score = response.logit_confidence
                else:
                    # Fallback to text-based if logits unavailable
                    removal_score = response.confidence if response.decision.value == "YES" else 0.0

                drop = baseline - removal_score

                # Removability: P(YES|not SKIP) > 0.5 means YES beats NO
                if self.mss_config.use_thresholds:
                    # Strict mode: also apply tau_final and delta thresholds
                    is_removable = (
                        removal_score > 0.5
                        and removal_score >= tau_final
                        and drop < delta
                    )
                else:
                    # Simple mode: removable if P(YES|not SKIP) > 0.5
                    is_removable = removal_score > 0.5

                score = SegmentScore(
                    segment_index=candidate_idx,
                    removal_score=removal_score,
                    drop=drop,
                    is_removable=is_removable,
                    response=response,
                )
                candidate_scores.append(score)

                logger.debug(
                    f"[Run {run_idx + 1}][Iter {iteration}] "
                    f"seg[{candidate_idx}]: score={removal_score:.3f}, drop={drop:.3f}, "
                    f"removable={is_removable}"
                )

            # Record iteration scores
            iter_record = {
                "iteration": iteration,
                "baseline": round(baseline, 4),
                "candidates": [s.to_dict() for s in candidate_scores],
            }

            # Find removable candidates
            removable = [s for s in candidate_scores if s.is_removable]

            if not removable:
                # No removable segments - terminate
                iter_record["removed"] = None
                iter_record["new_baseline"] = baseline
                iter_record["terminated"] = True
                iteration_scores.append(iter_record)

                elapsed = time.time() - start_time
                logger.info(
                    f"[Run {run_idx + 1}] Terminated: no removable segments. "
                    f"|K|={len(k_indices)}, {oracle_calls} oracle calls, {elapsed:.1f}s"
                )
                return MSSRunResult(
                    kept_indices=k_indices,
                    removal_order=removal_order,
                    oracle_calls=oracle_calls,
                    terminated_reason="no_removable_segments",
                    search_responses=search_responses,
                    iteration_scores=iteration_scores,
                )

            # Remove segment with smallest drop (most confident removal)
            best = min(removable, key=lambda s: s.drop)
            to_remove = best.segment_index
            new_baseline = best.removal_score

            iter_record["removed"] = to_remove
            iter_record["new_baseline"] = round(new_baseline, 4)
            iteration_scores.append(iter_record)

            # Remove segment
            k_indices.remove(to_remove)
            removal_order.append(to_remove)
            baseline = new_baseline

            iter_elapsed = time.time() - iter_start
            logger.debug(
                f"[Run {run_idx + 1}][Iter {iteration}] "
                f"Removed seg[{to_remove}] (drop={best.drop:.3f}) -> |K|={len(k_indices)}, "
                f"new_baseline={baseline:.3f} ({iter_elapsed:.2f}s)"
            )

            # Update progress bar if available
            if pbar_iteration:
                pbar_iteration.set_postfix({
                    "|K|": len(k_indices),
                    "removed": len(removal_order),
                    "calls": oracle_calls,
                    "baseline": f"{baseline:.2f}",
                })

        elapsed = time.time() - start_time
        logger.debug(
            f"[Run {run_idx + 1}] Complete: |K|={len(k_indices)}, "
            f"removed {len(removal_order)}/{initial_k_size - 1} segments, "
            f"{oracle_calls} oracle calls, {elapsed:.1f}s"
        )

        return MSSRunResult(
            kept_indices=k_indices,
            removal_order=removal_order,
            oracle_calls=oracle_calls,
            terminated_reason="single_segment" if len(k_indices) == 1 else "complete",
            search_responses=search_responses,
            iteration_scores=iteration_scores,
        )

    def _run_single_mss_random(
        self,
        all_frames: List[np.ndarray],
        segments: List[Segment],
        action_label: str,
        video_id: str,
        fps: float,
        run_idx: int,
        pbar_iteration: Optional["tqdm"] = None,
        prompt_context: Optional[Dict[str, str]] = None,
    ) -> MSSRunResult:
        """Execute random-strategy MSS extraction (legacy behavior).

        Uses random tie-breaking to select which segment to remove.

        Args:
            all_frames: All video frames
            segments: All video segments
            action_label: Action label to verify
            video_id: Video identifier
            fps: Video FPS
            run_idx: Run index for logging
            pbar_iteration: Optional tqdm progress bar for iteration updates
            prompt_context: Optional dict with additional prompt variables

        Returns:
            MSSRunResult with kept segments and metadata
        """
        start_time = time.time()

        # Initialize K = all segments
        k_indices = {seg.index for seg in segments}
        initial_k_size = len(k_indices)
        removal_order = []
        oracle_calls = 0
        iteration = 0
        search_responses = []  # Track all oracle responses during search

        logger.debug(
            f"[Run {run_idx + 1}] Starting greedy removal with |K|={len(k_indices)} segments"
        )

        while len(k_indices) > 1:
            iteration += 1
            iter_start = time.time()

            # Compute shortlist size
            shortlist_size = self.mss_config.compute_shortlist_size(len(k_indices))
            shortlist_size = min(shortlist_size, len(k_indices))

            # Sample shortlist A
            k_list = sorted(k_indices)
            shortlist = set(random.sample(k_list, shortlist_size))

            logger.debug(
                f"[Run {run_idx + 1}][Iter {iteration}] |K|={len(k_indices)}, "
                f"shortlist={shortlist_size} segments"
            )

            # Screen each candidate in shortlist
            removable = []
            expanded = False
            candidates_screened = 0

            while True:
                for candidate_idx in shortlist:
                    candidates_screened += 1

                    # Create K without candidate
                    kept_without = k_indices - {candidate_idx}
                    masked_frames, masking_info = self._create_masked_frames(
                        all_frames, segments, kept_without, fps
                    )

                    # Compute mask pattern hash for caching
                    removed_indices = get_kept_segment_mask(segments, kept_without)
                    mask_hash = compute_mask_pattern_hash(
                        removed_indices, len(segments)
                    )

                    # Query oracle (search mode) with removal note for CUT operator
                    response = self.oracle.query_search(
                        masked_frames, action_label, video_id, mask_hash,
                        removal_note=masking_info.removal_note,
                        prompt_context=prompt_context,
                    )
                    oracle_calls += 1

                    # Save response with context
                    search_responses.append({
                        "iteration": iteration,
                        "candidate_segment": candidate_idx,
                        "kept_segments": sorted(kept_without),
                        "response": response.to_dict(),
                    })

                    decision_str = response.decision.value
                    logger.debug(
                        f"[Run {run_idx + 1}][Iter {iteration}] "
                        f"Candidate seg[{candidate_idx}]: {decision_str} "
                        f"(conf={response.confidence:.2f})"
                    )

                    if self.oracle.ok_search(response):
                        removable.append(candidate_idx)

                # Check if we found any removable
                if removable:
                    logger.debug(
                        f"[Run {run_idx + 1}][Iter {iteration}] "
                        f"Found {len(removable)} removable segments: {removable}"
                    )
                    break

                # Expand once if no removable found
                if not expanded:
                    logger.debug(
                        f"[Run {run_idx + 1}][Iter {iteration}] "
                        f"No removable in shortlist ({len(shortlist)} screened), "
                        f"expanding to full K ({len(k_indices)} segments)"
                    )
                    shortlist = k_indices.copy()
                    expanded = True
                else:
                    # Still no removable after expansion - terminate
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Run {run_idx + 1}] Terminated: no removable after full scan. "
                        f"|K|={len(k_indices)}, {oracle_calls} oracle calls, {elapsed:.1f}s"
                    )
                    return MSSRunResult(
                        kept_indices=k_indices,
                        removal_order=removal_order,
                        oracle_calls=oracle_calls,
                        terminated_reason="no_removable",
                        search_responses=search_responses,
                    )

            # Select which removable to remove
            if self.mss_config.tie_breaker == "random" or len(removable) == 1:
                to_remove = random.choice(removable)
            else:
                # Vote-based tie-breaking (optional, more expensive)
                to_remove = random.choice(removable)

            # Remove segment
            k_indices.remove(to_remove)
            removal_order.append(to_remove)

            iter_elapsed = time.time() - iter_start
            logger.debug(
                f"[Run {run_idx + 1}][Iter {iteration}] "
                f"Removed seg[{to_remove}] -> |K|={len(k_indices)} "
                f"({candidates_screened} screened, {iter_elapsed:.2f}s)"
            )

            # Update progress bar if available
            if pbar_iteration:
                pbar_iteration.set_postfix({
                    "|K|": len(k_indices),
                    "removed": len(removal_order),
                    "calls": oracle_calls,
                })

        elapsed = time.time() - start_time
        logger.debug(
            f"[Run {run_idx + 1}] Complete: |K|={len(k_indices)}, "
            f"removed {len(removal_order)}/{initial_k_size - 1} segments, "
            f"{oracle_calls} oracle calls, {elapsed:.1f}s"
        )

        return MSSRunResult(
            kept_indices=k_indices,
            removal_order=removal_order,
            oracle_calls=oracle_calls,
            terminated_reason="single_segment" if len(k_indices) == 1 else "complete",
            search_responses=search_responses,
        )

    def _parse_direct_scores(
        self,
        raw_output: str,
        n_segments: int,
    ) -> Optional[Tuple[Dict[int, float], Set[int]]]:
        """Parse importance scores and MSS from oracle's direct annotation output.

        Extracts per-segment importance scores (0-100 normalized to 0.0-1.0)
        and the minimum_sufficient_set from the oracle's JSON response.

        Args:
            raw_output: Raw model output (may include <think> block)
            n_segments: Total number of segments in the video

        Returns:
            Tuple of (importance_weights, kept_indices) on success, None on failure:
            - importance_weights: Dict mapping segment index (0-based) to weight (0.0-1.0)
            - kept_indices: Set of 0-based segment indices in the minimum sufficient set
        """
        # Strip <think>...</think> block if present
        text = raw_output
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):]

        # Find and parse JSON using bracket matching (same approach as oracle.py)
        start = text.find("{")
        if start == -1:
            logger.warning("Direct scoring: no JSON found in oracle output")
            return None

        data = None
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                if in_string:
                    escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c in ("{", "["):
                depth += 1
            elif c in ("}", "]"):
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:i + 1])
                        break
                    except json.JSONDecodeError:
                        depth = 1
                        continue

        if data is None:
            logger.warning("Direct scoring: failed to parse JSON from oracle output")
            return None

        # Extract importance scores from segments array
        # Use None sentinel to detect omitted segments
        importance_weights: Dict[int, Optional[float]] = {i: None for i in range(n_segments)}
        segments_data = data.get("segments", [])

        if not segments_data:
            logger.warning("Direct scoring: no segments array in oracle output")
            return None

        for seg in segments_data:
            # segment_id is 1-based in the prompt output
            seg_id = seg.get("segment_id")
            importance = seg.get("importance", 0)
            if seg_id is not None:
                idx = int(seg_id) - 1  # Convert to 0-based
                if 0 <= idx < n_segments:
                    importance_weights[idx] = float(importance) / 100.0

        # Fill omitted segments via neighbor interpolation
        omitted = [i for i, w in importance_weights.items() if w is None]
        if omitted:
            logger.warning(
                f"Direct scoring: {len(omitted)}/{n_segments} segments omitted by model "
                f"(indices: {omitted}). Filling via neighbor interpolation."
            )
            for i in omitted:
                # Find nearest scored neighbors
                left_val = None
                for j in range(i - 1, -1, -1):
                    if importance_weights[j] is not None:
                        left_val = importance_weights[j]
                        break
                right_val = None
                for j in range(i + 1, n_segments):
                    if importance_weights[j] is not None:
                        right_val = importance_weights[j]
                        break
                if left_val is not None and right_val is not None:
                    importance_weights[i] = (left_val + right_val) / 2.0
                elif left_val is not None:
                    importance_weights[i] = left_val
                elif right_val is not None:
                    importance_weights[i] = right_val
                else:
                    # All segments omitted — shouldn't happen (caught by empty check above)
                    importance_weights[i] = 0.0

        # At this point all values are floats (no None remaining)
        importance_weights_final: Dict[int, float] = {
            i: w for i, w in importance_weights.items()  # type: ignore[misc]
        }

        # Extract minimum_sufficient_set (1-based segment IDs)
        mss_ids = data.get("minimum_sufficient_set", [])
        if mss_ids:
            kept_indices = set()
            for sid in mss_ids:
                idx = int(sid) - 1  # Convert to 0-based
                if 0 <= idx < n_segments:
                    kept_indices.add(idx)
        else:
            # Fallback: segments scoring >= 50 (importance >= 0.5 after normalization)
            kept_indices = {i for i, w in importance_weights_final.items() if w >= 0.5}

        if not kept_indices:
            logger.warning("Direct scoring: empty MSS after parsing")
            return None

        logger.debug(
            f"Direct scoring: parsed {len(segments_data)} segment scores, "
            f"MSS size={len(kept_indices)}, kept={sorted(kept_indices)}"
        )

        return importance_weights_final, kept_indices

    def extract(
        self,
        video_path: Path,
        action_label: str,
        video_id: Optional[str] = None,
        video_index: Optional[int] = None,
        total_videos: Optional[int] = None,
        prompt_context: Optional[Dict[str, str]] = None,
    ) -> MSSResult:
        """Extract MSS pseudo-labels from a video.

        Args:
            video_path: Path to video file
            action_label: Ground-truth action label
            video_id: Video identifier (uses filename if not provided)
            video_index: Current video index (for progress display)
            total_videos: Total number of videos (for progress display)
            prompt_context: Optional dict with additional prompt variables
                           (e.g., action_template, placeholders for SSv2)

        Returns:
            MSSResult with all extraction results
        """
        extraction_start = time.time()

        if video_id is None:
            video_id = video_path.stem

        # Store current video info for progress tracking
        self._current_video_info = {
            "video_id": video_id,
            "action_label": action_label,
            "video_path": str(video_path),
            "video_index": video_index,
            "total_videos": total_videos,
        }

        video_prefix = ""
        if video_index is not None and total_videos is not None:
            video_prefix = f"[{video_index + 1}/{total_videos}] "

        logger.info(
            f"{video_prefix}Starting MSS extraction for '{video_id}' "
            f"(label: '{action_label}')"
        )
        logger.info(f"{video_prefix}Video path: {video_path}")

        # Log SSv2 mode if prompt_context is provided
        if prompt_context and "action_template" in prompt_context:
            logger.info(
                f"{video_prefix}Using SSv2 template: '{prompt_context['action_template']}' "
                f"(objects: {prompt_context.get('placeholders', 'unknown')})"
            )

        # Read video and get segments
        logger.debug(f"{video_prefix}Loading video frames...")
        load_start = time.time()
        all_frames, fps, width, height = read_video_frames(video_path)
        duration_s = len(all_frames) / fps
        segments = segment_video(video_path, self.mss_config.delta_t, duration_s)
        load_elapsed = time.time() - load_start

        logger.info(
            f"{video_prefix}Video loaded: {len(all_frames)} frames, "
            f"{width}x{height}, {fps:.1f} fps, {duration_s:.2f}s duration, "
            f"{len(segments)} segments (Δt={self.mss_config.delta_t}s) "
            f"[{load_elapsed:.1f}s]"
        )

        # Inject video-level info into prompt_context for template formatting
        if prompt_context is not None:
            prompt_context["duration"] = f"{duration_s:.2f}"
            prompt_context["n_segments"] = str(len(segments))
            prompt_context["segment_duration"] = f"{self.mss_config.delta_t}"
            prompt_context["video_fps"] = fps

        # Pre-check: verify oracle can classify full video (single logit-based query)
        logger.info(f"{video_prefix}Running pre-check...")
        precheck_start = time.time()
        full_mask_hash = compute_mask_pattern_hash(set(), len(segments))
        precheck_response, precheck_p_yes, precheck_ok = self.oracle.query_final(
            all_frames, action_label, video_id, full_mask_hash,
            prompt_context=prompt_context,
        )
        precheck_elapsed = time.time() - precheck_start

        # Log precheck response
        logger.debug(
            f"{video_prefix}Pre-check: P(YES)={precheck_p_yes:.3f} "
            f"(text={precheck_response.decision.value}) - {precheck_response.evidence[:50]}..."
        )

        if not precheck_ok:
            logger.warning(
                f"{video_prefix}Pre-check FAILED: P(YES)={precheck_p_yes:.3f} "
                f"(threshold=0.5). "
                f"Oracle cannot reliably classify full video. [{precheck_elapsed:.1f}s]"
            )
            return MSSResult(
                video_id=video_id,
                segments=segments,
                mss_runs=[],
                inclusion_frequencies={},
                total_oracle_calls=1,
                precheck_passed=False,
                precheck_vote_yes=precheck_p_yes,
                precheck_responses=[precheck_response.to_dict()],
            )

        logger.info(
            f"{video_prefix}Pre-check PASSED: P(YES)={precheck_p_yes:.3f} "
            f"(threshold=0.5) [{precheck_elapsed:.1f}s]"
        )

        # Baseline confidence for score-based strategy is the pre-check P(YES)
        baseline_confidence = None
        if self.mss_config.removal_strategy == "score":
            baseline_confidence = precheck_p_yes
            logger.info(
                f"{video_prefix}Baseline confidence: {baseline_confidence:.3f}"
            )

        # Direct scoring: use oracle's single-query importance scores instead of removal loop
        if self.mss_config.direct_scoring:
            logger.info(
                f"{video_prefix}Direct scoring mode: parsing importance scores from precheck response"
            )

            max_direct_attempts = 3  # 1 initial (precheck) + up to 2 retries
            direct_oracle_calls = 1  # precheck already counted
            parsed = self._parse_direct_scores(
                precheck_response.raw_output, len(segments)
            )
            all_responses = [precheck_response.to_dict()]

            for retry in range(1, max_direct_attempts):
                if parsed is not None:
                    break
                logger.warning(
                    f"{video_prefix}Direct scoring parse failed, "
                    f"re-querying oracle (retry {retry}/2)"
                )
                retry_response = self.oracle.query_single(
                    all_frames, action_label, video_id,
                    prompt_context=prompt_context,
                )
                direct_oracle_calls += 1
                all_responses.append(retry_response.to_dict())
                parsed = self._parse_direct_scores(
                    retry_response.raw_output, len(segments)
                )

            if parsed is None:
                # All attempts failed — fall back to all segments at 0.5
                logger.warning(
                    f"{video_prefix}Direct scoring: all {max_direct_attempts} attempts "
                    f"failed to parse, falling back to all segments"
                )
                importance_weights = {i: 0.5 for i in range(len(segments))}
                kept_indices = set(range(len(segments)))
            else:
                importance_weights, kept_indices = parsed

            # Build a single MSSRunResult from the direct scores
            run_result = MSSRunResult(
                kept_indices=kept_indices,
                removal_order=[],
                oracle_calls=direct_oracle_calls - 1,  # exclude precheck
                terminated_reason="direct_scoring",
            )

            extraction_elapsed = time.time() - extraction_start
            logger.info(
                f"{video_prefix}Direct scoring complete for '{video_id}': "
                f"MSS size={len(kept_indices)}/{len(segments)}, "
                f"{direct_oracle_calls} oracle call(s) [{extraction_elapsed:.1f}s total]"
            )

            return MSSResult(
                video_id=video_id,
                segments=segments,
                mss_runs=[run_result],
                inclusion_frequencies=importance_weights,
                total_oracle_calls=direct_oracle_calls,
                precheck_passed=True,
                precheck_vote_yes=precheck_p_yes,
                precheck_responses=all_responses,
            )

        # Run R MSS extractions with progress bar
        mss_runs = []
        total_oracle_calls = 1  # Start with precheck call

        show_progress = self.mss_config.show_progress and TQDM_AVAILABLE

        run_iterator = range(self.mss_config.r_runs)
        if show_progress:
            run_iterator = tqdm(
                run_iterator,
                desc=f"{video_prefix}MSS runs",
                unit="run",
                leave=False,
                ncols=100,
            )

        strategy_info = (
            f"strategy={self.mss_config.removal_strategy}"
            if self.mss_config.removal_strategy == "score"
            else f"shortlist={self.mss_config.shortlist_min}-{self.mss_config.shortlist_max}"
        )
        logger.info(
            f"{video_prefix}Starting {self.mss_config.r_runs} MSS runs ({strategy_info})"
        )

        for r in run_iterator:
            run_start = time.time()

            run_result = self._run_single_mss(
                all_frames, segments, action_label, video_id, fps, r,
                pbar_iteration=run_iterator if show_progress else None,
                prompt_context=prompt_context,
                baseline_confidence=baseline_confidence,
            )
            mss_runs.append(run_result)
            total_oracle_calls += run_result.oracle_calls

            run_elapsed = time.time() - run_start

            # Update progress bar description
            if show_progress and hasattr(run_iterator, "set_postfix"):
                avg_k = sum(len(run.kept_indices) for run in mss_runs) / len(mss_runs)
                run_iterator.set_postfix({
                    "avg|K|": f"{avg_k:.1f}",
                    "calls": total_oracle_calls,
                })

            logger.info(
                f"{video_prefix}Run {r + 1}/{self.mss_config.r_runs}: "
                f"|K|={len(run_result.kept_indices)}, "
                f"removed={len(run_result.removal_order)}, "
                f"oracle_calls={run_result.oracle_calls}, "
                f"reason={run_result.terminated_reason} "
                f"[{run_elapsed:.1f}s]"
            )

            # Report progress
            self._report_progress(
                "mss_run",
                r + 1,
                self.mss_config.r_runs,
                {"run_result": run_result.to_dict()},
            )

        # Compute inclusion frequencies
        inclusion_counts: Dict[int, int] = {seg.index: 0 for seg in segments}
        for run in mss_runs:
            for idx in run.kept_indices:
                inclusion_counts[idx] += 1

        inclusion_frequencies = {
            idx: count / len(mss_runs) for idx, count in inclusion_counts.items()
        }

        # Log frequency summary
        high_freq = sum(1 for f in inclusion_frequencies.values() if f >= 0.8)
        low_freq = sum(1 for f in inclusion_frequencies.values() if f <= 0.2)
        mid_freq = len(segments) - high_freq - low_freq

        extraction_elapsed = time.time() - extraction_start
        avg_k_size = sum(len(run.kept_indices) for run in mss_runs) / len(mss_runs)

        logger.info(
            f"{video_prefix}MSS extraction complete for '{video_id}': "
            f"{len(mss_runs)} runs, avg|K|={avg_k_size:.1f}, "
            f"{total_oracle_calls} total oracle calls"
        )
        logger.info(
            f"{video_prefix}Frequency distribution: "
            f"high(>=0.8)={high_freq}, mid={mid_freq}, low(<=0.2)={low_freq} "
            f"[{extraction_elapsed:.1f}s total]"
        )

        return MSSResult(
            video_id=video_id,
            segments=segments,
            mss_runs=mss_runs,
            inclusion_frequencies=inclusion_frequencies,
            total_oracle_calls=total_oracle_calls,
            precheck_passed=True,
            precheck_vote_yes=precheck_p_yes,
            precheck_responses=[precheck_response.to_dict()],
        )

    def extract_batch_direct(
        self,
        videos: List[Dict],
    ) -> List[MSSResult]:
        """Batch extract direct scoring labels for multiple videos.

        Loads all videos, sends precheck queries in a single batched forward pass,
        then parses importance scores from each response. Much faster than calling
        extract() in a loop when using direct_scoring mode.

        Args:
            videos: List of dicts with keys:
                - video_path: Path to video file
                - action_label: Ground-truth action label
                - video_id: Optional video identifier
                - video_index: Optional index for progress display
                - total_videos: Optional total count for progress display
                - prompt_context: Optional prompt context dict

        Returns:
            List of MSSResult (same order as input videos)
        """
        batch_start = time.time()
        n = len(videos)
        logger.info(f"Batch direct scoring: loading {n} videos...")

        # Step 1: Load all videos and segment them (parallelized on CPU)
        loaded = []

        max_fps = self.mss_config.max_fps
        max_pixels = self.mss_config.max_pixels

        def load_one(v: Dict) -> Dict:
            video_path = Path(v["video_path"])
            video_id = v.get("video_id") or video_path.stem
            all_frames, fps, width, height = read_video_frames(video_path)
            duration_s = len(all_frames) / fps

            # Optional fps cap: uniformly subsample frames so the effective rate
            # is <= max_fps. Duration stays the same; the new fps becomes the
            # value Qwen sees as video metadata.
            if max_fps is not None and fps > max_fps:
                stride = fps / max_fps
                indices = [int(round(i * stride)) for i in range(int(len(all_frames) / stride))]
                indices = [i for i in indices if i < len(all_frames)]
                all_frames = [all_frames[i] for i in indices]
                fps = len(all_frames) / duration_s if duration_s > 0 else max_fps

            segments = segment_video(video_path, self.mss_config.delta_t, duration_s)

            # Inject video-level info into prompt_context
            prompt_context = v.get("prompt_context")
            if prompt_context is not None:
                prompt_context = dict(prompt_context)  # Don't mutate caller's dict
                prompt_context["duration"] = f"{duration_s:.2f}"
                prompt_context["n_segments"] = str(len(segments))
                prompt_context["segment_duration"] = f"{self.mss_config.delta_t}"
                prompt_context["video_fps"] = fps
                if max_pixels is not None:
                    prompt_context["max_pixels"] = max_pixels

            return {
                "video_path": video_path,
                "video_id": video_id,
                "action_label": v["action_label"],
                "all_frames": all_frames,
                "fps": fps,
                "segments": segments,
                "duration_s": duration_s,
                "prompt_context": prompt_context,
                "video_index": v.get("video_index"),
                "total_videos": v.get("total_videos"),
            }

        with ThreadPoolExecutor(max_workers=min(4, n)) as executor:
            loaded = list(executor.map(load_one, videos))

        load_elapsed = time.time() - batch_start
        logger.info(f"Batch direct scoring: {n} videos loaded [{load_elapsed:.1f}s]")

        # Step 2: Build batch query items for the oracle
        full_mask_hashes = []
        batch_items = []
        for info in loaded:
            mask_hash = compute_mask_pattern_hash(set(), len(info["segments"]))
            full_mask_hashes.append(mask_hash)
            batch_items.append({
                "frames": info["all_frames"],
                "action_label": info["action_label"],
                "video_id": info["video_id"],
                "mask_pattern_hash": mask_hash,
                "prompt_context": info["prompt_context"],
            })

        # Step 3: Batched oracle query (processes batch_size videos per forward pass)
        logger.info(
            f"Batch direct scoring: querying oracle for {n} videos "
            f"(batch_size={self.batch_size})..."
        )
        query_start = time.time()
        responses = self.oracle.query_batch(batch_items, batch_size=self.batch_size)
        query_elapsed = time.time() - query_start
        logger.info(f"Batch direct scoring: oracle queries complete [{query_elapsed:.1f}s]")

        # Step 4: Parse each response into MSSResult
        results = []
        for i, (info, response) in enumerate(zip(loaded, responses)):
            video_id = info["video_id"]
            segments = info["segments"]
            n_segments = len(segments)

            # Compute precheck P(YES) and pass/fail
            p_yes = get_logit_confidence(response)
            precheck_ok = not is_skip_flagged(response) and p_yes > 0.5

            if not precheck_ok:
                logger.warning(
                    f"Batch [{i+1}/{n}] '{video_id}': precheck FAILED "
                    f"(P(YES)={p_yes:.3f})"
                )
                results.append(MSSResult(
                    video_id=video_id,
                    segments=segments,
                    mss_runs=[],
                    inclusion_frequencies={},
                    total_oracle_calls=1,
                    precheck_passed=False,
                    precheck_vote_yes=p_yes,
                    precheck_responses=[response.to_dict()],
                ))
                continue

            # Parse direct scores from response
            parsed = self._parse_direct_scores(response.raw_output, n_segments)

            if parsed is None:
                # Retry up to 2 times
                all_responses = [response.to_dict()]
                oracle_calls = 1
                for retry in range(1, 3):
                    logger.warning(
                        f"Batch [{i+1}/{n}] '{video_id}': parse failed, "
                        f"re-querying oracle (retry {retry}/2)"
                    )
                    retry_resp = self.oracle.query_single(
                        info["all_frames"], info["action_label"], video_id,
                        prompt_context=info["prompt_context"],
                    )
                    oracle_calls += 1
                    all_responses.append(retry_resp.to_dict())
                    parsed = self._parse_direct_scores(retry_resp.raw_output, n_segments)
                    if parsed is not None:
                        break

                if parsed is None:
                    logger.warning(
                        f"Batch [{i+1}/{n}] '{video_id}': all parse attempts failed, "
                        f"falling back to all segments"
                    )
                    importance_weights = {j: 0.5 for j in range(n_segments)}
                    kept_indices = set(range(n_segments))
                else:
                    importance_weights, kept_indices = parsed
            else:
                all_responses = [response.to_dict()]
                oracle_calls = 1
                importance_weights, kept_indices = parsed

            run_result = MSSRunResult(
                kept_indices=kept_indices,
                removal_order=[],
                oracle_calls=oracle_calls - 1,
                terminated_reason="direct_scoring",
            )

            logger.info(
                f"Batch [{i+1}/{n}] '{video_id}': "
                f"P(YES)={p_yes:.3f}, MSS={len(kept_indices)}/{n_segments}"
            )

            results.append(MSSResult(
                video_id=video_id,
                segments=segments,
                mss_runs=[run_result],
                inclusion_frequencies=importance_weights,
                total_oracle_calls=oracle_calls,
                precheck_passed=True,
                precheck_vote_yes=p_yes,
                precheck_responses=all_responses,
            ))

        total_elapsed = time.time() - batch_start
        logger.info(
            f"Batch direct scoring complete: {n} videos, "
            f"{total_elapsed:.1f}s total ({total_elapsed/max(n,1):.1f}s/video)"
        )

        return results
