"""Six selector implementations + RNG seeding for the seven-condition experiment.

Per the development design notes Decisions 1–6.

Each selector returns a ``SelectionResult`` describing the chosen
``kept_segments`` (a list of ``(start_s, end_s)`` tuples in start_s order, no
overlap) plus provenance for sidecar recording. Duration-matched baselines that
cannot reach the tolerance band emit ``selector_failed=True``; the runner is
responsible for excluding such ``(video, *)`` rows from paired statistics across
all conditions for that video.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger(__name__)

DURATION_TOLERANCE_FLOOR_S = 0.05  # absolute floor regardless of segment quantum
# Per-video tolerance is computed as max(FLOOR, 0.5 * median(segment_durations_in_pool)).
# SSv2 (delta_t=0.5s) → ~0.25s tolerance; K400 (delta_t=1.0s) → ~0.5s tolerance.
# This adapts to the MSS-extraction granularity rather than imposing an unattainable
# fixed bound. See design.md Decision 3 (revised).
DURATION_TOLERANCE_S = DURATION_TOLERANCE_FLOOR_S  # legacy alias for tests; prefer _tolerance(pool)


def _tolerance(pool: list[tuple[int, float, float, str]]) -> float:
    if not pool:
        return DURATION_TOLERANCE_FLOOR_S
    durations = sorted(p[2] - p[1] for p in pool)
    median = durations[len(durations) // 2]
    return max(DURATION_TOLERANCE_FLOOR_S, 0.5 * median)

CONDITION_NAMES = (
    "full",
    "full-single",
    "vlm-selected",
    "vlm-weighted",
    "vlm-strict-0.9",
    "random",
    "uniform",
    "uniform-vs-strict-0.9",
    "motion",
    "lowest-evidence",
    "uniform-equal-segs",
    "vlm-aggregate",
    "vlm-anchored",
    "vlm-anchored-hybrid",
    # fast-forward family (positive + negative control); actual parsing via parse_*_condition
    "vlm-fastforward",
    "lowest-evidence-fastforward",
    # parametric score-threshold family; actual parsing via parse_score_threshold_condition
    "vlm-score-threshold",
)
NEEDS_MSS_SIDECAR = {
    "vlm-selected", "vlm-weighted", "vlm-strict-0.9",
    "random", "uniform", "uniform-vs-strict-0.9",
    "motion", "lowest-evidence", "uniform-equal-segs",
    "vlm-aggregate", "vlm-anchored", "vlm-anchored-hybrid",
    "lowest-evidence-fastforward",
    "vlm-score-threshold",
}
NEEDS_MOTION_CACHE = {"motion"}
VLM_STRICT_THRESHOLD = 0.9  # frequency cutoff for vlm-strict-0.9

# Fast-forward conditions: "vlm-fastforward-a<NN>" where NN is alpha × 100 (integer).
# Keeps the full timeline (no segments dropped, no boundary cuts) but allocates frames
# densely on important segments and sparsely (alpha-floored) on unimportant ones —
# like fast-forwarding through filler. alpha ∈ (0, 100]; alpha=100 ≡ uniform sampling.
FASTFORWARD_ALPHA_PERCENTS = (5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90)

# Score-threshold conditions: "vlm-score-threshold-t<NN>" where NN is the threshold × 100.
# Parametrically exposes vlm_strict(sidecar, threshold=NN/100): keep segments with
# continuous weight >= NN/100. Endpoints are anchors of the threshold curve, so the grid
# includes 0 ("keep-all under kept-set protocol", distinct from `full`) and 100
# ("only weight==1.0 segments"; falls back to single highest-weight segment when empty).
SCORE_THRESHOLD_PERCENTS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

# Insertion/Deletion curve grammar: "id-<ordering>-f<NN>" where NN is the
# percent of total-pool duration to keep, drawn top-of-ordering first.
# Orderings:
#   vlm        — segments ranked by MSS `weight` (= frequency) descending
#   anti-vlm   — `weight` ascending (least-important first)
#   random     — per-video deterministic shuffle (run_seed | video_id | "id-random")
#   temporal   — chronological (start_s ascending)
#   motion     — motion-cache mean-flow magnitude descending
ID_ORDERINGS = ("vlm", "anti-vlm", "random", "temporal", "motion")
ID_FRACTION_PERCENTS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
# f=0 (blank) and f=100 (full video) are endpoints handled by the analyzer:
# f=100 reuses the existing condition=full sidecar; f=0 reads as chance-level.


@dataclass
class SelectionResult:
    kept_segments: list[tuple[float, float]] = field(default_factory=list)
    kept_indices: list[int] = field(default_factory=list)
    kept_total_duration_s: float = 0.0
    kept_segment_count: int = 0
    selector_failed: bool = False
    selector_failure_reason: str | None = None
    rng_seed_hex: str | None = None
    # For vlm-weighted: per-kept-segment importance weight (parallel to kept_segments).
    # Used by the weighted-repack adapter path to allocate frames per segment.
    kept_segment_weights: list[float] | None = None


def per_video_rng(run_seed: int, video_id: str, condition: str) -> tuple[random.Random, str]:
    payload = f"{run_seed}|{video_id}|{condition}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed), digest[:8].hex()


def _segments_with_indices(sidecar: dict) -> list[tuple[int, float, float, str]]:
    out = []
    for seg in sidecar.get("segments", []):
        out.append((int(seg["index"]), float(seg["start_s"]), float(seg["end_s"]), str(seg.get("label", "unimportant"))))
    return out


def _duration(seg: tuple[int, float, float, str]) -> float:
    return seg[2] - seg[1]


def _to_kept(segs: list[tuple[int, float, float, str]]) -> tuple[list[tuple[float, float]], list[int], float]:
    segs_sorted = sorted(segs, key=lambda s: s[1])
    kept_segments = [(s[1], s[2]) for s in segs_sorted]
    kept_indices = [s[0] for s in segs_sorted]
    total = sum(_duration(s) for s in segs_sorted)
    return kept_segments, kept_indices, total


def _passes_tolerance(actual_total: float, target_total: float, tolerance: float | None = None) -> bool:
    if tolerance is None:
        tolerance = DURATION_TOLERANCE_FLOOR_S
    return abs(actual_total - target_total) <= tolerance


def vlm_selected(sidecar: dict) -> SelectionResult:
    chosen = [s for s in _segments_with_indices(sidecar) if s[3] == "important"]
    kept, idxs, total = _to_kept(chosen)
    return SelectionResult(kept_segments=kept, kept_indices=idxs, kept_total_duration_s=total, kept_segment_count=len(kept))


def vlm_strict(sidecar: dict, threshold: float = VLM_STRICT_THRESHOLD) -> SelectionResult:
    """Stricter version of vlm_selected: only segments with frequency >= threshold."""
    seg_records = sidecar.get("segments", [])
    chosen = []
    for seg in seg_records:
        f = seg.get("weight")
        if f is None:
            f = seg.get("frequency")
        f = float(f) if f is not None else 0.0
        if f >= threshold:
            chosen.append((int(seg["index"]), float(seg["start_s"]), float(seg["end_s"]), "important"))
    if not chosen:
        return SelectionResult(
            selector_failed=True,
            selector_failure_reason=f"no segments with frequency >= {threshold}",
        )
    kept, idxs, total = _to_kept(chosen)
    return SelectionResult(
        kept_segments=kept,
        kept_indices=idxs,
        kept_total_duration_s=total,
        kept_segment_count=len(kept),
    )


def vlm_weighted(sidecar: dict) -> SelectionResult:
    """Same kept-segment set as vlm-selected, plus per-segment importance weights.

    Weights come from the MSS sidecar's per-segment ``frequency`` (alias ``weight``),
    which is the inclusion frequency across MSS oracle runs and ranges in [0, 1].
    The weighted-repack adapter path uses these to allocate frames per segment.
    """
    seg_records = sidecar.get("segments", [])
    chosen_with_weight = []
    for seg in seg_records:
        if seg.get("label") != "important":
            continue
        idx = int(seg["index"])
        s = float(seg["start_s"])
        e = float(seg["end_s"])
        # Prefer 'weight'; fall back to 'frequency' (same value in MSS sidecars).
        w = seg.get("weight")
        if w is None:
            w = seg.get("frequency")
        w = float(w) if w is not None else 1.0
        chosen_with_weight.append((idx, s, e, "important", w))
    if not chosen_with_weight:
        return SelectionResult(selector_failed=True, selector_failure_reason="no important segments")
    chosen_with_weight.sort(key=lambda r: r[1])
    kept_segments = [(r[1], r[2]) for r in chosen_with_weight]
    kept_indices = [r[0] for r in chosen_with_weight]
    weights = [r[4] for r in chosen_with_weight]
    total = sum(r[2] - r[1] for r in chosen_with_weight)
    return SelectionResult(
        kept_segments=kept_segments,
        kept_indices=kept_indices,
        kept_total_duration_s=total,
        kept_segment_count=len(kept_segments),
        kept_segment_weights=weights,
    )


def vlm_fastforward(sidecar: dict, alpha: float) -> SelectionResult:
    """Fast-forward selector: keep ALL segments, with per-segment frame-density weights.

    Unlike ``vlm-selected`` / ``vlm-weighted`` which drop unimportant segments and
    concatenate the kept ones (introducing hard discontinuity boundaries), this
    selector preserves the full timeline. Per-segment importance is encoded as a
    frame-allocation weight: important segments get density 1.0 and unimportant
    segments get density ``alpha`` ∈ (0, 1]. The downstream weighted-repack adapter
    (HF) or weighted frame-index allocator (paper-ckpt) consumes these weights to
    distribute the fixed frame budget proportionally to ``duration × density``.

    Result: a temporally smooth, mostly-continuous video where important moments
    are densely sampled and filler segments are sparsely sampled — analogous to
    speeding up playback over the boring parts. No cuts, no boundary jumps;
    encoder sees full-timeline temporal context.

    Args:
        sidecar: MSS sidecar dict (must contain ``segments`` list).
        alpha: Density floor for unimportant segments. Must be in (0, 1]. alpha=1.0
            yields uniform sampling (no fast-forward); alpha→0 approaches the
            vlm-selected hard-cut behavior.

    Notes:
        - Duration is baked into the stored weights so the adapter's per-segment
          frame budget is ``duration × density / sum`` directly.
        - Falls back to ``selector_failed`` if the sidecar has no segments.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    seg_records = sidecar.get("segments", [])
    if not seg_records:
        return SelectionResult(selector_failed=True, selector_failure_reason="empty segments list")
    chosen: list[tuple[int, float, float, str, float]] = []
    for seg in seg_records:
        idx = int(seg["index"])
        s = float(seg["start_s"])
        e = float(seg["end_s"])
        label = str(seg.get("label", "unimportant"))
        density = 1.0 if label == "important" else alpha
        # Bake duration into the per-segment weight so adapters (which allocate by
        # weight only, not weight × duration) end up with the right density.
        eff_w = max(0.0, e - s) * density
        chosen.append((idx, s, e, label, eff_w))
    chosen.sort(key=lambda r: r[1])
    kept_segments = [(r[1], r[2]) for r in chosen]
    kept_indices = [r[0] for r in chosen]
    weights = [r[4] for r in chosen]
    total = sum(r[2] - r[1] for r in chosen)
    return SelectionResult(
        kept_segments=kept_segments,
        kept_indices=kept_indices,
        kept_total_duration_s=total,
        kept_segment_count=len(kept_segments),
        kept_segment_weights=weights,
    )


def parse_fastforward_condition(condition: str) -> float | None:
    """Parse "vlm-fastforward-a<NN>" → alpha (float in (0, 1]) or None.

    NN is alpha × 100 as an integer (e.g. ``vlm-fastforward-a25`` → alpha=0.25).
    Allowed NN values are in ``FASTFORWARD_ALPHA_PERCENTS``.
    """
    prefix = "vlm-fastforward-a"
    if not condition.startswith(prefix):
        return None
    tok = condition[len(prefix):]
    if not tok.isdigit():
        return None
    pct = int(tok)
    if pct not in FASTFORWARD_ALPHA_PERCENTS:
        return None
    return pct / 100.0


def lowest_evidence_fastforward(sidecar: dict, alpha: float) -> SelectionResult:
    """Negative-control fast-forward: inverts the density rule of ``vlm_fastforward``.

    Important segments get density ``alpha`` (sparse) and unimportant segments get
    density 1.0 (dense). All other logic is identical to ``vlm_fastforward``: the full
    timeline is preserved (no cuts), and weights encode ``duration × density`` so the
    downstream adapter allocates the frame budget proportionally.

    If VLM signal drives ``vlm-fastforward`` gains, this inverted condition should
    underperform both ``full`` and ``vlm-fastforward`` — confirming the signal is real.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    seg_records = sidecar.get("segments", [])
    if not seg_records:
        return SelectionResult(selector_failed=True, selector_failure_reason="empty segments list")
    chosen: list[tuple[int, float, float, str, float]] = []
    for seg in seg_records:
        idx = int(seg["index"])
        s = float(seg["start_s"])
        e = float(seg["end_s"])
        label = str(seg.get("label", "unimportant"))
        # Inverted: important → sparse (alpha), unimportant → dense (1.0)
        density = alpha if label == "important" else 1.0
        eff_w = max(0.0, e - s) * density
        chosen.append((idx, s, e, label, eff_w))
    chosen.sort(key=lambda r: r[1])
    kept_segments = [(r[1], r[2]) for r in chosen]
    kept_indices = [r[0] for r in chosen]
    weights = [r[4] for r in chosen]
    total = sum(r[2] - r[1] for r in chosen)
    return SelectionResult(
        kept_segments=kept_segments,
        kept_indices=kept_indices,
        kept_total_duration_s=total,
        kept_segment_count=len(kept_segments),
        kept_segment_weights=weights,
    )


def parse_lowest_evidence_fastforward_condition(condition: str) -> float | None:
    """Parse "lowest-evidence-fastforward-a<NN>" → alpha or None.

    Same grammar as ``parse_fastforward_condition`` but with the negctrl prefix.
    """
    prefix = "lowest-evidence-fastforward-a"
    if not condition.startswith(prefix):
        return None
    tok = condition[len(prefix):]
    if not tok.isdigit():
        return None
    pct = int(tok)
    if pct not in FASTFORWARD_ALPHA_PERCENTS:
        return None
    return pct / 100.0


def parse_score_threshold_condition(condition: str) -> int | None:
    """Parse "vlm-score-threshold-t<NN>" → NN (integer percent) or None.

    Allowed NN values are in ``SCORE_THRESHOLD_PERCENTS``. Returns ``None`` for any
    other prefix or off-grid value so the runner falls through to the dispatch-failed
    error path. Note the return type is *int* (the percent), not the float threshold —
    callers convert to float via ``NN / 100.0`` when invoking ``vlm_strict``.
    """
    prefix = "vlm-score-threshold-t"
    if not condition.startswith(prefix):
        return None
    tok = condition[len(prefix):]
    if not tok.isdigit():
        return None
    pct = int(tok)
    if pct not in SCORE_THRESHOLD_PERCENTS:
        return None
    return pct


def lowest_evidence(sidecar: dict) -> SelectionResult:
    """Negative-control selector: rank by ascending weight, take the bottom-N
    where N matches vlm-selected's segment count.

    Earlier behavior strict-filtered on ``label == "unimportant"`` and bailed
    with selector_failed when MSS kept everything (all segments labeled
    "important"). That excluded ~3-4% of SSv2 videos from paired analyses.
    The rank-based variant always succeeds: when weights are tied (degenerate
    "MSS=full" case), the tiebreak is segment index ascending, so
    lowest-evidence picks the first-N temporal segments — still a meaningful
    contrast against vlm-selected on those videos (vlm-selected keeps ALL),
    even if the contrast collapses when N == total.
    """
    pool = _segments_with_indices(sidecar)
    if not pool:
        return SelectionResult(selector_failed=True, selector_failure_reason="empty segment pool")

    # Target count = number of "important" segments (matches vlm-selected count).
    important = [s for s in pool if s[3] == "important"]
    n_target = len(important)
    if n_target == 0:
        # No vlm-selected segments to match against. Still emit the lowest-weight
        # segment (single-segment fallback) rather than fail — keeps paired analyses intact.
        n_target = 1

    # Rank ascending by weight (read from raw sidecar), tiebreak by segment index ascending.
    weight_by_idx: dict[int, float] = {}
    for seg in sidecar.get("segments", []):
        idx = int(seg["index"])
        w = seg.get("weight")
        if w is None:
            w = seg.get("frequency", 0.0)
        weight_by_idx[idx] = float(w if w is not None else 0.0)

    ranked = sorted(pool, key=lambda s: (weight_by_idx.get(s[0], 0.0), s[0]))
    chosen = ranked[: min(n_target, len(ranked))]
    kept, idxs, total = _to_kept(chosen)
    return SelectionResult(
        kept_segments=kept, kept_indices=idxs,
        kept_total_duration_s=total, kept_segment_count=len(kept),
    )


def _random_match_duration(
    sidecar: dict, target_duration_s: float, rng: random.Random
) -> SelectionResult:
    pool = _segments_with_indices(sidecar)
    if not pool:
        return SelectionResult(selector_failed=True, selector_failure_reason="empty segment pool")
    tol = _tolerance(pool)
    total_pool_dur = sum(_duration(s) for s in pool)
    # Degenerate fallback: vlm-selected kept ≈ all → random forced to keep all.
    if target_duration_s >= total_pool_dur - tol:
        pool_sorted = sorted(pool, key=lambda s: s[1])
        kept, idxs, total = _to_kept(pool_sorted)
        return SelectionResult(
            kept_segments=kept, kept_indices=idxs,
            kept_total_duration_s=total, kept_segment_count=len(kept),
        )
    # Degenerate fallback: vlm-selected kept ≈ nothing → one random segment.
    if target_duration_s <= tol:
        idx = rng.randrange(len(pool))
        kept, idxs, total = _to_kept([pool[idx]])
        return SelectionResult(
            kept_segments=kept, kept_indices=idxs,
            kept_total_duration_s=total, kept_segment_count=len(kept),
        )
    shuffled = list(pool)
    rng.shuffle(shuffled)
    chosen: list[tuple[int, float, float, str]] = []
    accumulated = 0.0
    for seg in shuffled:
        if _passes_tolerance(accumulated, target_duration_s, tol):
            break
        seg_dur = _duration(seg)
        if accumulated + seg_dur > target_duration_s + tol:
            continue  # skip; would overshoot
        chosen.append(seg)
        accumulated += seg_dur
    if not _passes_tolerance(accumulated, target_duration_s, tol):
        # Final fallback: take greedy random until we cross target or exhaust pool.
        chosen = []
        accumulated = 0.0
        for seg in shuffled:
            chosen.append(seg)
            accumulated += _duration(seg)
            if accumulated >= target_duration_s:
                break
        if not chosen:
            chosen = [shuffled[0]]
    kept, idxs, total = _to_kept(chosen)
    return SelectionResult(kept_segments=kept, kept_indices=idxs, kept_total_duration_s=total, kept_segment_count=len(kept))


def _uniform_match_duration(sidecar: dict, target_duration_s: float) -> SelectionResult:
    pool = _segments_with_indices(sidecar)
    if not pool:
        return SelectionResult(selector_failed=True, selector_failure_reason="empty segment pool")
    tol = _tolerance(pool)
    pool_sorted = sorted(pool, key=lambda s: s[1])
    n = len(pool_sorted)
    total_pool_dur = sum(_duration(s) for s in pool_sorted)

    # Degenerate-case fallback 1: vlm-selected kept (essentially) all segments.
    # Target duration ≈ pool duration → uniform must keep all to match.
    # Without this, stride-search can fail on tolerance even though "keep all"
    # is the obviously-correct answer.
    if target_duration_s >= total_pool_dur - tol:
        kept, idxs, total = _to_kept(pool_sorted)
        return SelectionResult(
            kept_segments=kept, kept_indices=idxs,
            kept_total_duration_s=total, kept_segment_count=len(kept),
        )
    # Degenerate-case fallback 2: vlm-selected kept (essentially) nothing.
    # Target ≈ 0. Return a single center segment so the recognizer gets a clip
    # at all (empty input would fail downstream).
    if target_duration_s <= tol:
        center_idx = n // 2
        kept, idxs, total = _to_kept([pool_sorted[center_idx]])
        return SelectionResult(
            kept_segments=kept, kept_indices=idxs,
            kept_total_duration_s=total, kept_segment_count=len(kept),
        )

    if total_pool_dur < target_duration_s - tol:
        return SelectionResult(
            selector_failed=True,
            selector_failure_reason=f"pool duration {total_pool_dur:.3f}s < target {target_duration_s:.3f}s",
        )
    n_target = max(1, round(target_duration_s / (total_pool_dur / n)))
    n_target = min(n_target, n)

    for trial_n in (n_target, n_target + 1, n_target - 1, n_target + 2, n_target - 2):
        if trial_n < 1 or trial_n > n:
            continue
        if trial_n == 1:
            chosen_indices = [n // 2]
        else:
            stride = (n - 1) / (trial_n - 1)
            chosen_indices = [round(i * stride) for i in range(trial_n)]
        chosen_indices = sorted(set(chosen_indices))
        chosen = [pool_sorted[i] for i in chosen_indices]
        actual = sum(_duration(s) for s in chosen)
        if _passes_tolerance(actual, target_duration_s, tol):
            kept, idxs, total = _to_kept(chosen)
            return SelectionResult(kept_segments=kept, kept_indices=idxs, kept_total_duration_s=total, kept_segment_count=len(kept))
    # Final fallback: pick the n_target value that comes closest to target;
    # better to return a slightly-off duration than to fail entirely.
    best = None; best_diff = float("inf")
    for trial_n in range(max(1, n_target - 3), min(n + 1, n_target + 4)):
        if trial_n == 1:
            chosen_indices = [n // 2]
        else:
            stride = (n - 1) / (trial_n - 1)
            chosen_indices = sorted({round(i * stride) for i in range(trial_n)})
        chosen = [pool_sorted[i] for i in chosen_indices]
        actual = sum(_duration(s) for s in chosen)
        diff = abs(actual - target_duration_s)
        if diff < best_diff:
            best_diff = diff
            best = chosen
    if best:
        kept, idxs, total = _to_kept(best)
        return SelectionResult(
            kept_segments=kept, kept_indices=idxs,
            kept_total_duration_s=total, kept_segment_count=len(kept),
        )
    return SelectionResult(
        selector_failed=True,
        selector_failure_reason=f"uniform-stride could not hit target {target_duration_s:.3f}s within tolerance {tol:.3f}s",
    )


def _motion_match_duration(
    sidecar: dict, target_duration_s: float, motion_scores: dict[int, float]
) -> SelectionResult:
    pool = _segments_with_indices(sidecar)
    if not pool:
        return SelectionResult(selector_failed=True, selector_failure_reason="empty segment pool")
    missing = [s[0] for s in pool if s[0] not in motion_scores]
    if missing:
        return SelectionResult(
            selector_failed=True,
            selector_failure_reason=f"motion scores missing for segments {missing[:5]}…",
        )
    tol = _tolerance(pool)
    total_pool_dur = sum(_duration(s) for s in pool)
    # Degenerate fallback: vlm-selected kept ≈ all → motion picks all (highest-
    # motion-first ordering doesn't matter when we're forced to keep everything).
    if target_duration_s >= total_pool_dur - tol:
        pool_sorted = sorted(pool, key=lambda s: s[1])
        kept, idxs, total = _to_kept(pool_sorted)
        return SelectionResult(
            kept_segments=kept, kept_indices=idxs,
            kept_total_duration_s=total, kept_segment_count=len(kept),
        )
    # Degenerate fallback: vlm-selected kept ≈ nothing → return single highest-motion segment.
    if target_duration_s <= tol:
        by_motion = sorted(pool, key=lambda s: -motion_scores[s[0]])
        kept, idxs, total = _to_kept([by_motion[0]])
        return SelectionResult(
            kept_segments=kept, kept_indices=idxs,
            kept_total_duration_s=total, kept_segment_count=len(kept),
        )
    by_motion = sorted(pool, key=lambda s: -motion_scores[s[0]])
    chosen: list[tuple[int, float, float, str]] = []
    accumulated = 0.0
    for seg in by_motion:
        if _passes_tolerance(accumulated, target_duration_s, tol):
            break
        seg_dur = _duration(seg)
        if accumulated + seg_dur > target_duration_s + tol:
            continue
        chosen.append(seg)
        accumulated += seg_dur
    if not _passes_tolerance(accumulated, target_duration_s, tol):
        # Final fallback: take greedy top-motion until we exceed target, then
        # accept whatever's closest. Better to return slightly-off duration
        # than to fail entirely on edge-case targets.
        chosen = []
        accumulated = 0.0
        for seg in by_motion:
            chosen.append(seg)
            accumulated += _duration(seg)
            if accumulated >= target_duration_s:
                break
        if not chosen:
            chosen = [by_motion[0]]
    kept, idxs, total = _to_kept(chosen)
    return SelectionResult(kept_segments=kept, kept_indices=idxs, kept_total_duration_s=total, kept_segment_count=len(kept))


def _uniform_equal_segs(sidecar: dict, n_target_segments: int) -> SelectionResult:
    pool = _segments_with_indices(sidecar)
    if not pool:
        return SelectionResult(selector_failed=True, selector_failure_reason="empty segment pool")
    pool_sorted = sorted(pool, key=lambda s: s[1])
    n = len(pool_sorted)
    # Degenerate-case fallback: vlm-selected kept 0 segments. Fall back to a
    # single center segment so the recognizer gets a clip rather than failing.
    # This matches the behavior of `target_duration_s ≈ 0` in
    # _uniform_match_duration above.
    if n_target_segments < 1:
        chosen_indices = [n // 2]
        chosen = [pool_sorted[i] for i in chosen_indices]
        kept, idxs, total = _to_kept(chosen)
        return SelectionResult(
            kept_segments=kept, kept_indices=idxs,
            kept_total_duration_s=total, kept_segment_count=len(kept),
        )
    n_target = min(n_target_segments, n)
    if n_target == 1:
        chosen_indices = [n // 2]
    else:
        stride = (n - 1) / (n_target - 1)
        chosen_indices = sorted({round(i * stride) for i in range(n_target)})
    chosen = [pool_sorted[i] for i in chosen_indices]
    kept, idxs, total = _to_kept(chosen)
    return SelectionResult(kept_segments=kept, kept_indices=idxs, kept_total_duration_s=total, kept_segment_count=len(kept))


def load_motion_scores(video_id: str, cache_dir: str | Path) -> dict[int, float]:
    p = Path(cache_dir) / f"{video_id}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"motion cache miss for video_id={video_id!r} at {p}. "
            "Run `python -m oracle.scripts.vjepa2.compute_motion --eval-csv ... --mss-run-dir ...` "
            "to populate. See vjepa2-seven-condition-experiment/design.md Decision 6."
        )
    payload = json.load(open(p))
    return {int(s["index"]): float(s["mean_flow_magnitude"]) for s in payload.get("segments", [])}


def parse_id_condition(condition: str) -> tuple[str, int] | None:
    """Parse "id-<ordering>-f<NN>" → (ordering, percent_int) or None if not an id-condition.

    Validates that the ordering is in ``ID_ORDERINGS`` and the percent is one of
    ``ID_FRACTION_PERCENTS``. Returns ``None`` for any string that doesn't match the
    grammar — including unknown orderings or out-of-grid fractions — so callers can
    fall through to the legacy condition table.
    """
    if not condition.startswith("id-"):
        return None
    rest = condition[len("id-"):]
    if "-f" not in rest:
        return None
    ordering, _, frac_token = rest.rpartition("-f")
    if not frac_token.isdigit():
        return None
    if ordering not in ID_ORDERINGS:
        return None
    pct = int(frac_token)
    if pct not in ID_FRACTION_PERCENTS:
        return None
    return ordering, pct


def _rank_segments(
    pool: list[tuple[int, float, float, str]],
    ordering: str,
    sidecar: dict,
    *,
    rng: random.Random,
    motion_scores: dict[int, float] | None = None,
) -> list[tuple[int, float, float, str]]:
    """Return ``pool`` reordered with most-important first under ``ordering``.

    Ties are broken by ascending segment index for determinism.
    """
    if ordering == "vlm" or ordering == "anti-vlm":
        weight_by_idx: dict[int, float] = {}
        for seg in sidecar.get("segments", []):
            idx = int(seg["index"])
            w = seg.get("weight")
            if w is None:
                w = seg.get("frequency")
            weight_by_idx[idx] = float(w) if w is not None else 0.0
        sign = -1.0 if ordering == "vlm" else 1.0
        return sorted(pool, key=lambda s: (sign * weight_by_idx.get(s[0], 0.0), s[0]))
    if ordering == "random":
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled
    if ordering == "temporal":
        return sorted(pool, key=lambda s: (s[1], s[0]))
    if ordering == "motion":
        if not motion_scores:
            return []  # caller handles failure
        return sorted(pool, key=lambda s: (-motion_scores.get(s[0], 0.0), s[0]))
    raise ValueError(f"Unknown id-curve ordering: {ordering!r}")


def id_curve(
    sidecar: dict,
    ordering: str,
    fraction_pct: int,
    *,
    rng: random.Random,
    motion_scores: dict[int, float] | None = None,
) -> SelectionResult:
    """Insertion/Deletion curve cell: keep top ``fraction_pct``% of total-pool duration.

    Segments are ranked under ``ordering`` (most-important first) and accumulated until
    cumulative duration ≥ target. The target is ``fraction_pct/100 * pool_total_duration``;
    the last segment may push past the target by up to one segment-length, which is
    expected — the I/D analyzer integrates over the actual fraction-kept, not the
    nominal grid percent. Endpoints f=0 and f=100 are not part of the running grid;
    they are handled by the analyzer (chance level + existing full-video sidecar).
    """
    pool = _segments_with_indices(sidecar)
    if not pool:
        return SelectionResult(selector_failed=True, selector_failure_reason="empty segment pool")
    if ordering == "motion" and not motion_scores:
        return SelectionResult(
            selector_failed=True,
            selector_failure_reason=f"motion ordering requires motion cache for fraction={fraction_pct}",
        )

    pool_total = sum(_duration(s) for s in pool)
    if pool_total <= 0:
        return SelectionResult(selector_failed=True, selector_failure_reason="zero pool duration")
    target = (fraction_pct / 100.0) * pool_total

    ranked = _rank_segments(pool, ordering, sidecar, rng=rng, motion_scores=motion_scores)
    if not ranked:
        return SelectionResult(
            selector_failed=True,
            selector_failure_reason=f"ranking returned empty for ordering={ordering}",
        )

    chosen: list[tuple[int, float, float, str]] = []
    accumulated = 0.0
    for seg in ranked:
        if accumulated >= target:
            break
        chosen.append(seg)
        accumulated += _duration(seg)
    # Edge case: target < smallest segment → keep at least one segment so build_input
    # has something to decode. The analyzer corrects for this via actual_fraction_kept.
    if not chosen:
        chosen = [ranked[0]]
        accumulated = _duration(ranked[0])

    kept, idxs, total = _to_kept(chosen)
    return SelectionResult(
        kept_segments=kept,
        kept_indices=idxs,
        kept_total_duration_s=total,
        kept_segment_count=len(kept),
    )


def select(
    condition: str,
    sidecar: dict,
    *,
    run_seed: int,
    video_id: str,
    motion_cache_dir: str | Path | None = None,
    target_duration_s: float | None = None,
    target_segment_count: int | None = None,
) -> SelectionResult:
    """Dispatch to the right selector. ``target_*`` are inferred from VLM-selected if None."""
    rng, seed_hex = per_video_rng(run_seed, video_id, condition)

    parsed_ff = parse_fastforward_condition(condition)
    if parsed_ff is not None:
        result = vlm_fastforward(sidecar, alpha=parsed_ff)
        result.rng_seed_hex = seed_hex
        return result

    parsed_le_ff = parse_lowest_evidence_fastforward_condition(condition)
    if parsed_le_ff is not None:
        result = lowest_evidence_fastforward(sidecar, alpha=parsed_le_ff)
        result.rng_seed_hex = seed_hex
        return result

    parsed_st = parse_score_threshold_condition(condition)
    if parsed_st is not None:
        result = vlm_strict(sidecar, threshold=parsed_st / 100.0)
        if result.selector_failed:
            # Empty-set fallback: keep the single highest-weight segment so downstream
            # decoding has at least one frame. Mirrors id_curve's edge-case behavior
            # (line ~712). Note: this only triggers when no segment has weight >= T;
            # the typical case (T <= 0.9 on D48, most T on SSv2/K400) yields a non-empty
            # kept set and is returned unchanged.
            pool = _segments_with_indices(sidecar)
            if pool:
                weight_by_idx: dict[int, float] = {}
                for seg in sidecar.get("segments", []):
                    idx = int(seg["index"])
                    w = seg.get("weight")
                    if w is None:
                        w = seg.get("frequency", 0.0)
                    weight_by_idx[idx] = float(w) if w is not None else 0.0
                best = max(pool, key=lambda s: (weight_by_idx.get(s[0], 0.0), -s[0]))
                kept, idxs, total = _to_kept([best])
                result = SelectionResult(
                    kept_segments=kept,
                    kept_indices=idxs,
                    kept_total_duration_s=total,
                    kept_segment_count=len(kept),
                )
        result.rng_seed_hex = seed_hex
        return result

    parsed_id = parse_id_condition(condition)
    if parsed_id is not None:
        ordering, frac_pct = parsed_id
        motion_scores = None
        if ordering == "motion":
            if motion_cache_dir is None:
                result = SelectionResult(selector_failed=True, selector_failure_reason="motion_cache_dir not provided")
                result.rng_seed_hex = seed_hex
                return result
            try:
                motion_scores = load_motion_scores(video_id, motion_cache_dir)
            except FileNotFoundError as e:
                result = SelectionResult(selector_failed=True, selector_failure_reason=str(e))
                result.rng_seed_hex = seed_hex
                return result
        result = id_curve(sidecar, ordering, frac_pct, rng=rng, motion_scores=motion_scores)
        result.rng_seed_hex = seed_hex
        return result

    if target_duration_s is None or target_segment_count is None:
        vlm = vlm_selected(sidecar)
        if target_duration_s is None:
            target_duration_s = vlm.kept_total_duration_s
        if target_segment_count is None:
            target_segment_count = vlm.kept_segment_count

    if condition == "full":
        result = SelectionResult()  # full-video, kept_segments=[] sentinel; runner reads condition=full and passes None
    elif condition == "full-single":
        # Like "full" but using predict_kept(None, num_segments=1, num_views=1) — single-clip whole-video uniform.
        # The runner dispatches this condition to predict_kept with kept_segments=None and 1×1 sampling.
        result = SelectionResult()
    elif condition == "vlm-aggregate":
        # Marker condition: actual logic is in wrapper.predict_full_with_clip_weighting.
        # Returns "all segments kept" so the runner doesn't trip the no-kept-segments check.
        result = SelectionResult(
            kept_segments=[(float(s["start_s"]), float(s["end_s"])) for s in sidecar.get("segments", [])],
            kept_indices=[int(s["index"]) for s in sidecar.get("segments", [])],
            kept_total_duration_s=float(sum(float(s["end_s"]) - float(s["start_s"]) for s in sidecar.get("segments", []))),
            kept_segment_count=len(sidecar.get("segments", [])),
        )
    elif condition == "vlm-anchored":
        # Marker condition: actual logic is in wrapper.predict_anchored.
        result = SelectionResult(
            kept_segments=[(float(s["start_s"]), float(s["end_s"])) for s in sidecar.get("segments", [])],
            kept_indices=[int(s["index"]) for s in sidecar.get("segments", [])],
            kept_total_duration_s=float(sum(float(s["end_s"]) - float(s["start_s"]) for s in sidecar.get("segments", []))),
            kept_segment_count=len(sidecar.get("segments", [])),
        )
    elif condition == "vlm-anchored-hybrid":
        # Marker condition: actual logic is in wrapper.predict_anchored_hybrid.
        result = SelectionResult(
            kept_segments=[(float(s["start_s"]), float(s["end_s"])) for s in sidecar.get("segments", [])],
            kept_indices=[int(s["index"]) for s in sidecar.get("segments", [])],
            kept_total_duration_s=float(sum(float(s["end_s"]) - float(s["start_s"]) for s in sidecar.get("segments", []))),
            kept_segment_count=len(sidecar.get("segments", [])),
        )
    elif condition == "vlm-selected":
        result = vlm_selected(sidecar)
    elif condition == "vlm-weighted":
        result = vlm_weighted(sidecar)
    elif condition == "vlm-strict-0.9":
        result = vlm_strict(sidecar, threshold=VLM_STRICT_THRESHOLD)
    elif condition == "uniform-vs-strict-0.9":
        # Target the duration of the strict variant (different from default vlm-selected)
        strict = vlm_strict(sidecar, threshold=VLM_STRICT_THRESHOLD)
        if strict.selector_failed:
            result = SelectionResult(selector_failed=True, selector_failure_reason=f"vlm-strict-0.9 failed: {strict.selector_failure_reason}")
        else:
            result = _uniform_match_duration(sidecar, strict.kept_total_duration_s)
    elif condition == "lowest-evidence":
        result = lowest_evidence(sidecar)
    elif condition == "random":
        result = _random_match_duration(sidecar, target_duration_s, rng)
    elif condition == "uniform":
        result = _uniform_match_duration(sidecar, target_duration_s)
    elif condition == "motion":
        if motion_cache_dir is None:
            result = SelectionResult(selector_failed=True, selector_failure_reason="motion_cache_dir not provided")
        else:
            try:
                scores = load_motion_scores(video_id, motion_cache_dir)
                result = _motion_match_duration(sidecar, target_duration_s, scores)
            except FileNotFoundError as e:
                result = SelectionResult(selector_failed=True, selector_failure_reason=str(e))
    elif condition == "uniform-equal-segs":
        result = _uniform_equal_segs(sidecar, target_segment_count)
    else:
        raise ValueError(f"Unknown condition: {condition!r}. Known: {CONDITION_NAMES}")

    result.rng_seed_hex = seed_hex
    return result


def _self_test():
    """Minimal sanity assertions; run via `python -m oracle.scripts.vjepa2.selectors`."""
    sidecar = {
        "segments": [
            {"index": 0, "start_s": 0.0, "end_s": 0.5, "label": "important"},
            {"index": 1, "start_s": 0.5, "end_s": 1.0, "label": "unimportant"},
            {"index": 2, "start_s": 1.0, "end_s": 1.5, "label": "important"},
            {"index": 3, "start_s": 1.5, "end_s": 2.0, "label": "important"},
            {"index": 4, "start_s": 2.0, "end_s": 2.5, "label": "unimportant"},
            {"index": 5, "start_s": 2.5, "end_s": 3.0, "label": "important"},
        ],
    }
    vlm = vlm_selected(sidecar)
    assert vlm.kept_segment_count == 4, vlm
    assert abs(vlm.kept_total_duration_s - 2.0) < 1e-6, vlm

    le = lowest_evidence(sidecar)
    # n_target = #important = 4; all weights 0.0 → tiebreak by idx → picks [0,1,2,3] = 2.0s
    assert le.kept_segment_count == 4 and abs(le.kept_total_duration_s - 2.0) < 1e-6, le

    rnd_a = select("random", sidecar, run_seed=42, video_id="vidA")
    rnd_b = select("random", sidecar, run_seed=42, video_id="vidA")
    assert rnd_a.kept_segments == rnd_b.kept_segments, "RNG determinism failed"
    assert _passes_tolerance(rnd_a.kept_total_duration_s, 2.0), rnd_a

    rnd_c = select("random", sidecar, run_seed=42, video_id="vidB")
    assert rnd_a.rng_seed_hex != rnd_c.rng_seed_hex, "different videos must have different rng seeds"

    uni = select("uniform", sidecar, run_seed=42, video_id="vidA")
    assert _passes_tolerance(uni.kept_total_duration_s, 2.0), uni

    motion_scores = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.7, 4: 0.2, 5: 0.6}
    res = _motion_match_duration(sidecar, 2.0, motion_scores)
    assert _passes_tolerance(res.kept_total_duration_s, 2.0), res
    expected_motion_top = {1, 3, 5, 2}  # top 4 by score
    assert set(res.kept_indices) == expected_motion_top, res

    ues = _uniform_equal_segs(sidecar, n_target_segments=4)
    assert ues.kept_segment_count == 4

    starts = [s[0] for s in vlm.kept_segments]
    assert starts == sorted(starts), "kept_segments not sorted"
    assert vlm.kept_indices == sorted(vlm.kept_indices), "kept_indices order mismatch"

    # I/D condition parsing
    assert parse_id_condition("id-vlm-f50") == ("vlm", 50)
    assert parse_id_condition("id-anti-vlm-f10") == ("anti-vlm", 10)
    assert parse_id_condition("id-random-f90") == ("random", 90)
    assert parse_id_condition("id-vlm-f5") is None  # not in grid
    assert parse_id_condition("id-bogus-f50") is None
    assert parse_id_condition("vlm-selected") is None

    # I/D selector — pool_total = 6 * 0.5 = 3.0 s
    sel_50 = select("id-vlm-f50", sidecar, run_seed=42, video_id="vidA")
    # vlm ranking on weights tied at default 0.0 (no weight in test sidecar) →
    # falls back to (0.0, idx) tiebreak → segments in index order [0..5]; target 1.5s.
    # First 3 segs accumulate 1.5s exactly → keep 3.
    assert abs(sel_50.kept_total_duration_s - 1.5) < 1e-6, sel_50
    assert sel_50.kept_segment_count == 3, sel_50

    sel_30_a = select("id-random-f30", sidecar, run_seed=42, video_id="vidA")
    sel_30_b = select("id-random-f30", sidecar, run_seed=42, video_id="vidA")
    assert sel_30_a.kept_indices == sel_30_b.kept_indices, "id-random determinism"

    sel_30_c = select("id-random-f30", sidecar, run_seed=42, video_id="vidB")
    assert sel_30_a.rng_seed_hex != sel_30_c.rng_seed_hex, "different videos must have distinct rng seeds"

    # Temporal ordering = chronological → first segments win
    sel_temporal = select("id-temporal-f50", sidecar, run_seed=42, video_id="vidA")
    assert sel_temporal.kept_indices == [0, 1, 2], sel_temporal

    # I/D with weighted sidecar (vlm should rank by weight desc)
    weighted_sc = {
        "segments": [
            {"index": 0, "start_s": 0.0, "end_s": 0.5, "weight": 0.1, "label": "unimportant"},
            {"index": 1, "start_s": 0.5, "end_s": 1.0, "weight": 0.9, "label": "important"},
            {"index": 2, "start_s": 1.0, "end_s": 1.5, "weight": 0.5, "label": "important"},
            {"index": 3, "start_s": 1.5, "end_s": 2.0, "weight": 0.7, "label": "important"},
        ],
    }
    sel_w = select("id-vlm-f50", weighted_sc, run_seed=42, video_id="w")
    # pool_total=2.0, target=1.0; ranked-by-weight-desc: [1(0.9), 3(0.7), 2(0.5), 0(0.1)]
    # first 2 segs → 1.0s exactly
    assert sorted(sel_w.kept_indices) == [1, 3], sel_w

    sel_aw = select("id-anti-vlm-f50", weighted_sc, run_seed=42, video_id="w")
    # ranked-by-weight-asc: [0(0.1), 2(0.5), 3(0.7), 1(0.9)] → first 2 → idx [0, 2]
    assert sorted(sel_aw.kept_indices) == [0, 2], sel_aw

    # Score-threshold condition parsing
    assert parse_score_threshold_condition("vlm-score-threshold-t50") == 50
    assert parse_score_threshold_condition("vlm-score-threshold-t0") == 0
    assert parse_score_threshold_condition("vlm-score-threshold-t100") == 100
    assert parse_score_threshold_condition("vlm-score-threshold-t55") is None  # off-grid
    assert parse_score_threshold_condition("vlm-score-threshold-tabc") is None  # non-digit
    # Adjacent families must be rejected
    assert parse_score_threshold_condition("vlm-selected") is None
    assert parse_score_threshold_condition("vlm-strict-0.9") is None
    assert parse_score_threshold_condition("vlm-fastforward-a50") is None
    assert parse_score_threshold_condition("lowest-evidence-fastforward-a50") is None
    assert parse_score_threshold_condition("id-vlm-f50") is None

    # Score-threshold selector — synthetic 5-segment sidecar
    st_sc = {
        "segments": [
            {"index": 0, "start_s": 0.0, "end_s": 0.5, "weight": 0.20, "label": "unimportant"},
            {"index": 1, "start_s": 0.5, "end_s": 1.0, "weight": 0.60, "label": "important"},
            {"index": 2, "start_s": 1.0, "end_s": 1.5, "weight": 0.85, "label": "important"},
            {"index": 3, "start_s": 1.5, "end_s": 2.0, "weight": 0.40, "label": "unimportant"},
            {"index": 4, "start_s": 2.0, "end_s": 2.5, "weight": 0.95, "label": "important"},
        ],
    }
    st_50 = select("vlm-score-threshold-t50", st_sc, run_seed=42, video_id="stA")
    assert st_50.kept_indices == [1, 2, 4], st_50  # weight >= 0.5
    st_100 = select("vlm-score-threshold-t100", st_sc, run_seed=42, video_id="stA")
    # No segment has weight >= 1.0 → fallback to single highest-weight segment (index 4, w=0.95)
    assert st_100.kept_indices == [4], st_100
    assert st_100.kept_segment_count == 1
    st_0 = select("vlm-score-threshold-t0", st_sc, run_seed=42, video_id="stA")
    assert st_0.kept_indices == [0, 1, 2, 3, 4], st_0  # all segments kept

    print("selectors self-test OK")


if __name__ == "__main__":
    _self_test()


__all__ = [
    "DURATION_TOLERANCE_S",
    "CONDITION_NAMES",
    "NEEDS_MSS_SIDECAR",
    "NEEDS_MOTION_CACHE",
    "FASTFORWARD_ALPHA_PERCENTS",
    "SCORE_THRESHOLD_PERCENTS",
    "ID_ORDERINGS",
    "ID_FRACTION_PERCENTS",
    "SelectionResult",
    "per_video_rng",
    "vlm_selected",
    "lowest_evidence",
    "vlm_fastforward",
    "parse_fastforward_condition",
    "lowest_evidence_fastforward",
    "parse_lowest_evidence_fastforward_condition",
    "vlm_strict",
    "parse_score_threshold_condition",
    "id_curve",
    "parse_id_condition",
    "select",
    "load_motion_scores",
]
