#!/usr/bin/env python
"""Temporal coverage / diversity analysis for seven-condition classifier-eval runs.

Computes, per (video, condition), how the kept segments are distributed across
the video timeline:
  - range: max(end_s) - min(start_s)
  - coverage_ratio: range / video_duration
  - com_position: mean(midpoint_s) / video_duration  (0 = early, 1 = late)
  - mean_gap: mean inter-segment gap in seconds
  - gini: Gini coefficient of selected duration across N=10 timeline bins
          (0 = uniform, 1 = clustered)

Aggregates across videos, separately for SSv2 and K400, separately for each
condition (vlm-selected, uniform, motion, random, lowest-evidence,
uniform-equal-segs).

Also reports the per-video correlation between (uniform.coverage - vlm.coverage)
and (uniform.top1_correct - vlm.top1_correct) — the test of whether videos
where vlm-selected clusters more also have larger uniform-vs-vlm recognition
gaps.

Inputs:
  --ssv2-dir / --k400-dir : seven-condition run dirs containing
                             <video_id>__<condition>.json sidecars
  --ssv2-mss-dir / --k400-mss-dir : MSS run dirs used to look up video duration
                                     (from the last segment's end_s)
  --output-dir            : where to write CSV outputs
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Conditions to analyze. "full" is excluded from clustering metrics (it's the
# whole video) but kept for recognition-gap denominator computations.
CLUSTERING_CONDITIONS = [
    "vlm-selected",
    "uniform",
    "motion",
    "random",
    "lowest-evidence",
    "uniform-equal-segs",
]

# Standard pairs to compare. (a, b) reports gap = a - b for each metric.
COMPARISON_PAIRS = [
    ("uniform", "vlm-selected"),
    ("uniform", "motion"),
    ("uniform", "random"),
    ("uniform", "lowest-evidence"),
]

GINI_BINS = 10


def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient of a non-negative value distribution.

    0 = perfectly uniform, near-1 = perfectly concentrated.
    Returns 0 if total is 0 or values is empty.
    """
    if not values:
        return 0.0
    xs = sorted(v for v in values if v >= 0)
    n = len(xs)
    total = sum(xs)
    if total == 0 or n == 0:
        return 0.0
    # Standard formula: G = sum((2i - n - 1) * x_i) / (n * total), 1-indexed
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += (2 * i - n - 1) * x
    return cum / (n * total)


def compute_metrics(
    kept_segments: List[List[float]],
    duration: float,
    n_bins: int = GINI_BINS,
) -> Optional[Dict[str, float]]:
    """Return dict of metrics for one (video, condition).

    kept_segments is a list of [start_s, end_s] pairs.
    duration is the video's total duration in seconds.
    Returns None if kept_segments is empty or duration <= 0.
    """
    if not kept_segments or duration <= 0:
        return None

    sorted_segs = sorted(
        ((float(s), float(e)) for s, e in kept_segments if e > s),
        key=lambda p: p[0],
    )
    if not sorted_segs:
        return None

    starts = [s for s, _ in sorted_segs]
    ends = [e for _, e in sorted_segs]
    midpoints = [(s + e) / 2 for s, e in sorted_segs]
    seg_durations = [e - s for s, e in sorted_segs]

    range_s = max(ends) - min(starts)
    coverage_ratio = range_s / duration if duration > 0 else 0.0
    com_position = (sum(midpoints) / len(midpoints)) / duration if duration > 0 else 0.0

    # Inter-segment gaps; only defined when >= 2 segments.
    if len(sorted_segs) >= 2:
        gaps = [
            sorted_segs[i + 1][0] - sorted_segs[i][1]
            for i in range(len(sorted_segs) - 1)
        ]
        # Negative gaps (overlapping) clamped to 0
        gaps = [max(0.0, g) for g in gaps]
        mean_gap = sum(gaps) / len(gaps)
    else:
        mean_gap = 0.0

    # Gini on bin-distribution of selected duration
    bin_width = duration / n_bins
    bin_durations = [0.0] * n_bins
    for s, e in sorted_segs:
        # Distribute this segment's mass across bins it touches
        # Clamp to video bounds
        s_clamped = max(0.0, min(s, duration))
        e_clamped = max(0.0, min(e, duration))
        if e_clamped <= s_clamped:
            continue
        for b in range(n_bins):
            b_start = b * bin_width
            b_end = (b + 1) * bin_width if b < n_bins - 1 else duration
            overlap = max(0.0, min(e_clamped, b_end) - max(s_clamped, b_start))
            if overlap > 0:
                bin_durations[b] += overlap
    gini = gini_coefficient(bin_durations)

    return {
        "range_s": range_s,
        "coverage_ratio": coverage_ratio,
        "com_position": com_position,
        "mean_gap": mean_gap,
        "gini": gini,
        "n_kept_segments": len(sorted_segs),
        "kept_total_s": sum(seg_durations),
    }


def lookup_duration_from_mss(mss_dir: Path, video_id: str) -> Optional[float]:
    """Read MSS sidecar for video_id and return last segment's end_s as duration.

    Returns None if MSS sidecar is missing or has no segments.
    """
    path = mss_dir / f"{video_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    segs = d.get("segments") or []
    if not segs:
        return None
    last = segs[-1]
    end_s = last.get("end_s")
    if end_s is None:
        return None
    return float(end_s)


def discover_videos_and_conditions(
    classifier_dir: Path,
) -> Tuple[List[str], List[str]]:
    """Scan classifier_dir for <video_id>__<condition>.json files.

    Returns (sorted unique video_ids, sorted unique conditions).
    """
    video_ids: set = set()
    conditions: set = set()
    for p in classifier_dir.glob("*__*.json"):
        name = p.stem
        if "__" not in name:
            continue
        # Last "__" splits video_id from condition (since v_id can contain "_")
        idx = name.rfind("__")
        video_ids.add(name[:idx])
        conditions.add(name[idx + 2:])
    return sorted(video_ids), sorted(conditions)


def load_sidecar(
    classifier_dir: Path,
    video_id: str,
    condition: str,
) -> Optional[dict]:
    path = classifier_dir / f"{video_id}__{condition}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def aggregate_dataset(
    dataset: str,
    classifier_dir: Path,
    mss_dir: Path,
    conditions: List[str],
) -> Tuple[List[dict], Dict[str, Dict[str, List[float]]]]:
    """Build per-video records + per-condition aggregate data.

    Returns:
        per_video_rows: list of dicts (one per (video, condition)) with metrics
        agg: agg[condition][metric] -> list of values across videos
    """
    video_ids, all_conditions = discover_videos_and_conditions(classifier_dir)
    print(f"[{dataset}] {len(video_ids)} videos, conditions found: {all_conditions}", file=sys.stderr)

    per_video_rows: List[dict] = []
    agg: Dict[str, Dict[str, List[float]]] = {
        c: defaultdict(list) for c in conditions
    }

    duration_cache: Dict[str, Optional[float]] = {}
    n_missing_duration = 0
    n_missing_sidecar = 0

    for vid in video_ids:
        if vid not in duration_cache:
            duration_cache[vid] = lookup_duration_from_mss(mss_dir, vid)
        duration = duration_cache[vid]
        if duration is None:
            n_missing_duration += 1
            continue

        # Pull top1/top5 from full.json so we have a recognition baseline per video
        full_sc = load_sidecar(classifier_dir, vid, "full")
        full_top1 = full_sc.get("top1_correct") if full_sc else None
        full_top5 = full_sc.get("top5_correct") if full_sc else None

        for cond in conditions:
            sc = load_sidecar(classifier_dir, vid, cond)
            if sc is None:
                n_missing_sidecar += 1
                continue
            kept = sc.get("kept_segments") or []
            if sc.get("selector_failed"):
                continue
            metrics = compute_metrics(kept, duration)
            if metrics is None:
                continue
            row = {
                "dataset": dataset,
                "video_id": vid,
                "condition": cond,
                "duration_s": duration,
                "top1_correct": int(bool(sc.get("top1_correct"))) if sc.get("top1_correct") is not None else None,
                "top5_correct": int(bool(sc.get("top5_correct"))) if sc.get("top5_correct") is not None else None,
                "full_top1_correct": int(bool(full_top1)) if full_top1 is not None else None,
                "full_top5_correct": int(bool(full_top5)) if full_top5 is not None else None,
                **metrics,
            }
            per_video_rows.append(row)
            for k in ("range_s", "coverage_ratio", "com_position", "mean_gap", "gini",
                      "n_kept_segments", "kept_total_s"):
                agg[cond][k].append(metrics[k])

    print(
        f"[{dataset}] missing duration: {n_missing_duration} videos; "
        f"missing condition sidecars: {n_missing_sidecar}",
        file=sys.stderr,
    )
    return per_video_rows, agg


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def per_video_correlation(
    per_video_rows: List[dict],
    a: str,
    b: str,
    metric: str,
    correct_field: str = "top1_correct",
) -> Optional[Dict[str, float]]:
    """Per-video correlation between (a.metric - b.metric) and (a.correct - b.correct).

    Aggregates over videos that have BOTH conditions present.
    Positive correlation means: when a has more (e.g., higher coverage) than b,
    a is also more often correct than b.
    """
    by_vid: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for row in per_video_rows:
        by_vid[row["video_id"]][row["condition"]] = row
    diffs_metric = []
    diffs_correct = []
    for vid, conds in by_vid.items():
        if a not in conds or b not in conds:
            continue
        ra, rb = conds[a], conds[b]
        if ra.get(metric) is None or rb.get(metric) is None:
            continue
        if ra.get(correct_field) is None or rb.get(correct_field) is None:
            continue
        diffs_metric.append(ra[metric] - rb[metric])
        diffs_correct.append(ra[correct_field] - rb[correct_field])
    if len(diffs_metric) < 10:
        return None
    r = pearson(diffs_metric, diffs_correct)
    return {
        "n": len(diffs_metric),
        "pearson_r": r if r is not None else float("nan"),
        "mean_metric_diff": sum(diffs_metric) / len(diffs_metric),
        "mean_correct_diff": sum(diffs_correct) / len(diffs_correct),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ssv2-dir", type=Path,
        default=Path("pseudo_labels/classifier_eval/ssv2_vjepa2-ssv2_20260505_025747"),
    )
    p.add_argument(
        "--k400-dir", type=Path,
        default=Path("pseudo_labels/classifier_eval/k400_videomae-k400-large_20260505_025747"),
    )
    p.add_argument(
        "--ssv2-mss-dir", type=Path,
        default=Path("pseudo_labels/mss/qwen3-vl-32b_20260412_133129"),
    )
    p.add_argument(
        "--k400-mss-dir", type=Path,
        default=Path("pseudo_labels/mss/qwen3-vl-32b_k400_test_20260502_122107"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("pseudo_labels/classifier_eval/temporal_coverage_analysis"),
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=CLUSTERING_CONDITIONS,
        help="Conditions to analyze (default: vlm-selected, uniform, motion, "
             "random, lowest-evidence, uniform-equal-segs)",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    per_dataset_agg: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for dataset, cls_dir, mss_dir in [
        ("ssv2", args.ssv2_dir, args.ssv2_mss_dir),
        ("k400", args.k400_dir, args.k400_mss_dir),
    ]:
        if not cls_dir.exists():
            print(f"[{dataset}] classifier dir missing: {cls_dir} — skipping",
                  file=sys.stderr)
            continue
        if not mss_dir.exists():
            print(f"[{dataset}] mss dir missing: {mss_dir} — skipping",
                  file=sys.stderr)
            continue
        rows, agg = aggregate_dataset(dataset, cls_dir, mss_dir, args.conditions)
        all_rows.extend(rows)
        per_dataset_agg[dataset] = agg

    # Per-video CSV
    per_video_csv = args.output_dir / "temporal_coverage_per_video.csv"
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(per_video_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"Wrote {per_video_csv}  ({len(all_rows)} rows)", file=sys.stderr)

    # Aggregate summary CSV
    summary_csv = args.output_dir / "temporal_coverage_summary.csv"
    summary_rows = []
    for dataset, agg in per_dataset_agg.items():
        for cond in args.conditions:
            for metric in ("range_s", "coverage_ratio", "com_position",
                           "mean_gap", "gini", "n_kept_segments", "kept_total_s"):
                vals = agg.get(cond, {}).get(metric, [])
                m, s = mean_std(vals)
                summary_rows.append({
                    "dataset": dataset,
                    "condition": cond,
                    "metric": metric,
                    "mean": m,
                    "std": s,
                    "n": len(vals),
                })
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "condition", "metric", "mean", "std", "n"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"Wrote {summary_csv}  ({len(summary_rows)} rows)", file=sys.stderr)

    # Pairwise comparisons + per-video correlations
    correlations_csv = args.output_dir / "temporal_coverage_correlations.csv"
    corr_rows = []
    for dataset, agg in per_dataset_agg.items():
        ds_rows = [r for r in all_rows if r["dataset"] == dataset]
        for a, b in COMPARISON_PAIRS:
            for metric in ("coverage_ratio", "mean_gap", "gini", "range_s"):
                # Aggregate gap
                a_vals = agg.get(a, {}).get(metric, [])
                b_vals = agg.get(b, {}).get(metric, [])
                am, _ = mean_std(a_vals)
                bm, _ = mean_std(b_vals)
                gap = am - bm if not (math.isnan(am) or math.isnan(bm)) else float("nan")
                # Per-video correlation between metric gap and recognition gap
                corr = per_video_correlation(ds_rows, a, b, metric, "top1_correct")
                corr_rows.append({
                    "dataset": dataset,
                    "condition_a": a,
                    "condition_b": b,
                    "metric": metric,
                    "mean_a": am,
                    "mean_b": bm,
                    "gap_a_minus_b": gap,
                    "n_paired_videos": corr["n"] if corr else 0,
                    "pearson_r_metric_vs_top1": corr["pearson_r"] if corr else float("nan"),
                    "mean_top1_diff_a_minus_b": corr["mean_correct_diff"] if corr else float("nan"),
                })
    with open(correlations_csv, "w", newline="") as f:
        if corr_rows:
            w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
            w.writeheader()
            for r in corr_rows:
                w.writerow(r)
    print(f"Wrote {correlations_csv}  ({len(corr_rows)} rows)", file=sys.stderr)

    # Print human-readable summary table to stdout
    print("\n========================================")
    print("TEMPORAL COVERAGE SUMMARY")
    print("========================================")
    for dataset in ("ssv2", "k400"):
        if dataset not in per_dataset_agg:
            continue
        agg = per_dataset_agg[dataset]
        print(f"\n=== {dataset.upper()} ===")
        print(f"{'condition':<22s} {'cov_ratio':>12s} {'mean_gap_s':>12s} {'gini':>8s} {'n_kept':>8s} {'n_videos':>10s}")
        for cond in args.conditions:
            cm, _ = mean_std(agg.get(cond, {}).get("coverage_ratio", []))
            gm, _ = mean_std(agg.get(cond, {}).get("mean_gap", []))
            gn, _ = mean_std(agg.get(cond, {}).get("gini", []))
            nk, _ = mean_std(agg.get(cond, {}).get("n_kept_segments", []))
            nvids = len(agg.get(cond, {}).get("coverage_ratio", []))
            print(f"{cond:<22s} {cm:>12.3f} {gm:>12.3f} {gn:>8.3f} {nk:>8.1f} {nvids:>10d}")

    print("\n========================================")
    print("PAIRWISE GAPS + PER-VIDEO CORRELATIONS (top1)")
    print("========================================")
    for r in corr_rows:
        if r["metric"] != "coverage_ratio":
            continue
        print(
            f"  {r['dataset']:<5s} {r['condition_a']:>12s} - {r['condition_b']:<18s} "
            f"{r['metric']:<14s} gap={r['gap_a_minus_b']:+.3f}  "
            f"r(metric_diff, top1_diff)={r['pearson_r_metric_vs_top1']:+.3f} "
            f"(n={r['n_paired_videos']})"
        )


if __name__ == "__main__":
    main()
