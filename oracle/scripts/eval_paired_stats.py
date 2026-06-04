#!/usr/bin/env python
"""Paired-bootstrap CIs + McNemar's test for the seven-condition experiment.

Per the development design notes Decision 7.

Reads sidecars from a run directory (one per ``(video_id, condition)``), aligns
across conditions, runs paired bootstrap and McNemar's test, applies Bonferroni
correction over the 6 baseline comparisons, and emits both a JSON metrics file
and a paper-ready Markdown report. Per-confidence-bucket and per-length-quartile
breakdowns come from the eval CSV's meta JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LOG = logging.getLogger("eval_paired_stats")

DEFAULT_CONDITIONS = ("full", "vlm-selected", "random", "uniform", "motion", "lowest-evidence", "uniform-equal-segs")
DEFAULT_REFERENCE = "vlm-selected"
RAW_ALPHA = 0.05


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=str)
    p.add_argument("--eval-meta", required=True, type=str, help="Path to <eval_csv>.meta.json for breakdown dims")
    p.add_argument("--mss-run-dir", required=True, type=str, help="Source MSS run dir for confidence/length lookup per video")
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conditions", default=None,
                   help="Comma-separated condition names to include. Default: all 7 (full, vlm-selected, random, uniform, motion, lowest-evidence, uniform-equal-segs).")
    p.add_argument("--reference-condition", default=DEFAULT_REFERENCE,
                   help=f"Reference condition for paired comparisons (default: {DEFAULT_REFERENCE}).")
    p.add_argument("--combined", action="store_true",
                   help="Run BOTH vlm-selected-as-reference AND full-video-as-reference "
                        "(12 paired tests total) with Holm-Bonferroni and N=12 Bonferroni "
                        "correction. Outputs paired_stats_combined.{json,md}.")
    p.add_argument("--stratify-by-kept-ratio", action="store_true",
                   help="In addition to the overall combined analysis, run the same combined "
                        "analysis (12 tests + Holm) within each kept_ratio bin (<0.4, 0.4-0.6, "
                        "0.6-0.8, ≥0.8). Requires --combined. Outputs paired_stats_stratified.{json,md}.")
    p.add_argument("--output-json", default=None, help="Default: <run-dir>/paired_stats.json (or paired_stats_combined.json with --combined)")
    p.add_argument("--output-md", default=None, help="Default: <run-dir>/paired_stats_report.md (or paired_stats_combined_report.md with --combined)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def load_records(run_dir: Path) -> dict[tuple[str, str], dict]:
    """Glob `*__*.json` sidecars; return {(video_id, condition): record}."""
    out = {}
    for p in run_dir.glob("*__*.json"):
        if "__" not in p.stem:
            continue
        if p.name.startswith(("results_shard_", "summary_shard_", "paired_stats_")):
            continue
        try:
            r = json.load(open(p))
        except json.JSONDecodeError:
            LOG.warning("Skipping malformed sidecar: %s", p)
            continue
        # Skip non-record JSONs (e.g., reporter output files that happen to glob-match)
        if not isinstance(r, dict) or "video_id" not in r or "condition" not in r:
            continue
        out[(r["video_id"], r["condition"])] = r
    return out


def align_records(records: dict[tuple[str, str], dict], conditions: tuple[str, ...]) -> tuple[list[str], dict[str, dict[str, dict]]]:
    """Return (video_ids_with_all_conditions, {video_id: {condition: record}})."""
    by_video: dict[str, dict[str, dict]] = defaultdict(dict)
    for (vid, cond), rec in records.items():
        by_video[vid][cond] = rec
    aligned_video_ids: list[str] = []
    for vid, conds in by_video.items():
        if set(conds.keys()) >= set(conditions):
            aligned_video_ids.append(vid)
    return sorted(aligned_video_ids), by_video


def video_failed_any(by_video_record: dict[str, dict], conditions: tuple[str, ...]) -> bool:
    """Per design.md Decision 3: a video failing any duration-matched selector is excluded across all conditions."""
    return any(by_video_record[c].get("selector_failed", False) for c in conditions)


def correctness_matrix(video_ids: list[str], by_video: dict[str, dict[str, dict]], conditions: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return (top1_matrix, top5_matrix) shaped (n_videos, n_conditions)."""
    n, k = len(video_ids), len(conditions)
    top1 = np.zeros((n, k), dtype=np.int8)
    top5 = np.zeros((n, k), dtype=np.int8)
    for i, vid in enumerate(video_ids):
        for j, cond in enumerate(conditions):
            rec = by_video[vid][cond]
            top1[i, j] = int(bool(rec.get("top1_correct", False)))
            top5[i, j] = int(bool(rec.get("top5_correct", False)))
    return top1, top5


def paired_bootstrap(matrix: np.ndarray, replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-condition mean accuracy + 95% CI lo/hi via paired (resample-by-video) bootstrap."""
    n_videos = matrix.shape[0]
    if n_videos == 0:
        zeros = np.zeros(matrix.shape[1])
        return zeros, zeros, zeros
    rng = np.random.default_rng(seed)
    means = matrix.mean(axis=0)
    boot_means = np.empty((replicates, matrix.shape[1]))
    for r in range(replicates):
        idx = rng.integers(0, n_videos, size=n_videos)
        boot_means[r] = matrix[idx].mean(axis=0)
    ci_lo = np.percentile(boot_means, 2.5, axis=0)
    ci_hi = np.percentile(boot_means, 97.5, axis=0)
    return means, ci_lo, ci_hi


def paired_diff_bootstrap(matrix: np.ndarray, vlm_idx: int, baseline_idx: int, replicates: int, seed: int) -> dict:
    n_videos = matrix.shape[0]
    if n_videos == 0:
        return {"mean_diff": 0.0, "diff_ci_lo": 0.0, "diff_ci_hi": 0.0}
    rng = np.random.default_rng(seed + baseline_idx)
    diffs = matrix[:, vlm_idx] - matrix[:, baseline_idx]
    boot_means = np.empty(replicates)
    for r in range(replicates):
        idx = rng.integers(0, n_videos, size=n_videos)
        boot_means[r] = diffs[idx].mean()
    return {
        "mean_diff": float(diffs.mean()),
        "diff_ci_lo": float(np.percentile(boot_means, 2.5)),
        "diff_ci_hi": float(np.percentile(boot_means, 97.5)),
    }


def mcnemar(matrix: np.ndarray, vlm_idx: int, baseline_idx: int) -> dict:
    """McNemar's test on paired binary outcomes with continuity correction."""
    a = matrix[:, vlm_idx]
    b = matrix[:, baseline_idx]
    b10 = int(np.sum((a == 1) & (b == 0)))  # vlm correct, baseline wrong
    b01 = int(np.sum((a == 0) & (b == 1)))  # baseline correct, vlm wrong
    if b10 + b01 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b10_vlm_only": b10, "b01_baseline_only": b01}
    chi2 = (abs(b10 - b01) - 1) ** 2 / (b10 + b01)
    # Survival of chi-square with df=1
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "b10_vlm_only": b10,
        "b01_baseline_only": b01,
    }


def holm_bonferroni(p_values: list[tuple[str, float]], alpha: float) -> dict[str, dict]:
    """Holm step-down on (test_name, p_value) tuples.

    Sort p-values ascending. Compare the k-th smallest (1-indexed) against α / (N − k + 1).
    The standard step-down rule: if the k-th test fails to reject (p_k ≥ threshold_k),
    then ALL subsequent tests in the rank order also fail to reject — even if their
    individual p-values would have rejected on their own thresholds.

    Returns: dict[test_name → {holm_significant, holm_threshold, holm_rank}]
    """
    n = len(p_values)
    ranked = sorted(p_values, key=lambda x: (x[1], x[0]))
    out: dict[str, dict] = {}
    cascaded_failure = False
    for k_zero, (name, p) in enumerate(ranked):
        threshold = alpha / (n - k_zero)  # equivalent to α / (N − k + 1) for 1-indexed k
        if cascaded_failure:
            sig = False
        elif p < threshold:
            sig = True
        else:
            sig = False
            cascaded_failure = True
        out[name] = {
            "holm_significant": sig,
            "holm_threshold": float(threshold),
            "holm_rank": k_zero + 1,
        }
    return out


def compute_breakdowns(
    video_ids: list[str], by_video: dict[str, dict[str, dict]],
    mss_run_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Map each video into a confidence bucket and length quartile by reading its MSS sidecar."""
    confs: dict[str, float] = {}
    durations: dict[str, float] = {}
    for vid in video_ids:
        sp = mss_run_dir / f"{vid}.json"
        try:
            sc = json.load(open(sp))
        except FileNotFoundError:
            continue
        responses = (sc.get("mss_result") or {}).get("precheck_responses") or []
        if responses and responses[0].get("confidence") is not None:
            confs[vid] = float(responses[0]["confidence"])
        segs = sc.get("segments") or []
        if segs:
            durations[vid] = float(segs[-1].get("end_s", 0.0))

    def tercile_label(values: list[float], v: float) -> str:
        sorted_v = sorted(values)
        n = len(sorted_v)
        if n == 0:
            return "unknown"
        q1 = sorted_v[n // 3]
        q2 = sorted_v[(2 * n) // 3]
        if v < q1:
            return "low"
        if v < q2:
            return "medium"
        return "high"

    def quartile_label(values: list[float], v: float) -> str:
        sorted_v = sorted(values)
        n = len(sorted_v)
        if n == 0:
            return "unknown"
        q1 = sorted_v[n // 4]
        q2 = sorted_v[(2 * n) // 4]
        q3 = sorted_v[(3 * n) // 4]
        if v < q1:
            return "Q1"
        if v < q2:
            return "Q2"
        if v < q3:
            return "Q3"
        return "Q4"

    conf_values = list(confs.values())
    dur_values = list(durations.values())

    by_conf: dict[str, list[str]] = defaultdict(list)
    by_len: dict[str, list[str]] = defaultdict(list)
    for vid in video_ids:
        c = confs.get(vid)
        d = durations.get(vid)
        bc = tercile_label(conf_values, c) if c is not None else "unknown"
        bl = quartile_label(dur_values, d) if d is not None else "unknown"
        by_conf[bc].append(vid)
        by_len[bl].append(vid)
    return by_conf, by_len


def stratified_eval(
    matrix_top1: np.ndarray, matrix_top5: np.ndarray,
    video_ids: list[str], subset_video_ids: list[str],
    replicates: int, seed: int,
    conditions: tuple[str, ...], reference: str, bonferroni_alpha: float,
) -> dict:
    if not subset_video_ids:
        return {"n": 0}
    idx_map = {v: i for i, v in enumerate(video_ids)}
    sel = np.array([idx_map[v] for v in subset_video_ids if v in idx_map])
    if sel.size == 0:
        return {"n": 0}
    sub1 = matrix_top1[sel]
    sub5 = matrix_top5[sel]
    means1, lo1, hi1 = paired_bootstrap(sub1, replicates, seed)
    means5, lo5, hi5 = paired_bootstrap(sub5, replicates, seed + 7919)

    per_cond = {}
    for j, cond in enumerate(conditions):
        per_cond[cond] = {
            "top1_acc": float(means1[j]),
            "top1_ci_lo": float(lo1[j]),
            "top1_ci_hi": float(hi1[j]),
            "top5_acc": float(means5[j]),
            "top5_ci_lo": float(lo5[j]),
            "top5_ci_hi": float(hi5[j]),
        }
    ref_j = conditions.index(reference)
    baselines = tuple(c for c in conditions if c != reference)
    pairwise = {}
    for cond in baselines:
        bj = conditions.index(cond)
        diff = paired_diff_bootstrap(sub1, ref_j, bj, replicates, seed + 1009)
        m = mcnemar(sub1, ref_j, bj)
        pairwise[f"{reference}_vs_{cond}"] = {
            **diff, **m,
            "significant": m["p_value"] < bonferroni_alpha,
        }
    return {"n": int(sel.size), "per_condition": per_cond, "pairwise_top1": pairwise}


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def render_markdown(metrics: dict) -> str:
    overall = metrics["overall"]
    conditions = tuple(metrics["conditions"])
    reference = metrics["reference_condition"]
    baselines = tuple(c for c in conditions if c != reference)
    bonferroni_alpha = metrics["bonferroni_alpha"]
    lines = [
        f"# {len(conditions)}-condition paired-stats report",
        "",
        f"- Run dir: `{metrics['run_dir']}`",
        f"- Recognizer: `{metrics['recognizer']}`",
        f"- Eval set: `{metrics['eval_meta_path']}`",
        f"- Conditions: {', '.join(conditions)}",
        f"- Reference for paired comparisons: `{reference}`",
        f"- Videos with all {len(conditions)} conditions and no selector failure: **{overall['n']}**",
        f"- Bootstrap replicates: {metrics['bootstrap_replicates']}",
        f"- α (raw): {RAW_ALPHA}, Bonferroni-corrected for {len(baselines)} comparisons: **α* = {bonferroni_alpha:.4f}**",
        "",
        "## Headline accuracy (overall)",
        "",
        md_table(
            ["condition", "top-1 acc", "top-1 95% CI", "top-5 acc", "top-5 95% CI"],
            [
                [
                    cond,
                    f"{overall['per_condition'][cond]['top1_acc']:.4f}",
                    f"[{overall['per_condition'][cond]['top1_ci_lo']:.4f}, {overall['per_condition'][cond]['top1_ci_hi']:.4f}]",
                    f"{overall['per_condition'][cond]['top5_acc']:.4f}",
                    f"[{overall['per_condition'][cond]['top5_ci_lo']:.4f}, {overall['per_condition'][cond]['top5_ci_hi']:.4f}]",
                ]
                for cond in conditions
            ],
        ),
        "",
        f"## Paired differences vs {reference} (top-1)",
        "",
        md_table(
            ["baseline", "mean diff", "95% CI", "McNemar χ²", "p-value", "significant @ α*"],
            [
                [
                    cond,
                    f"{overall['pairwise_top1'][f'{reference}_vs_{cond}']['mean_diff']:+.4f}",
                    f"[{overall['pairwise_top1'][f'{reference}_vs_{cond}']['diff_ci_lo']:+.4f}, {overall['pairwise_top1'][f'{reference}_vs_{cond}']['diff_ci_hi']:+.4f}]",
                    f"{overall['pairwise_top1'][f'{reference}_vs_{cond}']['chi2']:.2f}",
                    f"{overall['pairwise_top1'][f'{reference}_vs_{cond}']['p_value']:.4g}",
                    "✓" if overall['pairwise_top1'][f'{reference}_vs_{cond}']['significant'] else "—",
                ]
                for cond in baselines
            ],
        ),
        "",
        "## Per-confidence-bucket breakdown (top-1)",
        "",
    ]
    for bucket in ("low", "medium", "high"):
        b = metrics["by_confidence"].get(bucket, {})
        if not b or b.get("n", 0) == 0:
            continue
        lines += [
            f"### Confidence = `{bucket}` (n={b['n']})",
            "",
            md_table(
                ["condition", "top-1 acc", "top-1 95% CI"],
                [
                    [cond, f"{b['per_condition'][cond]['top1_acc']:.4f}",
                     f"[{b['per_condition'][cond]['top1_ci_lo']:.4f}, {b['per_condition'][cond]['top1_ci_hi']:.4f}]"]
                    for cond in conditions
                ],
            ),
            "",
        ]
    lines += [
        "## Per-length-quartile breakdown (top-1)",
        "",
    ]
    for q in ("Q1", "Q2", "Q3", "Q4"):
        b = metrics["by_length"].get(q, {})
        if not b or b.get("n", 0) == 0:
            continue
        lines += [
            f"### Length = `{q}` (n={b['n']})",
            "",
            md_table(
                ["condition", "top-1 acc", "top-1 95% CI"],
                [
                    [cond, f"{b['per_condition'][cond]['top1_acc']:.4f}",
                     f"[{b['per_condition'][cond]['top1_ci_lo']:.4f}, {b['per_condition'][cond]['top1_ci_hi']:.4f}]"]
                    for cond in conditions
                ],
            ),
            "",
        ]
    return "\n".join(lines)


def compute_pairwise_for_reference(
    top1_matrix: np.ndarray,
    conditions: tuple[str, ...],
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, dict]:
    """Compute paired_diff_bootstrap + mcnemar for a single reference vs each baseline."""
    ref_j = conditions.index(reference)
    baselines = tuple(c for c in conditions if c != reference)
    pairwise: dict[str, dict] = {}
    for cond in baselines:
        bj = conditions.index(cond)
        diff = paired_diff_bootstrap(top1_matrix, ref_j, bj, replicates, seed)
        m = mcnemar(top1_matrix, ref_j, bj)
        pairwise[f"{reference}_vs_{cond}"] = {**diff, **m}
    return pairwise


def render_combined_markdown(metrics: dict) -> str:
    overall = metrics["overall"]
    conditions = tuple(metrics["conditions"])
    bonferroni_alpha_global = metrics["bonferroni_alpha_global"]  # α / 12
    raw_alpha = metrics["raw_alpha"]

    def _row(comp_name: str, ref: str, baseline: str) -> list[str]:
        c = overall["pairwise_top1_combined"][comp_name]
        ref_acc = overall["per_condition"][ref]["top1_acc"]
        bas_acc = overall["per_condition"][baseline]["top1_acc"]
        bonf_n12 = "✓" if c["p_value"] < bonferroni_alpha_global else "—"
        holm = "✓" if c["holm_significant"] else "—"
        raw = "✓" if c["p_value"] < raw_alpha else "—"
        return [
            baseline,
            f"{ref_acc:.4f}",
            f"{bas_acc:.4f}",
            f"{c['mean_diff']:+.4f}",
            f"[{c['diff_ci_lo']:+.4f}, {c['diff_ci_hi']:+.4f}]",
            f"{c['p_value']:.4g}",
            raw,
            bonf_n12,
            holm,
        ]

    n_global = metrics["n_comparisons_global"]
    headers = ["compared", f"ref top-1", "compared top-1", "ref − cmp", "95% CI", "raw p", f"raw α<{raw_alpha:g}", f"Bonf α/{n_global}<{bonferroni_alpha_global:.4f}", f"Holm α={raw_alpha:g}"]

    lines = [
        f"# Combined paired-stats report (N={n_global}, Holm-Bonferroni)",
        "",
        f"- Run dir: `{metrics['run_dir']}`",
        f"- Recognizer: `{metrics['recognizer']}`",
        f"- Eval set: `{metrics['eval_meta_path']}`",
        f"- Conditions: {', '.join(conditions)}",
        f"- Eligible videos (all conditions, no selector failure): **{overall['n']}**",
        f"- Bootstrap replicates: {metrics['bootstrap_replicates']}",
        f"- Total comparisons: {n_global}",
        f"- α (raw) = {raw_alpha}, strict Bonferroni α/N = {bonferroni_alpha_global:.4f}, Holm-Bonferroni applied across all {n_global} tests",
        "",
        "## Headline accuracy",
        "",
        md_table(
            ["condition", "top-1 acc", "top-1 95% CI", "top-5 acc", "top-5 95% CI"],
            [
                [
                    cond,
                    f"{overall['per_condition'][cond]['top1_acc']:.4f}",
                    f"[{overall['per_condition'][cond]['top1_ci_lo']:.4f}, {overall['per_condition'][cond]['top1_ci_hi']:.4f}]",
                    f"{overall['per_condition'][cond]['top5_acc']:.4f}",
                    f"[{overall['per_condition'][cond]['top5_ci_lo']:.4f}, {overall['per_condition'][cond]['top5_ci_hi']:.4f}]",
                ]
                for cond in conditions
            ],
        ),
        "",
        "## Table A — vlm-selected as reference (top-1)",
        "",
        md_table(
            headers,
            [_row(f"vlm-selected_vs_{c}", "vlm-selected", c)
             for c in conditions if c != "vlm-selected" and f"vlm-selected_vs_{c}" in overall["pairwise_top1_combined"]],
        ),
        "",
        "## Table B — full as reference (top-1)",
        "",
        md_table(
            headers,
            [_row(f"full_vs_{c}", "full", c)
             for c in conditions if c != "full" and f"full_vs_{c}" in overall["pairwise_top1_combined"]],
        ),
        "",
        f"## Summary",
        "",
        f"- {metrics['recognizer']}: **{metrics['n_significant_holm']} of {metrics['n_comparisons_global']} comparisons** significant under Holm-Bonferroni at α={raw_alpha}.",
        f"- {metrics['n_significant_bonferroni']} of {metrics['n_comparisons_global']} significant under strict Bonferroni (α/N={bonferroni_alpha_global:.4f}).",
        "",
    ]
    return "\n".join(lines)


KEPT_RATIO_BINS = (
    ("bin1_lt_0.4",      0.0,    0.4,    "kept_ratio < 0.4 (highly selective)"),
    ("bin2_0.4_to_0.6",  0.4,    0.6,    "0.4 ≤ kept_ratio < 0.6 (moderately selective)"),
    ("bin3_0.6_to_0.8",  0.6,    0.8,    "0.6 ≤ kept_ratio < 0.8 (light selection)"),
    ("bin4_ge_0.8",      0.8,    1.001,  "kept_ratio ≥ 0.8 (almost everything kept)"),
)


def compute_kept_ratios(video_ids: list[str], mss_run_dir: Path) -> dict[str, float]:
    """Per video: kept_ratio = sum(important segment durations) / total video duration."""
    out: dict[str, float] = {}
    for vid in video_ids:
        sp = mss_run_dir / f"{vid}.json"
        try:
            sc = json.load(open(sp))
        except FileNotFoundError:
            continue
        segs = sc.get("segments") or []
        if not segs:
            continue
        important_dur = sum(
            float(s["end_s"]) - float(s["start_s"])
            for s in segs if s.get("label") == "important"
        )
        total_dur = float(segs[-1]["end_s"])
        if total_dur > 0:
            out[vid] = important_dur / total_dur
    return out


def bin_by_kept_ratio(video_ids: list[str], kept_ratios: dict[str, float]) -> dict[str, list[str]]:
    """Bucket video_ids into the four kept-ratio bins; returns empty list for missing."""
    bins: dict[str, list[str]] = {bin_name: [] for bin_name, _, _, _ in KEPT_RATIO_BINS}
    for vid in video_ids:
        kr = kept_ratios.get(vid)
        if kr is None:
            continue
        for bin_name, lo, hi, _ in KEPT_RATIO_BINS:
            if lo <= kr < hi:
                bins[bin_name].append(vid)
                break
    return bins


def render_stratified_combined_markdown(metrics: dict) -> str:
    """Combined-mode report extended with per-kept-ratio-bin sections."""
    base_md = render_combined_markdown(metrics)
    bins = metrics.get("by_kept_ratio", {})
    if not bins:
        return base_md
    lines = [base_md, "", "---", "", "# Stratified by kept_ratio", "",
             "Each bin gets its own combined analysis (12 tests, Holm-Bonferroni internal to that bin)."]
    bin_descriptions = {b[0]: b[3] for b in KEPT_RATIO_BINS}
    for bin_name, _, _, desc in KEPT_RATIO_BINS:
        b = bins.get(bin_name)
        if not b or b.get("n_eligible", 0) == 0:
            lines += ["", f"## {bin_name} — {desc} (n=0, skipped)", ""]
            continue
        lines += [
            "",
            f"## {bin_name} — {desc} (n={b['n_eligible']})",
            "",
            f"- {b['n_significant_holm']} of {b['n_comparisons_global']} comparisons significant under Holm-Bonferroni.",
            "",
            "### Headline accuracy",
            "",
            md_table(
                ["condition", "top-1 acc", "top-1 95% CI"],
                [
                    [c,
                     f"{b['per_condition'][c]['top1_acc']:.4f}",
                     f"[{b['per_condition'][c]['top1_ci_lo']:.4f}, {b['per_condition'][c]['top1_ci_hi']:.4f}]"]
                    for c in metrics["conditions"]
                ],
            ),
            "",
            "### Pairwise (Table A — vlm-selected as reference)",
            "",
            md_table(
                ["compared", "ref-cmp", "95% CI", "raw p", "Bonf α/12", "Holm"],
                [
                    [
                        c,
                        f"{b['pairwise'][f'vlm-selected_vs_{c}']['mean_diff']:+.4f}",
                        f"[{b['pairwise'][f'vlm-selected_vs_{c}']['diff_ci_lo']:+.4f}, {b['pairwise'][f'vlm-selected_vs_{c}']['diff_ci_hi']:+.4f}]",
                        f"{b['pairwise'][f'vlm-selected_vs_{c}']['p_value']:.4g}",
                        "✓" if b['pairwise'][f'vlm-selected_vs_{c}']['bonferroni_n12_significant'] else "—",
                        "✓" if b['pairwise'][f'vlm-selected_vs_{c}']['holm_significant'] else "—",
                    ]
                    for c in metrics["conditions"]
                    if c != "vlm-selected" and f'vlm-selected_vs_{c}' in b.get('pairwise', {})
                ],
            ),
            "",
            "### Pairwise (Table B — full as reference)",
            "",
            md_table(
                ["compared", "ref-cmp", "95% CI", "raw p", "Bonf α/12", "Holm"],
                [
                    [
                        c,
                        f"{b['pairwise'][f'full_vs_{c}']['mean_diff']:+.4f}",
                        f"[{b['pairwise'][f'full_vs_{c}']['diff_ci_lo']:+.4f}, {b['pairwise'][f'full_vs_{c}']['diff_ci_hi']:+.4f}]",
                        f"{b['pairwise'][f'full_vs_{c}']['p_value']:.4g}",
                        "✓" if b['pairwise'][f'full_vs_{c}']['bonferroni_n12_significant'] else "—",
                        "✓" if b['pairwise'][f'full_vs_{c}']['holm_significant'] else "—",
                    ]
                    for c in metrics["conditions"]
                    if c != "full" and f'full_vs_{c}' in b.get('pairwise', {})
                ],
            ),
        ]
    return "\n".join(lines)


def main_combined(args: argparse.Namespace) -> int:
    if args.conditions:
        conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    else:
        conditions = DEFAULT_CONDITIONS
    if "vlm-selected" not in conditions or "full" not in conditions:
        LOG.error("--combined requires both 'vlm-selected' and 'full' to be in --conditions; got %s", conditions)
        return 2

    run_dir = Path(args.run_dir)
    eval_meta_path = Path(args.eval_meta)
    mss_run_dir = Path(args.mss_run_dir)
    output_json = Path(args.output_json) if args.output_json else run_dir / "paired_stats_combined.json"
    output_md = Path(args.output_md) if args.output_md else run_dir / "paired_stats_combined_report.md"

    LOG.info("Loading records from %s …", run_dir)
    records = load_records(run_dir)
    LOG.info("Loaded %d (video, condition) records   conditions=%s", len(records), conditions)

    aligned_ids, by_video = align_records(records, conditions)
    eligible_ids = [vid for vid in aligned_ids if not video_failed_any(by_video[vid], conditions)]
    LOG.info("Aligned: %d videos.   Eligible (no selector failure): %d", len(aligned_ids), len(eligible_ids))
    if not eligible_ids:
        LOG.error("No eligible videos. Aborting.")
        return 2

    top1_matrix, top5_matrix = correctness_matrix(eligible_ids, by_video, conditions)

    LOG.info("Computing per-condition paired bootstrap (B=%d) …", args.bootstrap_replicates)
    means1, lo1, hi1 = paired_bootstrap(top1_matrix, args.bootstrap_replicates, args.seed)
    means5, lo5, hi5 = paired_bootstrap(top5_matrix, args.bootstrap_replicates, args.seed + 7919)
    per_condition = {
        cond: {
            "top1_acc": float(means1[j]),
            "top1_ci_lo": float(lo1[j]),
            "top1_ci_hi": float(hi1[j]),
            "top5_acc": float(means5[j]),
            "top5_ci_lo": float(lo5[j]),
            "top5_ci_hi": float(hi5[j]),
        }
        for j, cond in enumerate(conditions)
    }

    LOG.info("Computing pairwise (vlm-selected as reference) …")
    pairwise_vlm = compute_pairwise_for_reference(
        top1_matrix, conditions, "vlm-selected",
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1009,
    )
    LOG.info("Computing pairwise (full as reference) …")
    pairwise_full = compute_pairwise_for_reference(
        top1_matrix, conditions, "full",
        replicates=args.bootstrap_replicates,
        seed=args.seed + 5077,
    )

    pairwise_combined = {**pairwise_vlm, **pairwise_full}
    n_global = len(pairwise_combined)  # 12 expected (6 + 6)
    bonferroni_alpha_global = RAW_ALPHA / n_global

    p_values = [(name, c["p_value"]) for name, c in pairwise_combined.items()]
    holm = holm_bonferroni(p_values, RAW_ALPHA)
    for name, c in pairwise_combined.items():
        c["bonferroni_n12_significant"] = bool(c["p_value"] < bonferroni_alpha_global)
        c["bonferroni_n12_threshold"] = float(bonferroni_alpha_global)
        c["raw_significant"] = bool(c["p_value"] < RAW_ALPHA)
        c.update(holm[name])

    n_significant_holm = sum(1 for c in pairwise_combined.values() if c["holm_significant"])
    n_significant_bonferroni = sum(1 for c in pairwise_combined.values() if c["bonferroni_n12_significant"])

    sample_record = next(iter(records.values()))
    metrics = {
        "run_dir": str(run_dir),
        "recognizer": sample_record.get("recognizer", "unknown"),
        "model": sample_record.get("model", "unknown"),
        "eval_meta_path": str(eval_meta_path),
        "mss_run_dir": str(mss_run_dir),
        "conditions": list(conditions),
        "n_total_records": len(records),
        "n_aligned_videos": len(aligned_ids),
        "n_eligible_videos": len(eligible_ids),
        "bootstrap_replicates": args.bootstrap_replicates,
        "raw_alpha": RAW_ALPHA,
        "n_comparisons_global": n_global,
        "bonferroni_alpha_global": bonferroni_alpha_global,
        "n_significant_holm": n_significant_holm,
        "n_significant_bonferroni": n_significant_bonferroni,
        "overall": {
            "n": len(eligible_ids),
            "per_condition": per_condition,
            "pairwise_top1_combined": pairwise_combined,
        },
    }
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    output_md.write_text(render_combined_markdown(metrics))

    if args.stratify_by_kept_ratio:
        LOG.info("Stratifying by kept_ratio …")
        kept_ratios = compute_kept_ratios(eligible_ids, mss_run_dir)
        LOG.info("Computed kept_ratio for %d / %d eligible videos", len(kept_ratios), len(eligible_ids))
        bins_assignment = bin_by_kept_ratio(eligible_ids, kept_ratios)

        # Per-bin combined analysis
        idx_map = {v: i for i, v in enumerate(eligible_ids)}
        by_kept_ratio: dict[str, dict] = {}
        for bin_name, lo, hi, desc in KEPT_RATIO_BINS:
            bin_vids = bins_assignment.get(bin_name, [])
            if len(bin_vids) < 5:
                LOG.info("  %s: n=%d (skipped — too small)", bin_name, len(bin_vids))
                by_kept_ratio[bin_name] = {"n_eligible": len(bin_vids), "skipped": True, "kept_ratio_range": [lo, hi]}
                continue
            sel = np.array([idx_map[v] for v in bin_vids if v in idx_map])
            sub1 = top1_matrix[sel]
            sub5 = top5_matrix[sel]
            bm1, bl1, bh1 = paired_bootstrap(sub1, args.bootstrap_replicates, args.seed + 13)
            bm5, bl5, bh5 = paired_bootstrap(sub5, args.bootstrap_replicates, args.seed + 17)
            bin_per_cond = {
                cond: {
                    "top1_acc": float(bm1[j]),
                    "top1_ci_lo": float(bl1[j]),
                    "top1_ci_hi": float(bh1[j]),
                    "top5_acc": float(bm5[j]),
                    "top5_ci_lo": float(bl5[j]),
                    "top5_ci_hi": float(bh5[j]),
                }
                for j, cond in enumerate(conditions)
            }
            bin_pairwise_vlm = compute_pairwise_for_reference(
                sub1, conditions, "vlm-selected", args.bootstrap_replicates, args.seed + 1009 + 100,
            )
            bin_pairwise_full = compute_pairwise_for_reference(
                sub1, conditions, "full", args.bootstrap_replicates, args.seed + 5077 + 100,
            )
            bin_pairwise = {**bin_pairwise_vlm, **bin_pairwise_full}
            n_global_bin = len(bin_pairwise)
            bonf_thr_bin = RAW_ALPHA / n_global_bin
            holm_bin = holm_bonferroni([(name, c["p_value"]) for name, c in bin_pairwise.items()], RAW_ALPHA)
            for name, c in bin_pairwise.items():
                c["bonferroni_n12_significant"] = bool(c["p_value"] < bonf_thr_bin)
                c["bonferroni_n12_threshold"] = float(bonf_thr_bin)
                c["raw_significant"] = bool(c["p_value"] < RAW_ALPHA)
                c.update(holm_bin[name])
            n_sig_holm_bin = sum(1 for c in bin_pairwise.values() if c["holm_significant"])
            n_sig_bonf_bin = sum(1 for c in bin_pairwise.values() if c["bonferroni_n12_significant"])
            by_kept_ratio[bin_name] = {
                "n_eligible": int(sel.size),
                "kept_ratio_range": [lo, hi],
                "description": desc,
                "per_condition": bin_per_cond,
                "pairwise": bin_pairwise,
                "n_comparisons_global": n_global_bin,
                "n_significant_holm": n_sig_holm_bin,
                "n_significant_bonferroni": n_sig_bonf_bin,
            }
            LOG.info("  %s: n=%d, %d/%d Holm-significant", bin_name, sel.size, n_sig_holm_bin, n_global_bin)

        metrics["by_kept_ratio"] = by_kept_ratio
        # Distribution summary of kept_ratio
        kr_vals = list(kept_ratios.values())
        if kr_vals:
            kr_arr = np.array(kr_vals)
            metrics["kept_ratio_distribution"] = {
                "n": int(len(kr_arr)),
                "mean": float(kr_arr.mean()),
                "median": float(np.median(kr_arr)),
                "min": float(kr_arr.min()),
                "max": float(kr_arr.max()),
                "p10": float(np.percentile(kr_arr, 10)),
                "p25": float(np.percentile(kr_arr, 25)),
                "p75": float(np.percentile(kr_arr, 75)),
                "p90": float(np.percentile(kr_arr, 90)),
            }

        strat_output_json = run_dir / "paired_stats_stratified.json"
        strat_output_md = run_dir / "paired_stats_stratified_report.md"
        strat_output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        strat_output_md.write_text(render_stratified_combined_markdown(metrics))
        print()
        print(f"Wrote {strat_output_json}")
        print(f"Wrote {strat_output_md}")
        print()
        print(f"--- Stratified by kept_ratio ---")
        for bin_name, _, _, desc in KEPT_RATIO_BINS:
            b = by_kept_ratio.get(bin_name, {})
            if b.get("skipped"):
                print(f"  {bin_name}: n={b.get('n_eligible', 0)} (skipped, too small)")
            else:
                print(f"  {bin_name}: n={b['n_eligible']}, {b['n_significant_holm']}/{b['n_comparisons_global']} Holm-sig — {desc}")
        return 0

    print(render_combined_markdown(metrics))
    print()
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    print()
    print(f"{metrics['recognizer']}: {n_significant_holm} of {n_global} comparisons significant under Holm-Bonferroni at α={RAW_ALPHA}.")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if args.combined:
        return main_combined(args)

    if args.conditions:
        conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    else:
        conditions = DEFAULT_CONDITIONS
    reference = args.reference_condition
    if reference not in conditions:
        LOG.error("Reference condition %r not in conditions list %s", reference, conditions)
        return 2
    n_baselines = len(conditions) - 1
    if n_baselines < 1:
        LOG.error("Need at least 2 conditions (1 reference + ≥1 baseline)")
        return 2
    bonferroni_alpha = RAW_ALPHA / n_baselines

    run_dir = Path(args.run_dir)
    eval_meta_path = Path(args.eval_meta)
    mss_run_dir = Path(args.mss_run_dir)
    output_json = Path(args.output_json) if args.output_json else run_dir / "paired_stats.json"
    output_md = Path(args.output_md) if args.output_md else run_dir / "paired_stats_report.md"

    LOG.info("Loading records from %s …", run_dir)
    records = load_records(run_dir)
    LOG.info("Loaded %d (video, condition) records", len(records))
    LOG.info("Conditions: %s   reference: %s", conditions, reference)

    aligned_ids, by_video = align_records(records, conditions)
    LOG.info("%d videos have all %d conditions", len(aligned_ids), len(conditions))
    eligible_ids = [vid for vid in aligned_ids if not video_failed_any(by_video[vid], conditions)]
    LOG.info("%d videos eligible (no selector failure across any condition)", len(eligible_ids))
    if not eligible_ids:
        LOG.error("No eligible videos. Aborting.")
        return 2

    top1_matrix, top5_matrix = correctness_matrix(eligible_ids, by_video, conditions)

    LOG.info("Computing overall paired bootstrap (n=%d, B=%d) …", len(eligible_ids), args.bootstrap_replicates)
    overall = stratified_eval(
        top1_matrix, top5_matrix, eligible_ids, eligible_ids,
        args.bootstrap_replicates, args.seed, conditions, reference, bonferroni_alpha,
    )

    LOG.info("Computing per-confidence-bucket and per-length-quartile breakdowns …")
    by_conf, by_len = compute_breakdowns(eligible_ids, by_video, mss_run_dir)
    by_conf_metrics = {
        bucket: stratified_eval(
            top1_matrix, top5_matrix, eligible_ids, vids,
            args.bootstrap_replicates, args.seed + 31, conditions, reference, bonferroni_alpha,
        )
        for bucket, vids in by_conf.items()
    }
    by_len_metrics = {
        q: stratified_eval(
            top1_matrix, top5_matrix, eligible_ids, vids,
            args.bootstrap_replicates, args.seed + 91, conditions, reference, bonferroni_alpha,
        )
        for q, vids in by_len.items()
    }

    sample_record = next(iter(records.values()))
    metrics = {
        "run_dir": str(run_dir),
        "recognizer": sample_record.get("recognizer", "unknown"),
        "model": sample_record.get("model", "unknown"),
        "eval_meta_path": str(eval_meta_path),
        "mss_run_dir": str(mss_run_dir),
        "conditions": list(conditions),
        "reference_condition": reference,
        "n_total_records": len(records),
        "n_aligned_videos": len(aligned_ids),
        "n_eligible_videos": len(eligible_ids),
        "bootstrap_replicates": args.bootstrap_replicates,
        "raw_alpha": RAW_ALPHA,
        "n_comparisons": n_baselines,
        "bonferroni_alpha": bonferroni_alpha,
        "overall": overall,
        "by_confidence": by_conf_metrics,
        "by_length": by_len_metrics,
    }
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    output_md.write_text(render_markdown(metrics))

    print(render_markdown(metrics))
    print()
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
