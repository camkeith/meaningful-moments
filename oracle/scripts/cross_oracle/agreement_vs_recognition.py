#!/usr/bin/env python
"""Stage 4 — agreement vs recognition diagnostics.

For each video v in the pilot:
  disagreement_v   = 1 - mean pairwise Spearman across oracles for v
  difficulty_v     = -log P(true_class | full_video, classifier)
  correct_v[O]     = oracle O's vlm-selected condition top-1 correctness for v

Compute, per dataset:
  Pearson r (disagreement, difficulty) ± 95% CI
  Pearson r (disagreement, correct_qwen) ± 95% CI
  Pearson r (disagreement, correct_mean_across_oracles) ± 95% CI
  + stratified-quartile recognition rate per oracle

Inputs come from Stage 1 output sidecars (`__full.json`, `__vlm-selected.json`)
in pseudo_labels/cross_oracle_eval/by_oracle/<oracle>_<dataset>/, plus the
MSS sidecars themselves for the importance vectors.

Usage:
    python -m oracle.scripts.cross_oracle.agreement_vs_recognition \\
        --output-dir pseudo_labels/cross_oracle_eval/diagnostics/
"""

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle_agreement_eval import (  # type: ignore  # noqa: E402
    extract_features,
    load_sidecars,
    spearman,
)

from ._common import DATASETS, MSS_DIRS, ORACLES, OUT_ROOT, PILOT_CSVS

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("cross_oracle.agreement_vs_rec")


def per_video_disagreement(
    sidecars: Dict[str, Dict[str, dict]], video_id: str
) -> Optional[float]:
    """1 - mean(pairwise Spearman across oracles for this video).

    Returns None if fewer than 2 oracles have a valid feature vector.
    """
    feats = []
    for oracle in ORACLES:
        rec = sidecars[oracle].get(video_id)
        if rec is None:
            continue
        f = extract_features(rec)
        if f is not None:
            feats.append(f)
    if len(feats) < 2:
        return None
    rhos = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            if feats[i]["n_segments"] != feats[j]["n_segments"]:
                continue
            r = spearman(feats[i]["importance"], feats[j]["importance"])
            if not math.isnan(r):
                rhos.append(r)
    if not rhos:
        return None
    return 1.0 - sum(rhos) / len(rhos)


def per_video_difficulty(full_record: dict) -> Optional[float]:
    """-log(P(true_class | full_video)).

    run_classifier_eval.py's `__full.json` sidecars include `top5_probs` and
    `top5_label_ids`. If `ground_truth_id` is in `top5_label_ids`, use its
    probability; otherwise use a tiny epsilon to keep it well-defined.
    """
    try:
        gt = full_record.get("ground_truth_id")
        ids = full_record.get("top5_label_ids") or []
        probs = full_record.get("top5_probs") or []
        if gt is None or not probs:
            return None
        if gt in ids:
            p = float(probs[ids.index(gt)])
        else:
            # Class fell out of top-5 → upper bound on its prob is the tail (1 - sum(top5)).
            tail = max(0.0, 1.0 - sum(float(x) for x in probs))
            n_total = max(1, full_record.get("num_classes_total", 174) - len(ids))
            p = max(tail / n_total, 1e-9)
        if p <= 0:
            return None
        return -math.log(p)
    except Exception:
        return None


def load_cell_records(oracle: str, dataset: str, condition: str) -> Dict[str, dict]:
    """Load per-video sidecars from Stage 1 for one (oracle, dataset, condition)."""
    d = OUT_ROOT / "by_oracle" / f"{oracle}_{dataset}"
    out = {}
    for p in d.glob(f"*__{condition}.json"):
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception:
            continue
        vid = rec.get("video_id") or p.stem.split("__")[0]
        out[vid] = rec
    return out


def pearson_r_ci(x: List[float], y: List[float], n_boot: int = 2000, seed: int = 7) -> dict:
    """Pearson r + 95% bootstrap CI."""
    if len(x) != len(y) or len(x) < 3:
        return {"r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(x)}
    a = np.array(x, dtype=float)
    b = np.array(y, dtype=float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return {"r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(x)}
    r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    n = len(a)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aa, bb = a[idx], b[idx]
        if aa.std() < 1e-9 or bb.std() < 1e-9:
            continue
        rs.append(np.corrcoef(aa, bb)[0, 1])
    rs_sorted = sorted(rs)
    lo = rs_sorted[int(0.025 * len(rs_sorted))] if rs_sorted else float("nan")
    hi = rs_sorted[int(0.975 * len(rs_sorted))] if rs_sorted else float("nan")
    return {"r": r, "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_ROOT / "diagnostics",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    correlation_rows = []
    stratified_rows = []

    for ds in DATASETS:
        log.info(f"=== dataset={ds} ===")

        # MSS sidecars (for disagreement)
        mss_sidecars = {
            o: {v: r for v, r in load_sidecars(MSS_DIRS[(o, ds)]).items()}
            for o in ORACLES
        }

        # Stage 1 outputs
        full_records = {o: load_cell_records(o, ds, "full") for o in ORACLES}
        sel_records = {o: load_cell_records(o, ds, "vlm-selected") for o in ORACLES}

        # Pilot ids: union of videos that have a full record from any oracle
        pilot_ids = set()
        for o in ORACLES:
            pilot_ids |= set(full_records[o].keys())
        pilot_ids = sorted(pilot_ids)
        log.info(f"  pilot videos with at least one full record: {len(pilot_ids)}")

        # Per-video features
        per_video: Dict[str, dict] = {}
        for vid in pilot_ids:
            d = per_video_disagreement(mss_sidecars, vid)
            if d is None:
                continue
            # difficulty proxy from any available full record (prefer qwen's)
            difficulty = None
            for o in ("qwen", "intern", "azure"):
                rec = full_records[o].get(vid)
                if rec:
                    difficulty = per_video_difficulty(rec)
                    if difficulty is not None:
                        break
            correct = {}
            for o in ORACLES:
                rec = sel_records[o].get(vid)
                if rec is not None:
                    correct[o] = int(bool(rec.get("top1_correct", False)))
            per_video[vid] = {
                "disagreement": d,
                "difficulty": difficulty,
                "correct": correct,
            }

        # Build correlation arrays
        x_diff, y_difficulty = [], []
        x_diff_qwen, y_correct_qwen = [], []
        x_diff_mean, y_correct_mean = [], []
        for vid, d in per_video.items():
            if d["difficulty"] is not None:
                x_diff.append(d["disagreement"])
                y_difficulty.append(d["difficulty"])
            if "qwen" in d["correct"]:
                x_diff_qwen.append(d["disagreement"])
                y_correct_qwen.append(d["correct"]["qwen"])
            corrects = list(d["correct"].values())
            if corrects:
                x_diff_mean.append(d["disagreement"])
                y_correct_mean.append(sum(corrects) / len(corrects))

        for tag, x, y in [
            ("disagreement_vs_difficulty", x_diff, y_difficulty),
            ("disagreement_vs_correct_qwen", x_diff_qwen, y_correct_qwen),
            ("disagreement_vs_correct_mean", x_diff_mean, y_correct_mean),
        ]:
            res = pearson_r_ci(x, y)
            correlation_rows.append({
                "dataset": ds, "comparison": tag,
                "n": res["n"], "r": res["r"],
                "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
            })
            log.info(
                f"  {tag:<35s}  n={res['n']:>3d}  r={res['r']:+.3f}  "
                f"[{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}]"
            )

        # Stratified analysis: bin videos by disagreement quartile, mean correct per oracle
        if per_video:
            xs = np.array([d["disagreement"] for d in per_video.values()])
            quartiles = np.quantile(xs, [0.25, 0.5, 0.75])
            for vid, d in per_video.items():
                q_idx = int(np.searchsorted(quartiles, d["disagreement"]))
                row = {
                    "dataset": ds, "video_id": vid, "quartile": f"Q{q_idx + 1}",
                    "disagreement": d["disagreement"],
                }
                for o in ORACLES:
                    row[f"correct_{o}"] = d["correct"].get(o, "")
                stratified_rows.append(row)

    # Write outputs
    out_corr = args.output_dir / "correlations.csv"
    with open(out_corr, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "comparison", "n", "r", "ci_lo", "ci_hi"],
        )
        writer.writeheader()
        for r in correlation_rows:
            writer.writerow(r)
    log.info(f"wrote {out_corr}")

    out_strat = args.output_dir / "stratified.csv"
    with open(out_strat, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "dataset", "video_id", "quartile", "disagreement",
                "correct_qwen", "correct_intern", "correct_azure",
            ],
        )
        writer.writeheader()
        for r in stratified_rows:
            writer.writerow(r)
    log.info(f"wrote {out_strat}")


if __name__ == "__main__":
    main()
