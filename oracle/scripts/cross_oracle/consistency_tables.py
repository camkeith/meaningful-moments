#!/usr/bin/env python
"""Stage 3 — EPS/TDS consistency tables across the 9 (oracle × dataset) cells.

Reads run_classifier_eval.py per-video sidecars from
    pseudo_labels/cross_oracle_eval/by_oracle/<oracle>_<dataset>/
For each cell, builds the paired correctness matrix over the 5 conditions
(`full`, `vlm-selected`, `vlm-weighted`, `uniform`, `lowest-evidence`),
computes:
  EPS = acc(vlm-selected) − acc(lowest-evidence)
  TDS = acc(vlm-selected) − acc(uniform)
with paired-bootstrap 95% CI (1000 reps) + McNemar p, then applies
Holm-Bonferroni separately to the 9 EPS tests and the 9 TDS tests.

Also computes Cochran's Q heterogeneity per (dataset, metric) over the 3
oracle estimates, and emits a direction-of-finding consistency table.

Reuses paired_bootstrap, paired_diff_bootstrap, mcnemar, holm_bonferroni
from oracle/scripts/eval_paired_stats.py.

Usage:
    python -m oracle.scripts.cross_oracle.consistency_tables \\
        --output-dir pseudo_labels/cross_oracle_eval/consistency/
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
from eval_paired_stats import (  # type: ignore  # noqa: E402
    align_records,
    correctness_matrix,
    holm_bonferroni,
    load_records,
    mcnemar,
    paired_bootstrap,
    paired_diff_bootstrap,
    video_failed_any,
)

from ._common import (
    CLASSIFIER_CONDITIONS,
    DATASETS,
    ORACLES,
    OUT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("cross_oracle.consistency")

REQUIRED_CONDITIONS: Tuple[str, ...] = (
    "full", "vlm-selected", "vlm-weighted", "uniform", "lowest-evidence",
)
VLM_SEL = REQUIRED_CONDITIONS.index("vlm-selected")
LOWEST = REQUIRED_CONDITIONS.index("lowest-evidence")
UNIFORM = REQUIRED_CONDITIONS.index("uniform")
FULL = REQUIRED_CONDITIONS.index("full")


def cell_outdir(oracle: str, dataset: str) -> Path:
    return OUT_ROOT / "by_oracle" / f"{oracle}_{dataset}"


def load_cell(oracle: str, dataset: str) -> Optional[dict]:
    """Load + align one cell. Returns dict with top1/top5 matrices and per-condition means."""
    run_dir = cell_outdir(oracle, dataset)
    if not run_dir.exists():
        log.warning(f"missing run dir: {run_dir}")
        return None
    records = load_records(run_dir)
    if not records:
        log.warning(f"no records in {run_dir}")
        return None
    video_ids, by_video = align_records(records, REQUIRED_CONDITIONS)
    # Drop videos that had any selector failure across the conditions we use.
    video_ids = [
        v for v in video_ids
        if not video_failed_any(by_video[v], REQUIRED_CONDITIONS)
    ]
    if not video_ids:
        log.warning(f"  {oracle} × {dataset}: no aligned videos")
        return None
    top1, top5 = correctness_matrix(video_ids, by_video, REQUIRED_CONDITIONS)
    return {
        "oracle": oracle,
        "dataset": dataset,
        "video_ids": video_ids,
        "top1": top1,
        "top5": top5,
    }


def cell_metrics(cell: dict, replicates: int = 1000, seed: int = 42) -> dict:
    """Compute per-condition means + EPS/TDS diffs with bootstrap CIs."""
    out = {"oracle": cell["oracle"], "dataset": cell["dataset"], "n": len(cell["video_ids"])}

    for label, mat in [("top1", cell["top1"]), ("top5", cell["top5"])]:
        means, ci_lo, ci_hi = paired_bootstrap(mat.astype(float), replicates, seed)
        out[f"{label}_means"] = {
            cond: float(means[i]) for i, cond in enumerate(REQUIRED_CONDITIONS)
        }
        out[f"{label}_ci_lo"] = {
            cond: float(ci_lo[i]) for i, cond in enumerate(REQUIRED_CONDITIONS)
        }
        out[f"{label}_ci_hi"] = {
            cond: float(ci_hi[i]) for i, cond in enumerate(REQUIRED_CONDITIONS)
        }
        # EPS = vlm-selected − lowest-evidence
        eps = paired_diff_bootstrap(mat.astype(float), VLM_SEL, LOWEST, replicates, seed)
        eps_p = mcnemar(mat, VLM_SEL, LOWEST)
        out[f"{label}_EPS"] = {**eps, **eps_p}
        # TDS = vlm-selected − uniform
        tds = paired_diff_bootstrap(mat.astype(float), VLM_SEL, UNIFORM, replicates, seed)
        tds_p = mcnemar(mat, VLM_SEL, UNIFORM)
        out[f"{label}_TDS"] = {**tds, **tds_p}
    return out


def cochrans_q(estimates: List[float], variances: List[float]) -> Tuple[float, float, float]:
    """Cochran's Q heterogeneity test on 3 paired estimates with their variances.

    Returns (Q, df, p). Variances must be > 0; we floor at 1e-12 to avoid
    divide-by-zero on exactly-zero estimators.
    """
    weights = [1.0 / max(v, 1e-12) for v in variances]
    sum_w = sum(weights)
    weighted_mean = sum(w * e for w, e in zip(weights, estimates)) / sum_w
    Q = sum(w * (e - weighted_mean) ** 2 for w, e in zip(weights, estimates))
    df = len(estimates) - 1
    # Survival of chi-square — use scipy if available, else math.erfc fallback (df=2 only)
    try:
        from scipy.stats import chi2  # type: ignore
        p = float(chi2.sf(Q, df))
    except ImportError:
        # df=2 → P(X >= Q) = exp(-Q/2)
        if df == 2:
            p = math.exp(-Q / 2)
        else:
            p = float("nan")
    return float(Q), float(df), float(p)


def variance_from_ci(ci_lo: float, ci_hi: float) -> float:
    """SE ≈ (ci_hi − ci_lo) / 3.92 → var = SE^2."""
    se = (ci_hi - ci_lo) / 3.92
    return max(se * se, 1e-12)


def build_consistency_tables(
    metrics: List[dict], output_dir: Path, alpha: float = 0.05
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. EPS table ----
    rows_eps_top1, rows_eps_top5 = [], []
    pvals_eps_top1: List[Tuple[str, float]] = []
    pvals_eps_top5: List[Tuple[str, float]] = []
    for m in metrics:
        key = f"{m['oracle']}__{m['dataset']}"
        e1 = m["top1_EPS"]; e5 = m["top5_EPS"]
        rows_eps_top1.append({
            "oracle": m["oracle"], "dataset": m["dataset"], "n": m["n"],
            "eps": e1["mean_diff"], "ci_lo": e1["diff_ci_lo"], "ci_hi": e1["diff_ci_hi"],
            "p_value": e1["p_value"], "b10": e1["b10_vlm_only"], "b01": e1["b01_baseline_only"],
        })
        rows_eps_top5.append({
            "oracle": m["oracle"], "dataset": m["dataset"], "n": m["n"],
            "eps": e5["mean_diff"], "ci_lo": e5["diff_ci_lo"], "ci_hi": e5["diff_ci_hi"],
            "p_value": e5["p_value"], "b10": e5["b10_vlm_only"], "b01": e5["b01_baseline_only"],
        })
        pvals_eps_top1.append((key, e1["p_value"]))
        pvals_eps_top5.append((key, e5["p_value"]))

    holm_eps_top1 = holm_bonferroni(pvals_eps_top1, alpha)
    holm_eps_top5 = holm_bonferroni(pvals_eps_top5, alpha)
    for r in rows_eps_top1:
        r.update(holm_eps_top1[f"{r['oracle']}__{r['dataset']}"])
    for r in rows_eps_top5:
        r.update(holm_eps_top5[f"{r['oracle']}__{r['dataset']}"])

    for tag, rows in [("top1", rows_eps_top1), ("top5", rows_eps_top5)]:
        out_path = output_dir / f"eps_table_{tag}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "oracle", "dataset", "n", "eps", "ci_lo", "ci_hi",
                    "p_value", "b10", "b01",
                    "holm_significant", "holm_threshold", "holm_rank",
                ],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        log.info(f"wrote {out_path}")

    # ---- 2. TDS table ----
    rows_tds_top1, rows_tds_top5 = [], []
    pvals_tds_top1: List[Tuple[str, float]] = []
    pvals_tds_top5: List[Tuple[str, float]] = []
    for m in metrics:
        key = f"{m['oracle']}__{m['dataset']}"
        t1 = m["top1_TDS"]; t5 = m["top5_TDS"]
        rows_tds_top1.append({
            "oracle": m["oracle"], "dataset": m["dataset"], "n": m["n"],
            "tds": t1["mean_diff"], "ci_lo": t1["diff_ci_lo"], "ci_hi": t1["diff_ci_hi"],
            "p_value": t1["p_value"], "b10": t1["b10_vlm_only"], "b01": t1["b01_baseline_only"],
        })
        rows_tds_top5.append({
            "oracle": m["oracle"], "dataset": m["dataset"], "n": m["n"],
            "tds": t5["mean_diff"], "ci_lo": t5["diff_ci_lo"], "ci_hi": t5["diff_ci_hi"],
            "p_value": t5["p_value"], "b10": t5["b10_vlm_only"], "b01": t5["b01_baseline_only"],
        })
        pvals_tds_top1.append((key, t1["p_value"]))
        pvals_tds_top5.append((key, t5["p_value"]))

    holm_tds_top1 = holm_bonferroni(pvals_tds_top1, alpha)
    holm_tds_top5 = holm_bonferroni(pvals_tds_top5, alpha)
    for r in rows_tds_top1:
        r.update(holm_tds_top1[f"{r['oracle']}__{r['dataset']}"])
    for r in rows_tds_top5:
        r.update(holm_tds_top5[f"{r['oracle']}__{r['dataset']}"])

    for tag, rows in [("top1", rows_tds_top1), ("top5", rows_tds_top5)]:
        out_path = output_dir / f"tds_table_{tag}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "oracle", "dataset", "n", "tds", "ci_lo", "ci_hi",
                    "p_value", "b10", "b01",
                    "holm_significant", "holm_threshold", "holm_rank",
                ],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        log.info(f"wrote {out_path}")

    # ---- 3. Cochran's Q heterogeneity per dataset ----
    het_rows = []
    metrics_by_ds: Dict[str, List[dict]] = {}
    for m in metrics:
        metrics_by_ds.setdefault(m["dataset"], []).append(m)
    for ds, ms in metrics_by_ds.items():
        if len(ms) != len(ORACLES):
            log.warning(f"dataset {ds} has {len(ms)} oracles; skipping heterogeneity")
            continue
        for tag in ("top1", "top5"):
            for metric_label in ("EPS", "TDS"):
                ests, vars_ = [], []
                for m in ms:
                    d = m[f"{tag}_{metric_label}"]
                    ests.append(d["mean_diff"])
                    vars_.append(variance_from_ci(d["diff_ci_lo"], d["diff_ci_hi"]))
                Q, df, p = cochrans_q(ests, vars_)
                het_rows.append({
                    "dataset": ds, "metric": metric_label, "topk": tag,
                    "estimates": ests, "Q": Q, "df": df, "p_heterogeneity": p,
                    "consistent_at_alpha_0.05": p > 0.05,
                })
    out_path = output_dir / "heterogeneity.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "dataset", "metric", "topk",
                "estimates", "Q", "df", "p_heterogeneity",
                "consistent_at_alpha_0.05",
            ],
        )
        writer.writeheader()
        for r in het_rows:
            r["estimates"] = ",".join(f"{e:.4f}" for e in r["estimates"])
            writer.writerow(r)
    log.info(f"wrote {out_path}")

    # ---- 4. Direction-of-finding consistency ----
    direction_rows = []
    for ds, ms in metrics_by_ds.items():
        if len(ms) != len(ORACLES):
            continue
        for tag in ("top1", "top5"):
            eps_signs = [m[f"{tag}_EPS"]["mean_diff"] > 0 for m in ms]
            eps_holm_sig = []
            for m in ms:
                key = f"{m['oracle']}__{m['dataset']}"
                holm = holm_eps_top1 if tag == "top1" else holm_eps_top5
                eps_holm_sig.append(holm[key]["holm_significant"])
            tds_negative = [m[f"{tag}_TDS"]["mean_diff"] < 0 for m in ms]
            tds_ns = []
            for m in ms:
                key = f"{m['oracle']}__{m['dataset']}"
                holm = holm_tds_top1 if tag == "top1" else holm_tds_top5
                tds_ns.append(not holm[key]["holm_significant"])
            direction_rows.append({
                "dataset": ds, "topk": tag,
                "all_oracles_eps_positive": all(eps_signs),
                "all_oracles_eps_holm_sig": all(eps_holm_sig),
                "all_oracles_tds_negative": all(tds_negative),
                "all_oracles_tds_holm_ns": all(tds_ns),
            })
    out_path = output_dir / "direction.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "dataset", "topk",
                "all_oracles_eps_positive", "all_oracles_eps_holm_sig",
                "all_oracles_tds_negative", "all_oracles_tds_holm_ns",
            ],
        )
        writer.writeheader()
        for r in direction_rows:
            writer.writerow(r)
    log.info(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_ROOT / "consistency",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[dict] = []
    for oracle in ORACLES:
        for ds in DATASETS:
            cell = load_cell(oracle, ds)
            if cell is None:
                log.warning(f"  skipping {oracle} × {ds}")
                continue
            m = cell_metrics(cell, args.bootstrap_replicates, args.seed)
            log.info(
                f"  {oracle:>6s} × {ds:<8s}  n={m['n']:>3d}  "
                f"top1 EPS={m['top1_EPS']['mean_diff']:+.3f}  "
                f"TDS={m['top1_TDS']['mean_diff']:+.3f}"
            )
            metrics.append(m)

    # Persist raw cell metrics
    raw_path = args.output_dir / "cell_metrics.json"
    with open(raw_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info(f"wrote {raw_path}")

    if metrics:
        build_consistency_tables(metrics, args.output_dir, alpha=args.alpha)


if __name__ == "__main__":
    main()
