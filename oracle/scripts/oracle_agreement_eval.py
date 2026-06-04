#!/usr/bin/env python
"""N-way oracle agreement evaluation for the agnostic-validation study.

Loads the per-video MSS sidecar JSONs from N run directories (one per
oracle), validates that segmentation parameters agree (delta_t, n_segments
per video, mask_operator), then computes pairwise agreement metrics on:

  - per-segment importance scores (Spearman, Kendall tau)
  - kept-segment sets (Jaccard, symmetric F1, top-k overlap)
  - per-video keep ratios (MAE, Pearson)

Bootstrap CIs across videos are computed for every aggregate.

Usage:
    python oracle/scripts/oracle_agreement_eval.py \\
        --runs qwen=pseudo_labels/oracle_agreement/qwen_<ts>/ \\
        --runs gemini=pseudo_labels/oracle_agreement/gemini_<ts>/ \\
        --runs intern38=pseudo_labels/oracle_agreement/internvl3-38b_<ts>/ \\
        --runs intern78=pseudo_labels/oracle_agreement/internvl3-78b_<ts>/ \\
        --output-dir pseudo_labels/oracle_agreement/eval_<ts>/ \\
        --primary qwen
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("oracle_agreement_eval")


# ---------------------------------------------------------------------------
# Sidecar loading
# ---------------------------------------------------------------------------


def parse_run_arg(arg: str) -> Tuple[str, Path]:
    """Parse '<name>=<path>' into (name, Path)."""
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"--runs expects 'name=path', got {arg!r}"
        )
    name, _, path = arg.partition("=")
    name = name.strip()
    path = Path(path.strip())
    if not name:
        raise argparse.ArgumentTypeError(f"--runs missing name in {arg!r}")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"run dir not found: {path}")
    return name, path


def load_sidecars(d: Path) -> Dict[str, dict]:
    """Load every <video_id>.json sidecar from a run directory."""
    out: Dict[str, dict] = {}
    skipped = 0
    for p in d.glob("*.json"):
        if p.name in ("run_metadata.json", "config.json", "results.jsonl"):
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping unreadable sidecar {p}: {e}")
            skipped += 1
            continue
        vid = rec.get("video_id") or p.stem
        out[vid] = rec
    if skipped:
        logger.warning(f"  skipped {skipped} unreadable sidecars in {d}")
    return out


def load_metadata(d: Path) -> Optional[dict]:
    """Read run_metadata.json or config.json (whichever exists)."""
    for name in ("run_metadata.json", "config.json"):
        p = d / name
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Sidecar feature extraction
# ---------------------------------------------------------------------------


def extract_features(rec: dict) -> Optional[dict]:
    """Pull (importance_scores, kept_indices, keep_ratio, n_segments) out of a sidecar.

    Returns None for videos where precheck failed or features can't be
    extracted (so they're dropped from agreement analysis).
    """
    mss = rec.get("mss_result", {})
    if not mss.get("precheck_passed", False):
        return None

    segments = mss.get("segments") or []
    n_segments = len(segments)
    if n_segments == 0:
        return None

    incl = mss.get("inclusion_frequencies") or {}
    # JSON keys are strings — convert to int and verify they cover [0, n)
    importance: List[float] = [0.0] * n_segments
    for k, v in incl.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_segments:
            importance[idx] = float(v)

    runs = mss.get("mss_runs") or []
    kept: List[int] = []
    if runs:
        kept = sorted(int(i) for i in (runs[0].get("kept_indices") or []))

    keep_ratio = (len(kept) / n_segments) if n_segments else 0.0

    return {
        "video_id": rec.get("video_id"),
        "n_segments": n_segments,
        "importance": importance,
        "kept": kept,
        "keep_ratio": keep_ratio,
        "precheck_conf": float(mss.get("precheck_vote_yes", 0.0)),
    }


# ---------------------------------------------------------------------------
# Pairwise metrics
# ---------------------------------------------------------------------------


def spearman(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation. Returns NaN if either series is constant."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")

    def ranks(x: List[float]) -> List[float]:
        order = sorted(range(len(x)), key=lambda i: x[i])
        ranks = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    mean_ra = sum(ra) / n
    mean_rb = sum(rb) / n
    cov = sum((ra[i] - mean_ra) * (rb[i] - mean_rb) for i in range(n))
    var_a = sum((ra[i] - mean_ra) ** 2 for i in range(n))
    var_b = sum((rb[i] - mean_rb) ** 2 for i in range(n))
    denom = math.sqrt(var_a * var_b)
    if denom == 0:
        return float("nan")
    return cov / denom


def kendall_tau(a: List[float], b: List[float]) -> float:
    """Kendall tau-b. Returns NaN if either side is constant."""
    n = len(a)
    if n != len(b) or n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
                continue
            if db == 0:
                ties_b += 1
                continue
            if (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt(
        (concordant + discordant + ties_a) * (concordant + discordant + ties_b)
    )
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard similarity over two segment-index sets. Empty union → 0."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def set_f1(a: Iterable[int], b: Iterable[int]) -> float:
    """Symmetric F1 over two segment-index sets. Empty either → 0."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    p = inter / len(sa)
    r = inter / len(sb)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def top_k_overlap(a: List[float], b: List[float], k: int) -> float:
    """Fraction of a's top-k indices that are also in b's top-k.

    k is computed by the caller (typically the primary oracle's keep count).
    """
    if k <= 0 or len(a) != len(b):
        return float("nan")
    k = min(k, len(a))
    top_a = sorted(range(len(a)), key=lambda i: -a[i])[:k]
    top_b = set(sorted(range(len(b)), key=lambda i: -b[i])[:k])
    return sum(1 for i in top_a if i in top_b) / k


def pearson(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a)))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    denom = math.sqrt(va * vb)
    if denom == 0:
        return float("nan")
    return cov / denom


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def per_video_pair_metrics(
    feat_a: dict,
    feat_b: dict,
    primary_keep: Optional[int] = None,
) -> Dict[str, float]:
    """Compute the agreement metrics for one paired (video, oracle_a, oracle_b)."""
    if feat_a["n_segments"] != feat_b["n_segments"]:
        # Disagreeing segment counts means one oracle saw a different
        # segmentation — skip this pair for this video.
        return {}

    a = feat_a["importance"]
    b = feat_b["importance"]
    out = {
        "spearman": spearman(a, b),
        "kendall_tau": kendall_tau(a, b),
        "jaccard": jaccard(feat_a["kept"], feat_b["kept"]),
        "set_f1": set_f1(feat_a["kept"], feat_b["kept"]),
        "keep_ratio_a": feat_a["keep_ratio"],
        "keep_ratio_b": feat_b["keep_ratio"],
        "keep_ratio_abs_err": abs(feat_a["keep_ratio"] - feat_b["keep_ratio"]),
    }
    if primary_keep is not None:
        out["topk_overlap"] = top_k_overlap(a, b, primary_keep)
    return out


def bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (mean, low, high) percentile-bootstrap CI ignoring NaN."""
    import random
    clean = [v for v in values if isinstance(v, float) and not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(clean) / n
    if n < 2:
        return mean, mean, mean

    rng = random.Random(seed)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo_idx = int(alpha / 2 * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return mean, boots[lo_idx], boots[hi_idx]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


PARITY_KEYS_SOFT = ("model", "prompt_template_id", "delta_t", "mask_operator")


def soft_parity_check(metas: Dict[str, dict]) -> List[str]:
    """Compare run_metadata.json across runs. Models differ by design; we
    flag anything ELSE that differs as a warning."""
    warnings: List[str] = []
    keys_to_check = ("delta_t", "mask_operator", "direct_scoring", "prompt_template_id")
    seen: Dict[str, Dict[str, Any]] = {}
    for run_name, meta in metas.items():
        if not meta:
            warnings.append(f"{run_name}: no run_metadata.json / config.json")
            continue
        # Walk into mss_config / oracle_config / mask_config if present
        flat = {}
        for outer in ("mss_config", "oracle_config", "mask_config"):
            inner = meta.get(outer) or {}
            for k, v in inner.items():
                flat[k] = v
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                flat.setdefault(k, v)
        seen[run_name] = flat

    for k in keys_to_check:
        vals = {name: cfg.get(k) for name, cfg in seen.items()}
        unique = set(v for v in vals.values() if v is not None)
        if len(unique) > 1:
            warnings.append(f"  {k} differs across runs: {vals}")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", action="append", required=True, type=parse_run_arg,
        metavar="NAME=PATH",
        help="Repeatable. e.g. --runs qwen=pseudo_labels/.../qwen_<ts>/",
    )
    parser.add_argument(
        "--primary", type=str, default=None,
        help="Run name to treat as the primary oracle (drives top-k overlap "
             "and serves as the reference column in some tables). Defaults to "
             "the first --runs entry.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--ignore-parity", action="store_true",
        help="Run anyway when delta_t / mask_operator / prompt_template_id differ",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: Dict[str, Path] = {name: path for name, path in args.runs}
    run_names = list(runs.keys())
    if len(run_names) < 2:
        sys.exit("Need at least 2 --runs entries for pairwise agreement")
    primary = args.primary or run_names[0]
    if primary not in runs:
        sys.exit(f"--primary {primary!r} not in --runs names ({run_names})")

    logger.info(f"Runs: {run_names}, primary={primary}")

    # Load all sidecars + metadata
    sidecars: Dict[str, Dict[str, dict]] = {}
    metas: Dict[str, dict] = {}
    for name, path in runs.items():
        sc = load_sidecars(path)
        sidecars[name] = sc
        metas[name] = load_metadata(path) or {}
        logger.info(f"  {name}: {len(sc)} sidecars, metadata={'present' if metas[name] else 'MISSING'}")

    # Soft parity check (model name expected to differ; everything else flagged)
    warnings = soft_parity_check(metas)
    if warnings:
        msg = "Parity warnings:\n" + "\n".join(warnings)
        if args.ignore_parity:
            logger.warning(msg)
        else:
            logger.warning(msg)
            logger.warning(
                "Continuing anyway. Pass --ignore-parity to suppress, or "
                "address the differences before publishing the agreement numbers."
            )

    # Intersect on video_id
    common_ids = set(next(iter(sidecars.values())).keys())
    for sc in sidecars.values():
        common_ids &= set(sc.keys())
    common_ids = sorted(common_ids)
    logger.info(f"Common video_ids across all runs: {len(common_ids)}")

    # Extract features per (run, video_id), drop ones with no usable features
    features: Dict[str, Dict[str, dict]] = {n: {} for n in run_names}
    drops_per_run: Dict[str, int] = {n: 0 for n in run_names}
    for vid in common_ids:
        for name in run_names:
            f = extract_features(sidecars[name][vid])
            if f is None:
                drops_per_run[name] += 1
            else:
                features[name][vid] = f
    for name, n_drop in drops_per_run.items():
        if n_drop:
            logger.warning(f"  {name}: dropped {n_drop} videos (precheck failed or empty segments)")

    # Final usable set: video_ids present + valid in every run
    usable_ids = sorted(set.intersection(*[set(features[n].keys()) for n in run_names]))
    logger.info(f"Usable videos (all runs precheck-pass): {len(usable_ids)}")

    if not usable_ids:
        sys.exit("No usable videos — nothing to evaluate")

    # Read dataset annotation if present (sidecars usually carry video_path; fall
    # back to inferring dataset from path).
    def _dataset_of(rec: dict) -> str:
        path = (rec.get("video_path") or "").lower()
        if "ssv2" in path or "20bn-something-something" in path:
            return "ssv2"
        if "k400" in path or "kinetics" in path:
            return "k400"
        if "diving48" in path:
            return "diving48"
        if "charades" in path:
            return "charades"
        return "unknown"

    video_dataset: Dict[str, str] = {
        vid: _dataset_of(sidecars[primary][vid]) for vid in usable_ids
    }

    # Pairwise per-video metrics
    pairs = list(combinations(run_names, 2))
    raw: Dict[Tuple[str, str], Dict[str, List[float]]] = {
        p: defaultdict(list) for p in pairs
    }
    raw_by_dataset: Dict[Tuple[str, str], Dict[str, Dict[str, List[float]]]] = {
        p: defaultdict(lambda: defaultdict(list)) for p in pairs
    }
    metric_keys = ("spearman", "kendall_tau", "jaccard", "set_f1", "keep_ratio_abs_err", "topk_overlap")

    for vid in usable_ids:
        primary_keep = len(features[primary][vid]["kept"])
        ds = video_dataset[vid]
        for a, b in pairs:
            m = per_video_pair_metrics(features[a][vid], features[b][vid], primary_keep)
            for k in metric_keys:
                v = m.get(k)
                if v is None:
                    continue
                raw[(a, b)][k].append(v)
                raw_by_dataset[(a, b)][ds][k].append(v)

    # Aggregates with bootstrap CIs
    agg: Dict[str, Dict[str, dict]] = {}  # agg[metric][f"{a}__{b}"] = {mean, ci_low, ci_high, n}
    for k in metric_keys:
        agg[k] = {}
        for a, b in pairs:
            vals = raw[(a, b)][k]
            mean, lo, hi = bootstrap_ci(vals, n_boot=args.bootstrap_iters, seed=args.seed)
            agg[k][f"{a}__{b}"] = {
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n": len([v for v in vals if not math.isnan(v)]),
            }

    # Pearson on per-video keep-ratios (one number per pair, no bootstrap)
    keep_ratio_pearson: Dict[str, dict] = {}
    for a, b in pairs:
        ra = [features[a][v]["keep_ratio"] for v in usable_ids]
        rb = [features[b][v]["keep_ratio"] for v in usable_ids]
        keep_ratio_pearson[f"{a}__{b}"] = {
            "pearson": pearson(ra, rb),
            "n": len(ra),
        }

    # Per-dataset summary for spearman + jaccard (the two headline numbers)
    per_dataset: Dict[str, Dict[str, Dict[str, float]]] = {}
    for k in ("spearman", "jaccard", "set_f1"):
        per_dataset[k] = {}
        for a, b in pairs:
            for ds, vals_by_metric in raw_by_dataset[(a, b)].items():
                vals = vals_by_metric.get(k) or []
                clean = [v for v in vals if isinstance(v, float) and not math.isnan(v)]
                if not clean:
                    continue
                key = f"{a}__{b}__{ds}"
                per_dataset[k][key] = {
                    "mean": sum(clean) / len(clean),
                    "n": len(clean),
                }

    # Write outputs
    metrics = {
        "n_evaluated_videos": len(usable_ids),
        "runs": {name: str(path) for name, path in runs.items()},
        "primary": primary,
        "pairs": [f"{a}__{b}" for a, b in pairs],
        "agg": agg,
        "keep_ratio_pearson": keep_ratio_pearson,
        "per_dataset": per_dataset,
        "parity_warnings": warnings,
        "drops_per_run": drops_per_run,
    }
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Pairwise N×N CSV per metric
    import csv as _csv
    for k in metric_keys:
        path = args.output_dir / f"pairwise_{k}.csv"
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow([""] + run_names)
            for a in run_names:
                row = [a]
                for b in run_names:
                    if a == b:
                        row.append("1.000")
                        continue
                    key = f"{a}__{b}" if (a, b) in pairs else f"{b}__{a}"
                    cell = agg[k].get(key, {})
                    mean = cell.get("mean", float("nan"))
                    if isinstance(mean, float) and not math.isnan(mean):
                        row.append(f"{mean:.3f}")
                    else:
                        row.append("nan")
                w.writerow(row)

    # Console summary
    print("=" * 76)
    print(f"  Oracle agreement (n={len(usable_ids)} videos)")
    print(f"  Runs: {', '.join(run_names)}, primary={primary}")
    print("-" * 76)
    print(f"  {'pair':<28} {'spearman':>10} {'jaccard':>10} {'set_f1':>10} {'kr_mae':>10}")
    for a, b in pairs:
        sp = agg["spearman"][f"{a}__{b}"]["mean"]
        jc = agg["jaccard"][f"{a}__{b}"]["mean"]
        f1 = agg["set_f1"][f"{a}__{b}"]["mean"]
        kr = agg["keep_ratio_abs_err"][f"{a}__{b}"]["mean"]
        print(f"  {a + ' vs ' + b:<28} {sp:>10.3f} {jc:>10.3f} {f1:>10.3f} {kr:>10.3f}")
    print("=" * 76)
    print(f"  Wrote metrics.json + pairwise_*.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
