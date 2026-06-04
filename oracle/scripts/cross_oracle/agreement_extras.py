#!/usr/bin/env python
"""Stage 2 (extras) — augments oracle_agreement_eval.py with:

  * Per-class Cohen's κ (positive AND negative class separately)
  * Within-video permutation null distribution for Spearman / Jaccard
  * Format-failure-rate table per oracle × dataset

Reuses the sidecar loader and feature-extractor from oracle_agreement_eval.

Usage:
    python -m oracle.scripts.cross_oracle.agreement_extras \\
        --output-dir pseudo_labels/cross_oracle_eval/agreement/

Reads MSS dirs from cross_oracle._common.MSS_DIRS and pilot CSVs from
PILOT_CSVS. Writes one set of files per dataset:

    agreement/<dataset>/per_class_kappa.csv
    agreement/<dataset>/perm_null.json
    agreement/<dataset>/format_failures.csv
"""

import argparse
import csv
import json
import logging
import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse loader + extractor from the sibling script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle_agreement_eval import (  # type: ignore  # noqa: E402
    extract_features,
    jaccard,
    load_sidecars,
    spearman,
)

from ._common import (
    DATASETS,
    MSS_DIRS,
    ORACLES,
    OUT_ROOT,
    PILOT_CSVS,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("cross_oracle.extras")


# ---------------------------------------------------------------------------
# Cohen's κ for binary labels (segment level, pooled across videos in a class)
# ---------------------------------------------------------------------------


def cohen_kappa_binary(a: List[int], b: List[int]) -> float:
    """Cohen's κ for two binary label streams (1=positive, 0=negative).

    κ = (p_o − p_e) / (1 − p_e)  where p_o is observed agreement and p_e is
    chance agreement under independence. Returns NaN if either stream is
    constant (κ undefined).
    """
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    p_e = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if p_e >= 1 - 1e-9:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


# ---------------------------------------------------------------------------
# Per-class κ (positive AND negative class separately)
# ---------------------------------------------------------------------------


def _segment_label_streams(rec_a: dict, rec_b: dict) -> Optional[Tuple[List[int], List[int]]]:
    """Return parallel binary streams (kept_a, kept_b) over segments.

    Encodes "important" → 1, "unimportant" → 0 using segments[].label;
    falls back to mss_result.mss_runs[0].kept_indices if labels absent.
    """
    fa = extract_features(rec_a)
    fb = extract_features(rec_b)
    if fa is None or fb is None:
        return None
    # Use kept_indices to derive binary labels (we trust the extractor's
    # sort + dedup). If lengths disagree, abort.
    n = min(fa["n_segments"], fb["n_segments"])
    if n == 0:
        return None
    kept_a = set(fa["kept"])
    kept_b = set(fb["kept"])
    bin_a = [1 if i in kept_a else 0 for i in range(n)]
    bin_b = [1 if i in kept_b else 0 for i in range(n)]
    return bin_a, bin_b


def per_class_kappa(
    sidecars: Dict[str, Dict[str, dict]],   # oracle -> {video_id: rec}
    pilot_class_by_id: Dict[str, str],
    min_videos: int = 2,
) -> Dict[str, dict]:
    """Compute per-class binary κ for every oracle pair, separately for the
    positive (important) and negative (unimportant) classes.

    Returns a dict keyed by f"{oracle_a}__{oracle_b}":
        {
          class_label: {
            "n_videos": int,
            "kappa_positive": float,   # κ on important/important agreement
            "kappa_negative": float,   # κ on unimportant/unimportant agreement (label flip)
            "n_segments": int,
          }, ...
        }
    """
    out: Dict[str, dict] = {}
    for a, b in combinations(sidecars.keys(), 2):
        key = f"{a}__{b}"
        per_class_streams: Dict[str, Tuple[List[int], List[int], int]] = defaultdict(
            lambda: ([], [], 0)
        )
        common_ids = set(sidecars[a]) & set(sidecars[b])
        for vid in common_ids:
            cls = pilot_class_by_id.get(vid)
            if cls is None:
                continue
            streams = _segment_label_streams(sidecars[a][vid], sidecars[b][vid])
            if streams is None:
                continue
            bin_a, bin_b = streams
            cur = per_class_streams[cls]
            per_class_streams[cls] = (cur[0] + bin_a, cur[1] + bin_b, cur[2] + 1)

        cls_dict: Dict[str, dict] = {}
        for cls, (bin_a, bin_b, n_videos) in per_class_streams.items():
            if n_videos < min_videos:
                continue
            kappa_pos = cohen_kappa_binary(bin_a, bin_b)
            # Negative-class κ = treat unimportant as positive (label flip).
            inv_a = [1 - x for x in bin_a]
            inv_b = [1 - x for x in bin_b]
            kappa_neg = cohen_kappa_binary(inv_a, inv_b)
            cls_dict[cls] = {
                "n_videos": n_videos,
                "n_segments": len(bin_a),
                "kappa_positive": kappa_pos,
                "kappa_negative": kappa_neg,
            }
        out[key] = cls_dict
    return out


# ---------------------------------------------------------------------------
# Within-video permutation null
# ---------------------------------------------------------------------------


def _within_video_shuffle(importance: List[float], rng: random.Random) -> List[float]:
    out = importance[:]
    rng.shuffle(out)
    return out


def perm_null_for_pair(
    sidecars_a: Dict[str, dict],
    sidecars_b: Dict[str, dict],
    n_trials: int = 1000,
    seed: int = 7,
) -> dict:
    """Permutation null for pairwise Spearman + Jaccard.

    For each video v in the intersection:
      observed: spearman(importance_a, importance_b),
                jaccard(kept_a, kept_b)
    Null:
      Shuffle importance_a within each video, recompute metrics. 1000 trials.
      Aggregate with mean across videos (matching how oracle_agreement_eval
      reports per-video means).

    Returns:
      {
        "spearman_observed": float,
        "spearman_null_p95": float,
        "spearman_p_value": float,
        "jaccard_observed": float,
        "jaccard_null_p95": float,
        "jaccard_p_value": float,
        "n_videos": int,
        "n_trials": n_trials,
      }
    """
    common = set(sidecars_a) & set(sidecars_b)
    feats_a, feats_b = [], []
    for vid in common:
        fa = extract_features(sidecars_a[vid])
        fb = extract_features(sidecars_b[vid])
        if fa is None or fb is None:
            continue
        if fa["n_segments"] != fb["n_segments"] or fa["n_segments"] == 0:
            continue
        feats_a.append(fa)
        feats_b.append(fb)

    n = len(feats_a)
    if n == 0:
        return {"n_videos": 0, "error": "no comparable videos"}

    def _aggregate(fa_list, fb_list):
        rhos, jacs = [], []
        for fa, fb in zip(fa_list, fb_list):
            r = spearman(fa["importance"], fb["importance"])
            if not math.isnan(r):
                rhos.append(r)
            jacs.append(jaccard(fa["kept"], fb["kept"]))
        return (
            sum(rhos) / len(rhos) if rhos else float("nan"),
            sum(jacs) / len(jacs) if jacs else float("nan"),
        )

    obs_rho, obs_jac = _aggregate(feats_a, feats_b)

    null_rhos: List[float] = []
    null_jacs: List[float] = []
    rng = random.Random(seed)
    for t in range(n_trials):
        shuffled_a = []
        for fa in feats_a:
            sf = dict(fa)
            sf["importance"] = _within_video_shuffle(fa["importance"], rng)
            # For Jaccard we need kept sets. Derive from importance ordering:
            # take the top-len(fa["kept"]) indices of the shuffled importance.
            n_keep = len(fa["kept"])
            if n_keep > 0:
                idx_sorted = sorted(
                    range(len(sf["importance"])),
                    key=lambda i: sf["importance"][i],
                    reverse=True,
                )
                sf["kept"] = sorted(idx_sorted[:n_keep])
            shuffled_a.append(sf)
        rho, jac = _aggregate(shuffled_a, feats_b)
        null_rhos.append(rho)
        null_jacs.append(jac)

    null_rhos_sorted = sorted(x for x in null_rhos if not math.isnan(x))
    null_jacs_sorted = sorted(x for x in null_jacs if not math.isnan(x))

    def _percentile(xs: List[float], q: float) -> float:
        if not xs:
            return float("nan")
        k = int(q * (len(xs) - 1))
        return xs[k]

    def _p_value(observed: float, nulls: List[float]) -> float:
        if not nulls or math.isnan(observed):
            return float("nan")
        # one-sided: P(null >= observed)
        n_ge = sum(1 for x in nulls if x >= observed)
        return (n_ge + 1) / (len(nulls) + 1)

    return {
        "n_videos": n,
        "n_trials": n_trials,
        "spearman_observed": obs_rho,
        "spearman_null_p95": _percentile(null_rhos_sorted, 0.95),
        "spearman_p_value": _p_value(obs_rho, null_rhos_sorted),
        "jaccard_observed": obs_jac,
        "jaccard_null_p95": _percentile(null_jacs_sorted, 0.95),
        "jaccard_p_value": _p_value(obs_jac, null_jacs_sorted),
    }


# ---------------------------------------------------------------------------
# Format-failure table
# ---------------------------------------------------------------------------


def format_failure_table(
    sidecars: Dict[str, Dict[str, dict]],
    pilot_ids: List[str],
) -> List[dict]:
    """Per-oracle counts of: precheck_passed, precheck_failed, missing entirely."""
    rows = []
    pilot_set = set(pilot_ids)
    for name, recs in sidecars.items():
        present = pilot_set & set(recs)
        passed = sum(
            1 for v in present
            if recs[v].get("mss_result", {}).get("precheck_passed", False)
        )
        failed_precheck = len(present) - passed
        missing = len(pilot_set) - len(present)
        rows.append({
            "oracle": name,
            "n_pilot": len(pilot_set),
            "n_present": len(present),
            "n_passed": passed,
            "n_failed_precheck": failed_precheck,
            "n_missing": missing,
            "precheck_pass_rate": passed / max(1, len(present)),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_pilot(csv_path: Path) -> Dict[str, str]:
    """video_id -> class label (uses 'label' column for all 3 datasets)."""
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            out[r["video_id"]] = r.get("label") or "?"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_ROOT / "agreement",
    )
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--datasets", default="ssv2,k400,diving48",
        help="Subset of datasets to run (default: all).",
    )
    parser.add_argument(
        "--min-videos-per-class", type=int, default=2,
        help="Minimum videos per class for per-class κ (default: 2 — pilot is "
             "stratified by class so most classes have only 1-2 videos in SSv2/K400).",
    )
    parser.add_argument(
        "--skip-perm-null", action="store_true",
        help="Skip the permutation null (already computed; saves time on re-runs).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d for d in args.datasets.split(",") if d.strip()]

    for ds in datasets:
        log.info(f"=== dataset={ds} ===")
        out_ds = args.output_dir / ds
        out_ds.mkdir(parents=True, exist_ok=True)
        pilot_class = load_pilot(PILOT_CSVS[ds])

        sidecars: Dict[str, Dict[str, dict]] = {}
        for oracle in ORACLES:
            sidecars[oracle] = load_sidecars(MSS_DIRS[(oracle, ds)])
            # Restrict to pilot only — saves a lot in the Qwen case.
            sidecars[oracle] = {
                vid: rec for vid, rec in sidecars[oracle].items()
                if vid in pilot_class
            }
            log.info(f"  {oracle:>6s}: {len(sidecars[oracle])} pilot sidecars")

        # ---- Per-class κ ----
        log.info("computing per-class κ ...")
        kappa = per_class_kappa(sidecars, pilot_class, min_videos=args.min_videos_per_class)
        with open(out_ds / "per_class_kappa.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pair", "class", "n_videos", "n_segments",
                "kappa_positive", "kappa_negative",
            ])
            for pair, cls_dict in kappa.items():
                for cls, m in cls_dict.items():
                    writer.writerow([
                        pair, cls, m["n_videos"], m["n_segments"],
                        f"{m['kappa_positive']:.4f}",
                        f"{m['kappa_negative']:.4f}",
                    ])
        log.info(f"  wrote {out_ds / 'per_class_kappa.csv'}")

        # ---- Permutation null ----
        if args.skip_perm_null:
            log.info("skipping permutation null (--skip-perm-null)")
        else:
            log.info(f"computing permutation null ({args.n_trials} trials) ...")
            nulls = {}
            for a, b in combinations(ORACLES, 2):
                log.info(f"  {a} vs {b}")
                nulls[f"{a}__{b}"] = perm_null_for_pair(
                    sidecars[a], sidecars[b],
                    n_trials=args.n_trials, seed=args.seed,
                )
            with open(out_ds / "perm_null.json", "w") as f:
                json.dump(nulls, f, indent=2)
            log.info(f"  wrote {out_ds / 'perm_null.json'}")

        # ---- Format-failure table ----
        log.info("building format-failure table ...")
        pilot_ids = list(pilot_class.keys())
        # Re-load full sidecars (including precheck-failed) for failure stats.
        full_sidecars: Dict[str, Dict[str, dict]] = {}
        for oracle in ORACLES:
            recs = load_sidecars(MSS_DIRS[(oracle, ds)])
            full_sidecars[oracle] = {v: r for v, r in recs.items() if v in pilot_class}
        rows = format_failure_table(full_sidecars, pilot_ids)
        with open(out_ds / "format_failures.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "oracle", "n_pilot", "n_present", "n_passed",
                    "n_failed_precheck", "n_missing", "precheck_pass_rate",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        log.info(f"  wrote {out_ds / 'format_failures.csv'}")


if __name__ == "__main__":
    main()
