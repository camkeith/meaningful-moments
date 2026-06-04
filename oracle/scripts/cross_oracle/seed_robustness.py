#!/usr/bin/env python
"""Stage 5 — cross-seed robustness for the agreement metrics.

We have MSS sidecars only for the seed=42 200-video pilot, so a true
re-draw isn't possible without re-running each oracle on a fresh sample.
Instead, sub-sample WITHIN the existing 200 with seeds {43, 44, 45}
(50% of the pilot each — n=100), recompute mean Spearman / Jaccard /
keep-ratio MAE for every pair, and report the spread across seeds.

This bounds how sensitive the cross-oracle metrics are to which exact
videos make the pilot. If spread is small (<0.05 absolute on Spearman),
agreement is robust to subset draw. If large, flag and discuss.

Usage:
    python -m oracle.scripts.cross_oracle.seed_robustness \\
        --output-dir pseudo_labels/cross_oracle_eval/robustness/ \\
        --seeds 42,43,44,45 --frac 0.5
"""

import argparse
import csv
import logging
import math
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle_agreement_eval import (  # type: ignore  # noqa: E402
    extract_features,
    jaccard,
    load_sidecars,
    spearman,
)

from ._common import DATASETS, MSS_DIRS, ORACLES, OUT_ROOT, PILOT_CSVS

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("cross_oracle.seed_robustness")


def aggregate_pair_metrics(
    sidecars_a: Dict[str, dict], sidecars_b: Dict[str, dict], video_ids: List[str]
) -> dict:
    rhos, jacs, ratio_diffs = [], [], []
    for vid in video_ids:
        ra = sidecars_a.get(vid); rb = sidecars_b.get(vid)
        if ra is None or rb is None:
            continue
        fa = extract_features(ra); fb = extract_features(rb)
        if fa is None or fb is None:
            continue
        if fa["n_segments"] != fb["n_segments"] or fa["n_segments"] == 0:
            continue
        r = spearman(fa["importance"], fb["importance"])
        if not math.isnan(r):
            rhos.append(r)
        jacs.append(jaccard(fa["kept"], fb["kept"]))
        ratio_diffs.append(abs(fa["keep_ratio"] - fb["keep_ratio"]))
    return {
        "n": len(rhos),
        "spearman_mean": sum(rhos) / len(rhos) if rhos else float("nan"),
        "jaccard_mean": sum(jacs) / len(jacs) if jacs else float("nan"),
        "keep_ratio_mae": sum(ratio_diffs) / len(ratio_diffs) if ratio_diffs else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_ROOT / "robustness",
    )
    parser.add_argument(
        "--seeds", default="42,43,44,45",
        help="Comma-separated seeds for sub-sampling (42 reproduces full pilot for sanity).",
    )
    parser.add_argument(
        "--frac", type=float, default=0.5,
        help="Fraction of pilot to retain per seed sub-sample (default 0.5 → n=100).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = []
    for ds in DATASETS:
        log.info(f"=== dataset={ds} ===")
        pilot_ids = []
        with open(PILOT_CSVS[ds]) as f:
            for r in csv.DictReader(f):
                pilot_ids.append(r["video_id"])
        n_keep = max(1, int(round(args.frac * len(pilot_ids))))

        sidecars: Dict[str, Dict[str, dict]] = {
            o: {v: r for v, r in load_sidecars(MSS_DIRS[(o, ds)]).items() if v in set(pilot_ids)}
            for o in ORACLES
        }

        for seed in seeds:
            rng = random.Random(seed)
            ids = pilot_ids[:] if seed == 42 and args.frac >= 0.999 else rng.sample(pilot_ids, n_keep)
            for a, b in combinations(ORACLES, 2):
                m = aggregate_pair_metrics(sidecars[a], sidecars[b], ids)
                rows.append({
                    "dataset": ds, "seed": seed, "frac": args.frac,
                    "n_pilot": len(pilot_ids), "n_subsample": n_keep,
                    "pair": f"{a}__{b}", "n_compared": m["n"],
                    "spearman_mean": m["spearman_mean"],
                    "jaccard_mean": m["jaccard_mean"],
                    "keep_ratio_mae": m["keep_ratio_mae"],
                })

    out_path = args.output_dir / "seeds.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "dataset", "seed", "frac", "n_pilot", "n_subsample",
                "pair", "n_compared",
                "spearman_mean", "jaccard_mean", "keep_ratio_mae",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    log.info(f"wrote {out_path}")

    # Per-(dataset, pair) spread report
    spread_rows = []
    by_key: Dict[tuple, List[dict]] = {}
    for r in rows:
        by_key.setdefault((r["dataset"], r["pair"]), []).append(r)
    for (ds, pair), rs in by_key.items():
        for metric in ("spearman_mean", "jaccard_mean", "keep_ratio_mae"):
            vals = [r[metric] for r in rs if not math.isnan(r[metric])]
            if not vals:
                continue
            spread_rows.append({
                "dataset": ds, "pair": pair, "metric": metric,
                "min": min(vals), "max": max(vals),
                "spread": max(vals) - min(vals),
                "n_seeds": len(vals),
            })
    spread_path = args.output_dir / "spread.csv"
    with open(spread_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "pair", "metric", "min", "max", "spread", "n_seeds"],
        )
        writer.writeheader()
        for r in spread_rows:
            writer.writerow(r)
    log.info(f"wrote {spread_path}")


if __name__ == "__main__":
    main()
