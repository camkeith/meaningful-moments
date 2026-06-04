#!/usr/bin/env python
"""
Compare two classify_ssv2.py output dirs (full-video arm vs MSS-kept arm).

Reads every per-video sidecar from each dir, intersects on video_id, verifies
the two arms used the same model/prompt/gen config, then writes:

  <output-dir>/metrics.json  — aggregate top-1/top-5 per arm + delta
  <output-dir>/per_class.csv — per-template breakdown

Usage:
    python oracle/scripts/classify_ssv2_eval.py \\
        --full-dir pseudo_labels/classify_ssv2/full_<ts>/ \\
        --mss-kept-dir pseudo_labels/classify_ssv2/mss_kept_<ts>/ \\
        --output-dir pseudo_labels/classify_ssv2/eval_<ts>/
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from classify.prompt import SSV2_LABELS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("classify_ssv2_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--mss-kept-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ignore-metadata-mismatch", action="store_true",
                   help="Proceed even if run_metadata.json differs between arms (not recommended)")
    return p.parse_args()


def _load_sidecars(d: Path) -> Dict[str, dict]:
    out = {}
    for p in d.glob("*.json"):
        if p.name == "run_metadata.json":
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping unreadable sidecar {p}: {e}")
            continue
        out[rec["video_id"]] = rec
    return out


def _load_metadata(d: Path) -> Optional[dict]:
    p = d / "run_metadata.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


_PARITY_KEYS = ("model", "prompt_hash", "max_new_tokens", "do_sample", "temperature", "n_labels")


def _assert_parity(m_full: Optional[dict], m_mss: Optional[dict], allow_mismatch: bool):
    if m_full is None or m_mss is None:
        msg = "run_metadata.json missing in one or both dirs"
        if allow_mismatch:
            logger.warning(msg)
            return
        raise SystemExit(msg)
    diffs = []
    for k in _PARITY_KEYS:
        if m_full.get(k) != m_mss.get(k):
            diffs.append(f"  {k}: full={m_full.get(k)!r}  mss_kept={m_mss.get(k)!r}")
    if diffs:
        msg = "Run metadata mismatch between arms:\n" + "\n".join(diffs)
        if allow_mismatch:
            logger.warning(msg)
            return
        raise SystemExit(
            msg + "\n(pass --ignore-metadata-mismatch to override; results won't be comparable)"
        )


def _summary(records: List[dict]) -> dict:
    n = len(records)
    n_parsed = sum(1 for r in records if r.get("parsed"))
    n_top1 = sum(1 for r in records if r.get("top1_correct"))
    n_top5 = sum(1 for r in records if r.get("top5_correct"))
    return {
        "n": n,
        "n_parsed": n_parsed,
        "parse_failure_rate": 0.0 if n == 0 else (n - n_parsed) / n,
        "top1_accuracy": 0.0 if n == 0 else n_top1 / n,
        "top5_accuracy": 0.0 if n == 0 else n_top5 / n,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_records = _load_sidecars(args.full_dir)
    mss_records = _load_sidecars(args.mss_kept_dir)
    logger.info(f"full arm:     {len(full_records)} sidecars")
    logger.info(f"mss_kept arm: {len(mss_records)} sidecars")

    meta_full = _load_metadata(args.full_dir)
    meta_mss = _load_metadata(args.mss_kept_dir)
    _assert_parity(meta_full, meta_mss, args.ignore_metadata_mismatch)

    shared = sorted(set(full_records) & set(mss_records))
    only_full = set(full_records) - set(mss_records)
    only_mss = set(mss_records) - set(full_records)
    if only_full or only_mss:
        logger.warning(
            f"Video sets differ: {len(only_full)} only in full, {len(only_mss)} only in mss_kept. "
            f"Metrics computed over intersection of {len(shared)}."
        )

    full_list = [full_records[v] for v in shared]
    mss_list = [mss_records[v] for v in shared]

    summary_full = _summary(full_list)
    summary_mss = _summary(mss_list)

    # Per-video agreement (top-1 picks match between arms)
    agree = sum(
        1 for f, m in zip(full_list, mss_list)
        if f.get("parsed") and m.get("parsed") and f.get("top5") and m.get("top5")
        and f["top5"][0] == m["top5"][0]
    )

    # Per-class rollup
    per_class = defaultdict(lambda: {
        "n_videos": 0,
        "top1_full": 0, "top5_full": 0,
        "top1_mss_kept": 0, "top5_mss_kept": 0,
    })
    for f, m in zip(full_list, mss_list):
        gt = f.get("gt_label") or m.get("gt_label")
        if gt is None:
            continue
        d = per_class[gt]
        d["n_videos"] += 1
        if f.get("top1_correct"): d["top1_full"] += 1
        if f.get("top5_correct"): d["top5_full"] += 1
        if m.get("top1_correct"): d["top1_mss_kept"] += 1
        if m.get("top5_correct"): d["top5_mss_kept"] += 1

    rows = []
    for label, d in per_class.items():
        n = d["n_videos"] or 1
        rows.append({
            "class_template": label,
            "n_videos": d["n_videos"],
            "top1_full": d["top1_full"] / n,
            "top5_full": d["top5_full"] / n,
            "top1_mss_kept": d["top1_mss_kept"] / n,
            "top5_mss_kept": d["top5_mss_kept"] / n,
            "delta_top1": (d["top1_full"] - d["top1_mss_kept"]) / n,
            "delta_top5": (d["top5_full"] - d["top5_mss_kept"]) / n,
        })
    per_class_df = pd.DataFrame(rows).sort_values("class_template")
    per_class_path = args.output_dir / "per_class.csv"
    per_class_df.to_csv(per_class_path, index=False)

    metrics = {
        "n_evaluated": len(shared),
        "n_only_in_full_dir": len(only_full),
        "n_only_in_mss_dir": len(only_mss),
        "full": summary_full,
        "mss_kept": summary_mss,
        "delta": {
            "top1_accuracy": summary_full["top1_accuracy"] - summary_mss["top1_accuracy"],
            "top5_accuracy": summary_full["top5_accuracy"] - summary_mss["top5_accuracy"],
        },
        "top1_agreement_rate": 0.0 if not shared else agree / len(shared),
        "run_metadata_full": meta_full,
        "run_metadata_mss_kept": meta_mss,
    }
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Human summary
    print("=" * 72)
    print(f"  Evaluated videos: {len(shared)}")
    print(f"  Parse failures:   full={summary_full['n'] - summary_full['n_parsed']}   "
          f"mss_kept={summary_mss['n'] - summary_mss['n_parsed']}")
    print("-" * 72)
    print(f"  top-1 accuracy:   full={summary_full['top1_accuracy']:.4f}   "
          f"mss_kept={summary_mss['top1_accuracy']:.4f}   "
          f"Δ={metrics['delta']['top1_accuracy']:+.4f}")
    print(f"  top-5 accuracy:   full={summary_full['top5_accuracy']:.4f}   "
          f"mss_kept={summary_mss['top5_accuracy']:.4f}   "
          f"Δ={metrics['delta']['top5_accuracy']:+.4f}")
    print(f"  top-1 agreement:  {metrics['top1_agreement_rate']:.4f}")
    print("=" * 72)
    print(f"  Wrote: {metrics_path}")
    print(f"  Wrote: {per_class_path}")


if __name__ == "__main__":
    main()
