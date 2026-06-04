#!/usr/bin/env python
"""Shuffle-damage analysis: temporal-order shuffle test on classifier-eval runs.

For each (dataset, condition) pair, compares unshuffled vs shuffled top1/top5
recognition. Reports:
  - Per-condition top1/top5 with and without shuffle
  - Shuffle damage = unshuffled - shuffled (in accuracy points)
  - McNemar's test on per-video flips (b = unshuffled-correct→shuffled-wrong,
    c = unshuffled-wrong→shuffled-correct, p-value)
  - Paired bootstrap CI on the per-video shuffle damage
  - Differential shuffle damage between two conditions (e.g., uniform vs
    vlm-selected): paired bootstrap on damage(uniform) - damage(vlm-selected)

Hypothesis being tested (Option B from the user's spec):
  - On SSv2: damage(uniform) > damage(vlm-selected). Uniform was relying on
    temporal info that shuffling destroys; vlm-selected wasn't using that info
    anyway.
  - On K400: damage(uniform) ≈ damage(vlm-selected) ≈ 0. Neither uses temporal
    info, so shuffling doesn't matter much.

Reads sidecars from one or more --run-dir directories. Sidecars use the schema
written by run_classifier_eval.py: <video_id>__<condition>.json or
<video_id>__<condition>--shuffled.json.

Inputs:
  --run-dir       (repeatable) classifier_eval run dir; videos can come from
                  multiple dirs but typically one SSv2 dir + one K400 dir
  --dataset-tag   (repeatable, parallel to --run-dir) "ssv2" or "k400"
  --conditions    base condition names to analyze (e.g. vlm-selected,uniform)
  --output-dir    where to write CSV + summary
  --bootstrap-reps  default 10000

Output:
  shuffle_damage_summary.csv  — per (dataset, condition) shuffle damage stats
  shuffle_damage_pairwise.csv — differential damage between condition pairs
  paired_records.csv          — per-(dataset, video_id, condition) raw flips
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_sidecar(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def discover_paired_videos(
    run_dir: Path,
    base_condition: str,
) -> List[Tuple[str, dict, dict]]:
    """Find videos with both unshuffled and shuffled sidecars for `base_condition`.

    Returns list of (video_id, unshuffled_sidecar, shuffled_sidecar).
    Drops videos where either side is missing or selector_failed.
    """
    pairs = []
    for unshuffled_p in run_dir.glob(f"*__{base_condition}.json"):
        vid = unshuffled_p.name.split(f"__{base_condition}.json")[0]
        shuffled_p = run_dir / f"{vid}__{base_condition}--shuffled.json"
        if not shuffled_p.exists():
            continue
        u = load_sidecar(unshuffled_p)
        s = load_sidecar(shuffled_p)
        if u is None or s is None:
            continue
        if u.get("selector_failed") or s.get("selector_failed"):
            continue
        pairs.append((vid, u, s))
    return pairs


def mcnemar_pvalue(b: int, c: int) -> float:
    """Exact binomial McNemar p-value (two-sided).

    b = pairs where condition_a correct, condition_b wrong
    c = pairs where condition_a wrong, condition_b correct
    """
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided exact binomial test on min(b, c) under p=0.5
    k = min(b, c)
    # P(X <= k or X >= n-k) under Binomial(n, 0.5)
    log2 = math.log(2)
    log_total = n * log2  # sum of binomial coeffs * 0.5^n = 1; but we want P
    # Cumulative
    p_one_tail = 0.0
    log_coef = 0.0  # log C(n, 0)
    for i in range(0, k + 1):
        if i > 0:
            log_coef += math.log(n - i + 1) - math.log(i)
        p_one_tail += math.exp(log_coef - n * log2)
    p_two_tail = min(1.0, 2.0 * p_one_tail)
    return p_two_tail


def paired_bootstrap_ci(
    diffs: List[float],
    reps: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired bootstrap CI for the mean of diffs.

    Returns (mean, lower, upper) at (1-alpha) confidence.
    """
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(reps):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * reps)]
    hi = means[int((1 - alpha / 2) * reps)]
    return sum(diffs) / n, lo, hi


def analyze_shuffle_damage(
    dataset: str,
    run_dir: Path,
    base_condition: str,
    bootstrap_reps: int,
) -> Optional[Dict]:
    """Per-condition shuffle damage with McNemar + paired bootstrap."""
    pairs = discover_paired_videos(run_dir, base_condition)
    if not pairs:
        return None

    n = len(pairs)
    # Per-video flip outcomes (top1)
    u_top1, s_top1 = [], []
    u_top5, s_top5 = [], []
    diffs_top1, diffs_top5 = [], []
    b_top1 = c_top1 = 0  # unshuffled correct & shuffled wrong / vice versa
    b_top5 = c_top5 = 0
    for _, u, s in pairs:
        ut1 = bool(u.get("top1_correct"))
        st1 = bool(s.get("top1_correct"))
        ut5 = bool(u.get("top5_correct"))
        st5 = bool(s.get("top5_correct"))
        u_top1.append(ut1); s_top1.append(st1)
        u_top5.append(ut5); s_top5.append(st5)
        diffs_top1.append(int(ut1) - int(st1))
        diffs_top5.append(int(ut5) - int(st5))
        if ut1 and not st1: b_top1 += 1
        elif st1 and not ut1: c_top1 += 1
        if ut5 and not st5: b_top5 += 1
        elif st5 and not ut5: c_top5 += 1

    u_acc1 = sum(u_top1) / n
    s_acc1 = sum(s_top1) / n
    u_acc5 = sum(u_top5) / n
    s_acc5 = sum(s_top5) / n

    mean_d1, lo1, hi1 = paired_bootstrap_ci(diffs_top1, bootstrap_reps)
    mean_d5, lo5, hi5 = paired_bootstrap_ci(diffs_top5, bootstrap_reps)
    p_top1 = mcnemar_pvalue(b_top1, c_top1)
    p_top5 = mcnemar_pvalue(b_top5, c_top5)

    return {
        "dataset": dataset,
        "condition": base_condition,
        "n_paired_videos": n,
        "unshuffled_top1": u_acc1,
        "shuffled_top1": s_acc1,
        "shuffle_damage_top1": u_acc1 - s_acc1,
        "shuffle_damage_top1_ci_lo": lo1,
        "shuffle_damage_top1_ci_hi": hi1,
        "mcnemar_top1_b": b_top1,
        "mcnemar_top1_c": c_top1,
        "mcnemar_top1_p": p_top1,
        "unshuffled_top5": u_acc5,
        "shuffled_top5": s_acc5,
        "shuffle_damage_top5": u_acc5 - s_acc5,
        "shuffle_damage_top5_ci_lo": lo5,
        "shuffle_damage_top5_ci_hi": hi5,
        "mcnemar_top5_b": b_top5,
        "mcnemar_top5_c": c_top5,
        "mcnemar_top5_p": p_top5,
        # Keep the per-video diffs for differential-damage computation
        "_diffs_top1": diffs_top1,
        "_diffs_top5": diffs_top5,
        "_pairs": pairs,
    }


def differential_damage(
    a: Dict,
    b: Dict,
    bootstrap_reps: int,
) -> Optional[Dict]:
    """Differential shuffle damage on the videos paired-present in BOTH conditions.

    For videos where both `a` and `b` have unshuffled+shuffled sidecars, compute
    per-video shuffle damage in each condition and return paired bootstrap CI on
    damage(a) - damage(b).
    """
    a_pairs = {vid: (u, s) for vid, u, s in a["_pairs"]}
    b_pairs = {vid: (u, s) for vid, u, s in b["_pairs"]}
    shared = sorted(set(a_pairs) & set(b_pairs))
    if len(shared) < 10:
        return None
    diffs_top1 = []
    diffs_top5 = []
    for vid in shared:
        ua, sa = a_pairs[vid]
        ub, sb = b_pairs[vid]
        da1 = int(bool(ua.get("top1_correct"))) - int(bool(sa.get("top1_correct")))
        db1 = int(bool(ub.get("top1_correct"))) - int(bool(sb.get("top1_correct")))
        diffs_top1.append(da1 - db1)
        da5 = int(bool(ua.get("top5_correct"))) - int(bool(sa.get("top5_correct")))
        db5 = int(bool(ub.get("top5_correct"))) - int(bool(sb.get("top5_correct")))
        diffs_top5.append(da5 - db5)
    m1, lo1, hi1 = paired_bootstrap_ci(diffs_top1, bootstrap_reps)
    m5, lo5, hi5 = paired_bootstrap_ci(diffs_top5, bootstrap_reps)
    return {
        "dataset": a["dataset"],
        "condition_a": a["condition"],
        "condition_b": b["condition"],
        "n_shared": len(shared),
        "diff_damage_top1": m1,
        "diff_damage_top1_ci_lo": lo1,
        "diff_damage_top1_ci_hi": hi1,
        "diff_damage_top5": m5,
        "diff_damage_top5_ci_lo": lo5,
        "diff_damage_top5_ci_hi": hi5,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", action="append", required=True,
        help="Classifier-eval run dir (repeatable). Pair with --dataset-tag.",
    )
    p.add_argument(
        "--dataset-tag", action="append", required=True,
        help='"ssv2" or "k400" — one per --run-dir, in matching order',
    )
    p.add_argument(
        "--conditions", nargs="+",
        default=["vlm-selected", "uniform"],
        help="Base conditions to analyze (default: vlm-selected uniform)",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write CSV outputs",
    )
    p.add_argument("--bootstrap-reps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if len(args.run_dir) != len(args.dataset_tag):
        print(f"--run-dir count ({len(args.run_dir)}) != --dataset-tag count "
              f"({len(args.dataset_tag)})", file=sys.stderr)
        sys.exit(2)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Per-condition damage analysis
    summaries: List[Dict] = []
    by_dataset: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for run_dir, tag in zip(args.run_dir, args.dataset_tag):
        run_dir = Path(run_dir)
        for cond in args.conditions:
            res = analyze_shuffle_damage(tag, run_dir, cond, args.bootstrap_reps)
            if res is None:
                print(f"[{tag}/{cond}] no paired videos in {run_dir} — skipping",
                      file=sys.stderr)
                continue
            by_dataset[tag][cond] = res
            summaries.append(res)

    # Strip private fields before writing CSV
    summary_csv = args.output_dir / "shuffle_damage_summary.csv"
    if summaries:
        public_fields = [k for k in summaries[0].keys() if not k.startswith("_")]
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=public_fields)
            w.writeheader()
            for r in summaries:
                w.writerow({k: r[k] for k in public_fields})
        print(f"Wrote {summary_csv}  ({len(summaries)} rows)", file=sys.stderr)

    # Pairwise differential damage (e.g., uniform vs vlm-selected)
    pairwise = []
    for tag, conds in by_dataset.items():
        cond_names = list(conds.keys())
        for i in range(len(cond_names)):
            for j in range(len(cond_names)):
                if i == j: continue
                a, b = conds[cond_names[i]], conds[cond_names[j]]
                d = differential_damage(a, b, args.bootstrap_reps)
                if d is not None:
                    pairwise.append(d)

    pairwise_csv = args.output_dir / "shuffle_damage_pairwise.csv"
    if pairwise:
        with open(pairwise_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairwise[0].keys()))
            w.writeheader()
            for r in pairwise:
                w.writerow(r)
        print(f"Wrote {pairwise_csv}  ({len(pairwise)} rows)", file=sys.stderr)

    # Per-video paired records (raw flips)
    paired_csv = args.output_dir / "paired_records.csv"
    with open(paired_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "video_id", "condition",
                    "unshuffled_top1", "shuffled_top1",
                    "unshuffled_top5", "shuffled_top5"])
        for r in summaries:
            for vid, u, s in r["_pairs"]:
                w.writerow([
                    r["dataset"], vid, r["condition"],
                    int(bool(u.get("top1_correct"))),
                    int(bool(s.get("top1_correct"))),
                    int(bool(u.get("top5_correct"))),
                    int(bool(s.get("top5_correct"))),
                ])
    print(f"Wrote {paired_csv}", file=sys.stderr)

    # Human-readable summary to stdout
    print("\n========================================")
    print("SHUFFLE DAMAGE SUMMARY")
    print("========================================")
    for r in summaries:
        sig1 = "**" if r["mcnemar_top1_p"] < 0.05 else "  "
        print(
            f"  {r['dataset']:<5s} {r['condition']:<22s}  n={r['n_paired_videos']:5d}  "
            f"top1: {r['unshuffled_top1']:.3f} → {r['shuffled_top1']:.3f}  "
            f"damage={r['shuffle_damage_top1']:+.3f} "
            f"CI=[{r['shuffle_damage_top1_ci_lo']:+.3f}, {r['shuffle_damage_top1_ci_hi']:+.3f}]  "
            f"McNemar p={r['mcnemar_top1_p']:.4f} {sig1}"
        )

    if pairwise:
        print("\n========================================")
        print("DIFFERENTIAL SHUFFLE DAMAGE (a - b)")
        print("========================================")
        for d in pairwise:
            sig = "**" if (d["diff_damage_top1_ci_lo"] > 0 or d["diff_damage_top1_ci_hi"] < 0) else "  "
            print(
                f"  {d['dataset']:<5s} damage({d['condition_a']:>14s}) - damage({d['condition_b']:<14s})  "
                f"n={d['n_shared']:5d}  "
                f"top1 diff={d['diff_damage_top1']:+.3f} "
                f"CI=[{d['diff_damage_top1_ci_lo']:+.3f}, {d['diff_damage_top1_ci_hi']:+.3f}] {sig}"
            )


if __name__ == "__main__":
    main()
