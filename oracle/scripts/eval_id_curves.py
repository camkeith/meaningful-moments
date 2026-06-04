#!/usr/bin/env python
"""Insertion/Deletion-curve analyzer over a single I/D-sweep run dir.

For each video × ordering, reconstructs ``score(fraction_kept)`` at the inner
grid {10..90 percent} plus two endpoints:
    f=0   → chance level (1/n_classes for top-1, k/n_classes for top-k)
    f=100 → full-video score from ``<video_id>__full.json`` (if present;
            otherwise that video is excluded from AUC integration)

Then computes per-video insertion-AUC for each ordering using the trapezoidal
rule. Deletion-AUC under ordering O equals insertion-AUC under reverse(O), so
running both ``vlm`` and ``anti-vlm`` (or ``random`` twice — equivalent in
distribution) yields both directions.

Headline test: paired bootstrap for ΔAUC = AUC(vlm) − AUC(random), per
metric (top-1 / top-5). 10k resamples by default; 95% CI + two-sided p-value.

Outputs (under ``<run-dir>/id_curve_analysis/``):
    summary.json          — headline AUC + ΔAUC + CIs + provenance
    per_video.csv         — one row per (video_id, ordering, metric) with AUC
    per_fraction.csv      — mean ± SE of score at each grid point per ordering
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

LOG = logging.getLogger("eval_id_curves")

INNER_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
GRID_FRACTIONS = (0.0,) + INNER_FRACTIONS + (1.0,)
ID_RE = re.compile(r"^(?P<vid>.+)__id-(?P<ord>.+)-f(?P<pct>\d{2,3})\.json$")
FULL_RE = re.compile(r"^(?P<vid>.+)__full\.json$")
N_CLASSES = {
    "vjepa2-ssv2": 174,
    "vjepa2-diving48": 48,
    "videomae-k400-large": 400,
    "videomae-k400-huge": 400,
}


def chance_topk(n_classes: int, k: int) -> float:
    return min(1.0, k / n_classes)


def trapezoidal_auc(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    pairs = sorted(zip(xs, ys))
    out = 0.0
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        out += 0.5 * (y1 + y0) * (x1 - x0)
    return out


def load_sidecars(run_dir: Path) -> tuple[
    dict[tuple[str, str, int], dict],   # (vid, ordering, pct) → record
    dict[str, dict],                     # vid → full record
]:
    id_records: dict[tuple[str, str, int], dict] = {}
    full_records: dict[str, dict] = {}
    for p in run_dir.glob("*.json"):
        name = p.name
        if name.startswith("results_shard_") or name.startswith("summary_shard_"):
            continue
        if name in {"paired_records.jsonl", "run_metadata.json"}:
            continue
        m_id = ID_RE.match(name)
        if m_id:
            try:
                rec = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            id_records[(m_id["vid"], m_id["ord"], int(m_id["pct"]))] = rec
            continue
        m_full = FULL_RE.match(name)
        if m_full:
            try:
                full_records[m_full["vid"]] = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
    return id_records, full_records


def per_video_curves(
    id_records: dict[tuple[str, str, int], dict],
    full_records: dict[str, dict],
    n_classes: int,
) -> tuple[
    dict[str, dict[str, dict[float, dict[str, float]]]],
    dict[str, set[float]],
]:
    """Return (curves, inner_grid_per_ord).

    ``curves[vid][ordering][fraction] = {top1, top5}``.

    The "complete grid" for each ordering is derived from the run itself —
    the set of fractions for which any video produced an I/D sidecar under
    that ordering. A (vid, ordering) curve is included only when:
      - no I/D cell for that pair set ``selector_failed=True``;
      - every fraction in the ordering's inner grid is present;
      - the video has a ``__full.json`` sidecar (f=1 endpoint).
    f=0 is added as chance level (k/n_classes).
    """
    by_vid_ord: dict[tuple[str, str], dict[float, dict[str, float]]] = defaultdict(dict)
    by_vid_ord_failed: set[tuple[str, str]] = set()
    fractions_per_ord: dict[str, set[float]] = defaultdict(set)

    for (vid, ord_, pct), rec in id_records.items():
        f = pct / 100.0
        if rec.get("selector_failed", False):
            by_vid_ord_failed.add((vid, ord_))
            continue
        fractions_per_ord[ord_].add(f)
        by_vid_ord[(vid, ord_)][f] = {
            "top1": float(rec.get("top1_correct") or 0.0),
            "top5": float(rec.get("top5_correct") or 0.0),
        }

    chance_top1 = chance_topk(n_classes, 1)
    chance_top5 = chance_topk(n_classes, 5)

    curves: dict[str, dict[str, dict[float, dict[str, float]]]] = defaultdict(dict)
    for (vid, ord_), points in by_vid_ord.items():
        if (vid, ord_) in by_vid_ord_failed:
            continue
        inner_set = fractions_per_ord.get(ord_, set())
        if not inner_set.issubset(points.keys()):
            continue
        full = full_records.get(vid)
        if full is None:
            continue
        points[0.0] = {"top1": chance_top1, "top5": chance_top5}
        points[1.0] = {
            "top1": float(full.get("top1_correct") or 0.0),
            "top5": float(full.get("top5_correct") or 0.0),
        }
        curves[vid][ord_] = dict(points)
    return curves, dict(fractions_per_ord)


def per_video_auc(
    curves: dict[str, dict[str, dict[float, dict[str, float]]]],
) -> dict[tuple[str, str, str], float]:
    """Returns auc[(vid, ord, metric)] = trapezoidal AUC over [0, 1]."""
    out: dict[tuple[str, str, str], float] = {}
    for vid, by_ord in curves.items():
        for ord_, points in by_ord.items():
            xs = sorted(points.keys())
            for metric in ("top1", "top5"):
                ys = [points[x][metric] for x in xs]
                out[(vid, ord_, metric)] = trapezoidal_auc(xs, ys)
    return out


def paired_bootstrap_delta(
    aucs: dict[tuple[str, str, str], float],
    ord_a: str,
    ord_b: str,
    metric: str,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """ΔAUC = AUC(ord_a) − AUC(ord_b), paired by video. Returns mean, CI95, p_two."""
    import numpy as np

    diffs: list[float] = []
    for (vid, ord_, m) in aucs:
        if m != metric or ord_ != ord_a:
            continue
        if (vid, ord_b, metric) not in aucs:
            continue
        a = aucs[(vid, ord_a, metric)]
        b = aucs[(vid, ord_b, metric)]
        if math.isnan(a) or math.isnan(b):
            continue
        diffs.append(a - b)

    n = len(diffs)
    if n < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_two_sided": float("nan"), "n_paired": n}

    arr = np.array(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        boot[i] = arr[rng.integers(0, n, size=n)].mean()
    mean = float(arr.mean())
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    # Two-sided p ≈ 2 * min(P[boot ≥ 0 under H0], P[boot ≤ 0 under H0]).
    # Under H0 (mean=0), shift bootstrap by the observed mean:
    boot_centered = boot - mean
    p_two = 2.0 * min(
        float((boot_centered >= abs(mean)).mean()),
        float((boot_centered <= -abs(mean)).mean()),
    )
    p_two = min(1.0, max(p_two, 1.0 / n_resamples))
    return {"mean": mean, "ci_low": ci_low, "ci_high": ci_high,
            "p_two_sided": p_two, "n_paired": n}


def per_fraction_means(
    curves: dict[str, dict[str, dict[float, dict[str, float]]]],
) -> list[dict]:
    """Per-(ordering, fraction, metric) mean and SE across videos with that pair."""
    import math as _m
    rows = []
    by_ord_frac: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for vid, by_ord in curves.items():
        for ord_, points in by_ord.items():
            for frac, scores in points.items():
                for metric in ("top1", "top5"):
                    by_ord_frac[(ord_, frac, metric)].append(scores[metric])
    for (ord_, frac, metric), vals in sorted(by_ord_frac.items()):
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            se = _m.sqrt(var / n)
        else:
            se = float("nan")
        rows.append({"ordering": ord_, "fraction": frac, "metric": metric,
                     "n": n, "mean": mean, "se": se})
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=str,
                   help="I/D sweep run directory (contains *__id-*.json sidecars + optional *__full.json endpoints).")
    p.add_argument("--eval-csv", required=True, type=str,
                   help="The CSV used for the run (for provenance + counting).")
    p.add_argument("--recognizer", required=True, choices=sorted(N_CLASSES.keys()))
    p.add_argument("--reference-ordering", default="random",
                   help="Baseline ordering for ΔAUC = AUC(vlm) − AUC(reference). Default: random.")
    p.add_argument("--vlm-ordering", default="vlm")
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None,
                   help="Default: <run-dir>/id_curve_analysis/")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        LOG.error("Run dir missing: %s", run_dir); return 2
    eval_csv = Path(args.eval_csv)
    if not eval_csv.exists():
        LOG.error("Eval CSV missing: %s", eval_csv); return 2

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "id_curve_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_classes = N_CLASSES[args.recognizer]
    LOG.info("Loading sidecars from %s …", run_dir)
    t0 = time.time()
    id_records, full_records = load_sidecars(run_dir)
    LOG.info(
        "Loaded %d I/D sidecars + %d full-video sidecars in %.1fs",
        len(id_records), len(full_records), time.time() - t0,
    )

    orderings_present = sorted({k[1] for k in id_records.keys()})
    fractions_present = sorted({k[2] for k in id_records.keys()})
    LOG.info("Orderings: %s   Fractions: %s", orderings_present, fractions_present)

    curves, inner_grid_per_ord = per_video_curves(id_records, full_records, n_classes=n_classes)
    LOG.info(
        "Complete (vid, ord) curves: %d   per-ord inner grid (frac): %s",
        sum(len(v) for v in curves.values()),
        {o: sorted(round(f, 2) for f in fs) for o, fs in inner_grid_per_ord.items()},
    )

    aucs = per_video_auc(curves)

    # Per-video CSV
    pv_csv = out_dir / "per_video.csv"
    with pv_csv.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["video_id", "ordering", "metric", "auc"])
        for (vid, ord_, metric), v in sorted(aucs.items()):
            w.writerow([vid, ord_, metric, f"{v:.6f}"])
    LOG.info("Wrote %s", pv_csv)

    # Per-fraction CSV
    pf_csv = out_dir / "per_fraction.csv"
    pf_rows = per_fraction_means(curves)
    with pf_csv.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["ordering", "fraction", "metric", "n", "mean", "se"])
        for r in pf_rows:
            w.writerow([r["ordering"], f"{r['fraction']:.2f}", r["metric"], r["n"],
                        f"{r['mean']:.6f}", f"{r['se']:.6f}" if not math.isnan(r["se"]) else ""])
    LOG.info("Wrote %s", pf_csv)

    # Headline: ΔAUC for vlm vs reference (top-1 + top-5)
    headline: dict[str, dict] = {}
    for ord_ in orderings_present:
        if ord_ == args.reference_ordering:
            continue
        for metric in ("top1", "top5"):
            res = paired_bootstrap_delta(
                aucs, ord_, args.reference_ordering, metric,
                n_resamples=args.n_resamples, seed=args.seed,
            )
            headline.setdefault(ord_, {})[metric] = res

    # Mean AUC per ordering × metric (paired n)
    mean_auc: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for ord_ in orderings_present:
        for metric in ("top1", "top5"):
            vals = [v for (vid, o, m), v in aucs.items()
                    if o == ord_ and m == metric and not math.isnan(v)]
            if not vals:
                mean_auc[ord_][metric] = {"mean": float("nan"), "n": 0}
                continue
            mean_auc[ord_][metric] = {"mean": sum(vals) / len(vals), "n": len(vals)}

    summary = {
        "run_dir": str(run_dir),
        "eval_csv": str(eval_csv),
        "recognizer": args.recognizer,
        "n_classes": n_classes,
        "orderings_present": orderings_present,
        "fractions_present_pct": fractions_present,
        "n_videos_in_csv": sum(1 for _ in eval_csv.open()) - 1,
        "n_id_sidecars_loaded": len(id_records),
        "n_full_sidecars_loaded": len(full_records),
        "n_complete_curves": sum(len(v) for v in curves.values()),
        "mean_auc_by_ordering": {o: dict(m) for o, m in mean_auc.items()},
        "delta_auc_vs_reference": {
            "reference_ordering": args.reference_ordering,
            "deltas": headline,
        },
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "method": (
            "Per-video trapezoidal AUC over fraction-kept ∈ {0, 0.1,…, 0.9, 1.0}; "
            "f=0 = chance (k/n_classes); f=1 from <vid>__full.json. "
            "ΔAUC = AUC(vlm) − AUC(reference) paired by video; bootstrap with replacement, "
            "centered for two-sided p."
        ),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    LOG.info("Wrote %s", summary_path)

    # Pretty-print headline
    print()
    print("=" * 80)
    print(f"I/D-curve analysis — {args.recognizer}")
    print(f"  run-dir: {run_dir}")
    print(f"  complete curves: {summary['n_complete_curves']}")
    print(f"  orderings: {orderings_present}   fractions(pct): {fractions_present}")
    print()
    print("Mean AUC by ordering (paired n):")
    for ord_ in orderings_present:
        for metric in ("top1", "top5"):
            r = mean_auc[ord_][metric]
            print(f"  {ord_:>10s}  {metric}: AUC={r['mean']:.4f}  n={r['n']}")
    print()
    print(f"ΔAUC = AUC(<ord>) − AUC({args.reference_ordering})  (paired bootstrap, B={args.n_resamples})")
    for ord_, by_metric in headline.items():
        for metric, r in by_metric.items():
            print(f"  {ord_:>10s} − {args.reference_ordering:<10s}  {metric}: "
                  f"Δ={r['mean']:+.4f}  CI95=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
                  f"p={r['p_two_sided']:.4f}  n={r['n_paired']}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
