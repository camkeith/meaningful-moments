#!/usr/bin/env python
"""Stratified eval-subset sampler for SSv2 and K400 test pools.

Pinned MSS sidecar fields (verified against actual sidecars in the pinned runs):
    - Eligibility filter: ``mss_result.precheck_passed`` (bool)
    - Confidence (analysis-time bucketing): ``mss_result.precheck_responses[0].confidence`` (float in [0, 1])
    - Clip duration (analysis-time quartiles): ``segments[-1].end_s`` (float, seconds)

See the development design notes for protocol rationale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("sample_eval_subsets")

# TODO: diving-48 — full test set, no subsetting; track in next change
DATASETS: dict[str, dict[str, object]] = {
    "ssv2": {
        "default_cap_per_class": 12,
        "default_source_csv": "data/csvs/ssv2/test.csv",
        "default_mss_run": "pseudo_labels/mss/qwen3-vl-32b_20260412_133129",
        "default_output_csv": "data/csvs/ssv2/eval_2k_stratified.csv",
    },
    "k400": {
        "default_cap_per_class": 5,
        "default_source_csv": "data/csvs/k400/test.csv",
        "default_mss_run": "pseudo_labels/mss/qwen3-vl-32b_k400_test_20260502_122107",
        "default_output_csv": "data/csvs/k400/eval_2k_stratified.csv",
    },
}


def video_id_from_path(video_path: str) -> str:
    return Path(video_path).stem


def sidecar_path_for(mss_run_dir: Path, video_id: str) -> Path:
    return mss_run_dir / f"{video_id}.json"


def load_sidecar(path: Path) -> dict | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def extract_eligibility_and_meta(sidecar: dict) -> tuple[bool, float | None, float | None]:
    mss = sidecar.get("mss_result", {})
    precheck = bool(mss.get("precheck_passed", False))
    confidence: float | None = None
    responses = mss.get("precheck_responses") or []
    if responses:
        raw = responses[0].get("confidence")
        if raw is not None:
            confidence = float(raw)
    duration: float | None = None
    segments = sidecar.get("segments") or []
    if segments:
        end_s = segments[-1].get("end_s")
        if end_s is not None:
            duration = float(end_s)
    return precheck, confidence, duration


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_mss_run_hash(eligible_sidecars: list[Path], mss_run_dir: Path) -> str:
    """SHA-256 over sorted ``(relative_path, file_sha256)`` tuples for the eligible pool."""
    h = hashlib.sha256()
    for path in sorted(eligible_sidecars):
        rel = path.relative_to(mss_run_dir)
        h.update(str(rel).encode("utf-8"))
        h.update(b"\x00")
        h.update(sha256_file(path).encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()


def stratified_within_class(
    eligible: dict[str, list[tuple[str, str, float | None, float | None]]],
    cap_per_class: int,
    seed: int,
) -> list[tuple[str, str, str, float | None, float | None]]:
    """Draw ``min(cap, len)`` per class. Within-class sorted by video_id; classes alphabetical."""
    rng = random.Random(seed)
    out: list[tuple[str, str, str, float | None, float | None]] = []
    for cls in sorted(eligible.keys()):
        rows = list(eligible[cls])
        rng.shuffle(rows)
        chosen = rows[: min(cap_per_class, len(rows))]
        chosen.sort(key=lambda r: r[0])
        out.extend((cls, *r) for r in chosen)
    return out


def tercile_buckets(values: list[float | None]) -> tuple[dict[str, int], dict[str, tuple[float, float]]]:
    finite = sorted(v for v in values if v is not None)
    counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
    edges: dict[str, tuple[float, float]] = {}
    if not finite:
        return counts, edges
    n = len(finite)
    q1 = finite[n // 3]
    q2 = finite[(2 * n) // 3]
    edges = {"low": (finite[0], q1), "medium": (q1, q2), "high": (q2, finite[-1])}
    for v in values:
        if v is None:
            counts["unknown"] += 1
        elif v < q1:
            counts["low"] += 1
        elif v < q2:
            counts["medium"] += 1
        else:
            counts["high"] += 1
    return counts, edges


def quartile_buckets(values: list[float | None]) -> tuple[dict[str, int], dict[str, tuple[float, float]]]:
    finite = sorted(v for v in values if v is not None)
    counts: dict[str, int] = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0, "unknown": 0}
    edges: dict[str, tuple[float, float]] = {}
    if not finite:
        return counts, edges
    n = len(finite)
    q1 = finite[n // 4]
    q2 = finite[(2 * n) // 4]
    q3 = finite[(3 * n) // 4]
    edges = {
        "Q1": (finite[0], q1),
        "Q2": (q1, q2),
        "Q3": (q2, q3),
        "Q4": (q3, finite[-1]),
    }
    for v in values:
        if v is None:
            counts["unknown"] += 1
        elif v < q1:
            counts["Q1"] += 1
        elif v < q2:
            counts["Q2"] += 1
        elif v < q3:
            counts["Q3"] += 1
        else:
            counts["Q4"] += 1
    return counts, edges


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    p.add_argument("--n", type=int, default=2000, help="Target sample size (informational; actual size = sum of within-class draws)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap-per-class", type=int, default=None)
    p.add_argument("--mss-run-dir", type=str, default=None)
    p.add_argument("--source-csv", type=str, default=None)
    p.add_argument("--output-csv", type=str, default=None)
    p.add_argument("--meta-json", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    defaults = DATASETS[args.dataset]
    cap_per_class = args.cap_per_class if args.cap_per_class is not None else int(defaults["default_cap_per_class"])
    source_csv = Path(args.source_csv or str(defaults["default_source_csv"]))
    mss_run_dir = Path(args.mss_run_dir or str(defaults["default_mss_run"]))
    output_csv = Path(args.output_csv or str(defaults["default_output_csv"]))
    meta_json = Path(args.meta_json) if args.meta_json else Path(str(output_csv) + ".meta.json")

    if not args.overwrite:
        existing = [p for p in (output_csv, meta_json) if p.exists()]
        if existing:
            LOG.error("Refusing to overwrite existing files without --overwrite: %s", existing)
            return 2

    LOG.info("Reading source CSV: %s", source_csv)
    df = pd.read_csv(source_csv)
    pool_size = len(df)

    eligible: dict[str, list[tuple[str, str, float | None, float | None]]] = defaultdict(list)
    eligible_sidecars: list[Path] = []
    skipped_no_sidecar = 0
    skipped_precheck = 0

    LOG.info("Scanning %d candidates against MSS run %s", pool_size, mss_run_dir)
    for video_path, label in zip(df.video_path, df.label):
        vid = video_id_from_path(video_path)
        sp = sidecar_path_for(mss_run_dir, vid)
        sc = load_sidecar(sp)
        if sc is None:
            skipped_no_sidecar += 1
            continue
        precheck, conf, dur = extract_eligibility_and_meta(sc)
        if not precheck:
            skipped_precheck += 1
            continue
        eligible[label].append((vid, video_path, conf, dur))
        eligible_sidecars.append(sp)

    eligible_count = sum(len(v) for v in eligible.values())
    LOG.info(
        "Eligible: %d (skipped: no_sidecar=%d, precheck_failed=%d) across %d classes",
        eligible_count, skipped_no_sidecar, skipped_precheck, len(eligible),
    )

    sampled = stratified_within_class(eligible, cap_per_class, args.seed)
    LOG.info("Sampled %d videos (target n=%d, cap=%d/class)", len(sampled), args.n, cap_per_class)

    LOG.info("Hashing %d eligible sidecars (SHA-256 each + manifest hash)…", len(eligible_sidecars))
    mss_hash = compute_mss_run_hash(eligible_sidecars, mss_run_dir)
    LOG.info("MSS-run content hash: %s", mss_hash)

    sampled_confs = [r[3] for r in sampled]
    sampled_durs = [r[4] for r in sampled]
    conf_counts, conf_edges = tercile_buckets(sampled_confs)
    len_counts, len_edges = quartile_buckets(sampled_durs)
    per_class_counts = Counter(r[0] for r in sampled)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"video_path": [r[2] for r in sampled], "label": [r[0] for r in sampled]})
    out_df.to_csv(output_csv, index=False, lineterminator="\n")

    meta = {
        "dataset": args.dataset,
        "n_target": args.n,
        "actual_n": len(sampled),
        "seed": args.seed,
        "cap_per_class": cap_per_class,
        "source_csv": str(source_csv),
        "mss_run_dir": str(mss_run_dir),
        "mss_run_content_hash": mss_hash,
        "mss_run_eligible_count": len(eligible_sidecars),
        "source_pool_size": pool_size,
        "skipped_no_sidecar": skipped_no_sidecar,
        "skipped_precheck_failed": skipped_precheck,
        "n_classes_in_pool": len(eligible),
        "per_class_counts": dict(sorted(per_class_counts.items())),
        "confidence_bucket_counts": conf_counts,
        "confidence_bucket_edges": {k: list(v) for k, v in conf_edges.items()},
        "length_quartile_counts": len_counts,
        "length_quartile_edges_s": {k: list(v) for k, v in len_edges.items()},
        "pinned_fields": {
            "eligibility": "mss_result.precheck_passed",
            "confidence": "mss_result.precheck_responses[0].confidence",
            "duration_s": "segments[-1].end_s",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with meta_json.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    LOG.info("Wrote CSV: %s (%d rows)", output_csv, len(sampled))
    LOG.info("Wrote meta: %s", meta_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
