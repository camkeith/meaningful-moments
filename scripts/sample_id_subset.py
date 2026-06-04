#!/usr/bin/env python
"""Stratified n=500 sub-sampler from the pinned eval_2k_stratified CSVs.

Optional fallback when compute is constrained: draws a fixed n (default 500)
from ``data/csvs/{ssv2,k400}/eval_2k_stratified.csv`` while preserving the
per-class distribution of the parent. Per-class quotas use floor + probabilistic
top-up so the output hits the target n exactly.

Outputs (default):
    data/csvs/<dataset>/id_<n>_stratified.csv
    data/csvs/<dataset>/id_<n>_stratified.csv.meta.json

The meta JSON records: parent CSV path + sha256, seed, n_target, actual_n,
per-class counts, and parent meta hash (so we can verify provenance later).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("sample_id_subset")

DATASET_DEFAULTS: dict[str, dict[str, str]] = {
    "ssv2": {"parent_csv": "data/csvs/ssv2/eval_2k_stratified.csv"},
    "k400": {"parent_csv": "data/csvs/k400/eval_2k_stratified.csv"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def stratified_subsample(
    rows: list[dict[str, str]], n_target: int, seed: int
) -> list[dict[str, str]]:
    """Draw n_target rows preserving per-class distribution.

    Algorithm (deterministic given seed):
      1. Group rows by ``label``.
      2. Per-class quota = n_target * (class_size / total_size).
      3. Floor the quota → integer base; track the fractional remainder.
      4. Distribute the residual (n_target − sum(floors)) to classes with the
         largest fractional remainders (ties broken by alphabetical class name).
      5. Within each class, shuffle (seeded) and take the first ``quota`` rows.
    """
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)
    classes = sorted(by_class.keys())
    total = sum(len(by_class[c]) for c in classes)
    if total == 0:
        return []

    raw_quota = {c: n_target * len(by_class[c]) / total for c in classes}
    floor = {c: int(math.floor(raw_quota[c])) for c in classes}
    residual = n_target - sum(floor.values())
    # Cap floors at class size; ranked top-up by fractional remainder
    # then by descending class_size (more headroom) then alphabetical.
    fractionals = sorted(
        classes,
        key=lambda c: (
            -(raw_quota[c] - floor[c]),  # largest remainder first
            -len(by_class[c]),
            c,
        ),
    )
    quota = dict(floor)
    i = 0
    while residual > 0 and i < len(fractionals):
        c = fractionals[i % len(fractionals)]
        if quota[c] < len(by_class[c]):
            quota[c] += 1
            residual -= 1
        i += 1
        # If we ran out of headroom this lap, do another pass; if no class can
        # take more, break to avoid infinite loop.
        if i >= len(fractionals) and residual > 0:
            saturated = all(quota[c] >= len(by_class[c]) for c in classes)
            if saturated:
                LOG.warning(
                    "Residual %d > 0 but all classes saturated (parent has %d total). "
                    "Output will be smaller than n_target.", residual, total,
                )
                break
            i = 0  # another pass

    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    for c in classes:
        bucket = list(by_class[c])
        rng.shuffle(bucket)
        out.extend(bucket[: quota[c]])
    out.sort(key=lambda r: (r["label"], r["video_path"]))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS.keys()))
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parent-csv", default=None, help="Override parent CSV path.")
    p.add_argument("--output-csv", default=None, help="Override output CSV path.")
    p.add_argument("--meta-json", default=None, help="Override output meta path.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    parent_csv = Path(args.parent_csv or DATASET_DEFAULTS[args.dataset]["parent_csv"])
    output_csv = Path(args.output_csv or f"data/csvs/{args.dataset}/id_{args.n}_stratified.csv")
    meta_json = Path(args.meta_json or str(output_csv) + ".meta.json")

    if not parent_csv.exists():
        LOG.error("Parent CSV missing: %s", parent_csv)
        return 2

    if not args.overwrite:
        existing = [p for p in (output_csv, meta_json) if p.exists()]
        if existing:
            LOG.error("Refusing to overwrite without --overwrite: %s", existing)
            return 2

    rows = load_rows(parent_csv)
    LOG.info("Parent CSV %s: %d rows, %d classes", parent_csv, len(rows),
             len({r["label"] for r in rows}))

    sampled = stratified_subsample(rows, args.n, args.seed)
    LOG.info("Sampled %d rows (target n=%d, seed=%d)", len(sampled), args.n, args.seed)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "label"], lineterminator="\n")
        w.writeheader()
        for r in sampled:
            w.writerow({"video_path": r["video_path"], "label": r["label"]})

    parent_meta = parent_csv.with_suffix(parent_csv.suffix + ".meta.json")
    parent_meta_hash = sha256_file(parent_meta) if parent_meta.exists() else None

    meta = {
        "dataset": args.dataset,
        "n_target": args.n,
        "actual_n": len(sampled),
        "seed": args.seed,
        "parent_csv": str(parent_csv),
        "parent_csv_sha256": sha256_file(parent_csv),
        "parent_meta_sha256": parent_meta_hash,
        "n_classes_in_parent": len({r["label"] for r in rows}),
        "n_classes_in_sample": len({r["label"] for r in sampled}),
        "per_class_counts": dict(sorted(Counter(r["label"] for r in sampled).items())),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "I/D-curve fallback subsample (n=500). Parent is the pinned 2k stratified eval pool.",
    }
    with meta_json.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    LOG.info("Wrote CSV: %s", output_csv)
    LOG.info("Wrote meta: %s", meta_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
