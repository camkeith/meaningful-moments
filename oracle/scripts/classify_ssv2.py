#!/usr/bin/env python
"""
Closed-set top-5 SSv2 classification with Qwen3-VL.

Two modes:
  --mode full       Classify the full video.
  --mode mss_kept   Build a CUT+concat video from MSS kept_indices and classify it.

Both modes read the same MSS labels dir (required) to filter out precheck-failed
videos, so the evaluated set is identical across arms.

Usage (full video arm):
    python oracle/scripts/classify_ssv2.py \\
        --dataset data/csvs/ssv2/test.csv \\
        --mss-labels-dir pseudo_labels/mss/qwen3-vl-32b_20260412_133129/ \\
        --output-dir pseudo_labels/classify_ssv2/full_<ts>/ \\
        --mode full \\
        --model qwen3-vl-32b \\
        --parallel --visible-gpus 0,1,2,3 \\
        --gpus-per-instance 4 --batch-size 35

Usage (MSS-kept arm):
    python oracle/scripts/classify_ssv2.py \\
        --dataset data/csvs/ssv2/test.csv \\
        --mss-labels-dir pseudo_labels/mss/qwen3-vl-32b_20260412_133129/ \\
        --output-dir pseudo_labels/classify_ssv2/mss_kept_<ts>/ \\
        --mode mss_kept \\
        --model qwen3-vl-32b \\
        --parallel --visible-gpus 0,1,2,3 \\
        --gpus-per-instance 4 --batch-size 35
"""

import argparse
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from classify.mss_loader import MSSRecord, load_mss_record  # noqa: E402
from classify.parallel import ParallelClassifier  # noqa: E402
from classify.prompt import SSV2_LABELS, prompt_hash  # noqa: E402
from classify.qwen_classifier import create_classifier  # noqa: E402
from classify.video_io import load_full_frames, load_mss_kept_frames  # noqa: E402

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("classify_ssv2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-set top-5 SSv2 classification with Qwen3-VL")
    p.add_argument("--dataset", type=Path, required=True, help="CSV with columns video_path,label")
    p.add_argument("--mss-labels-dir", type=Path, required=True,
                   help="Directory of MSS output JSONs (one per video). Used for precheck filter "
                        "in all modes and for kept_indices in --mode mss_kept.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where per-video sidecar JSONs are written. "
                        "If omitted, defaults to pseudo_labels/classify_ssv2/{mode}_{YYYYMMDD_HHMMSS}/")
    p.add_argument("--output-root", type=Path, default=Path("pseudo_labels/classify_ssv2"),
                   help="Parent directory used when auto-generating --output-dir (default: "
                        "pseudo_labels/classify_ssv2)")
    p.add_argument("--mode", choices=["full", "mss_kept"], required=True)
    p.add_argument("--model", default="qwen3-vl-32b",
                   help="Model key from mss.qwen_oracle.MODEL_CONFIGS (default: qwen3-vl-32b)")
    p.add_argument("--cache-dir", type=str, default="./models/pretrained_oracle_models/hf_cache")
    p.add_argument("--parallel", action="store_true", help="Use multi-process workers")
    p.add_argument("--visible-gpus", type=str, default=None, help="Comma-separated GPU IDs (e.g. '0,1,2,3')")
    p.add_argument("--gpus-per-instance", type=int, default=4,
                   help="GPUs per worker (default: 4 = single worker, tensor-parallel across all GPUs)")
    p.add_argument("--batch-size", type=int, default=35, help="Videos per forward pass (default: 35)")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="Max tokens to generate per video (default: 128 — IDs-only output ~30 tokens + margin)")
    p.add_argument("--max-retries", type=int, default=1,
                   help="Max parse attempts per video (default: 1 = greedy only; no sampling — classification task)")
    p.add_argument("--limit", type=int, default=None, help="Process only first N videos")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecar JSONs")
    p.add_argument("--log-dir", type=Path, default=Path("logs/classify_ssv2"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def setup_file_logging(log_dir: Path, tag: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{tag}_{ts}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)
    return log_path


def ground_truth_normalization_map() -> Dict[str, str]:
    """Map canonical SSv2 template -> canonical template (identity) + trivial variants."""
    m = {}
    for label in SSV2_LABELS:
        m[label] = label
        m[label.strip().lower()] = label
    return m


def build_pending_rows(
    dataset_df: pd.DataFrame,
    mss_labels_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> tuple:
    """Resolve each CSV row to (MSSRecord, gt_label). Skip if no MSS, precheck failed, or output exists."""
    rows = []
    gt_map = ground_truth_normalization_map()
    stats = {
        "total": len(dataset_df),
        "skipped_missing_mss": 0,
        "skipped_precheck_failed": 0,
        "skipped_existing_output": 0,
        "skipped_gt_mismatch": 0,
        "to_process": 0,
    }
    for _, row in dataset_df.iterrows():
        video_path = row["video_path"]
        video_id = Path(video_path).stem
        gt_raw = str(row["label"])
        gt = gt_map.get(gt_raw) or gt_map.get(gt_raw.strip().lower())
        if gt is None:
            stats["skipped_gt_mismatch"] += 1
            continue

        mss_json = mss_labels_dir / f"{video_id}.json"
        if not mss_json.exists():
            stats["skipped_missing_mss"] += 1
            continue

        try:
            rec = load_mss_record(mss_json)
        except Exception as e:
            logger.warning(f"Failed to load MSS JSON for {video_id}: {e}")
            stats["skipped_missing_mss"] += 1
            continue

        if not rec.precheck_passed:
            stats["skipped_precheck_failed"] += 1
            continue

        out_path = output_dir / f"{video_id}.json"
        if out_path.exists() and not overwrite:
            stats["skipped_existing_output"] += 1
            continue

        rows.append({"record": rec, "gt_label": gt, "out_path": out_path})

    stats["to_process"] = len(rows)
    return rows, stats


def load_frames_for_item(pending: dict, mode: str):
    """Return (frames, fps). Does disk I/O."""
    rec: MSSRecord = pending["record"]
    if mode == "full":
        frames, fps = load_full_frames(rec.video_path)
    else:  # mss_kept
        frames, fps = load_mss_kept_frames(rec)
    return frames, fps


def write_sidecar(out_path: Path, payload: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)


def summarize_output_dir(output_dir: Path) -> dict:
    """Walk the sidecars in output_dir and compute aggregate top-1 / top-5 accuracy."""
    n = 0
    n_top1 = 0
    n_top5 = 0
    n_parse_failures = 0
    for p in output_dir.glob("*.json"):
        if p.name == "run_metadata.json":
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception:
            continue
        n += 1
        if not rec.get("parsed"):
            n_parse_failures += 1
        if rec.get("top1_correct"):
            n_top1 += 1
        if rec.get("top5_correct"):
            n_top5 += 1
    return {
        "n": n,
        "n_top1": n_top1,
        "n_top5": n_top5,
        "n_parse_failures": n_parse_failures,
        "top1_accuracy": (n_top1 / n) if n else 0.0,
        "top5_accuracy": (n_top5 / n) if n else 0.0,
    }


def write_run_metadata(output_dir: Path, args: argparse.Namespace, start_ts: str):
    meta = {
        "mode": args.mode,
        "model": args.model,
        "prompt_hash": prompt_hash(),
        "n_labels": len(SSV2_LABELS),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "batch_size": args.batch_size,
        "gpus_per_instance": args.gpus_per_instance,
        "visible_gpus": args.visible_gpus,
        "dataset": str(args.dataset),
        "mss_labels_dir": str(args.mss_labels_dir),
        "host": socket.gethostname(),
        "start_ts": start_ts,
        "cli_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    write_sidecar(output_dir / "run_metadata.json", meta)


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
        gpu_ids = [int(x) for x in args.visible_gpus.split(",")]
    else:
        gpu_ids = None

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = args.output_root / f"{args.mode}_{ts}"
        logger.info(f"--output-dir not provided; auto-generated: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_file_logging(args.log_dir, tag=f"{args.mode}")
    logger.info(f"Logging to {log_path}")
    start_ts = datetime.now().isoformat()
    logger.info(f"classify_ssv2 starting mode={args.mode} dataset={args.dataset} "
                f"output={args.output_dir}")

    # Load dataset
    df = pd.read_csv(args.dataset)
    if args.limit:
        df = df.head(args.limit)

    # Resolve pending rows (applies precheck filter + resume)
    pending, stats = build_pending_rows(
        df, args.mss_labels_dir, args.output_dir, args.overwrite
    )
    logger.info(f"Dataset stats: {stats}")
    write_run_metadata(args.output_dir, args, start_ts)

    if not pending:
        logger.info("Nothing to process.")
        return

    # Build classifier (parallel or single-process)
    if args.parallel:
        if gpu_ids is None:
            import torch
            gpu_ids = list(range(torch.cuda.device_count()))
        classifier = ParallelClassifier(
            model_key=args.model,
            gpu_ids=gpu_ids,
            gpus_per_instance=args.gpus_per_instance,
            cache_dir=args.cache_dir,
            max_new_tokens=args.max_new_tokens,
            worker_batch_size=args.batch_size,
        )
        dispatch = lambda items: classifier.classify(items)
        shutdown = classifier.shutdown
        def peak_mem(): return classifier.last_peak_mem_gb
    else:
        num_gpus = len(gpu_ids) if gpu_ids else 1
        clf = create_classifier(
            model_key=args.model,
            num_gpus=num_gpus,
            cache_dir=args.cache_dir,
            max_new_tokens=args.max_new_tokens,
        )
        clf.max_retries = args.max_retries
        dispatch = lambda items: clf.classify_batch(items, batch_size=args.batch_size)
        shutdown = lambda: None
        def peak_mem(): return clf.last_peak_mem_gb

    try:
        pbar = tqdm(total=len(pending), desc=f"Classify [{args.mode}]", unit="vid")
        for chunk_start in range(0, len(pending), args.batch_size):
            chunk = pending[chunk_start:chunk_start + args.batch_size]

            # Load frames for all items in this chunk (disk I/O)
            items = []
            chunk_pending = []
            for p in chunk:
                try:
                    frames, fps = load_frames_for_item(p, args.mode)
                    if not frames:
                        logger.warning(f"{p['record'].video_id}: empty frame list (mode={args.mode}); writing parsed=false")
                        write_sidecar(p["out_path"], {
                            "video_id": p["record"].video_id,
                            "gt_label": p["gt_label"],
                            "top5": [],
                            "top1_correct": False,
                            "top5_correct": False,
                            "parsed": False,
                            "raw_output": "",
                            "mode": args.mode,
                            "elapsed_s": 0.0,
                            "normalization_applied": False,
                            "error": "empty frames",
                        })
                        pbar.update(1)
                        continue
                    items.append({
                        "video_id": p["record"].video_id,
                        "frames": frames,
                        "fps": fps,
                    })
                    chunk_pending.append(p)
                except Exception as e:
                    logger.exception(f"Failed to load frames for {p['record'].video_id}: {e}")
                    write_sidecar(p["out_path"], {
                        "video_id": p["record"].video_id,
                        "gt_label": p["gt_label"],
                        "top5": [],
                        "top1_correct": False,
                        "top5_correct": False,
                        "parsed": False,
                        "raw_output": "",
                        "mode": args.mode,
                        "elapsed_s": 0.0,
                        "normalization_applied": False,
                        "error": f"frame load failed: {e}",
                    })
                    pbar.update(1)

            if not items:
                continue

            t0 = time.perf_counter()
            results = dispatch(items)
            batch_elapsed = time.perf_counter() - t0

            peak = peak_mem()
            if peak:
                peak_str = ", ".join(f"gpu{g}={peak[g]:.1f}GB" for g in sorted(peak))
                logger.info(
                    f"Batch {chunk_start // args.batch_size + 1} "
                    f"peak VRAM: {peak_str} (n={len(items)}, t={batch_elapsed:.1f}s)"
                )

            # Write sidecars.
            result_by_id = {r["video_id"]: r for r in results}
            for p in chunk_pending:
                vid = p["record"].video_id
                r = result_by_id.get(vid)
                if r is None:
                    logger.warning(f"No result returned for {vid}")
                    pbar.update(1)
                    continue
                top5 = r["top5"]
                gt = p["gt_label"]
                payload = {
                    "video_id": vid,
                    "gt_label": gt,
                    "top5": top5,
                    "top1_correct": bool(top5 and top5[0] == gt),
                    "top5_correct": gt in set(top5),
                    "parsed": r["parsed"],
                    "raw_output": r["raw_output"],
                    "mode": args.mode,
                    "elapsed_s": r["elapsed_s"],
                    "normalization_applied": r.get("normalization_applied", False),
                    "retries": r.get("retries", 0),
                }
                write_sidecar(p["out_path"], payload)
                pbar.update(1)

        pbar.close()
    finally:
        shutdown()

    summary = summarize_output_dir(args.output_dir)
    logger.info(
        f"Accuracy over {summary['n']} sidecar(s): "
        f"top-1 = {summary['top1_accuracy']:.4f} "
        f"({summary['n_top1']}/{summary['n']}); "
        f"top-5 = {summary['top5_accuracy']:.4f} "
        f"({summary['n_top5']}/{summary['n']}); "
        f"parse failures = {summary['n_parse_failures']}"
    )
    print("=" * 72)
    print(f"  Mode:             {args.mode}")
    print(f"  Evaluated videos: {summary['n']}")
    print(f"  Parse failures:   {summary['n_parse_failures']}")
    print(f"  top-1 accuracy:   {summary['top1_accuracy']:.4f} "
          f"({summary['n_top1']}/{summary['n']})")
    print(f"  top-5 accuracy:   {summary['top5_accuracy']:.4f} "
          f"({summary['n_top5']}/{summary['n']})")
    print("=" * 72)

    logger.info(f"Done. Output dir: {args.output_dir}  Log: {log_path}")


if __name__ == "__main__":
    main()
