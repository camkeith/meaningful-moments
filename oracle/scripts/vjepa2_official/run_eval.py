#!/usr/bin/env python
"""Run the V-JEPA2 official-checkpoint pipeline against the seven-condition
selectors and write per-(video, condition) sidecars matching the existing
`run_classifier_eval.py` schema.

This runner replaces the simplified HF `VJEPA2ForVideoClassification` head
with the paper's full encoder + 4-block AttentiveClassifier (loaded from
the official checkpoints under `external/vjepa2_checkpoints/`). It produces
sidecars that are drop-in compatible with `shuffle_damage_analysis.py` and
the existing seven-condition reporters.

Recognizer keys recognized:
    vjepa2-official-diving48  → 4-block multilevel probe, 32×4×3 sampling for full
    vjepa2-official-ssv2      → 4-block single-layer probe, 16×2×3 sampling for full

For selection conditions (vlm-selected, uniform, motion, random, etc.):
    Single-clip from kept_segments (1 segment × 1 view), 32 frames at the
    config's frame_step. Optional --shuffle-segments permutes segment order
    before frame index aggregation (replicates the shuffle test).

Usage example:
    python -m oracle.scripts.vjepa2_official.run_eval \
        --recognizer vjepa2-official-diving48 \
        --eval-csv data/csvs/diving48/val.csv \
        --output-dir pseudo_labels/classifier_eval/diving48_vjepa2-official_<TS>/ \
        --device cuda:6 \
        --shard-id 0 --num-shards 2 \
        --condition vlm-selected \
        --mss-run-dir pseudo_labels/mss/qwen3-vl-32b_diving48_val_20260506_002245
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch

LOG = logging.getLogger("vjepa2_official_run_eval")

MM_ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[3]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recognizer", required=True,
                   choices=["vjepa2-official-diving48", "vjepa2-official-ssv2"])
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--condition", default="full",
                   help="full or any seven-condition name (vlm-selected, uniform, "
                        "motion, random, lowest-evidence, uniform-equal-segs, "
                        "vlm-weighted, vlm-strict-0.9, uniform-vs-strict-0.9) "
                        "or a fast-forward cell 'vlm-fastforward-a<NN>' with "
                        "NN ∈ {5, 10, 25, 50, 75} (alpha = NN/100, density floor "
                        "for unimportant segments while keeping the full timeline) "
                        "or vlm-aggregate (full 4×3 sampling + importance-weighted "
                        "softmax aggregation across clips) or vlm-anchored (4 clip "
                        "starts placed at MSS-importance quantiles × 3 views).")
    p.add_argument("--num-segments", type=int, default=None,
                   help="Override num_segments for predict_kept (kept-segment "
                        "conditions). Default = 1 (single-clip). Set equal to the "
                        "recognizer's paper protocol value (4 for d48, 2 for ssv2) "
                        "to match predict_full's compute regime. Ignored for "
                        "condition=full / vlm-aggregate / vlm-anchored which use "
                        "their own paper-protocol sampling.")
    p.add_argument("--num-views", type=int, default=None,
                   help="Override num_views_per_segment for predict_kept. Default 1. "
                        "Set to 3 for paper-protocol 3-crop spatial views.")
    p.add_argument("--frame-step", type=int, default=None,
                   help="Override the wrapper's frame_step (default = paper protocol "
                        "from config: 2 for D48, 4 for SSv2). Smaller value tightens "
                        "the per-clip temporal span (clip-span = frames_per_clip × frame_step). "
                        "Useful when paper-protocol span > video length (e.g. short SSv2 clips).")
    p.add_argument("--mss-run-dir", default=None,
                   help="Required for non-full conditions")
    p.add_argument("--motion-cache-dir", default=None,
                   help="Required for condition=motion")
    p.add_argument(
        "--head-checkpoint", default=None,
        help=(
            "Optional finetuned head_best.pt. When set the wrapper keeps only probe[0] "
            "and replaces its final Linear with the trained one — single-probe inference."
        ),
    )
    p.add_argument(
        "--attentive-probe-checkpoint", default=None,
        help=(
            "Optional fully-retrained attentive_probe_best.pt (4-block AttentiveClassifier "
            "trained on MSS-cut tokens). Replaces the entire LR-sweep ensemble with this one "
            "trained classifier. Mutually exclusive with --head-checkpoint."
        ),
    )
    p.add_argument("--run-seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shuffle-segments", action="store_true",
                   help="Permute kept_segment order with deterministic seed before "
                        "decoding/feeding to recognizer. Sidecar suffix becomes "
                        "<cond>--shuffled.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def shard_rows(rows, shard_id, num_shards):
    if num_shards <= 1:
        return list(rows)
    return [r for i, r in enumerate(rows) if i % num_shards == shard_id]


def video_id_from_path(p: str) -> str:
    return Path(p).stem


def sidecar_path(out: Path, vid: str, cond: str) -> Path:
    return out / f"{vid}__{cond}.json"


def shuffle_segments(
    kept_segments: list,
    kept_segment_weights: list | None,
    run_seed: int,
    video_id: str,
    condition: str,
) -> tuple[list, list | None, str]:
    if not kept_segments:
        return kept_segments, kept_segment_weights, ""
    h = hashlib.sha256(f"{run_seed}|{video_id}|{condition}|shuffle".encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = random.Random(seed)
    paired = list(zip(kept_segments, kept_segment_weights or [None] * len(kept_segments)))
    rng.shuffle(paired)
    new_segs = [p[0] for p in paired]
    new_w = [p[1] for p in paired] if kept_segment_weights else None
    return new_segs, new_w, f"{seed:016x}"


def load_csv_with_header(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if args.condition not in ("full", "full-single") and not args.mss_run_dir:
        LOG.error("--mss-run-dir is required for condition=%s", args.condition)
        return 2

    # Late import (uses the official repo via PYTHONPATH inside the wrapper)
    sys.path.insert(0, str(MM_ROOT))
    from oracle.scripts.vjepa2_official.wrapper import VJepa2OfficialWrapper

    wrapper = VJepa2OfficialWrapper(
        args.recognizer,
        device=args.device,
        head_checkpoint=args.head_checkpoint,
        attentive_probe_checkpoint=args.attentive_probe_checkpoint,
        frame_step=args.frame_step,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_out = args.condition + ("--shuffled" if args.shuffle_segments else "")
    results_jsonl = out_dir / f"results_shard_{args.shard_id:02d}.jsonl"
    paired_jsonl = out_dir / "paired_records.jsonl"

    # Load CSV + shard
    rows = load_csv_with_header(args.eval_csv)
    rows = shard_rows(rows, args.shard_id, args.num_shards)
    if args.limit is not None:
        rows = rows[: args.limit]
    LOG.info("recognizer=%s condition=%s shard %d/%d  %d videos",
             args.recognizer, cond_out, args.shard_id, args.num_shards, len(rows))

    # Resume: skip done sidecars
    todo = [r for r in rows if not sidecar_path(out_dir, video_id_from_path(r["video_path"]), cond_out).exists()]
    skipped = len(rows) - len(todo)
    LOG.info("Resume: %d already done, %d to process", skipped, len(todo))
    if not todo:
        LOG.info("Nothing to do.")
        return 0

    # MSS sidecar selector dispatch — for non-full conditions we need kept_segments
    if args.condition not in ("full", "full-single"):
        sys.path.insert(0, str(MM_ROOT))
        from oracle.scripts.vjepa2 import selectors as sel
        # Pre-load MSS sidecars
        mss_dir = Path(args.mss_run_dir)
        sidecars: dict[str, dict] = {}
        for r in todo:
            vid = video_id_from_path(r["video_path"])
            p = mss_dir / f"{vid}.json"
            if p.exists():
                try:
                    sidecars[vid] = json.load(open(p))
                except Exception:
                    pass

    # Class-id resolver: prefer numeric class_id from CSV; otherwise look up the
    # text label against a known labels-map (SSv2 ships data/SSv2/labels.json).
    label_text_to_id: dict[str, int] = {}
    if args.recognizer == "vjepa2-official-ssv2":
        ssv2_labels_path = MM_ROOT / "data" / "SSv2" / "labels.json"
        if ssv2_labels_path.exists():
            raw = json.loads(ssv2_labels_path.read_text())
            for k, v in raw.items():
                label_text_to_id[k.strip().lower()] = int(v)

    def gt_id(r: dict) -> int | None:
        cid = r.get("class_id")
        if cid not in (None, ""):
            try:
                return int(cid)
            except (TypeError, ValueError):
                pass
        lbl = (r.get("label") or "").strip()
        if lbl:
            cleaned = lbl.replace("[", "").replace("]", "").strip().lower()
            if cleaned in label_text_to_id:
                return label_text_to_id[cleaned]
            if lbl.lower() in label_text_to_id:
                return label_text_to_id[lbl.lower()]
        return None

    counts = [0, 0, 0, 0]  # n_total, n_top1, n_top5, n_selfail
    n_unmapped = 0
    t0 = time.time()
    from tqdm.auto import tqdm
    pbar = tqdm(todo, desc=f"{args.recognizer}/{cond_out}/sh{args.shard_id}",
                unit="clip", dynamic_ncols=True)

    for r in pbar:
        vid = video_id_from_path(r["video_path"])
        gt = gt_id(r)
        if gt is None:
            n_unmapped += 1

        # Select kept_segments
        if args.condition in ("full", "full-single"):
            kept_segments = None
            kept_segment_weights = None
            sel_meta = {
                "kept_segments": None,
                "kept_segment_indices_in_sidecar": None,
                "kept_segment_weights": None,
                "kept_total_duration_s": None,
                "kept_segment_count": None,
                "selector_failed": False,
                "selector_failure_reason": None,
                "rng_seed_for_video_condition": None,
                "shuffle_segments": False,
                "shuffle_seed_hex": "",
            }
        else:
            sc = sidecars.get(vid)
            if sc is None:
                # Missing MSS — write selector_failed
                record = _make_record(r, vid, cond_out, args.recognizer,
                                      wrapper.cfg["probe_ckpt"].name, gt,
                                      None, None, args.run_seed,
                                      head_checkpoint=args.head_checkpoint,
                                      attentive_probe_checkpoint=args.attentive_probe_checkpoint,
                                      selector_failed=True,
                                      selector_failure_reason="MSS sidecar missing")
                _write_sidecar(record, out_dir, results_jsonl, paired_jsonl)
                counts[0] += 1; counts[3] += 1
                continue
            result = sel.select(args.condition, sc,
                                run_seed=args.run_seed, video_id=vid,
                                motion_cache_dir=args.motion_cache_dir)
            if result.selector_failed:
                # Skip motion cache miss (transient); write fail otherwise
                if (result.selector_failure_reason
                        and "motion cache miss" in result.selector_failure_reason):
                    LOG.info("skip %s — motion cache pending", vid)
                    continue
                record = _make_record(r, vid, cond_out, args.recognizer,
                                      wrapper.cfg["probe_ckpt"].name, gt,
                                      None, None, args.run_seed,
                                      head_checkpoint=args.head_checkpoint,
                                      attentive_probe_checkpoint=args.attentive_probe_checkpoint,
                                      selector_failed=True,
                                      selector_failure_reason=result.selector_failure_reason)
                _write_sidecar(record, out_dir, results_jsonl, paired_jsonl)
                counts[0] += 1; counts[3] += 1
                continue
            kept_segments = list(result.kept_segments)
            kept_segment_weights = list(result.kept_segment_weights) if result.kept_segment_weights else None
            shuffle_seed_hex = ""
            if args.shuffle_segments:
                kept_segments, kept_segment_weights, shuffle_seed_hex = shuffle_segments(
                    kept_segments, kept_segment_weights,
                    args.run_seed, vid, args.condition,
                )
            sel_meta = {
                "kept_segments": [list(s) for s in kept_segments],
                "kept_segment_indices_in_sidecar": result.kept_indices,
                "kept_segment_weights": list(kept_segment_weights) if kept_segment_weights else None,
                "kept_total_duration_s": result.kept_total_duration_s,
                "kept_segment_count": result.kept_segment_count,
                "selector_failed": False,
                "selector_failure_reason": None,
                "rng_seed_for_video_condition": result.rng_seed_hex,
                "shuffle_segments": bool(args.shuffle_segments),
                "shuffle_seed_hex": shuffle_seed_hex,
            }

        # Forward
        try:
            t_fwd = time.time()
            if args.condition == "full":
                pred = wrapper.predict_full(r["video_path"])
            elif args.condition == "full-single":
                # Whole-video uniform sampling, single clip, single view.
                # The protocol-symmetric counterpart to vlm-selected @ 1×1.
                pred = wrapper.predict_kept(
                    r["video_path"], None,
                    num_segments=1, num_views_per_segment=1,
                )
            elif args.condition == "vlm-aggregate":
                pred = wrapper.predict_full_with_clip_weighting(r["video_path"], sc)
            elif args.condition == "vlm-anchored":
                pred = wrapper.predict_anchored(r["video_path"], sc)
            elif args.condition == "vlm-anchored-hybrid":
                pred = wrapper.predict_anchored_hybrid(r["video_path"], sc)
            else:
                # Default num_segments / num_views = 1 (single-clip). Use CLI override
                # for paper-protocol kept eval (set to wrapper.num_segments / .num_views_per_segment
                # to match predict_full).
                ns = args.num_segments if args.num_segments is not None else 1
                nv = args.num_views if args.num_views is not None else 1
                pred = wrapper.predict_kept(
                    r["video_path"], kept_segments,
                    kept_segment_weights=kept_segment_weights,
                    num_segments=ns, num_views_per_segment=nv,
                )
            elapsed = time.time() - t_fwd
        except Exception as e:
            LOG.warning("forward failed for %s/%s: %s", vid, cond_out, e)
            continue

        top5_ids = pred["top5_label_ids"]
        top5_probs = pred["top5_probs"]
        top1 = gt is not None and top5_ids[0] == gt
        top5 = gt is not None and gt in top5_ids

        record = _make_record(
            r, vid, cond_out, args.recognizer,
            wrapper.cfg["probe_ckpt"].name, gt,
            top5_ids, top5_probs, args.run_seed,
            head_checkpoint=args.head_checkpoint,
            attentive_probe_checkpoint=args.attentive_probe_checkpoint,
            top1=top1, top5=top5, elapsed_s=elapsed, **sel_meta,
        )
        _write_sidecar(record, out_dir, results_jsonl, paired_jsonl)

        counts[0] += 1
        if top1: counts[1] += 1
        if top5: counts[2] += 1
        if (counts[0] % 10) == 0:
            denom = max(1, counts[0])
            pbar.set_postfix(top1=f"{counts[1]/denom:.3f}",
                             top5=f"{counts[2]/denom:.3f}",
                             failed=counts[3])

    pbar.close()
    n_total, n_top1, n_top5, n_selfail = counts
    n_scored = max(1, n_total - n_selfail)
    summary = {
        "recognizer": args.recognizer,
        "condition": cond_out,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_total": n_total,
        "n_scored": n_scored,
        "n_selector_failed": n_selfail,
        "n_unmapped_gt": n_unmapped,
        "n_top1_correct": n_top1,
        "n_top5_correct": n_top5,
        "top1_accuracy": n_top1 / n_scored,
        "top5_accuracy": n_top5 / n_scored,
        "wallclock_s": time.time() - t0,
    }
    summary_path = out_dir / f"summary_shard_{args.shard_id:02d}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOG.info("=" * 60)
    LOG.info("recognizer=%s  condition=%s  shard %d/%d", args.recognizer, cond_out, args.shard_id, args.num_shards)
    LOG.info("  n_scored: %d  n_selector_failed: %d", n_scored, n_selfail)
    LOG.info("  top-1: %.4f (%d/%d)", n_top1 / n_scored, n_top1, n_scored)
    LOG.info("  top-5: %.4f (%d/%d)", n_top5 / n_scored, n_top5, n_scored)
    LOG.info("  wallclock: %.1f s", summary["wallclock_s"])
    LOG.info("  summary: %s", summary_path)
    return 0


def _make_record(
    r: dict, vid: str, cond_out: str, recognizer: str, model_name: str,
    gt: int | None, top5_ids: list | None, top5_probs: list | None, run_seed: int,
    *, top1: bool = False, top5: bool = False, elapsed_s: float = 0.0,
    selector_failed: bool = False, selector_failure_reason: str | None = None,
    head_checkpoint: str | None = None,
    attentive_probe_checkpoint: str | None = None,
    **sel_meta,
) -> dict:
    base = {
        "video_id": vid,
        "video_path": r["video_path"],
        "ground_truth": r.get("label"),
        "ground_truth_id": gt,
        "condition": cond_out,
        "top5_label_ids": top5_ids,
        "top5_probs": top5_probs,
        "top1_correct": bool(top1),
        "top5_correct": bool(top5),
        "recognizer": recognizer,
        "model": model_name,
        "head_checkpoint": head_checkpoint,
        "attentive_probe_checkpoint": attentive_probe_checkpoint,
        "elapsed_fwd_batch_s": elapsed_s,
        "batch_size": 1,
        "run_seed": run_seed,
        "selector_failed": selector_failed,
        "selector_failure_reason": selector_failure_reason,
    }
    base.update(sel_meta)
    return base


def _write_sidecar(record: dict, out_dir: Path, results_jsonl: Path, paired_jsonl: Path):
    sp = out_dir / f"{record['video_id']}__{record['condition']}.json"
    sp.write_text(json.dumps(record, sort_keys=True))
    with results_jsonl.open("a") as f:
        f.write(json.dumps(record) + "\n")
    with paired_jsonl.open("a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
