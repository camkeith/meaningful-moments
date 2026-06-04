#!/usr/bin/env python
"""Recognizer-agnostic classifier evaluation over an eval CSV — one shard, one GPU.

Dispatches on ``--recognizer``:
    - ``vjepa2-ssv2``         → ``VJepa2Wrapper`` with ``facebook/vjepa2-vitl-fpc16-256-ssv2``
    - ``videomae-k400-large`` → ``VideoMaeWrapper`` with ``MCG-NJU/videomae-large-finetuned-kinetics``
    - ``videomae-k400-huge``  → ``VideoMaeWrapper`` with ``MCG-NJU/videomae-huge-finetuned-kinetics``

Multiple workers can contribute to the same run by sharing ``--output-dir`` with
disjoint ``--shard-id``/``--num-shards``. Each writes:
    <output-dir>/<video_id>__<condition>.json  — one sidecar per (video, condition)
    <output-dir>/results_shard_NN.jsonl
    <output-dir>/summary_shard_NN.json
    <output-dir>/paired_records.jsonl          — append-only, all conditions/all shards

Resume is automatic: a worker skips ``(video, condition)`` pairs with an existing sidecar.

Conditions: ``full, vlm-selected, random, uniform, motion, lowest-evidence, uniform-equal-segs``.
See the development design notes for protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import torch

LOG = logging.getLogger("run_classifier_eval")

RECOGNIZERS = {
    "vjepa2-ssv2": {
        "model": "facebook/vjepa2-vitl-fpc16-256-ssv2",
        "wrapper": "oracle.scripts.vjepa2.inference:VJepa2Wrapper",
    },
    "vjepa2-diving48": {
        "model": "facebook/vjepa2-vitl-fpc32-256-diving48",
        "wrapper": "oracle.scripts.vjepa2.inference:VJepa2Wrapper",
    },
    "videomae-k400-large": {
        "model": "MCG-NJU/videomae-large-finetuned-kinetics",
        "wrapper": "oracle.scripts.videomae.inference:VideoMaeWrapper",
    },
    "videomae-k400-huge": {
        "model": "MCG-NJU/videomae-huge-finetuned-kinetics",
        "wrapper": "oracle.scripts.videomae.inference:VideoMaeWrapper",
    },
}

CONDITIONS = (
    "full",
    "vlm-selected", "vlm-weighted", "vlm-strict-0.9",
    "random", "uniform", "uniform-vs-strict-0.9",
    "motion", "lowest-evidence", "uniform-equal-segs",
)
# I/D-curve conditions are accepted as free-form "id-<ordering>-f<NN>" strings;
# the selector module validates them at dispatch time.


def _import_wrapper(spec: str):
    mod_name, cls_name = spec.split(":")
    import importlib
    return getattr(importlib.import_module(mod_name), cls_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recognizer", required=True, choices=sorted(RECOGNIZERS.keys()))
    p.add_argument("--eval-csv", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--protocol", default="repack", choices=["repack", "mask", "sparse"])
    p.add_argument(
        "--condition", default="full",
        help=(
            "Selection condition. Either one of the legacy seven-condition names "
            f"({', '.join(CONDITIONS)}) or an I/D-curve cell of the form "
            "'id-<ordering>-f<NN>' where ordering is one of vlm/anti-vlm/random/temporal/motion "
            "and NN is one of 10..90 in steps of 10."
        ),
    )
    p.add_argument("--mss-run-dir", default=None, help="Required for non-full conditions")
    p.add_argument("--motion-cache-dir", default=None, help="Required for condition=motion or id-motion-*")
    p.add_argument(
        "--head-checkpoint", default=None,
        help=(
            "Optional path to a finetuned classifier-head checkpoint. Currently supported "
            "for VideoMAE recognizers only; loaded via VideoMaeWrapper.from_pretrained("
            "head_checkpoint=...). Recorded in the sidecar so downstream comparisons can "
            "distinguish off-the-shelf vs finetuned-head runs."
        ),
    )
    p.add_argument("--run-seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--shuffle-segments", action="store_true",
        help=(
            "After the selector returns kept_segments (sorted by start_s), randomly "
            "permute their order before passing to the recognizer. The shuffle is "
            "per-video deterministic, seeded by sha256(run_seed|video_id|'shuffle'). "
            "Sidecars are written under condition='<base>--shuffled' so they don't "
            "collide with the unshuffled run. Not applicable to condition=full."
        ),
    )
    return p.parse_args(argv)


def shard_rows(rows, shard_id, num_shards):
    if num_shards <= 1:
        return list(rows)
    return [r for i, r in enumerate(rows) if i % num_shards == shard_id]


def normalize_label(s: str) -> str:
    return s.replace("[", "").replace("]", "").strip().lower()


def video_id_from_path(p: str) -> str:
    return Path(p).stem


def sidecar_path(output_dir: Path, video_id: str, condition: str) -> Path:
    return output_dir / f"{video_id}__{condition}.json"


def _shuffle_seed(run_seed: int, video_id: str, condition: str) -> int:
    """Deterministic per-(video, condition) shuffle seed.

    Per the spec: sha256(run_seed|video_id|condition|'shuffle')[:8]. Including
    the condition gives independent shuffle streams for vlm-selected vs uniform
    on the same video.
    """
    import hashlib
    h = hashlib.sha256(f"{run_seed}|{video_id}|{condition}|shuffle".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _shuffle_segments(
    kept_segments: list,
    kept_segment_weights: list | None,
    run_seed: int,
    video_id: str,
    condition: str,
) -> tuple[list, list | None, str]:
    """Shuffle kept_segments + parallel weights with a deterministic seed.

    Returns (shuffled_segments, shuffled_weights, seed_hex). The seed_hex is
    written to the sidecar's shuffle_seed_hex for reproducibility.
    """
    import random
    if not kept_segments:
        return kept_segments, kept_segment_weights, ""
    seed = _shuffle_seed(run_seed, video_id, condition)
    rng = random.Random(seed)
    # Build paired list so weights stay aligned with their segments under shuffle
    paired = list(zip(kept_segments, kept_segment_weights or [None] * len(kept_segments)))
    rng.shuffle(paired)
    new_segs = [p[0] for p in paired]
    new_w = [p[1] for p in paired] if kept_segment_weights else None
    return new_segs, new_w, f"{seed:016x}"


def migrate_legacy_full_sidecars(output_dir: Path) -> int:
    """Rename old `<video_id>.json` ceiling sidecars to `<video_id>__full.json`. Idempotent."""
    n = 0
    for p in output_dir.glob("*.json"):
        name = p.name
        if name.startswith("results_shard_") or name.startswith("summary_shard_") or name == "paired_records.jsonl":
            continue
        if "__" in name:
            continue
        vid = p.stem
        new_p = output_dir / f"{vid}__full.json"
        if new_p.exists():
            continue
        p.rename(new_p)
        n += 1
    return n


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    spec = RECOGNIZERS[args.recognizer]
    model_id = spec["model"]
    wrapper_cls = _import_wrapper(spec["wrapper"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / f"results_shard_{args.shard_id:02d}.jsonl"
    paired_jsonl = output_dir / "paired_records.jsonl"

    # One-time migration of legacy ceiling sidecars (only relevant when condition=full).
    n_migrated = migrate_legacy_full_sidecars(output_dir)
    if n_migrated:
        LOG.info("Migrated %d legacy <video_id>.json sidecars → <video_id>__full.json", n_migrated)

    # Validate --condition: must be a known seven-cond name OR a parsable I/D-curve cell
    # OR a parsable fast-forward cell (vlm-fastforward-a<NN> / lowest-evidence-fastforward-a<NN>)
    # OR a parsable score-threshold cell (vlm-score-threshold-t<NN>).
    from oracle.scripts.vjepa2 import selectors as _sel
    parsed_id = _sel.parse_id_condition(args.condition)
    parsed_ff = _sel.parse_fastforward_condition(args.condition)
    parsed_le_ff = _sel.parse_lowest_evidence_fastforward_condition(args.condition)
    parsed_st = _sel.parse_score_threshold_condition(args.condition)
    if (args.condition not in CONDITIONS
            and parsed_id is None
            and parsed_ff is None
            and parsed_le_ff is None
            and parsed_st is None):
        LOG.error(
            "Unknown --condition %r. Known: %s, or 'id-<ordering>-f<NN>' "
            "(ordering ∈ %s; NN ∈ %s), or 'vlm-fastforward-a<NN>' / "
            "'lowest-evidence-fastforward-a<NN>' (NN ∈ %s), or "
            "'vlm-score-threshold-t<NN>' (NN ∈ %s).",
            args.condition, list(CONDITIONS), list(_sel.ID_ORDERINGS),
            list(_sel.ID_FRACTION_PERCENTS), list(_sel.FASTFORWARD_ALPHA_PERCENTS),
            list(_sel.SCORE_THRESHOLD_PERCENTS),
        )
        return 2

    if args.condition != "full" and not args.mss_run_dir:
        LOG.error("--mss-run-dir is required for condition=%s", args.condition)
        return 2

    if args.shuffle_segments and args.condition == "full":
        LOG.error("--shuffle-segments is incompatible with --condition full (no segments to shuffle)")
        return 2

    # Sidecar tag: when shuffle is on, append "--shuffled" so the output set is
    # disjoint from the unshuffled run and can be paired against it.
    condition_out = args.condition + ("--shuffled" if args.shuffle_segments else "")
    if (args.condition == "motion" or (parsed_id and parsed_id[0] == "motion")) and not args.motion_cache_dir:
        LOG.error("--motion-cache-dir is required for condition=%s", args.condition)
        return 2

    mss_run_dir = Path(args.mss_run_dir) if args.mss_run_dir else None

    with open(args.eval_csv) as f:
        all_rows = list(csv.DictReader(f))
    rows = shard_rows(all_rows, args.shard_id, args.num_shards)
    if args.limit is not None:
        rows = rows[: args.limit]

    LOG.info(
        "recognizer=%s condition=%s shard %d/%d  %d videos (of %d)  device=%s seed=%d  shuffle=%s",
        args.recognizer, condition_out, args.shard_id, args.num_shards,
        len(rows), len(all_rows), args.device, args.run_seed, args.shuffle_segments,
    )

    todo = [r for r in rows if not sidecar_path(output_dir, video_id_from_path(r["video_path"]), condition_out).exists()]
    skipped_existing = len(rows) - len(todo)
    LOG.info("Resume: %d already done for condition=%s, %d to process", skipped_existing, condition_out, len(todo))
    if not todo:
        LOG.info("Nothing to do.")
        return 0

    from tqdm.auto import tqdm
    from oracle.scripts.vjepa2.adapter import build_input
    from oracle.scripts.vjepa2 import selectors as sel

    # Pre-load MSS sidecars for non-full conditions
    sidecars: dict[str, dict] = {}
    if args.condition != "full":
        LOG.info("Loading MSS sidecars from %s …", mss_run_dir)
        t0 = time.time()
        for r in todo:
            vid = video_id_from_path(r["video_path"])
            p = mss_run_dir / f"{vid}.json"
            try:
                sidecars[vid] = json.load(open(p))
            except FileNotFoundError:
                LOG.warning("MSS sidecar missing for %s — will skip with selector_failed", vid)
        LOG.info("Loaded %d sidecars in %.1fs", len(sidecars), time.time() - t0)

    torch.cuda.init()
    device_idx = int(args.device.split(":")[-1]) if args.device.startswith("cuda") else None
    if device_idx is not None:
        torch.cuda.reset_peak_memory_stats(device_idx)

    LOG.info("Loading wrapper: %s on %s …", model_id, args.device)
    t_load_0 = time.time()
    if args.head_checkpoint is not None:
        if not (args.recognizer.startswith("videomae") or args.recognizer.startswith("vjepa2")):
            LOG.error("--head-checkpoint is only supported for videomae-* and vjepa2-* recognizers (got %s)", args.recognizer)
            return 2
        wrapper = wrapper_cls.from_pretrained(model_id, device=args.device, head_checkpoint=args.head_checkpoint)
    else:
        wrapper = wrapper_cls.from_pretrained(model_id, device=args.device)
    t_load = time.time() - t_load_0
    LOG.info("Wrapper loaded in %.1fs (frames_per_clip=%d, image_size=%d)", t_load, wrapper.frames_per_clip, wrapper.image_size)

    id2label = wrapper.model.config.id2label
    label_norm = {normalize_label(v): k for k, v in id2label.items()}

    counts = [0, 0, 0, 0]   # total, top1_correct, top5_correct, selector_failed
    unmapped = [0]
    fwd_times: list[float] = []
    BATCH = max(1, args.batch_size)
    buffer: list[tuple[dict, torch.Tensor, dict]] = []  # (row, pixel_values, selection_meta)

    pbar = tqdm(total=len(todo), desc=f"{args.recognizer}/{condition_out}/sh{args.shard_id}", unit="clip", dynamic_ncols=True)
    t_run_0 = time.time()

    def write_sidecar(record: dict):
        vid = record["video_id"]
        cond = record["condition"]
        sp = sidecar_path(output_dir, vid, cond)
        sp.write_text(json.dumps(record, sort_keys=True))
        with results_jsonl.open("a") as fjl:
            fjl.write(json.dumps(record) + "\n")
        with paired_jsonl.open("a") as fjl:
            fjl.write(json.dumps(record) + "\n")

    def flush(buf):
        if not buf:
            return
        rows_b = [b[0] for b in buf]
        meta_b = [b[2] for b in buf]
        pv = torch.cat([b[1] for b in buf], dim=0).to(args.device, dtype=torch.bfloat16)
        if device_idx is not None:
            torch.cuda.synchronize(args.device)
        t0 = time.time()
        with torch.inference_mode():
            logits = wrapper.forward_batch(pv)
        if device_idx is not None:
            torch.cuda.synchronize(args.device)
        elapsed = time.time() - t0
        fwd_times.append(elapsed)

        logits_cpu = logits.float().cpu()
        for i, (r, sel_meta) in enumerate(zip(rows_b, meta_b)):
            row_logits = logits_cpu[i]
            probs = torch.softmax(row_logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=5)
            top_ids_l = [int(x) for x in top_ids.tolist()]
            top_probs_l = [float(x) for x in top_probs.tolist()]
            top_labels = [id2label[i] for i in top_ids_l]

            # Prefer numeric class_id when the CSV provides it (e.g., Diving-48
            # where the probe's id2label format diverges from the CSV's `label` text).
            gt_id = None
            cls_id_raw = r.get("class_id")
            if cls_id_raw not in (None, ""):
                try:
                    gt_id = int(cls_id_raw)
                except (TypeError, ValueError):
                    gt_id = None
            if gt_id is None:
                gt_norm = normalize_label(r["label"])
                gt_id = label_norm.get(gt_norm)
            if gt_id is None:
                unmapped[0] += 1
            top1 = gt_id is not None and top_ids_l[0] == gt_id
            top5 = gt_id is not None and gt_id in top_ids_l

            record = {
                "video_id": video_id_from_path(r["video_path"]),
                "video_path": r["video_path"],
                "ground_truth": r["label"],
                "ground_truth_id": gt_id,
                "condition": condition_out,
                **sel_meta,
                "top5_label_ids": top_ids_l,
                "top5_labels": top_labels,
                "top5_probs": top_probs_l,
                "top1_correct": bool(top1),
                "top5_correct": bool(top5),
                "recognizer": args.recognizer,
                "model": model_id,
                "head_checkpoint": args.head_checkpoint,
                "protocol_requested": args.protocol,
                "protocol_resolved": "repack" if args.protocol == "sparse" else args.protocol,
                "shard_id": args.shard_id,
                "elapsed_fwd_batch_s": elapsed,
                "batch_size": len(buf),
                "run_seed": args.run_seed,
            }
            write_sidecar(record)
            counts[0] += 1
            if top1:
                counts[1] += 1
            if top5:
                counts[2] += 1
        pbar.update(len(buf))
        denom = max(1, counts[0])
        pbar.set_postfix(top1=f"{counts[1]/denom:.3f}", top5=f"{counts[2]/denom:.3f}", failed=counts[3])
        buf.clear()

    for r in todo:
        vid = video_id_from_path(r["video_path"])

        # Selector dispatch
        if args.condition == "full":
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
            }
        else:
            sc = sidecars.get(vid)
            if sc is None:
                # Missing MSS sidecar → record selector_failed
                record = {
                    "video_id": vid,
                    "video_path": r["video_path"],
                    "ground_truth": r["label"],
                    "ground_truth_id": label_norm.get(normalize_label(r["label"])),
                    "condition": condition_out,
                    "kept_segments": None,
                    "kept_segment_indices_in_sidecar": None,
                    "kept_total_duration_s": None,
                    "kept_segment_count": None,
                    "selector_failed": True,
                    "selector_failure_reason": "MSS sidecar missing",
                    "rng_seed_for_video_condition": None,
                    "top5_label_ids": None,
                    "top5_labels": None,
                    "top5_probs": None,
                    "top1_correct": False,
                    "top5_correct": False,
                    "recognizer": args.recognizer,
                    "model": model_id,
                    "head_checkpoint": args.head_checkpoint,
                    "protocol_requested": args.protocol,
                    "protocol_resolved": "repack" if args.protocol == "sparse" else args.protocol,
                    "shard_id": args.shard_id,
                    "elapsed_fwd_batch_s": 0.0,
                    "batch_size": 0,
                    "run_seed": args.run_seed,
                }
                write_sidecar(record)
                counts[0] += 1
                counts[3] += 1
                pbar.update(1)
                continue
            result = sel.select(
                args.condition, sc,
                run_seed=args.run_seed, video_id=vid,
                motion_cache_dir=args.motion_cache_dir,
            )
            # Special case: motion cache miss is transient (cache may still be populating)
            # — skip without writing a permanent sidecar so a future run can retry.
            if (result.selector_failed and result.selector_failure_reason
                    and "motion cache miss" in result.selector_failure_reason):
                LOG.info("skipping %s — motion cache not yet populated; will retry next run", vid)
                continue
            if result.selector_failed:
                record = {
                    "video_id": vid,
                    "video_path": r["video_path"],
                    "ground_truth": r["label"],
                    "ground_truth_id": label_norm.get(normalize_label(r["label"])),
                    "condition": condition_out,
                    "kept_segments": [],
                    "kept_segment_indices_in_sidecar": [],
                    "kept_total_duration_s": 0.0,
                    "kept_segment_count": 0,
                    "selector_failed": True,
                    "selector_failure_reason": result.selector_failure_reason,
                    "rng_seed_for_video_condition": result.rng_seed_hex,
                    "top5_label_ids": None,
                    "top5_labels": None,
                    "top5_probs": None,
                    "top1_correct": False,
                    "top5_correct": False,
                    "recognizer": args.recognizer,
                    "model": model_id,
                    "head_checkpoint": args.head_checkpoint,
                    "protocol_requested": args.protocol,
                    "protocol_resolved": "repack" if args.protocol == "sparse" else args.protocol,
                    "shard_id": args.shard_id,
                    "elapsed_fwd_batch_s": 0.0,
                    "batch_size": 0,
                    "run_seed": args.run_seed,
                }
                write_sidecar(record)
                counts[0] += 1
                counts[3] += 1
                pbar.update(1)
                continue
            kept_segments = result.kept_segments
            kept_segment_weights = result.kept_segment_weights
            shuffle_seed_hex = ""
            if args.shuffle_segments:
                kept_segments, kept_segment_weights, shuffle_seed_hex = _shuffle_segments(
                    list(kept_segments),
                    list(kept_segment_weights) if kept_segment_weights else None,
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

        try:
            pv = build_input(
                r["video_path"], kept_segments, args.protocol,
                wrapper.frames_per_clip, wrapper.image_size, wrapper.processor,
                kept_segment_weights=kept_segment_weights if args.condition != "full" else None,
                respect_segment_order=bool(args.shuffle_segments),
            )
        except Exception as e:
            LOG.warning("build_input failed for %s/%s: %s", vid, args.condition, e)
            continue
        buffer.append((r, pv, sel_meta))
        if len(buffer) >= BATCH:
            flush(buffer)
    flush(buffer)
    pbar.close()

    n_total, n_top1, n_top5, n_selfail = counts
    n_unmapped_gt = unmapped[0]
    n_scored = max(1, n_total - n_selfail)
    t_run = time.time() - t_run_0
    fwd_total = sum(fwd_times)
    forwards = len(fwd_times)
    clips_per_s = n_total / t_run if t_run > 0 else 0.0
    fwd_clips_per_s = (n_total - n_selfail) / fwd_total if fwd_total > 0 else 0.0
    peak_gb = (
        torch.cuda.max_memory_allocated(device_idx) / 1024**3
        if device_idx is not None else 0.0
    )

    summary = {
        "recognizer": args.recognizer,
        "model": model_id,
        "head_checkpoint": args.head_checkpoint,
        "condition": args.condition,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "device": args.device,
        "protocol": args.protocol,
        "batch_size": BATCH,
        "run_seed": args.run_seed,
        "n_total": n_total,
        "n_scored": n_total - n_selfail,
        "n_selector_failed": n_selfail,
        "n_top1_correct": n_top1,
        "n_top5_correct": n_top5,
        "top1_acc_over_scored": n_top1 / n_scored,
        "top5_acc_over_scored": n_top5 / n_scored,
        "n_unmapped_gt": n_unmapped_gt,
        "n_skipped_existing": skipped_existing,
        "wallclock_s": t_run,
        "fwd_only_s": fwd_total,
        "forwards": forwards,
        "throughput_clips_per_s_wallclock": clips_per_s,
        "throughput_clips_per_s_fwd_only": fwd_clips_per_s,
        "wrapper_load_s": t_load,
        "peak_vram_gb": peak_gb,
    }
    (output_dir / f"summary_shard_{args.shard_id:02d}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print()
    print("=" * 80)
    print(f"recognizer={args.recognizer}  model={model_id}")
    print(f"  condition={args.condition}  shard {args.shard_id}/{args.num_shards}  device={args.device}  protocol={args.protocol}  batch={BATCH}  seed={args.run_seed}")
    print(f"  videos:           {n_total}     (skipped-existing: {skipped_existing}, unmapped-gt: {n_unmapped_gt})")
    print(f"  scored:           {n_total - n_selfail}      selector-failed: {n_selfail}")
    print(f"  top-1 (scored):   {summary['top1_acc_over_scored']:.4f}  ({n_top1}/{n_total - n_selfail})")
    print(f"  top-5 (scored):   {summary['top5_acc_over_scored']:.4f}  ({n_top5}/{n_total - n_selfail})")
    print(f"  wallclock:        {t_run:.1f}s     ({clips_per_s:.2f} clips/s)")
    print(f"  fwd-only:         {fwd_total:.1f}s     ({fwd_clips_per_s:.2f} clips/s)  forwards={forwards}")
    print(f"  peak VRAM:        {peak_gb:.2f} GB")
    print(f"  summary:          {output_dir}/summary_shard_{args.shard_id:02d}.json")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
