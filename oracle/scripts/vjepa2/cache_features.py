#!/usr/bin/env python
"""Phase A — extract V-JEPA 2 post-pooler features for linear-probe finetuning.

For each row in ``--csv`` (train or val), loads the matching MSS sidecar from
``--mss-run-dir``, builds the cut+concat input via ``build_input`` with the
recognizer-specific frames_per_clip / image_size, then runs:

    encoder_out = model.vjepa2(pixel_values_videos=pv)
    pooled      = model.pooler(encoder_out.last_hidden_state)   # (B, 1024)

and saves ``pooled`` to a per-shard ``.pt`` file. Encoder + pooler stay frozen.

Sharding mirrors ``run_classifier_eval.py``: launch ``--num-shards`` workers in
parallel (one per GPU), each with a distinct ``--shard-id``. Resume is automatic.

Output:
    <output-dir>/shard_<NN>.pt   — {"video_ids": [...], "features": (N,D), "labels": (N,)}
    <output-dir>/cache_metadata_shard_<NN>.json
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

from oracle.scripts.vjepa2 import selectors as sel
from oracle.scripts.vjepa2.adapter import build_input
from oracle.scripts.vjepa2.inference import VJepa2Wrapper

LOG = logging.getLogger("vjepa2.cache_features")

# Known recognizers we support feature-caching for (matches run_classifier_eval RECOGNIZERS).
MODEL_IDS = {
    "vjepa2-ssv2": "facebook/vjepa2-vitl-fpc16-256-ssv2",
    "vjepa2-diving48": "facebook/vjepa2-vitl-fpc32-256-diving48",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recognizer", required=True, choices=sorted(MODEL_IDS.keys()))
    p.add_argument("--csv", required=True, type=str, help="Train or val CSV (same schema as eval CSVs)")
    p.add_argument("--mss-run-dir", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="Cap rows after sharding (smoke runs)")
    p.add_argument(
        "--flush-every", type=int, default=128,
        help="Persist the shard .pt file every N successfully-extracted videos",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def video_id_from_path(p: str) -> str:
    return Path(p).stem


def _load_existing_video_ids(output_dir: Path) -> set[str]:
    seen: set[str] = set()
    for shard_pt in sorted(output_dir.glob("shard_*.pt")):
        try:
            blob = torch.load(str(shard_pt), map_location="cpu", weights_only=False)
            seen.update(blob.get("video_ids", []))
        except Exception as e:
            LOG.warning("Could not read %s for resume (%s); continuing", shard_pt, e)
    return seen


def _save_shard(
    shard_path: Path,
    video_ids: list[str],
    features: list[torch.Tensor],
    labels: list[int],
) -> None:
    if not features:
        return
    blob = {
        "video_ids": list(video_ids),
        "features": torch.stack(features, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    tmp = shard_path.with_suffix(".pt.tmp")
    torch.save(blob, str(tmp))
    tmp.rename(shard_path)


def normalize_label(s: str) -> str:
    return s.replace("[", "").replace("]", "").strip().lower()


def _resolve_gt_id(row: dict, label_norm: dict[str, int]) -> int | None:
    """Prefer numeric class_id, then label, then template (SSv2 train/val_full schema)."""
    cls_id_raw = row.get("class_id")
    if cls_id_raw not in (None, ""):
        try:
            return int(cls_id_raw)
        except (TypeError, ValueError):
            pass
    gt = label_norm.get(normalize_label(row["label"]))
    if gt is None and row.get("template"):
        gt = label_norm.get(normalize_label(row["template"]))
    return gt


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    model_id = MODEL_IDS[args.recognizer]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard_{args.shard_id:02d}.pt"
    mss_run_dir = Path(args.mss_run_dir)

    with open(args.csv) as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for i, r in enumerate(all_rows) if i % max(1, args.num_shards) == args.shard_id]
    if args.limit is not None:
        rows = rows[: args.limit]

    LOG.info(
        "shard %d/%d  %d rows of %d  device=%s  model=%s",
        args.shard_id, args.num_shards, len(rows), len(all_rows), args.device, model_id,
    )

    seen = _load_existing_video_ids(output_dir)
    todo = [r for r in rows if video_id_from_path(r["video_path"]) not in seen]
    LOG.info("Resume: %d already cached, %d to extract", len(rows) - len(todo), len(todo))
    if not todo:
        return 0

    LOG.info("Loading wrapper %s on %s …", model_id, args.device)
    wrapper = VJepa2Wrapper.from_pretrained(model_id, device=args.device)
    model = wrapper.model
    model.eval()
    encoder = model.vjepa2     # frozen
    pooler = model.pooler      # frozen
    id2label = model.config.id2label
    label_norm = {normalize_label(v): int(k) for k, v in id2label.items()}

    # Resume: preload this-shard's prior contents.
    if shard_path.exists():
        prior = torch.load(str(shard_path), map_location="cpu", weights_only=False)
        video_ids = list(prior["video_ids"])
        features = [prior["features"][i].clone() for i in range(prior["features"].shape[0])]
        labels = [int(x) for x in prior["labels"].tolist()]
    else:
        video_ids, features, labels = [], [], []

    from tqdm.auto import tqdm
    pbar = tqdm(total=len(todo), desc=f"cache/sh{args.shard_id}", unit="clip", dynamic_ncols=True)

    n_missing_sidecar = 0
    n_unlabeled = 0
    n_no_kept = 0
    n_build_fail = 0
    n_done_since_flush = 0
    t0 = time.time()

    buf_rows: list[dict] = []
    buf_pv: list[torch.Tensor] = []
    buf_labels: list[int] = []

    def flush_batch():
        nonlocal n_done_since_flush
        if not buf_pv:
            return
        pv = torch.cat(buf_pv, dim=0).to(args.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            enc_out = encoder(pixel_values_videos=pv)
            pooled = pooler(enc_out.last_hidden_state).float().cpu()  # (B, 1024)
        for i, r in enumerate(buf_rows):
            vid = video_id_from_path(r["video_path"])
            video_ids.append(vid)
            features.append(pooled[i].contiguous())
            labels.append(buf_labels[i])
            n_done_since_flush += 1
        buf_rows.clear()
        buf_pv.clear()
        buf_labels.clear()
        if n_done_since_flush >= args.flush_every:
            _save_shard(shard_path, video_ids, features, labels)
            n_done_since_flush = 0

    for r in todo:
        vid = video_id_from_path(r["video_path"])

        gt_id = _resolve_gt_id(r, label_norm)
        if gt_id is None:
            n_unlabeled += 1
            pbar.update(1)
            continue

        sc_path = mss_run_dir / f"{vid}.json"
        try:
            sidecar = json.load(open(sc_path))
        except FileNotFoundError:
            n_missing_sidecar += 1
            pbar.update(1)
            continue

        result = sel.vlm_selected(sidecar)
        if result.selector_failed or not result.kept_segments:
            n_no_kept += 1
            pbar.update(1)
            continue

        try:
            pv = build_input(
                r["video_path"], result.kept_segments, "repack",
                wrapper.frames_per_clip, wrapper.image_size, wrapper.processor,
            )
        except Exception as e:
            LOG.warning("build_input failed for %s (%s)", vid, e)
            n_build_fail += 1
            pbar.update(1)
            continue

        buf_rows.append(r)
        buf_pv.append(pv)
        buf_labels.append(gt_id)
        pbar.update(1)

        if len(buf_pv) >= max(1, args.batch_size):
            flush_batch()

    flush_batch()
    _save_shard(shard_path, video_ids, features, labels)
    pbar.close()

    elapsed = time.time() - t0
    meta = {
        "recognizer": args.recognizer,
        "model_id": model_id,
        "mss_run_dir": str(mss_run_dir),
        "csv": str(args.csv),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_cached_shard": len(video_ids),
        "n_missing_sidecar": n_missing_sidecar,
        "n_unlabeled": n_unlabeled,
        "n_no_kept_segments": n_no_kept,
        "n_build_input_failed": n_build_fail,
        "hidden_size": int(features[0].shape[-1]) if features else None,
        "num_labels": len(id2label),
        "wallclock_s": elapsed,
    }
    (output_dir / f"cache_metadata_shard_{args.shard_id:02d}.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )

    print(
        f"[shard {args.shard_id}] cached={len(video_ids)} "
        f"missing_sidecar={n_missing_sidecar} unlabeled={n_unlabeled} "
        f"no_kept={n_no_kept} build_fail={n_build_fail} elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
