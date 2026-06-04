#!/usr/bin/env python
"""Phase A — extract paper-ckpt pre-final-Linear features for MSS-cut probe finetuning.

For each row in ``--csv`` (train or val), loads the matching MSS sidecar from
``--mss-run-dir``, runs the V-JEPA 2 paper-checkpoint encoder + ``probe[0].pooler``
on the MSS-kept frames (single-segment, single-view — same path as predict_kept),
and caches the ``(embed_dim,)`` feature to disk.

Output schema matches ``oracle/scripts/vjepa2/cache_features.py`` so the existing
``oracle/scripts/train_head.py`` trainer can consume it with ``--head-arch vjepa2``
(same LinearHead module — the head feature space is still 1024-d).

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
from oracle.scripts.vjepa2_official.wrapper import PROBE_CONFIGS, VJepa2OfficialWrapper

LOG = logging.getLogger("vjepa2_official.cache_features")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recognizer", required=True, choices=sorted(PROBE_CONFIGS.keys()))
    p.add_argument("--csv", required=True, type=str, help="Train or val CSV (same schema as eval CSVs)")
    p.add_argument("--mss-run-dir", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Ignored — the paper-ckpt forward path is single-video. Accepted for launcher compat.",
    )
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


# Mapping: vjepa2-official-* → corresponding HF id2label source for label name normalization.
# We need an id2label mapping but the official wrapper doesn't expose one; pull it from the
# HF port (same num_classes; same class index ordering per Meta's released label maps).
HF_LABEL_SOURCE = {
    "vjepa2-official-diving48": "facebook/vjepa2-vitl-fpc32-256-diving48",
    "vjepa2-official-ssv2": "facebook/vjepa2-vitl-fpc16-256-ssv2",
}


def _id2label_for(recognizer: str) -> dict[int, str]:
    """Resolve the (class_id → label string) mapping for the recognizer."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(HF_LABEL_SOURCE[recognizer])
    return {int(k): v for k, v in cfg.id2label.items()}


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

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
        "shard %d/%d  %d rows of %d  device=%s  recognizer=%s",
        args.shard_id, args.num_shards, len(rows), len(all_rows), args.device, args.recognizer,
    )

    seen = _load_existing_video_ids(output_dir)
    todo = [r for r in rows if video_id_from_path(r["video_path"]) not in seen]
    LOG.info("Resume: %d already cached, %d to extract", len(rows) - len(todo), len(todo))
    if not todo:
        return 0

    LOG.info("Loading wrapper %s on %s …", args.recognizer, args.device)
    wrapper = VJepa2OfficialWrapper(recognizer_key=args.recognizer, device=args.device)
    id2label = _id2label_for(args.recognizer)
    label_norm = {normalize_label(v): int(k) for k, v in id2label.items()}
    num_labels = len(id2label)

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
            feature = wrapper.predict_kept_features(r["video_path"], result.kept_segments)
        except Exception as e:
            LOG.warning("feature extraction failed for %s (%s)", vid, e)
            n_build_fail += 1
            pbar.update(1)
            continue

        video_ids.append(vid)
        features.append(feature.contiguous())
        labels.append(gt_id)
        n_done_since_flush += 1
        pbar.update(1)

        if n_done_since_flush >= args.flush_every:
            _save_shard(shard_path, video_ids, features, labels)
            n_done_since_flush = 0

    _save_shard(shard_path, video_ids, features, labels)
    pbar.close()

    elapsed = time.time() - t0
    meta = {
        "recognizer": args.recognizer,
        "mss_run_dir": str(mss_run_dir),
        "csv": str(args.csv),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_cached_shard": len(video_ids),
        "n_missing_sidecar": n_missing_sidecar,
        "n_unlabeled": n_unlabeled,
        "n_no_kept_segments": n_no_kept,
        "n_extract_failed": n_build_fail,
        "hidden_size": int(features[0].shape[-1]) if features else None,
        "num_labels": num_labels,
        "wallclock_s": elapsed,
    }
    (output_dir / f"cache_metadata_shard_{args.shard_id:02d}.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )

    print(
        f"[shard {args.shard_id}] cached={len(video_ids)} "
        f"missing_sidecar={n_missing_sidecar} unlabeled={n_unlabeled} "
        f"no_kept={n_no_kept} extract_fail={n_build_fail} elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
