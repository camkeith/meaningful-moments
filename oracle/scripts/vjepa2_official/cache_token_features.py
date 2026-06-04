#!/usr/bin/env python
"""Phase A — extract V-JEPA 2 paper-ckpt encoder-output tokens (pre-classifier).

This is the input to the AttentiveClassifier — what the 4-block probe attends over.
Caching these lets us retrain the full AttentiveClassifier (not just its final
Linear) on MSS-cut inputs without re-running the encoder every training step.

Output: one .pt file per video at
    <output-dir>/<video_id>.pt   = {"tokens": Tensor (or list/stack), "label": int, "video_id": str}

Schema notes:
  - SSv2 (single-level, `out_layers=None`): `tokens` is Tensor[num_tokens, embed_dim]
    (e.g., (2048, 1024) for fpc16, 256×256, patch=16, tubelet=2 → 8 × 16 × 16 = 2048).
  - Diving-48 (multilevel, `out_layers=[17,19,21,23]`): `tokens` is Tensor[num_layers,
    num_tokens, embed_dim] (e.g., (4, 4096, 1024) for fpc32). Stored stacked, matches
    the encoder's multilevel output shape so the AttentiveClassifier can consume directly.
  - All stored as bf16 to halve disk vs fp32 (training casts back to fp32 as needed).

Sharding is video-level (one file per video) rather than tensor-stacked-per-shard
because token tensors are ~4-16 MB each; stacking into one file would blow up RAM
at load time. Per-video files keep the dataloader simple and resume trivially.

CSV / MSS / args mirror oracle.scripts.vjepa2_official.cache_features (post-pooler
features) so the launcher can dispatch via a flag.
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

LOG = logging.getLogger("vjepa2_official.cache_token_features")

# HF id2label source (one per recognizer) — used to map CSV `label` text → class id.
HF_LABEL_SOURCE = {
    "vjepa2-official-diving48": "facebook/vjepa2-vitl-fpc32-256-diving48",
    "vjepa2-official-ssv2": "facebook/vjepa2-vitl-fpc16-256-ssv2",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recognizer", required=True, choices=sorted(PROBE_CONFIGS.keys()))
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--mss-run-dir", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--num-segments", type=int, default=1,
        help="Number of temporal sub-clips per video. Paper uses 2 at train and val.",
    )
    p.add_argument(
        "--num-views", type=int, default=1,
        help="Number of spatial views per sub-clip. Paper uses 1 at train, 3 at val.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def video_id_from_path(p: str) -> str:
    return Path(p).stem


def normalize_label(s: str) -> str:
    return s.replace("[", "").replace("]", "").strip().lower()


def _resolve_gt_id(row: dict, label_norm: dict[str, int]) -> int | None:
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


def _id2label_for(recognizer: str) -> dict[int, str]:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(HF_LABEL_SOURCE[recognizer])
    return {int(k): v for k, v in cfg.id2label.items()}


def _per_video_path(out_dir: Path, video_id: str) -> Path:
    return out_dir / f"{video_id}.pt"


def _atomic_save(tensor_dict: dict, path: Path) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save(tensor_dict, str(tmp))
    tmp.rename(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    # Skip videos already cached.
    todo = [r for r in rows if not _per_video_path(output_dir, video_id_from_path(r["video_path"])).exists()]
    LOG.info("Resume: %d already cached, %d to extract", len(rows) - len(todo), len(todo))
    if not todo:
        return 0

    LOG.info("Loading wrapper %s on %s …", args.recognizer, args.device)
    wrapper = VJepa2OfficialWrapper(recognizer_key=args.recognizer, device=args.device)
    id2label = _id2label_for(args.recognizer)
    label_norm = {normalize_label(v): int(k) for k, v in id2label.items()}
    num_labels = len(id2label)

    from tqdm.auto import tqdm
    pbar = tqdm(total=len(todo), desc=f"tokcache/sh{args.shard_id}", unit="clip", dynamic_ncols=True)

    n_missing_sidecar = 0
    n_unlabeled = 0
    n_no_kept = 0
    n_extract_fail = 0
    n_cached = 0
    sample_shape = None
    t0 = time.time()

    # Reuse the wrapper's decoding path but stop before the pooler.
    import av
    from oracle.scripts.vjepa2_official.wrapper import (
        _gather_frames, _resize_short_side, _spatial_views, _to_tensor_normalize,
        _multi_clip_kept_indices,
    )

    num_segments = args.num_segments
    num_views = args.num_views
    LOG.info("Per-video clips: num_segments=%d × num_views=%d = %d", num_segments, num_views, num_segments * num_views)

    for r in todo:
        vid = video_id_from_path(r["video_path"])
        out_path = _per_video_path(output_dir, vid)

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
            with av.open(r["video_path"]) as c:
                stream = c.streams.video[0]
                fps = float(stream.average_rate or 25.0)
                all_frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            if not all_frames:
                raise RuntimeError(f"empty video: {r['video_path']}")

            # Paper-style multi-segment sampling on the MSS-kept frame pool.
            seg_idx_list = _multi_clip_kept_indices(
                len(all_frames), fps, list(result.kept_segments),
                wrapper.frames_per_clip, wrapper.frame_step, num_segments,
            )

            clip_tokens: list[torch.Tensor] = []
            for indices in seg_idx_list:
                seg_frames = _gather_frames(all_frames, indices)
                seg_resized = _resize_short_side(seg_frames, wrapper.resolution)
                views = _spatial_views(seg_resized, wrapper.resolution, num_views)
                for view in views:
                    view_tensor = _to_tensor_normalize(view, wrapper.device, wrapper.dtype)
                    clips = [[view_tensor]]  # one (segment, view) at a time → 1 clip in encoder list
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                        outputs = wrapper.encoder(clips, clip_indices=None)
                        tok = outputs[0]
                        if tok.dim() == 3:
                            tok = tok.squeeze(0)
                        elif tok.dim() == 4:
                            tok = tok.squeeze(0)
                        else:
                            raise RuntimeError(
                                f"unexpected encoder output dim {tok.dim()} (shape {tuple(tok.shape)})"
                            )
                    clip_tokens.append(tok.to(torch.bfloat16).cpu())
            # Stack → (num_clips, num_tokens, embed_dim)
            stacked = torch.stack(clip_tokens, dim=0)
        except Exception as e:
            LOG.warning("token extraction failed for %s (%s)", vid, e)
            n_extract_fail += 1
            pbar.update(1)
            continue

        if sample_shape is None:
            sample_shape = tuple(stacked.shape)
            LOG.info("sample stacked token shape: %s, dtype: %s", sample_shape, stacked.dtype)

        _atomic_save(
            {
                "tokens": stacked,
                "label": int(gt_id),
                "video_id": vid,
                "recognizer": args.recognizer,
                "num_segments": num_segments,
                "num_views": num_views,
            },
            out_path,
        )
        n_cached += 1
        pbar.update(1)

    pbar.close()

    elapsed = time.time() - t0
    meta = {
        "recognizer": args.recognizer,
        "mss_run_dir": str(mss_run_dir),
        "csv": str(args.csv),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_cached_shard": n_cached,
        "n_missing_sidecar": n_missing_sidecar,
        "n_unlabeled": n_unlabeled,
        "n_no_kept_segments": n_no_kept,
        "n_extract_failed": n_extract_fail,
        "sample_token_shape": list(sample_shape) if sample_shape else None,
        "num_labels": num_labels,
        "wallclock_s": elapsed,
    }
    (output_dir / f"cache_metadata_shard_{args.shard_id:02d}.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )

    print(
        f"[shard {args.shard_id}] cached={n_cached} "
        f"missing_sidecar={n_missing_sidecar} unlabeled={n_unlabeled} "
        f"no_kept={n_no_kept} extract_fail={n_extract_fail} elapsed={elapsed:.1f}s "
        f"sample_token_shape={sample_shape}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
