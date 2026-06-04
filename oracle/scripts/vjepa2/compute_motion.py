#!/usr/bin/env python
"""Pre-compute per-segment optical-flow magnitude for the motion baseline.

Per the development design notes Decision 6:
Farneback dense flow on CPU, one cache JSON per video.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger("compute_motion")

DEFAULT_CACHE_DIR = "pseudo_labels/motion_cache"


def video_id_from_path(p: str) -> str:
    return Path(p).stem


def compute_for_video(args_tuple) -> dict:
    video_path_str, sidecar_path_str, cache_path_str = args_tuple
    video_path = Path(video_path_str)
    sidecar_path = Path(sidecar_path_str)
    cache_path = Path(cache_path_str)

    if cache_path.exists():
        return {"video_id": video_id_from_path(video_path_str), "status": "skipped_existing"}

    try:
        with sidecar_path.open() as f:
            sidecar = json.load(f)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"video_id": video_id_from_path(video_path_str), "status": "failed", "error": "could not open video"}
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 1:
            cap.release()
            return {"video_id": video_id_from_path(video_path_str), "status": "failed", "error": f"too few frames: {total_frames}"}

        frames_gray = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        if len(frames_gray) < 2:
            return {"video_id": video_id_from_path(video_path_str), "status": "failed", "error": "need ≥2 frames for flow"}

        per_frame_magnitude = np.zeros(len(frames_gray), dtype=np.float32)
        for i in range(1, len(frames_gray)):
            flow = cv2.calcOpticalFlowFarneback(
                frames_gray[i - 1], frames_gray[i],
                None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            per_frame_magnitude[i] = float(mag.mean())
        per_frame_magnitude[0] = per_frame_magnitude[1] if len(per_frame_magnitude) > 1 else 0.0

        seg_outputs = []
        for seg in sidecar.get("segments", []):
            i0 = max(0, int(round(float(seg["start_s"]) * fps)))
            i1 = min(len(per_frame_magnitude) - 1, int(round(float(seg["end_s"]) * fps)))
            if i1 < i0:
                seg_mean = 0.0
            else:
                seg_mean = float(per_frame_magnitude[i0 : i1 + 1].mean()) if i1 + 1 > i0 else 0.0
            seg_outputs.append({
                "index": int(seg["index"]),
                "start_s": float(seg["start_s"]),
                "end_s": float(seg["end_s"]),
                "mean_flow_magnitude": seg_mean,
            })

        out = {
            "video_id": video_id_from_path(video_path_str),
            "video_path": video_path_str,
            "fps": float(fps),
            "total_frames": int(len(frames_gray)),
            "segments": seg_outputs,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "method": "Farneback (cv2.calcOpticalFlowFarneback) — pyr_scale=0.5, levels=3, winsize=15, iters=3, poly_n=5, poly_sigma=1.2",
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, sort_keys=True))
        return {"video_id": video_id_from_path(video_path_str), "status": "ok", "n_segments": len(seg_outputs)}
    except Exception as e:
        return {
            "video_id": video_id_from_path(video_path_str),
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--mss-run-dir", required=True, help="Source of segment definitions (per-video JSON)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    cache_dir = Path(args.cache_dir)
    mss_dir = Path(args.mss_run_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(args.eval_csv) as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    LOG.info("Total videos: %d   workers: %d   cache: %s", len(rows), args.num_workers, cache_dir)

    work = []
    for r in rows:
        vp = r["video_path"]
        vid = video_id_from_path(vp)
        sp = mss_dir / f"{vid}.json"
        cp = cache_dir / f"{vid}.json"
        if not sp.exists():
            LOG.warning("missing sidecar for %s, skipping", vid)
            continue
        work.append((vp, str(sp), str(cp)))

    LOG.info("Eligible: %d (skipped %d for missing sidecars)", len(work), len(rows) - len(work))

    from tqdm.auto import tqdm
    t0 = time.time()
    n_ok = n_skipped = n_failed = 0
    failures = []

    if args.num_workers <= 1:
        results = (compute_for_video(w) for w in work)
        for res in tqdm(results, total=len(work), desc="motion", unit="vid", dynamic_ncols=True):
            if res["status"] == "ok":
                n_ok += 1
            elif res["status"] == "skipped_existing":
                n_skipped += 1
            else:
                n_failed += 1
                failures.append(res)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            for res in tqdm(pool.imap_unordered(compute_for_video, work, chunksize=4), total=len(work), desc="motion", unit="vid", dynamic_ncols=True):
                if res["status"] == "ok":
                    n_ok += 1
                elif res["status"] == "skipped_existing":
                    n_skipped += 1
                else:
                    n_failed += 1
                    failures.append(res)

    elapsed = time.time() - t0
    print()
    print("=" * 64)
    print(f"motion pre-compute  cache={cache_dir}")
    print(f"  total:   {len(work)}")
    print(f"  ok:      {n_ok}")
    print(f"  skipped: {n_skipped}  (already cached)")
    print(f"  failed:  {n_failed}")
    print(f"  wall:    {elapsed:.1f}s   ({len(work) / max(elapsed, 1e-9):.2f} vid/s)")
    print("=" * 64)
    for f in failures[:10]:
        print(f"  FAILED {f['video_id']}: {f.get('error')}")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
