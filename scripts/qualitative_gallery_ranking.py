"""Pick 5 wins + 5 losses per substrate for the qualitative gallery (§Q).

A "win" is a video where the MSS-cut classifier is correct AND the full-video
classifier is wrong. A "loss" is the reverse. Picks are deterministically
sampled with seed=42, stratified by ground-truth class so we don't end up
with duplicate templates in one bucket.

Output: paper_tables_20260514_140732/qualitative_gallery_picks.json
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

SAGE = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[1]))
SEED = 42
N_PICKS_PER_BUCKET = 5
MIN_KEPT = 3
MIN_DROPPED = 3
N_WORKERS = 32  # parallel file reads (we're NFS-bound, not CPU-bound)


# ─── substrate configs ────────────────────────────────────────────────────

SSV2_CFG = {
    "name": "ssv2",
    "full_dir": SAGE / "pseudo_labels/classify_ssv2/full_20260423_005053",
    "mss_dir":  SAGE / "pseudo_labels/classify_ssv2/mss_kept_20260423_005053",
    "mss_run_dir": SAGE / "pseudo_labels/mss/qwen3-vl-32b_20260412_133129",
    "schema": "ssv2",
}

K400_CFG = {
    "name": "k400",
    "eval_dir": SAGE / "pseudo_labels/classifier_eval/k400_videomae-k400-large_20260505_025747",
    "mss_run_dir": SAGE / "pseudo_labels/mss/qwen3-vl-32b_k400_test_20260502_122107",
    "schema": "seven_condition",
}

D48_CFG = {
    "name": "d48",
    "eval_dir": SAGE / "pseudo_labels/classifier_eval/diving48_vjepa2-diving48_20260506_013210",
    "mss_run_dir": SAGE / "pseudo_labels/mss/qwen3-vl-32b_diving48_val_20260506_002245",
    "schema": "seven_condition",
}


# ─── parallel JSON readers ────────────────────────────────────────────────

def _read_classify_ssv2(path_str):
    """Worker: read one SSv2 closed-set sidecar; return slim dict or None."""
    try:
        d = json.loads(Path(path_str).read_text())
        if not d.get("parsed", False):
            return None
        return {
            "video_id": d.get("video_id") or Path(path_str).stem,
            "gt_label": d.get("gt_label"),
            "top1_correct": bool(d.get("top1_correct", False)),
            "top5": d.get("top5") or [],
        }
    except Exception:
        return None


def _read_eval_seven(path_str):
    """Worker: read one classifier_eval seven-condition sidecar; only return if
    condition is in {full, vlm-selected}."""
    p = Path(path_str)
    name = p.name
    if "__" not in name:
        return None
    cond = name.rsplit(".json", 1)[0].split("__", 1)[1]
    if cond not in ("full", "vlm-selected"):
        return None
    try:
        d = json.loads(p.read_text())
        vid = name.split("__")[0]
        labels = d.get("top5_labels") or []
        ids = d.get("top5_label_ids") or []
        top1 = labels[0] if labels else (f"class_{ids[0]}" if ids else "?")
        return {
            "video_id": vid,
            "condition": cond,
            "ground_truth_id": d.get("ground_truth_id"),
            "ground_truth": d.get("ground_truth") or d.get("gt_label"),
            "top1_correct": bool(d.get("top1_correct", False)),
            "top1_label": top1,
        }
    except Exception:
        return None


def _read_mss(path_str):
    """Worker: read MSS sidecar; return slim dict suitable for gallery rendering."""
    try:
        d = json.loads(Path(path_str).read_text())
        segs = d.get("segments", [])
        if not segs:
            return None
        if not d.get("mss_result", {}).get("precheck_passed", True):
            return None
        runs = d.get("mss_result", {}).get("mss_runs", [])
        if not runs:
            return None
        kept = sorted(runs[0].get("kept_indices", []))
        if not kept:
            return None
        total = len(segs)
        if len(kept) < MIN_KEPT or (total - len(kept)) < MIN_DROPPED:
            return None
        times = [(float(s.get("start_s") or 0.0), float(s.get("end_s") or 0.0)) for s in segs]
        return {
            "video_id": d.get("video_id") or Path(path_str).stem,
            "kept_indices": kept,
            "total_segments": total,
            "segment_times": times,
            "video_path": d.get("video_path", ""),
            "duration_s": times[-1][1] if times else 0.0,
            "mss_sidecar_path": str(path_str),
        }
    except Exception:
        return None


def parallel_read(paths, worker_fn, label):
    print(f"  [{label}] reading {len(paths):,} files with {N_WORKERS} workers …", flush=True)
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        results = pool.map(worker_fn, paths, chunksize=200)
    elapsed = time.time() - t0
    kept = [r for r in results if r is not None]
    print(f"    done in {elapsed:.1f}s, kept {len(kept):,} of {len(paths):,}", flush=True)
    return kept


# ─── per-substrate collectors ─────────────────────────────────────────────

def collect_ssv2(cfg):
    full_paths = [str(p) for p in cfg["full_dir"].glob("*.json")]
    full_records = parallel_read(full_paths, _read_classify_ssv2, "ssv2-full")
    full_by_vid = {r["video_id"]: r for r in full_records if r["gt_label"]}

    mss_paths = [str(p) for p in cfg["mss_dir"].glob("*.json")]
    mss_records = parallel_read(mss_paths, _read_classify_ssv2, "ssv2-mss")
    mss_by_vid = {r["video_id"]: r for r in mss_records if r["gt_label"]}

    common_vids = sorted(set(full_by_vid) & set(mss_by_vid))
    print(f"  [ssv2] common videos: {len(common_vids):,}")

    # Read corresponding MSS sidecars (only for common video_ids)
    mss_run_paths = [str(cfg["mss_run_dir"] / f"{v}.json") for v in common_vids]
    # Filter to those that exist on disk (cheap stat)
    mss_run_paths = [p for p in mss_run_paths if os.path.exists(p)]
    print(f"  [ssv2] MSS sidecars on disk: {len(mss_run_paths):,}")
    mss_run_records = parallel_read(mss_run_paths, _read_mss, "ssv2-mss-run")
    mss_run_by_vid = {r["video_id"]: r for r in mss_run_records}

    pairs = []
    for vid in common_vids:
        if vid not in mss_run_by_vid:
            continue
        fd, md, mr = full_by_vid[vid], mss_by_vid[vid], mss_run_by_vid[vid]
        pred_full = (fd.get("top5") or ["?"])[0]
        pred_mss = (md.get("top5") or ["?"])[0]
        pairs.append({
            "video_id": vid,
            "gt_label": fd["gt_label"],
            "pred_full": pred_full,
            "pred_mss": pred_mss,
            "full_top1_correct": fd["top1_correct"],
            "mss_top1_correct": md["top1_correct"],
            **mr,
        })
    return pairs


def collect_seven(cfg):
    eval_paths = [str(p) for p in cfg["eval_dir"].glob("*__*.json")]
    records = parallel_read(eval_paths, _read_eval_seven, f"{cfg['name']}-eval")
    by_vid_cond = defaultdict(dict)
    for r in records:
        if r is None: continue
        by_vid_cond[r["video_id"]][r["condition"]] = r
    paired_vids = [v for v, cs in by_vid_cond.items() if "full" in cs and "vlm-selected" in cs]
    print(f"  [{cfg['name']}] paired (full + vlm-selected) videos: {len(paired_vids):,}")

    mss_run_paths = [str(cfg["mss_run_dir"] / f"{v}.json") for v in paired_vids]
    mss_run_paths = [p for p in mss_run_paths if os.path.exists(p)]
    print(f"  [{cfg['name']}] MSS sidecars on disk: {len(mss_run_paths):,}")
    mss_records = parallel_read(mss_run_paths, _read_mss, f"{cfg['name']}-mss-run")
    mss_by_vid = {r["video_id"]: r for r in mss_records}

    pairs = []
    for vid in paired_vids:
        if vid not in mss_by_vid:
            continue
        conds = by_vid_cond[vid]
        fd, md = conds["full"], conds["vlm-selected"]
        gt = fd.get("ground_truth") or f"class_{fd.get('ground_truth_id')}"
        pairs.append({
            "video_id": vid,
            "gt_label": gt,
            "gt_class_id": fd.get("ground_truth_id"),
            "pred_full": fd["top1_label"],
            "pred_mss": md["top1_label"],
            "full_top1_correct": fd["top1_correct"],
            "mss_top1_correct": md["top1_correct"],
            **mss_by_vid[vid],
        })
    return pairs


# ─── stratified sampling ───────────────────────────────────────────────────

def stratified_sample(items, n, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for it in items:
        by_label[it["gt_label"]].append(it)
    labels = sorted(by_label.keys())
    rng.shuffle(labels)
    for lab in labels:
        by_label[lab].sort(key=lambda x: x["video_id"])
        rng.shuffle(by_label[lab])
    picks, used = [], set()
    for lab in labels:
        if len(picks) >= n: break
        for cand in by_label[lab]:
            if cand["video_id"] not in used:
                picks.append(cand)
                used.add(cand["video_id"])
                break
    if len(picks) < n:
        remaining = [it for lab in labels for it in by_label[lab] if it["video_id"] not in used]
        rng.shuffle(remaining)
        for cand in remaining:
            if len(picks) >= n: break
            picks.append(cand)
            used.add(cand["video_id"])
    return picks[:n]


def split_buckets(pairs):
    return {
        "wins":         [p for p in pairs if (not p["full_top1_correct"]) and p["mss_top1_correct"]],
        "losses":       [p for p in pairs if p["full_top1_correct"] and (not p["mss_top1_correct"])],
        "both_correct": [p for p in pairs if p["full_top1_correct"] and p["mss_top1_correct"]],
        "both_wrong":   [p for p in pairs if (not p["full_top1_correct"]) and (not p["mss_top1_correct"])],
    }


def main():
    out = {}
    summary = []
    for cfg, loader in [
        (SSV2_CFG, collect_ssv2),
        (K400_CFG, collect_seven),
        (D48_CFG,  collect_seven),
    ]:
        name = cfg["name"]
        print(f"\n=== {name.upper()} ===", flush=True)
        pairs = loader(cfg)
        buckets = split_buckets(pairs)
        bsz = {k: len(v) for k, v in buckets.items()}
        print(f"  buckets: wins={bsz['wins']}  losses={bsz['losses']}  "
              f"both_T={bsz['both_correct']}  both_F={bsz['both_wrong']}  total_pairs={len(pairs)}")
        wins = stratified_sample(buckets["wins"], N_PICKS_PER_BUCKET, SEED)
        losses = stratified_sample(buckets["losses"], N_PICKS_PER_BUCKET, SEED + 1)
        print(f"  sampled: wins={len(wins)} losses={len(losses)}")
        out[name] = {"bucket_sizes": bsz, "wins": wins, "losses": losses}
        summary.append((name, len(pairs), bsz["wins"], bsz["losses"], bsz["both_correct"], bsz["both_wrong"]))

    print("\n" + "=" * 70)
    print(f"{'substrate':<10} {'n_pairs':>8} {'wins':>6} {'losses':>7} {'both_T':>8} {'both_F':>8}")
    for r in summary:
        print(f"{r[0]:<10} {r[1]:>8d} {r[2]:>6d} {r[3]:>7d} {r[4]:>8d} {r[5]:>8d}")

    out_path = SAGE / "paper_tables_20260514_140732/qualitative_gallery_picks.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
