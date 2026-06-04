"""Extract dataset-distribution arrays from the three canonical Qwen3-VL-32B train MSS runs.

Outputs (under scripts/distributions/out/):
  distributions.json
      ssv2 / k400 / d48 ->
        class_counts            : [[class_name, count], ...]      (full)
        segments_per_video      : [int, ...]                      (downsampled, seeded)
        segments_per_video_full : [int, ...]                      (full, used for stats)
        selected_fraction       : [float, ...]                    (downsampled, seeded)
        selected_fraction_full  : [float, ...]                    (full, used for stats)
        importance_scores       : [float, ...]                    (downsampled, seeded)
        n_videos                : int
        n_videos_precheck_passed: int
        n_classes               : int
  example_records.json
      ssv2 / k400 / d48 -> list of 3 records each (real annotations)

Selected-fraction = sum(duration of segments labeled "important") / sum(duration of all segments).
Importance-score = the segment `weight` (continuous, equals inclusion frequency).
Class name:
  ssv2  -> metadata.template     (canonical 174-class space)
  k400  -> action_label
  d48   -> metadata.raw_class_name
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MM_ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[2]))
PSEUDO_ROOT = MM_ROOT / "pseudo_labels/mss"
# Each substrate maps to ALL of its split run dirs, so the extracted
# distributions cover the full headline corpus (train+val+test), matching the
# 499,299 precheck-passed / 536,181 attempt counts reported in the thesis.
RUNS = {
    "ssv2": [
        PSEUDO_ROOT / "qwen3-vl-32b_ssv2_train_20260427_094557",
        PSEUDO_ROOT / "qwen3-vl-32b_ssv2_val_20260430_134831",
        PSEUDO_ROOT / "qwen3-vl-32b_20260412_133129",  # SSv2 test
    ],
    "k400": [
        PSEUDO_ROOT / "qwen3-vl-32b_k400_train_20260503_004606",
        PSEUDO_ROOT / "qwen3-vl-32b_k400_val_20260502_014549",
        PSEUDO_ROOT / "qwen3-vl-32b_k400_test_20260502_122107",
    ],
    "d48": [
        PSEUDO_ROOT / "qwen3-vl-32b_diving48_train_20260506_003855",
        PSEUDO_ROOT / "qwen3-vl-32b_diving48_val_20260506_002245",
    ],
}

DEFAULT_PLOT_SAMPLE = 5000        # downsample for plotting arrays
DEFAULT_IMPORTANCE_SAMPLE = 10000  # downsample for importance scores per substrate


def _ssv2_canonical(name: str | None) -> str | None:
    """Normalize SSv2 class names to the canonical 174-template space.

    Train/val sidecars store the bracketed template ("Putting [something] on a
    surface"); the test run stores a bracket-stripped variant ("Putting
    something on a surface"). Strip brackets + casing + whitespace so the two
    variants merge instead of doubling to ~348 classes.
    """
    if name is None:
        return None
    import re as _re
    return _re.sub(r"\s+", " ", name.replace("[", "").replace("]", "")).strip().lower()


def class_name_of(j: dict, substrate: str) -> str | None:
    if substrate == "ssv2":
        meta = j.get("metadata") or {}
        return _ssv2_canonical(meta.get("template") or j.get("action_label"))
    if substrate == "k400":
        return j.get("action_label")
    if substrate == "d48":
        meta = j.get("metadata") or {}
        return meta.get("raw_class_name") or j.get("action_label")
    return None


def per_video(path_substrate: tuple[str, str]) -> tuple[str | None, int, float, list[float], bool, float] | None:
    """Returns (class_name, n_segments, selected_fraction, importance_scores, precheck_passed, clip_duration_s)."""
    path, substrate = path_substrate
    try:
        with open(path, "r") as f:
            j = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    segments = j.get("segments") or []
    n_segments = len(segments)
    if n_segments == 0:
        return None

    total_dur = 0.0
    important_dur = 0.0
    weights: list[float] = []
    for seg in segments:
        start = float(seg.get("start_s") or 0.0)
        end = float(seg.get("end_s") or 0.0)
        dur = max(0.0, end - start)
        total_dur += dur
        if seg.get("label") == "important":
            important_dur += dur
        w = seg.get("weight")
        if w is None:
            w = seg.get("frequency")
        if w is not None:
            weights.append(float(w))

    selected_fraction = (important_dur / total_dur) if total_dur > 0 else 0.0
    mss_result = j.get("mss_result") or {}
    precheck_passed = bool(mss_result.get("precheck_passed", True))

    return (class_name_of(j, substrate), n_segments, selected_fraction, weights, precheck_passed, total_dur)


def list_json_files(run_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for run_dir in run_dirs:
        files.extend(run_dir.glob("*.json"))
    return sorted(files)


def process_substrate(substrate: str, run_dirs: list[Path], n_workers: int,
                       plot_sample: int, importance_sample: int, seed: int) -> dict[str, Any]:
    files = list_json_files(run_dirs)
    print(f"[{substrate}] {len(files):,} json files across "
          f"{len(run_dirs)} run dir(s): {', '.join(d.name for d in run_dirs)}", flush=True)
    rng = random.Random(seed)

    # All plotting arrays and class counts are restricted to precheck-passed
    # videos, so the figures characterize the released headline corpus. The
    # total scored count is tracked separately for the summary table.
    class_counter: Counter[str] = Counter()
    segs_per_video: list[int] = []
    sel_frac: list[float] = []
    clip_dur: list[float] = []
    importance_reservoir: list[float] = []  # reservoir sample of segment weights
    importance_seen = 0
    n_scored = 0
    n_precheck_passed = 0

    args = [(str(p), substrate) for p in files]

    # Reservoir sample for importance scores (memory-bounded across ~1.5M segments)
    irng = random.Random(seed + 1)

    with mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(per_video, args, chunksize=128)):
            if result is None:
                continue
            cls, n_seg, frac, weights, precheck, dur = result
            n_scored += 1
            if not precheck:
                continue
            n_precheck_passed += 1
            if cls is not None:
                class_counter[cls] += 1
            segs_per_video.append(n_seg)
            sel_frac.append(frac)
            clip_dur.append(dur)
            for w in weights:
                importance_seen += 1
                if len(importance_reservoir) < importance_sample:
                    importance_reservoir.append(w)
                else:
                    k = irng.randint(0, importance_seen - 1)
                    if k < importance_sample:
                        importance_reservoir[k] = w
            if (i + 1) % 25000 == 0:
                print(f"[{substrate}]  processed {i+1:,} / {len(files):,}", flush=True)

    print(f"[{substrate}] done: {n_scored:,} scored, "
          f"{n_precheck_passed:,} precheck-passed, "
          f"{len(class_counter):,} classes, {importance_seen:,} passed segments",
          flush=True)

    # Downsample plotting arrays
    def downsample(xs: list, k: int) -> list:
        if len(xs) <= k:
            return list(xs)
        return rng.sample(xs, k)

    return {
        "n_videos": n_scored,
        "n_videos_precheck_passed": n_precheck_passed,
        "n_classes": len(class_counter),
        "n_segments_total": importance_seen,
        "class_counts": sorted(class_counter.items(), key=lambda kv: -kv[1]),
        "segments_per_video_full": segs_per_video,
        "selected_fraction_full": sel_frac,
        "clip_duration_full": clip_dur,
        "segments_per_video": downsample(segs_per_video, plot_sample),
        "selected_fraction": downsample(sel_frac, plot_sample),
        "clip_duration": downsample(clip_dur, plot_sample),
        "importance_scores": importance_reservoir,
    }


def pick_examples(substrate: str, run_dirs: list[Path], n: int, seed: int) -> list[dict]:
    """Pick n example records that passed precheck, deterministically by seed."""
    files = list_json_files(run_dirs)
    rng = random.Random(seed + 42)
    rng.shuffle(files)
    out: list[dict] = []
    for p in files:
        try:
            with open(p, "r") as f:
                j = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        mss = j.get("mss_result") or {}
        if not mss.get("precheck_passed", False):
            continue
        segments = j.get("segments") or []
        if len(segments) < 4:
            continue
        # Trim each segment to the fields a reviewer cares about.
        seg_view = [
            {
                "index": s.get("index"),
                "start_s": s.get("start_s"),
                "end_s": s.get("end_s"),
                "label": s.get("label"),
                "weight": s.get("weight"),
            }
            for s in segments
        ]
        important = [s["index"] for s in segments if s.get("label") == "important"]
        precheck_resp = (mss.get("precheck_responses") or [{}])[0]
        out.append({
            "video_id": j.get("video_id"),
            "video_path": j.get("video_path"),
            "class_name": class_name_of(j, substrate),
            "action_label_filled": j.get("action_label"),
            "n_segments": len(segments),
            "important_segment_indices": important,
            "segments": seg_view,
            "oracle_evidence": precheck_resp.get("evidence"),
            "oracle_rationale": precheck_resp.get("rationale"),
        })
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, mp.cpu_count()))
    ap.add_argument("--plot-sample", type=int, default=DEFAULT_PLOT_SAMPLE)
    ap.add_argument("--importance-sample", type=int, default=DEFAULT_IMPORTANCE_SAMPLE)
    ap.add_argument("--examples-per-substrate", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=MM_ROOT / "scripts/distributions/out")
    ap.add_argument("--substrates", nargs="*", default=list(RUNS.keys()),
                    help="Subset of substrates to run (ssv2 k400 d48). Default: all.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    distributions: dict[str, Any] = {}
    examples: dict[str, Any] = {}
    for substrate in args.substrates:
        run_dirs = RUNS[substrate]
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                print(f"[{substrate}] missing run_dir: {run_dir}", file=sys.stderr)
                return 2
        distributions[substrate] = process_substrate(
            substrate, run_dirs, args.workers,
            args.plot_sample, args.importance_sample, args.seed,
        )
        examples[substrate] = pick_examples(
            substrate, run_dirs, args.examples_per_substrate, args.seed,
        )

    out_distributions = args.out_dir / "distributions.json"
    out_examples = args.out_dir / "example_records.json"
    with open(out_distributions, "w") as f:
        json.dump(distributions, f)
    with open(out_examples, "w") as f:
        json.dump(examples, f, indent=2)

    sizes = {k: out_distributions.parent / k for k in distributions}
    print()
    print(f"wrote {out_distributions}  ({out_distributions.stat().st_size/1e6:.2f} MB)")
    print(f"wrote {out_examples}  ({out_examples.stat().st_size/1e3:.2f} KB)")
    print()
    print("Summary:")
    for s, d in distributions.items():
        print(f"  {s}: {d['n_videos']:,} videos | {d['n_classes']} classes | "
              f"{d['n_segments_total']:,} segments | "
              f"precheck_passed={d['n_videos_precheck_passed']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
