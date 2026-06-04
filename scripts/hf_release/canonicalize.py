#!/usr/bin/env python3
"""Canonicalize raw MSS sidecars into the App.-I release schema.

Transforms per-video JSONs from the 12 release run dirs (8 production +
4 cross-oracle) into the canonical Meaningful Moments sidecar format:

  - segments[i] gains time_range / phase / reason lifted from the oracle's
    raw_output (1-based segment_id join, asserted via weight == importance/100)
  - summary becomes the oracle's prose action_summary; the on-disk stats dict
    is released as label_counts
  - mss_result is flattened (kept_indices from mss_runs[0]; empty greedy-mode
    scaffolding dropped)
  - video_path is scrubbed to substrate-relative (starts with SSv2/, k400/,
    or diving48/)

Read-only over pseudo_labels/. Resumable (skips existing outputs). Atomic
writes (.tmp -> rename). Per-run report with recovery rate; aborts a run dir
below MIN_RECOVERY.

Usage:
  python scripts/hf_release/canonicalize.py --run ssv2_val --limit 500
  python scripts/hf_release/canonicalize.py --all
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SAGE_ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[2]))
STAGING_WORK = SAGE_ROOT / "hf_release_staging" / "v1.0_work" / "canonical"

# release_name -> source run dir (relative to sage root)
RELEASE_RUNS = {
    "ssv2_train": "pseudo_labels/mss/qwen3-vl-32b_ssv2_train_20260427_094557",
    "ssv2_val": "pseudo_labels/mss/qwen3-vl-32b_ssv2_val_20260430_134831",
    "ssv2_test": "pseudo_labels/mss/qwen3-vl-32b_20260412_133129",
    "k400_train": "pseudo_labels/mss/qwen3-vl-32b_k400_train_20260503_004606",
    "k400_val": "pseudo_labels/mss/qwen3-vl-32b_k400_val_20260502_014549",
    "k400_test": "pseudo_labels/mss/qwen3-vl-32b_k400_test_20260502_122107",
    "diving48_train": "pseudo_labels/mss/qwen3-vl-32b_diving48_train_20260506_003855",
    "diving48_val": "pseudo_labels/mss/qwen3-vl-32b_diving48_val_20260506_002245",
    # cross-oracle supplement (pinned by eval_complete_gemini_20260520_003221/metrics.json)
    "cross_oracle/qwen": "pseudo_labels/oracle_agreement/_qwen_sample600_20260507_043616",
    "cross_oracle/gemini": "pseudo_labels/oracle_agreement/gemini_20260506_020002",
    "cross_oracle/internvl3-38b": "pseudo_labels/oracle_agreement/internvl3-38b_20260506_020002",
    "cross_oracle/gpt-5.5": "pseudo_labels/oracle_agreement/azure_gpt-5.5_20260506_162942",
}

EXCLUDE_FILES = {"config.json", ".human_annotated.json"}
EXCLUDE_PREFIXES = ("_batch_state",)  # Gemini Batch-API bookkeeping, not sidecars
MIN_RECOVERY = 0.995  # hard gate for production runs; report-only for cross_oracle/*
WEIGHT_TOL = 1e-6

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_raw_output(raw):
    """Extract the oracle's JSON object from a raw_output string. None on failure."""
    if not raw or not isinstance(raw, str):
        return None
    text = _THINK_RE.sub("", raw)
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def scrub_path(p):
    """Absolute or repo-relative source path -> substrate-relative (SSv2/..., k400/..., diving48/...)."""
    if not p:
        return p
    p = str(p)
    for prefix in (str(SAGE_ROOT) + "/", "data/"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    # second strip handles absolute paths that contained .../sage/data/...
    if p.startswith("data/"):
        p = p[len("data/"):]
    return p


def canonicalize(doc):
    """Transform one raw sidecar dict -> (canonical dict, status str).

    status: 'enriched'            raw_output parsed, segments joined + asserted
            'degraded:<reason>'   App.-I-legal nulls emitted (counted vs MIN_RECOVERY)
    """
    mss = doc.get("mss_result") or {}
    prs = mss.get("precheck_responses") or []
    raw_obj = parse_raw_output(prs[0].get("raw_output")) if prs else None

    status = "enriched"
    by_id = {}
    if raw_obj is None:
        status = "degraded:raw_unparseable" if prs else "degraded:no_precheck_response"
    else:
        raw_segs = raw_obj.get("segments")
        if isinstance(raw_segs, list) and raw_segs:
            by_id = {
                s.get("segment_id"): s
                for s in raw_segs
                if isinstance(s, dict) and isinstance(s.get("segment_id"), int)
            }
        else:
            status = "degraded:no_raw_segments"

    src_segments = doc.get("segments") or []
    # Join-correctness assertion: weight == importance/100 for every joined segment.
    # Any mismatch -> treat the whole video as unenriched (never emit mis-joined text).
    if status == "enriched":
        for seg in src_segments:
            r = by_id.get(seg["index"] + 1)
            if r is None:
                status = "degraded:segment_count_mismatch"
                break
            imp = r.get("importance")
            if not isinstance(imp, (int, float)) or abs(seg["weight"] - imp / 100.0) > WEIGHT_TOL:
                status = "degraded:weight_join_mismatch"
                break

    enrich = status == "enriched"
    segments = []
    for seg in src_segments:
        r = by_id.get(seg["index"] + 1) if enrich else None
        start_s, end_s = seg.get("start_s"), seg.get("end_s")
        if r is not None and r.get("time_range"):
            time_range = r["time_range"]
        elif start_s is not None and end_s is not None:
            time_range = f"{start_s:g}-{end_s:g}"
        else:
            time_range = None
        # NOTE: source `frequency` is intentionally dropped — it equals `weight`
        # in every direct-scoring sidecar (greedy-era multi-run artifact).
        segments.append({
            "index": seg["index"],
            "time_range": time_range,
            "start_s": start_s,
            "end_s": end_s,
            "label": seg.get("label"),
            "weight": seg.get("weight"),
            "phase": (r.get("phase") if r else None),
            "reason": (r.get("reason", "") if r else ""),
        })

    # summary remap: prose -> summary, on-disk stats dict -> label_counts
    src_summary = doc.get("summary")
    label_counts = src_summary if isinstance(src_summary, dict) else None
    prose = None
    if raw_obj is not None and isinstance(raw_obj.get("action_summary"), str):
        prose = raw_obj["action_summary"]
    elif prs and isinstance(prs[0].get("evidence"), str):
        prose = prs[0]["evidence"]
    elif isinstance(src_summary, str):
        prose = src_summary

    mss_runs = mss.get("mss_runs") or []
    kept = mss_runs[0].get("kept_indices", []) if mss_runs else []
    mss_out = {
        "kept_indices": kept,
        "total_oracle_calls": mss.get("total_oracle_calls"),
        "precheck_passed": mss.get("precheck_passed"),
        "precheck_vote_yes": mss.get("precheck_vote_yes"),
        "precheck_responses": prs,  # verbatim, incl. raw_output (sidecars/tars only)
    }
    if "scores_recovered_from_raw_output" in mss:
        mss_out["scores_recovered_from_raw_output"] = mss["scores_recovered_from_raw_output"]

    out = {
        "video_id": doc.get("video_id"),
        "action_label": doc.get("action_label"),
        "video_path": scrub_path(doc.get("video_path")),
        "timestamp": doc.get("timestamp"),
        "summary": prose,
        "label_counts": label_counts,
        "segments": segments,
        "mss_result": mss_out,
        "elapsed_s": doc.get("elapsed_s"),
    }
    if "metadata" in doc:
        out["metadata"] = doc["metadata"]
    return out, status


def list_sidecars(src_dir):
    return sorted(
        f for f in os.listdir(src_dir)
        if f.endswith(".json") and f not in EXCLUDE_FILES
        and not f.startswith(".") and not f.startswith(EXCLUDE_PREFIXES)
    )


def process_run(release_name, limit=None, workers=32, force=False):
    src_dir = SAGE_ROOT / RELEASE_RUNS[release_name]
    out_dir = STAGING_WORK / release_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir.parent / f"{out_dir.name}.report.json"

    names = list_sidecars(src_dir)
    if limit:
        names = names[:limit]

    counts = {"total": len(names), "enriched": 0, "skipped_existing": 0,
              "degraded_precheck_passed": 0, "degraded_precheck_failed": 0}
    degraded = {}  # reason -> [video_ids]
    errors = []
    lock = threading.Lock()

    def one(name):
        out_path = out_dir / name
        if not force and out_path.exists():
            with lock:
                counts["skipped_existing"] += 1
            return
        try:
            doc = json.load(open(src_dir / name))
            canon, status = canonicalize(doc)
        except Exception as e:  # unreadable source counts as degraded, not crash
            with lock:
                errors.append((name, repr(e)))
            return
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(canon, f, indent=1)
        os.replace(tmp, out_path)
        with lock:
            if status == "enriched":
                counts["enriched"] += 1
            else:
                degraded.setdefault(status, []).append(name[:-5])
                if canon["mss_result"].get("precheck_passed"):
                    counts["degraded_precheck_passed"] += 1
                else:
                    counts["degraded_precheck_failed"] += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, n) for n in names]
        for i, fut in enumerate(as_completed(futures)):
            fut.result()
            if (i + 1) % 20000 == 0:
                print(f"  [{release_name}] {i + 1}/{len(names)}", flush=True)

    processed = counts["total"] - counts["skipped_existing"]
    n_degraded = sum(len(v) for v in degraded.values())
    recovery_all = counts["enriched"] / processed if processed else 1.0
    # Gate metric: enrichment success among precheck-passed videos only —
    # precheck-failed degradations are source-recorded oracle failures
    # (content-policy refusals etc.), released as-is per App. D.
    eligible = counts["enriched"] + counts["degraded_precheck_passed"]
    recovery_passed = counts["enriched"] / eligible if eligible else 1.0
    is_production = not release_name.startswith("cross_oracle/")
    gate_ok = (recovery_passed >= MIN_RECOVERY and not errors) if is_production else not errors
    report = {
        "release_name": release_name,
        "source_dir": str(src_dir.relative_to(SAGE_ROOT)),
        "counts": counts,
        "recovery_rate_all": recovery_all,
        "recovery_rate_precheck_passed": recovery_passed,
        "degraded": {k: {"n": len(v), "video_ids": v[:200]} for k, v in degraded.items()},
        "errors": errors[:50],
        "min_recovery": MIN_RECOVERY,
        "gated": is_production,
        "passed": gate_ok,
    }
    tmp = report_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1)
    os.replace(tmp, report_path)
    print(f"[{release_name}] processed={processed} enriched={counts['enriched']} "
          f"degraded={n_degraded} errors={len(errors)} "
          f"recovery(passed-only)={recovery_passed:.4%} all={recovery_all:.4%} "
          f"-> {'PASS' if gate_ok else 'FAIL'}{'' if is_production else ' (report-only)'}",
          flush=True)
    if not gate_ok:
        print(f"[{release_name}] ABORT: recovery below {MIN_RECOVERY:.1%} or read errors; "
              f"see {report_path}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=sorted(RELEASE_RUNS), help="single release run")
    ap.add_argument("--all", action="store_true", help="all 12 release runs")
    ap.add_argument("--limit", type=int, default=None, help="first N sidecars (validation)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="rewrite existing outputs")
    args = ap.parse_args()

    targets = sorted(RELEASE_RUNS) if args.all else ([args.run] if args.run else [])
    if not targets:
        ap.error("--run or --all required")
    ok = True
    for name in targets:
        ok &= process_run(name, limit=args.limit, workers=args.workers, force=args.force)["passed"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
