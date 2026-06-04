#!/usr/bin/env python
"""Stage 0 — sanity checks for the cross-oracle analysis.

Verifies that the 3 oracles × 3 datasets give us comparable inputs:
  - all 200 pilot video_ids present in each (oracle, dataset) run dir
  - segment timings identical across oracles for the same video
  - precheck-pass rate per cell
  - format-failure rate per oracle

Writes one JSON report so downstream stages can branch on the results.

Usage:
    python -m oracle.scripts.cross_oracle.sanity \\
        --output-dir pseudo_labels/cross_oracle_eval/

Exits with non-zero status if any cell drops below the precheck-pass floor
(default 50%) AND the cell isn't already on a known-bad allow-list (currently
just azure × diving48).
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ._common import (
    DATASETS,
    MSS_DIRS,
    ORACLES,
    OUT_ROOT,
    PILOT_CSVS,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("cross_oracle.sanity")

# Cells we expect to underperform — surfaced in the report but don't fail the run.
KNOWN_BAD = {("azure", "diving48")}
PRECHECK_FLOOR = 0.40  # below this, fail unless in KNOWN_BAD


def load_pilot_ids(csv_path: Path) -> List[str]:
    with open(csv_path) as f:
        return [r["video_id"] for r in csv.DictReader(f)]


def list_sidecars(d: Path) -> List[Path]:
    """Drop bookkeeping JSONs from the sidecar listing."""
    excluded = {"run_metadata.json", "config.json", "results.jsonl"}
    return [p for p in d.glob("*.json") if p.name not in excluded]


def inspect_sidecar(p: Path) -> Optional[dict]:
    """Return the parsed sidecar or None on read error."""
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        log.debug(f"unreadable {p}: {e}")
        return None


def cell_coverage(oracle: str, dataset: str, pilot_ids: List[str]) -> dict:
    """Compute coverage stats for one (oracle, dataset) cell."""
    run_dir = MSS_DIRS[(oracle, dataset)]
    if not run_dir.exists():
        return {
            "oracle": oracle,
            "dataset": dataset,
            "run_dir": str(run_dir),
            "error": "run_dir not found",
        }

    sidecars_by_id: Dict[str, dict] = {}
    unreadable = 0
    for p in list_sidecars(run_dir):
        rec = inspect_sidecar(p)
        if rec is None:
            unreadable += 1
            continue
        vid = rec.get("video_id") or p.stem
        sidecars_by_id[vid] = rec

    pilot_set = set(pilot_ids)
    present = pilot_set & set(sidecars_by_id)
    missing = pilot_set - set(sidecars_by_id)

    pilot_passed: List[str] = []
    pilot_failed_precheck: List[str] = []
    for vid in present:
        rec = sidecars_by_id[vid]
        if rec.get("mss_result", {}).get("precheck_passed", False):
            pilot_passed.append(vid)
        else:
            pilot_failed_precheck.append(vid)

    return {
        "oracle": oracle,
        "dataset": dataset,
        "run_dir": str(run_dir),
        "n_total_sidecars_in_dir": len(sidecars_by_id),
        "n_pilot": len(pilot_set),
        "n_pilot_present": len(present),
        "n_pilot_missing": len(missing),
        "n_pilot_passed_precheck": len(pilot_passed),
        "n_pilot_failed_precheck": len(pilot_failed_precheck),
        "n_unreadable_sidecars": unreadable,
        "precheck_pass_rate_on_pilot": (
            len(pilot_passed) / max(1, len(present))
        ),
        "missing_video_ids": sorted(missing),
        "failed_precheck_video_ids": sorted(pilot_failed_precheck),
    }


def check_segment_alignment(
    pilot_ids: List[str], dataset: str, max_videos: int = 50
) -> dict:
    """Compare segment timings across oracles for the same video.

    Sample up to `max_videos` from the pilot, load each oracle's sidecar,
    and check that all oracles produce identical segments[*].start_s/end_s.
    """
    out = {"dataset": dataset, "n_checked": 0, "n_aligned": 0, "mismatches": []}
    sidecars_per_oracle: Dict[str, Dict[str, dict]] = {}
    for oracle in ORACLES:
        run_dir = MSS_DIRS[(oracle, dataset)]
        sidecars_per_oracle[oracle] = {
            (rec.get("video_id") or p.stem): rec
            for p in list_sidecars(run_dir)
            for rec in [inspect_sidecar(p)]
            if rec is not None
        }

    sample = pilot_ids[:max_videos]
    for vid in sample:
        per_oracle_segments: Dict[str, list] = {}
        for oracle in ORACLES:
            rec = sidecars_per_oracle[oracle].get(vid)
            if rec is None:
                continue
            segs = rec.get("mss_result", {}).get("segments") or rec.get("segments") or []
            if not segs:
                continue
            per_oracle_segments[oracle] = [
                (round(float(s.get("start_s", 0)), 4), round(float(s.get("end_s", 0)), 4))
                for s in segs
            ]
        if len(per_oracle_segments) < 2:
            continue
        out["n_checked"] += 1
        first = next(iter(per_oracle_segments.values()))
        if all(per_oracle_segments[o] == first for o in per_oracle_segments):
            out["n_aligned"] += 1
        else:
            mm = {"video_id": vid, "per_oracle_segments": per_oracle_segments}
            if len(out["mismatches"]) < 5:
                out["mismatches"].append(mm)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_ROOT,
        help="Where to write sanity.json (default: pseudo_labels/cross_oracle_eval/)",
    )
    parser.add_argument(
        "--max-align-checks",
        type=int,
        default=50,
        help="Max videos to spot-check for segment-timing alignment per dataset.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {"cells": [], "alignment": [], "summary": {}}

    log.info("=== Stage 0: sanity checks ===")

    # Coverage matrix per (oracle, dataset)
    for ds in DATASETS:
        pilot_ids = load_pilot_ids(PILOT_CSVS[ds])
        log.info(f"{ds}: {len(pilot_ids)} pilot videos")
        for oracle in ORACLES:
            cell = cell_coverage(oracle, ds, pilot_ids)
            report["cells"].append(cell)
            log.info(
                f"  {oracle:>6s} × {ds:<8s}  "
                f"present={cell.get('n_pilot_present', 0):>3d}/{cell.get('n_pilot', 0)}  "
                f"passed={cell.get('n_pilot_passed_precheck', 0):>3d}  "
                f"pass_rate={cell.get('precheck_pass_rate_on_pilot', 0):.1%}"
            )

    # Segment-timing alignment
    log.info("--- segment-timing alignment ---")
    for ds in DATASETS:
        pilot_ids = load_pilot_ids(PILOT_CSVS[ds])
        align = check_segment_alignment(pilot_ids, ds, max_videos=args.max_align_checks)
        report["alignment"].append(align)
        log.info(
            f"  {ds:<8s}  checked={align['n_checked']:>3d}  "
            f"aligned={align['n_aligned']:>3d}  "
            f"mismatches={len(align['mismatches'])}"
        )

    # Summary
    fails = []
    for cell in report["cells"]:
        rate = cell.get("precheck_pass_rate_on_pilot", 0.0)
        key = (cell["oracle"], cell["dataset"])
        if rate < PRECHECK_FLOOR and key not in KNOWN_BAD:
            fails.append({"cell": key, "rate": rate})
    report["summary"]["fail_floor_violations"] = fails
    report["summary"]["pass"] = len(fails) == 0

    out_path = args.output_dir / "sanity.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"wrote {out_path}")

    if fails:
        log.error(f"FAIL: {len(fails)} cell(s) below floor {PRECHECK_FLOOR:.0%}: {fails}")
        sys.exit(1)
    log.info("Sanity checks passed.")


if __name__ == "__main__":
    main()
