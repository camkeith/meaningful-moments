"""
Load MSS direct-scoring records produced by mss_extract.py.

Each MSS JSON describes a single video: its segments, which segments were
kept in the minimum-sufficient-subset, and whether the precheck passed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set


@dataclass
class MSSRecord:
    video_id: str
    video_path: Path
    action_label: str
    precheck_passed: bool
    segments: List[dict]           # list of {index, start_s, end_s}
    kept_indices: Set[int]         # indices from mss_result.mss_runs[0].kept_indices


def load_mss_record(json_path: Path) -> MSSRecord:
    """Parse a single MSS output JSON into the minimal record we need."""
    with open(json_path, "r") as f:
        d = json.load(f)

    mss_result = d.get("mss_result", {})
    runs = mss_result.get("mss_runs", [])
    kept = set(runs[0].get("kept_indices", [])) if runs else set()

    segs = mss_result.get("segments")
    if not segs:
        segs = [
            {"index": s["index"], "start_s": s["start_s"], "end_s": s["end_s"]}
            for s in d.get("segments", [])
        ]

    return MSSRecord(
        video_id=str(d["video_id"]),
        video_path=Path(d["video_path"]),
        action_label=d.get("action_label", ""),
        precheck_passed=bool(mss_result.get("precheck_passed", False)),
        segments=list(segs),
        kept_indices=kept,
    )
