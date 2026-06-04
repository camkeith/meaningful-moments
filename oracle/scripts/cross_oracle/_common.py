"""Shared constants + helpers for the cross-oracle analysis scripts.

Four oracles, three datasets, fixed pilot CSVs. Centralized so every script
sees the same paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[3]))

ORACLES: Tuple[str, ...] = ("qwen", "intern", "azure", "gemini")
DATASETS: Tuple[str, ...] = ("ssv2", "k400", "diving48")

# (oracle, dataset) → MSS run dir holding per-video sidecars.
MSS_DIRS: Dict[Tuple[str, str], Path] = {
    ("qwen",   "ssv2"):     REPO_ROOT / "pseudo_labels/mss/qwen3-vl-32b_20260412_133129",
    ("qwen",   "k400"):     REPO_ROOT / "pseudo_labels/mss/qwen3-vl-32b_k400_test_20260502_122107",
    ("qwen",   "diving48"): REPO_ROOT / "pseudo_labels/mss/qwen3-vl-32b_diving48_val_20260506_002245",
    ("intern", "ssv2"):     REPO_ROOT / "pseudo_labels/oracle_agreement/internvl3-38b_20260506_020002",
    ("intern", "k400"):     REPO_ROOT / "pseudo_labels/oracle_agreement/internvl3-38b_20260506_020002",
    ("intern", "diving48"): REPO_ROOT / "pseudo_labels/oracle_agreement/internvl3-38b_20260506_020002",
    ("azure",  "ssv2"):     REPO_ROOT / "pseudo_labels/oracle_agreement/azure_gpt-5.5_20260506_162942",
    ("azure",  "k400"):     REPO_ROOT / "pseudo_labels/oracle_agreement/azure_gpt-5.5_20260506_162942",
    ("azure",  "diving48"): REPO_ROOT / "pseudo_labels/oracle_agreement/azure_gpt-5.5_20260506_162942",
    ("gemini", "ssv2"):     REPO_ROOT / "pseudo_labels/oracle_agreement/gemini_20260506_020002",
    ("gemini", "k400"):     REPO_ROOT / "pseudo_labels/oracle_agreement/gemini_20260506_020002",
    ("gemini", "diving48"): REPO_ROOT / "pseudo_labels/oracle_agreement/gemini_20260506_020002",
}

# Pilot CSVs (200 videos each, schema: video_path, label, video_id, dataset, ...)
# Diving-48 uses an enriched derived CSV that adds the `class_id` column the
# recognizer needs for label matching (the original pilot CSV doesn't include it).
PILOT_CSVS: Dict[str, Path] = {
    "ssv2":     REPO_ROOT / "data/csvs/oracle_agreement/sample_ssv2_200.csv",
    "k400":     REPO_ROOT / "data/csvs/oracle_agreement/sample_k400_200.csv",
    "diving48": REPO_ROOT / "pseudo_labels/cross_oracle_eval/sample_diving48_200_with_class_id.csv",
}

# Recognizer per dataset (matches oracle/scripts/run_classifier_eval.py:RECOGNIZERS).
RECOGNIZERS: Dict[str, str] = {
    "ssv2":     "vjepa2-ssv2",
    "k400":     "videomae-k400-large",
    "diving48": "vjepa2-diving48",
}

# Conditions evaluated per (oracle, dataset) cell. EPS = vlm-selected − lowest-evidence.
# TDS = vlm-selected − uniform. We also keep `full` (for difficulty proxy in Stage 4)
# and `vlm-weighted` (for completeness).
CLASSIFIER_CONDITIONS: Tuple[str, ...] = (
    "full",
    "vlm-selected",
    "vlm-weighted",
    "uniform",
    "lowest-evidence",
)

OUT_ROOT = REPO_ROOT / "pseudo_labels/cross_oracle_eval"
