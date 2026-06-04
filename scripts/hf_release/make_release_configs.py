#!/usr/bin/env python3
"""Generate released config.json per release run (task 1.5).

Copies each source run's config.json and injects a `release` provenance block:
model revision (recovered from the local HF cache refs — NOT recorded at
extraction time, hence the release_note), full prompt SHA-256 (cross-checked
against the App.-J short hashes), and dataset version. Synthesizes a config
for the cross_oracle/qwen dir, which has none.

Output: hf_release_staging/v1.0_work/configs/<release_name>/config.json
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from canonicalize import RELEASE_RUNS, SAGE_ROOT

OUT_ROOT = SAGE_ROOT / "hf_release_staging" / "v1.0_work" / "configs"
PROMPT_DIR = SAGE_ROOT / "oracle/scripts/mss/prompts"

# App. J short hashes (first 16 hex of file SHA-256, verified at commit 9f688b2)
APP_J_SHORT = {
    "direct_scoring": "91166e24b4bd2980",
    "direct_scoring_ssv2": "2b19223aa8c20171",
    "direct_scoring_k400": "3f0f55cbceefdf83",
    "direct_scoring_diving48": "e537b25dc8fd3ca6",
}

# Oracle model revision per release run. Local HF models pinned by cache ref;
# API oracles have no pinnable weight revision.
MODEL_REVISIONS = {
    "qwen3-vl-32b": {
        "model_id": "Qwen/Qwen3-VL-32B-Instruct",
        "revision": "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
    },
    "internvl3-38b": {
        "model_id": "OpenGVLab/InternVL3-38B",
        "revision": "b99bcba4fe0c51d9e475af25ec000dbfea6284af",
    },
    "gemini-3.1-pro": {
        "model_id": "gemini-3.1-pro (Dartmouth ChatAPI Batch)",
        "revision": None,
    },
    "gpt-5.5": {
        "model_id": "gpt-5.5 (Azure OpenAI)",
        "revision": None,
    },
}

REVISION_NOTE = (
    "model revision recovered from the local HuggingFace cache ref at release "
    "time; it was not recorded in config.json at extraction time. API oracles "
    "(Gemini, GPT-5.5) have no pinnable weight revision."
)


def prompt_sha(template_id):
    path = PROMPT_DIR / f"{template_id}.md"
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    short = h[:16]
    expected = APP_J_SHORT.get(template_id)
    if expected and short != expected:
        sys.exit(f"FATAL: prompt {template_id} short hash {short} != App.-J {expected} "
                 f"— prompt file drifted since commit 9f688b2")
    return h, short


def main():
    for release_name, src_rel in RELEASE_RUNS.items():
        src_cfg_path = SAGE_ROOT / src_rel / "config.json"
        notes = [REVISION_NOTE]
        if src_cfg_path.exists():
            cfg = json.load(open(src_cfg_path))
        else:
            # cross_oracle/qwen (_qwen_sample600) has no config.json. Parameters
            # inferred from the oracle_agreement launcher and the three sibling
            # oracle configs of the same study (all record the generic
            # direct_scoring prompt and identical mss_config).
            assert release_name == "cross_oracle/qwen", release_name
            base = json.load(open(
                SAGE_ROOT / RELEASE_RUNS["cross_oracle/internvl3-38b"] / "config.json"))
            cfg = dict(base)
            cfg["provider"] = "local"
            cfg["model"] = "qwen3-vl-32b"
            cfg["timestamp"] = "_qwen_sample600_20260507_043616"
            notes.append(
                "synthesized at release time: the source run dir has no config.json; "
                "parameters copied from the sibling internvl3-38b config of the same "
                "oracle-agreement study (identical mss_config / generic direct_scoring "
                "prompt, per scripts/oracle_agreement/launch.sh)."
            )
        if release_name.startswith("cross_oracle/"):
            notes.append(
                "the `dataset` field reflects one launch shard; the full study pool "
                "is manifests/eval_pools/sample_600.csv (n=600 across SSv2/K400/D48)."
            )

        template_id = cfg.get("oracle_config", {}).get("prompt_template_id", "direct_scoring")
        full, short = prompt_sha(template_id)
        rev = MODEL_REVISIONS[cfg["model"]]
        cfg["release"] = {
            "dataset_version": "v1.0",
            "model_id": rev["model_id"],
            "model_revision": rev["revision"],
            "prompt_template_file": f"prompts/{template_id}.md",
            "prompt_sha256": full,
            "prompt_sha256_short16": short,
            "release_note": " ".join(notes),
        }

        out_path = OUT_ROOT / release_name / "config.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=1)
        os.replace(tmp, out_path)
        print(f"{release_name:28s} prompt={template_id:24s} short={short} "
              f"model={cfg['model']}")
    print("OK: all release configs written; prompt hashes match App. J")


if __name__ == "__main__":
    main()
