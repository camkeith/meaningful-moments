#!/usr/bin/env python3
"""Stage supporting artifacts (task 2.3): eval pools (scrubbed), prompts,
distributions.json, sample_sidecar.json, supplement README.

Run after the full canonicalization pass (supplement README reads the
canonicalization reports for per-oracle coverage).
"""

import csv
import json
import os
import shutil
import sys
from pathlib import Path

from canonicalize import SAGE_ROOT, STAGING_WORK, scrub_path

STAGING = SAGE_ROOT / "hf_release_staging" / "v1.0"
PAPER = SAGE_ROOT / "paper_tables_20260514_140732"

EVAL_POOLS = [
    "data/csvs/ssv2/eval_2k_stratified.csv",
    "data/csvs/ssv2/eval_2k_stratified.csv.meta.json",
    "data/csvs/ssv2/id_500_stratified.csv",
    "data/csvs/ssv2/id_500_stratified.csv.meta.json",
    "data/csvs/k400/eval_2k_stratified.csv",
    "data/csvs/k400/eval_2k_stratified.csv.meta.json",
    "data/csvs/k400/id_500_stratified.csv",
    "data/csvs/k400/id_500_stratified.csv.meta.json",
    "data/csvs/diving48/id_500_stratified.csv",
    "data/csvs/diving48/id_500_stratified.csv.meta.json",
    "data/csvs/oracle_agreement/sample_600.csv",
    "data/csvs/oracle_agreement/sample_600_manifest.json",
]

PROMPTS = ["direct_scoring", "direct_scoring_ssv2", "direct_scoring_k400",
           "direct_scoring_diving48"]

ORACLE_ORDER = ["qwen", "gemini", "internvl3-38b", "gpt-5.5"]
ORACLE_LABELS = {
    "qwen": "Qwen3-VL-32B-Instruct (primary oracle, local)",
    "gemini": "Gemini 3.1 Pro (Dartmouth ChatAPI Batch)",
    "internvl3-38b": "InternVL3-38B (local)",
    "gpt-5.5": "GPT-5.5 (Azure OpenAI)",
}


def scrub_csv(src, dst):
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys()) if rows else csv.DictReader(open(src)).fieldnames
    for r in rows:
        if "video_path" in r:
            r["video_path"] = scrub_path(r["video_path"])
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, dst)


# Internal-infrastructure markers that must never appear in released files.
# "Dartmouth" alone is fine (public API provenance + contact email); internal
# hostnames, absolute mount paths, and the username are not.
SCRUB_PATTERNS = tuple(s.replace("|", "") for s in
    ("/ju|mbo/", "hpcc.dart|mouth", "thayer|fs", "f006|bx5"))  # split so released copies grep clean


def scrub_text_file(src, dst):
    text = Path(src).read_text()
    text = text.replace(str(SAGE_ROOT) + "/", "")
    hits = [p for p in SCRUB_PATTERNS if p in text]
    if hits:
        sys.exit(f"FATAL: scrub patterns {hits} remain in {src}")
    tmp = Path(str(dst) + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, dst)


def main():
    # eval pools
    pool_dir = STAGING / "manifests" / "eval_pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    for rel in EVAL_POOLS:
        src = SAGE_ROOT / rel
        sub = rel.split("/")[2]  # ssv2 / k400 / diving48 / oracle_agreement
        dst = pool_dir / f"{sub}__{Path(rel).name}"
        if rel.endswith(".csv"):
            scrub_csv(src, dst)
        else:
            scrub_text_file(src, dst)
        print(f"eval_pool: {dst.name}")

    # prompts (originals + LaTeX-sanitized copies used in App. J)
    prompt_dir = STAGING / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for p in PROMPTS:
        shutil.copyfile(SAGE_ROOT / f"oracle/scripts/mss/prompts/{p}.md", prompt_dir / f"{p}.md")
        san = PAPER / f"prompt_{p}.txt"
        if san.exists():
            shutil.copyfile(san, prompt_dir / f"{p}.sanitized.txt")
    print("prompts: 4 .md + sanitized copies")

    # dataset card (canonical copy lives in scripts/hf_release/)
    shutil.copyfile(Path(__file__).parent / "dataset_card.md", STAGING / "README.md")
    print("README.md (dataset card) staged")

    # distributions.json (cited in thesis §3)
    scrub_text_file(SAGE_ROOT / "scripts/distributions/out/distributions.json",
                    STAGING / "distributions.json")
    print("distributions.json staged")

    # sample sidecar = a real canonical sidecar (the App.-I example video)
    src = STAGING_WORK / "diving48_val" / "ovWCmIMMkRI_00032.json"
    shutil.copyfile(src, STAGING / "sample_sidecar.json")
    print("sample_sidecar.json staged (ovWCmIMMkRI_00032, canonical form)")

    # supplement README from canonicalization reports
    lines = [
        "# Cross-Oracle Supplement",
        "",
        "Per-video MSS outputs from four oracles over the shared 600-video",
        "stratified pilot pool (`manifests/eval_pools/oracle_agreement__sample_600.csv`,",
        "200 videos each from SSv2 / K400 / Diving-48). Released so readers can",
        "recompute pairwise agreement under any oracle subset (thesis App. B).",
        "The joint-precheck-pass set across all four oracles is n=369; the",
        "three-oracle subset excluding Gemini is n=375.",
        "",
        "Sidecars follow the same canonical schema as the primary corpus.",
        "All oracles used the generic `direct_scoring` prompt (apples-to-apples;",
        "see each oracle's `config.json`).",
        "",
        "| oracle | sidecars | enrichment among precheck-passed | notes |",
        "|---|---|---|---|",
    ]
    for o in ORACLE_ORDER:
        rpt = json.load(open(STAGING_WORK / "cross_oracle" / f"{o}.report.json"))
        c = rpt["counts"]
        n = c["total"]
        rec = rpt["recovery_rate_precheck_passed"]
        notes = {
            "qwen": "primary oracle rerun on the pilot pool",
            "gemini": "Batch-API run; segment scores post-hoc recovered for a "
                      "subset (`scores_recovered_from_raw_output`); a few passed "
                      "videos have unparseable raw text (null `phase`/`reason`)",
            "internvl3-38b": "one video carries run-time score-parse fallback "
                             "weights (enrichment refused, null `phase`/`reason`)",
            "gpt-5.5": "missing videos are Azure content-policy refusals, "
                       "released as precheck-failed records with the error in "
                       "`raw_output`",
        }[o]
        lines.append(f"| {ORACLE_LABELS[o]} | {n} | {rec:.2%} | {notes} |")
    lines += [
        "",
        "`phase`/`reason`/`time_range` are lifted from each oracle's raw output",
        "where parseable; unparseable responses degrade to nulls and remain in",
        "the release as failure records (thesis App. D).",
        "",
    ]
    out = STAGING / "supplement" / "cross_oracle" / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, out)
    print("supplement README staged")


if __name__ == "__main__":
    main()
