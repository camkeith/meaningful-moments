#!/usr/bin/env python3
"""Assemble the HF staging tree from canonical sidecars (tasks 2.1-2.4).

One pass per release run: each canonical sidecar is read once and fans out to
  - a deterministic tar.gz shard (sorted video_id, <=10k members, pinned
    mtime/uid/gid, gzip mtime=0)               -> sidecars/<run>/  (production)
                                                  supplement/cross_oracle/<oracle>/
  - a parquet row (production runs only; raw_output omitted; fixed nullable
    schema per config so features match across splits)  -> data/<config>/
  - a manifest CSV row                                   -> manifests/
  - a sha256 line (pre-tar bytes)                        -> sha256sums.txt

Refuses to stage a run whose canonical dir is incomplete vs its source run dir.

Usage:
  venv python scripts/hf_release/stage.py --runs diving48_val cross_oracle/qwen
  venv python scripts/hf_release/stage.py --all
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from canonicalize import RELEASE_RUNS, SAGE_ROOT, STAGING_WORK, list_sidecars

STAGING = SAGE_ROOT / "hf_release_staging" / "v1.0"
CONFIGS = SAGE_ROOT / "hf_release_staging" / "v1.0_work" / "configs"
SHA_WORK = SAGE_ROOT / "hf_release_staging" / "v1.0_work" / "sha"

FILES_PER_TAR = 10_000
ROWS_PER_PARQUET = 100_000
PINNED_MTIME = 1767225600  # 2026-01-01T00:00:00Z

# release_name -> (config, hf split) for parquet; supplement runs are tars-only
PARQUET_SPLITS = {
    "ssv2_train": ("ssv2", "train"),
    "ssv2_val": ("ssv2", "validation"),
    "ssv2_test": ("ssv2", "test"),
    "k400_train": ("k400", "train"),
    "k400_val": ("k400", "validation"),
    "k400_test": ("k400", "test"),
    "diving48_train": ("diving48", "train"),
    "diving48_val": ("diving48", "validation"),
}

_PRECHECK_STRUCT = pa.struct([
    ("decision", pa.string()), ("confidence", pa.float64()),
    ("evidence", pa.string()), ("rationale", pa.string()),
    ("logit_yes", pa.float64()), ("logit_no", pa.float64()),
    ("logit_skip", pa.float64()), ("logit_confidence", pa.float64()),
    ("logit_p_skip", pa.float64()),
])

_BASE_FIELDS = [
    ("video_id", pa.string()), ("action_label", pa.string()),
    ("video_path", pa.string()), ("timestamp", pa.string()),
    ("summary", pa.string()),
    ("label_counts", pa.struct([
        ("total_segments", pa.int64()), ("important_count", pa.int64()),
        ("unimportant_count", pa.int64()), ("important_ratio", pa.float64()),
        ("unimportant_ratio", pa.float64()),
    ])),
    ("segments", pa.list_(pa.struct([
        ("index", pa.int32()), ("time_range", pa.string()),
        ("start_s", pa.float64()), ("end_s", pa.float64()),
        ("label", pa.string()), ("weight", pa.float64()),
        ("phase", pa.string()), ("reason", pa.string()),
    ]))),
    ("mss_result", pa.struct([
        ("kept_indices", pa.list_(pa.int32())),
        ("total_oracle_calls", pa.int64()),
        ("precheck_passed", pa.bool_()),
        ("precheck_vote_yes", pa.float64()),  # vote fraction, not a bool
        ("scores_recovered_from_raw_output", pa.bool_()),
        ("precheck_responses", pa.list_(_PRECHECK_STRUCT)),
    ])),
    ("elapsed_s", pa.float64()),
]

# fixed nullable extras per config, identical across that config's splits
_CONFIG_EXTRAS = {
    "ssv2": [("template", pa.string()), ("placeholders", pa.string())],
    "k400": [],
    "diving48": [("class_id", pa.int32()), ("raw_class_name", pa.string())],
}


def schema_for(config):
    return pa.schema(_BASE_FIELDS + _CONFIG_EXTRAS[config])


def parquet_row(doc, config):
    mss = doc["mss_result"]
    row = {
        "video_id": doc.get("video_id"),
        "action_label": doc.get("action_label"),
        "video_path": doc.get("video_path"),
        "timestamp": doc.get("timestamp"),
        "summary": doc.get("summary"),
        "label_counts": doc.get("label_counts"),
        "segments": doc.get("segments"),
        "mss_result": {
            "kept_indices": mss.get("kept_indices"),
            "total_oracle_calls": mss.get("total_oracle_calls"),
            "precheck_passed": mss.get("precheck_passed"),
            "precheck_vote_yes": mss.get("precheck_vote_yes"),
            "scores_recovered_from_raw_output": mss.get("scores_recovered_from_raw_output"),
            "precheck_responses": [
                {k: r.get(k) for k, _ in zip(
                    ("decision", "confidence", "evidence", "rationale",
                     "logit_yes", "logit_no", "logit_skip",
                     "logit_confidence", "logit_p_skip"),
                    range(9))}
                for r in (mss.get("precheck_responses") or [])
            ],
        },
        "elapsed_s": doc.get("elapsed_s"),
    }
    meta = doc.get("metadata") or {}
    if config == "ssv2":
        row["template"] = meta.get("template")
        ph = meta.get("placeholders")
        row["placeholders"] = ph if isinstance(ph, str) or ph is None else json.dumps(ph)
    elif config == "diving48":
        row["class_id"] = meta.get("class_id")
        row["raw_class_name"] = meta.get("raw_class_name")
    return row


def manifest_row(doc, shard_name, config):
    segs = doc.get("segments") or []
    kept = doc["mss_result"].get("kept_indices") or []
    prs = doc["mss_result"].get("precheck_responses") or []
    row = {
        "video_id": doc.get("video_id"),
        "action_label": doc.get("action_label"),
        "n_segments": len(segs),
        "precheck_passed": doc["mss_result"].get("precheck_passed"),
        "precheck_decision": prs[0].get("decision") if prs else None,
        "kept_fraction": round(len(kept) / len(segs), 4) if segs else 0.0,
        "sidecar_shard": shard_name,
    }
    meta = doc.get("metadata") or {}
    if config == "ssv2":
        row["template"] = meta.get("template")
        row["placeholders"] = meta.get("placeholders")
    elif config == "diving48":
        row["class_id"] = meta.get("class_id")
        row["raw_class_name"] = meta.get("raw_class_name")
    return row


def open_det_tar(path):
    """Deterministic tar.gz writer: pinned gzip mtime + member metadata."""
    raw = open(path, "wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    tf = tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT)
    return tf, gz, raw


def add_det_member(tf, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = PINNED_MTIME
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    tf.addfile(info, io.BytesIO(data))


def stage_run(release_name, force=False):
    canon_dir = STAGING_WORK / release_name
    src_dir = SAGE_ROOT / RELEASE_RUNS[release_name]
    names = sorted(f for f in os.listdir(canon_dir) if f.endswith(".json"))
    expected = list_sidecars(src_dir)
    if names != expected:
        sys.exit(f"FATAL [{release_name}]: canonical dir incomplete "
                 f"({len(names)} vs {len(expected)} source sidecars) — "
                 f"finish canonicalization first")

    is_supplement = release_name.startswith("cross_oracle/")
    config_split = PARQUET_SPLITS.get(release_name)
    if is_supplement:
        out_dir = STAGING / "supplement" / release_name
    else:
        out_dir = STAGING / "sidecars" / release_name
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir = (out_dir if is_supplement else STAGING / "manifests")
    man_dir.mkdir(parents=True, exist_ok=True)
    SHA_WORK.mkdir(parents=True, exist_ok=True)

    n_tars = (len(names) + FILES_PER_TAR - 1) // FILES_PER_TAR
    tar_names = [f"sidecars-{i:03d}.tar.gz" for i in range(n_tars)]
    done_marker = SHA_WORK / f"{release_name.replace('/', '__')}.done"
    if done_marker.exists() and not force:
        print(f"[{release_name}] already staged, skipping")
        return

    # parquet writers (production only)
    writers, pq_paths = [], []
    if config_split:
        config, split = config_split
        pq_dir = STAGING / "data" / config
        pq_dir.mkdir(parents=True, exist_ok=True)
        n_pq = (len(names) + ROWS_PER_PARQUET - 1) // ROWS_PER_PARQUET
        pq_paths = [pq_dir / f"{split}-{i:05d}-of-{n_pq:05d}.parquet" for i in range(n_pq)]
        schema = schema_for(config)

    sha_lines, man_rows, pq_batch = [], [], []
    pq_idx = 0
    writer = None

    def flush_parquet(final=False):
        nonlocal pq_batch, pq_idx, writer
        while pq_batch and (len(pq_batch) >= ROWS_PER_PARQUET or final):
            chunk, pq_batch = pq_batch[:ROWS_PER_PARQUET], pq_batch[ROWS_PER_PARQUET:]
            table = pa.Table.from_pylist(chunk, schema=schema)
            pq.write_table(table, pq_paths[pq_idx], compression="snappy")
            pq_idx += 1
            if final and not pq_batch:
                break

    for t_i in range(n_tars):
        chunk = names[t_i * FILES_PER_TAR:(t_i + 1) * FILES_PER_TAR]
        tar_path = out_dir / tar_names[t_i]
        tmp = tar_path.with_suffix(".gz.tmp")
        tf, gz, raw = open_det_tar(tmp)
        for name in chunk:
            data = (canon_dir / name).read_bytes()
            add_det_member(tf, name, data)
            sha_lines.append(f"{hashlib.sha256(data).hexdigest()}  {release_name}/{name}")
            doc = json.loads(data)
            cfg = config_split[0] if config_split else release_name.split("/")[0]
            man_rows.append(manifest_row(doc, tar_names[t_i], cfg if config_split else "supplement"))
            if config_split:
                pq_batch.append(parquet_row(doc, config_split[0]))
        tf.close(); gz.close(); raw.close()
        os.replace(tmp, tar_path)
        if config_split:
            flush_parquet()
        print(f"  [{release_name}] tar {t_i + 1}/{n_tars}", flush=True)
    if config_split:
        flush_parquet(final=True)

    # manifest
    man_path = (out_dir / "manifest.csv") if is_supplement \
        else (man_dir / f"{release_name}.csv")
    fields = list(man_rows[0].keys())
    tmpm = man_path.with_suffix(".csv.tmp")
    with open(tmpm, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(man_rows)
    os.replace(tmpm, man_path)

    # released config.json
    cfg_src = CONFIGS / release_name / "config.json"
    (out_dir / "config.json").write_bytes(cfg_src.read_bytes())

    # per-run sha fragment (concatenated by finalize)
    sha_path = SHA_WORK / f"{release_name.replace('/', '__')}.sha"
    tmps = sha_path.with_suffix(".sha.tmp")
    tmps.write_text("\n".join(sha_lines) + "\n")
    os.replace(tmps, sha_path)
    done_marker.touch()
    print(f"[{release_name}] staged: {len(names)} sidecars, {n_tars} tars, "
          f"{pq_idx} parquet, manifest -> {man_path.name}", flush=True)


def finalize_sha():
    """Concatenate per-run fragments + hash every released file -> sha256sums.txt."""
    parts = ["# Meaningful Moments v1.0 — SHA-256 manifest",
             "# Section 1: canonical per-video sidecars (paths as <run>/<video_id>.json,",
             "#   matching extraction of sidecars/<run>/sidecars-*.tar.gz into <run>/)"]
    for name in sorted(RELEASE_RUNS):
        frag = SHA_WORK / f"{name.replace('/', '__')}.sha"
        if not frag.exists():
            sys.exit(f"FATAL: missing sha fragment for {name} — stage it first")
        parts.append(frag.read_text().rstrip("\n"))
    parts.append("# Section 2: released files")
    # croissant.json is excluded: it embeds these hashes (would be circular);
    # it carries its own integrity via the repo's git history instead.
    skip = {"sha256sums.txt", "croissant.json"}
    for p in sorted(STAGING.rglob("*")):
        if p.is_file() and p.name not in skip:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            parts.append(f"{h}  {p.relative_to(STAGING)}")
    out = STAGING / "sha256sums.txt"
    tmp = out.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(parts) + "\n")
    os.replace(tmp, out)
    print(f"sha256sums.txt written ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], choices=sorted(RELEASE_RUNS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--finalize-sha", action="store_true",
                    help="write sha256sums.txt after all runs are staged")
    args = ap.parse_args()
    targets = sorted(RELEASE_RUNS) if args.all else args.runs
    for name in targets:
        stage_run(name, force=args.force)
    if args.finalize_sha:
        finalize_sha()


if __name__ == "__main__":
    main()
