#!/usr/bin/env python3
"""Stage the recognizer-eval sidecar supplement.

Reads the committed enumeration (scripts/code_release/eval_sidecar_runs.json),
and for each cited run dir packs every file into deterministic tar shards with
a SHA-256 manifest. JSON files containing absolute internal paths get their
`video_path` scrubbed to substrate-relative (all other files are copied
byte-verbatim, so restaging is byte-identical).

Output tree (uploaded later to the HF dataset as a versioned addition):
  hf_release_staging/eval_supplement/<run-name>/sidecars-NNN.tar.gz
  hf_release_staging/eval_supplement/<run-name>/manifest.json
  hf_release_staging/eval_supplement/sha256sums-eval.txt
  hf_release_staging/eval_supplement/eval_sidecar_runs.json   (the enumeration)

Usage:
  python scripts/hf_release/stage_eval_sidecars.py            # non-provisional runs
  python scripts/hf_release/stage_eval_sidecars.py --include-provisional
  python scripts/hf_release/stage_eval_sidecars.py --finalize # write sha manifest
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from canonicalize import SAGE_ROOT, scrub_path
from stage import FILES_PER_TAR, add_det_member, open_det_tar

ENUM = SAGE_ROOT / "scripts/code_release/eval_sidecar_runs.json"
OUT = SAGE_ROOT / "hf_release_staging" / "eval_supplement"
SCRUB_MARKERS = tuple(s.replace("|", "") for s in
    ("/ju|mbo/", "hpcc.dart|mouth", "thayer|fs", "f006|bx5"))  # split so released copies grep clean


def run_name(entry):
    return entry["path"].rstrip("/").split("/")[-1] if not entry["path"].startswith(
        "pseudo_labels/cross_oracle_eval") else "cross_oracle_stage1"


def iter_files(root, metrics_only=False):
    keep_small = {"val_metrics.json", "config.json", "train_log.csv"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for f in sorted(filenames):
            if f.startswith(".") or f.endswith((".log", ".tmp", ".lock")):
                continue
            if metrics_only and f not in keep_small:
                continue
            p = Path(dirpath) / f
            yield p, str(p.relative_to(root))


_ROOT_PREFIX = str(SAGE_ROOT) + "/"


def _scrub_doc(doc):
    """Recursively rewrite any absolute under-root path to repo-relative;
    video_path additionally gets the substrate-relative treatment."""
    if isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, str):
                if v.startswith(_ROOT_PREFIX):
                    v = v[len(_ROOT_PREFIX):]
                if k == "video_path":
                    v = scrub_path(v)
                doc[k] = v
            elif isinstance(v, (dict, list)):
                _scrub_doc(v)
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            if isinstance(v, str) and v.startswith(_ROOT_PREFIX):
                doc[i] = v[len(_ROOT_PREFIX):]
            elif isinstance(v, (dict, list)):
                _scrub_doc(v)
    return doc


def _assert_clean(out_bytes, p):
    rem = [m for m in SCRUB_MARKERS if m in out_bytes.decode("utf-8", errors="replace")]
    if rem:
        sys.exit(f"FATAL: scrub markers {rem} remain in {p} after video_path scrub "
                 f"— extend scrub_bytes for this field")


def scrub_bytes(p, data):
    """Scrub absolute internal paths from JSON/JSONL; everything else must be clean."""
    text = data.decode("utf-8", errors="replace")
    if not any(m in text for m in SCRUB_MARKERS):
        return data
    if p.suffix == ".json":
        out = json.dumps(_scrub_doc(json.loads(data)), indent=1).encode()
        _assert_clean(out, p)
        return out
    if p.suffix == ".jsonl":
        out = ("\n".join(json.dumps(_scrub_doc(json.loads(line)))
                         for line in text.splitlines() if line.strip()) + "\n").encode()
        _assert_clean(out, p)
        return out
    sys.exit(f"FATAL: non-JSON file {p} contains scrub markers — handle explicitly")


def stage_entry(entry, force=False):
    name = run_name(entry)
    src = SAGE_ROOT / entry["path"]
    out_dir = OUT / name
    done = out_dir / ".staged"
    if done.exists() and not force:
        print(f"[{name}] already staged, skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(iter_files(src, metrics_only=entry.get("metrics_only", False)))
    if not files:
        sys.exit(f"FATAL [{name}]: no files found under {src}")

    sha_lines, n_scrubbed = [], 0
    n_tars = (len(files) + FILES_PER_TAR - 1) // FILES_PER_TAR
    for t in range(n_tars):
        chunk = files[t * FILES_PER_TAR:(t + 1) * FILES_PER_TAR]
        tar_path = out_dir / f"sidecars-{t:03d}.tar.gz"
        tmp = tar_path.with_suffix(".gz.tmp")
        tf, gz, raw = open_det_tar(tmp)
        for p, rel in chunk:
            data = p.read_bytes()
            scrubbed = scrub_bytes(p, data)
            if scrubbed is not data:
                n_scrubbed += 1
            add_det_member(tf, rel, scrubbed)
            sha_lines.append(f"{hashlib.sha256(scrubbed).hexdigest()}  {name}/{rel}")
        tf.close(); gz.close(); raw.close()
        os.replace(tmp, tar_path)
    manifest = {
        "source_dir": entry["path"],
        "cited_by": entry["cited_by"],
        "n_files": len(files),
        "n_scrubbed": n_scrubbed,
        "n_tars": n_tars,
        "metrics_only": entry.get("metrics_only", False),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    sha_frag = out_dir / ".sha_fragment"
    sha_frag.write_text("\n".join(sha_lines) + "\n")
    done.touch()
    print(f"[{name}] staged: {len(files)} files, {n_tars} tars, {n_scrubbed} scrubbed",
          flush=True)


def finalize():
    parts = ["# Meaningful Moments eval-sidecar supplement — SHA-256 manifest",
             "# Paths as <run>/<relative-path>, matching tar extraction into <run>/"]
    for d in sorted(OUT.iterdir()):
        frag = d / ".sha_fragment"
        if d.is_dir() and frag.exists():
            parts.append(frag.read_text().rstrip("\n"))
    (OUT / "sha256sums-eval.txt").write_text("\n".join(parts) + "\n")
    (OUT / "eval_sidecar_runs.json").write_bytes(ENUM.read_bytes())
    print(f"sha256sums-eval.txt written "
          f"({(OUT / 'sha256sums-eval.txt').stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-provisional", action="store_true")
    ap.add_argument("--only", help="stage a single run name")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    entries = json.load(open(ENUM))["runs"]
    for e in entries:
        if e.get("provisional") and not args.include_provisional:
            continue
        if args.only and run_name(e) != args.only:
            continue
        stage_entry(e, force=args.force)
    if args.finalize:
        finalize()


if __name__ == "__main__":
    main()
