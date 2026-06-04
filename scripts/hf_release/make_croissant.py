#!/usr/bin/env python3
"""Generate croissant.json (task 2.5).

Hand-authored Croissant 1.0 JSON-LD: one FileObject per released file with its
SHA-256 (cross-referencing sha256sums.txt, per thesis App. S), one FileSet +
RecordSet per parquet config, license, and version. Validated with mlcroissant
after generation.

Run AFTER stage.py --finalize-sha (reads section 2 of sha256sums.txt).

  venv python scripts/hf_release/make_croissant.py --repo-id <ns>/meaningful-moments
"""

import argparse
import json
import os
import re
from pathlib import Path

from canonicalize import SAGE_ROOT

STAGING = SAGE_ROOT / "hf_release_staging" / "v1.0"

CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "sc": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileObject": "cr:fileObject",
    "fileProperty": "cr:fileProperty",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}

CONFIGS = {"ssv2": ["train", "validation", "test"],
           "k400": ["train", "validation", "test"],
           "diving48": ["train", "validation"]}

SCALAR_FIELDS = [
    ("video_id", "sc:Text", "Substrate-native video identifier"),
    ("action_label", "sc:Text", "Substrate class string for this video"),
    ("video_path", "sc:Text", "Substrate-relative source video path"),
    ("summary", "sc:Text", "Oracle's one-paragraph description of the video"),
    ("elapsed_s", "sc:Float", "Annotation wall-clock seconds"),
]


def read_released_hashes():
    """path -> sha256 from section 2 of sha256sums.txt."""
    hashes = {}
    in_s2 = False
    for line in (STAGING / "sha256sums.txt").read_text().splitlines():
        if line.startswith("# Section 2"):
            in_s2 = True
            continue
        if not in_s2 or line.startswith("#") or not line.strip():
            continue
        h, path = re.split(r"\s+", line, maxsplit=1)
        hashes[path.strip()] = h
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="<namespace>/meaningful-moments")
    args = ap.parse_args()
    base_url = f"https://huggingface.co/datasets/{args.repo_id}/resolve/main"

    hashes = read_released_hashes()
    distribution = [{
        "@type": "cr:FileObject",
        "@id": "repo",
        "name": "repo",
        "description": "The Hugging Face git repository.",
        "contentUrl": f"https://huggingface.co/datasets/{args.repo_id}",
        "encodingFormat": "git+https",
        "sha256": "https://github.com/mlcommons/croissant/issues/80",
    }]
    fmt = {".parquet": "application/x-parquet", ".gz": "application/gzip",
           ".csv": "text/csv", ".json": "application/json", ".md": "text/markdown",
           ".txt": "text/plain"}
    for path, h in sorted(hashes.items()):
        if path == "croissant.json":
            continue
        distribution.append({
            "@type": "cr:FileObject",
            "@id": f"file/{path}",
            "name": path,
            "contentUrl": f"{base_url}/{path}",
            "encodingFormat": fmt.get(Path(path).suffix, "application/octet-stream"),
            "sha256": h,
        })
    record_sets = []
    for config, splits in CONFIGS.items():
        distribution.append({
            "@type": "cr:FileSet",
            "@id": f"parquet/{config}",
            "name": f"parquet-{config}",
            "description": f"Parquet shards for the {config} config "
                           f"(splits: {', '.join(splits)}).",
            "containedIn": {"@id": "repo"},
            "encodingFormat": "application/x-parquet",
            "includes": f"data/{config}/*.parquet",
        })
        fields = [{
            "@type": "cr:Field",
            "@id": f"{config}/{name}",
            "name": name,
            "description": desc,
            "dataType": dtype,
            "source": {"fileSet": {"@id": f"parquet/{config}"},
                       "extract": {"column": name}},
        } for name, dtype, desc in SCALAR_FIELDS]
        record_sets.append({
            "@type": "cr:RecordSet",
            "@id": f"records/{config}",
            "name": config,
            "description":
                f"One record per {config} video. Nested columns (segments "
                "list with per-segment weight/phase/reason, mss_result with "
                "kept_indices and precheck responses) are documented in the "
                "dataset card and present in the parquet schema.",
            "field": fields,
        })

    croissant = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "name": "meaningful-moments",
        "description":
            "Meaningful Moments (MM): ~4.58M per-segment temporal-importance "
            "pseudo-labels over 536,181 videos and 622 action classes across "
            "SSv2, Kinetics-400, and Diving-48, produced by a Qwen3-VL-32B "
            "oracle with single-call direct scoring. Labels only; source "
            "videos are obtained from the upstream substrates.",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": f"https://huggingface.co/datasets/{args.repo_id}",
        "version": "1.0.0",
        "datePublished": "2026-06-03",
        "citeAs": "Keith, Cameron. Meaningful Moments: VLM-Derived Per-Segment "
                  "Importance Labels for Video Action Recognition. 2026. v1.0.",
        "distribution": distribution,
        "recordSet": record_sets,
    }
    out = STAGING / "croissant.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(croissant, indent=1))
    os.replace(tmp, out)
    print(f"croissant.json written: {len(distribution) - 1} file objects/sets, "
          f"{len(record_sets)} record sets")

    import mlcroissant as mlc
    ds = mlc.Dataset(jsonld=str(out))
    n_issues = len([i for i in (ds.metadata.issues.errors or [])])
    print(f"mlcroissant validation: {'OK' if not n_issues else f'{n_issues} ERRORS'}")
    if n_issues:
        for e in list(ds.metadata.issues.errors)[:5]:
            print("  -", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
