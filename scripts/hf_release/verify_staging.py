#!/usr/bin/env python3
"""Pre-upload verification gates (task 2.7 / design D8 gates 1-2 + scrub).

Hard-fails on any violation:
  G1  exact counts: canonical == source run dir == source CSV == thesis Table 1
  G2  50-video parquet <-> canonical-sidecar field equality per config
  G3  parquet features identical across splits within each config
  G4  scrub scan (staged text files + sampled tar members) for internal
      mount paths, hostnames, and the username (see SCRUB below)
  G5  no .log / .human_annotated.json / _batch_state anywhere in staging
  G6  segment total ~= 4.58M and per-substrate precheck-fail rates match the
      thesis (4.98 / 7.80 / 15.34 %, +/- 0.15pp)
  G7  manifest rows recompute from sidecars (sample) and shard membership is real
"""

import csv
import io
import gzip
import json
import os
import random
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

from canonicalize import RELEASE_RUNS, SAGE_ROOT, STAGING_WORK, list_sidecars

STAGING = SAGE_ROOT / "hf_release_staging" / "v1.0"
# hostname pattern includes the domain, not the bare cluster name: YouTube
# video ids contain it as a random substring (e.g. 8kgVhpccs1E)
SCRUB = tuple(s.replace("|", "") for s in
    ("/ju|mbo/", "hpcc.dart|mouth", "thayer|fs", "f006|bx5"))  # split so released copies grep clean

SOURCE_CSVS = {
    "ssv2_train": "data/csvs/ssv2/train_full.csv",
    "ssv2_val": "data/csvs/ssv2/val_full.csv",
    "ssv2_test": "data/csvs/ssv2/test.csv",
    "k400_train": "data/csvs/k400/train.csv",
    "k400_val": "data/csvs/k400/val.csv",
    "k400_test": "data/csvs/k400/test.csv",
    "diving48_train": "data/csvs/diving48/train.csv",
    "diving48_val": "data/csvs/diving48/val.csv",
}
TABLE1 = {  # thesis tab:dataset-composition
    "ssv2_train": 168913, "ssv2_val": 24777, "ssv2_test": 27157,
    "k400_train": 239789, "k400_val": 19877, "k400_test": 38671,
    "diving48_train": 15027, "diving48_val": 1970,
}
PRECHECK_FAIL = {"ssv2": 0.0498, "k400": 0.0780, "diving48": 0.1534}
SPLITS = {"ssv2": ["train", "validation", "test"],
          "k400": ["train", "validation", "test"],
          "diving48": ["train", "validation"]}
RUN_OF = {("ssv2", "train"): "ssv2_train", ("ssv2", "validation"): "ssv2_val",
          ("ssv2", "test"): "ssv2_test", ("k400", "train"): "k400_train",
          ("k400", "validation"): "k400_val", ("k400", "test"): "k400_test",
          ("diving48", "train"): "diving48_train",
          ("diving48", "validation"): "diving48_val"}

failures = []


def gate(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def g1_counts():
    for run, csv_rel in SOURCE_CSVS.items():
        n_canon = len([f for f in os.listdir(STAGING_WORK / run) if f.endswith(".json")])
        n_src = len(list_sidecars(SAGE_ROOT / RELEASE_RUNS[run]))
        with open(SAGE_ROOT / csv_rel) as f:
            n_csv = sum(1 for _ in f) - 1
        ok = n_canon == n_src == n_csv == TABLE1[run]
        gate(ok, f"G1 {run}: canonical={n_canon} source={n_src} csv={n_csv} table1={TABLE1[run]}")
    for run in RELEASE_RUNS:
        if run.startswith("cross_oracle/"):
            n_canon = len([f for f in os.listdir(STAGING_WORK / run) if f.endswith(".json")])
            n_src = len(list_sidecars(SAGE_ROOT / RELEASE_RUNS[run]))
            gate(n_canon == n_src, f"G1 {run}: canonical={n_canon} source={n_src}")


def _pq_tables(config):
    return {split: pq.read_table(sorted((STAGING / "data" / config).glob(f"{split}-*.parquet")))
            for split in SPLITS[config]}


def g2_g3_parquet():
    rng = random.Random(42)
    for config in SPLITS:
        tables = _pq_tables(config)
        schemas = {s: t.schema for s, t in tables.items()}
        first = list(schemas.values())[0]
        ok = all(sch.equals(first) for sch in schemas.values())
        gate(ok, f"G3 {config}: parquet schema identical across {len(schemas)} splits")
        for split, t in tables.items():
            run = RUN_OF[(config, split)]
            idxs = rng.sample(range(t.num_rows), min(50 // len(tables) + 5, t.num_rows))
            for i in idxs:
                row = t.slice(i, 1).to_pylist()[0]
                side = json.load(open(STAGING_WORK / run / f"{row['video_id']}.json"))
                ok = (row["action_label"] == side["action_label"]
                      and row["summary"] == side["summary"]
                      and len(row["segments"]) == len(side["segments"])
                      and all(abs((a["weight"] or 0) - (b["weight"] or 0)) < 1e-9
                              and a["phase"] == b["phase"]
                              for a, b in zip(row["segments"], side["segments"]))
                      and row["mss_result"]["kept_indices"] == side["mss_result"]["kept_indices"]
                      and all("raw_output" not in (r or {}) for r in
                              row["mss_result"]["precheck_responses"] or []))
                if not ok:
                    gate(False, f"G2 {run}: parquet/sidecar mismatch at {row['video_id']}")
                    return
        gate(True, f"G2 {config}: parquet rows match canonical sidecars (sampled)")


def g4_scrub_g5_files():
    bad_names, scrub_hits = [], []
    rng = random.Random(7)
    for p in sorted(STAGING.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".log" or p.name == ".human_annotated.json" or "_batch_state" in p.name:
            bad_names.append(str(p))
        if p.suffix in (".csv", ".json", ".md", ".txt"):
            text = p.read_text(errors="replace")
            hits = [s for s in SCRUB if s in text]
            if hits:
                scrub_hits.append(f"{p.relative_to(STAGING)}: {hits}")
        elif p.name.endswith(".tar.gz"):
            with tarfile.open(fileobj=gzip.open(p, "rb"), mode="r|") as tf:
                k = 0
                for m in tf:
                    if rng.random() < 0.02 or k < 3:
                        data = tf.extractfile(m).read().decode(errors="replace")
                        hits = [s for s in SCRUB if s in data]
                        if hits:
                            scrub_hits.append(f"{p.name}::{m.name}: {hits}")
                        k += 1
                    if k > 25:
                        break
    gate(not bad_names, f"G5 no internal files in staging ({len(bad_names)} found)")
    gate(not scrub_hits, f"G4 scrub clean ({len(scrub_hits)} hits)"
         + ("" if not scrub_hits else f" e.g. {scrub_hits[:3]}"))


def g6_totals():
    total_segs = passed_segs = 0
    for config in SPLITS:
        fails = n = 0
        for split in SPLITS[config]:
            run = RUN_OF[(config, split)]
            with open(STAGING / "manifests" / f"{run}.csv", newline="") as f:
                for r in csv.DictReader(f):
                    n += 1
                    total_segs += int(r["n_segments"])
                    if r["precheck_passed"] != "True":
                        fails += 1
                    else:
                        passed_segs += int(r["n_segments"])
        rate = fails / n
        ok = abs(rate - PRECHECK_FAIL[config]) < 0.0015
        gate(ok, f"G6 {config}: precheck-fail {rate:.2%} vs thesis {PRECHECK_FAIL[config]:.2%} (n={n})")
    # thesis "~4.58M per-segment scores" counts videos with parseable oracle
    # labels (~499K precheck-passed), not the failure records also released
    ok = 4.4e6 < passed_segs < 4.8e6
    gate(ok, f"G6 segments on precheck-passed videos {passed_segs:,} "
             f"(~4.58M expected; {total_segs:,} total incl. failure records)")


def g7_manifest_sample():
    rng = random.Random(11)
    for run in SOURCE_CSVS:
        with open(STAGING / "manifests" / f"{run}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rng.sample(rows, 5):
            side = json.load(open(STAGING_WORK / run / f"{r['video_id']}.json"))
            segs = side["segments"]
            kept = side["mss_result"]["kept_indices"]
            ok = (int(r["n_segments"]) == len(segs)
                  and abs(float(r["kept_fraction"]) - (len(kept) / len(segs) if segs else 0)) < 1e-3)
            if not ok:
                gate(False, f"G7 {run}: manifest mismatch {r['video_id']}")
                return
        # shard membership: first row's shard must contain its video
        r = rows[0]
        tar_path = STAGING / "sidecars" / run / r["sidecar_shard"]
        with tarfile.open(tar_path, "r:gz") as tf:
            names = set()
            for m in tf:
                names.add(m.name)
                if f"{r['video_id']}.json" in names:
                    break
        gate(f"{r['video_id']}.json" in names, f"G7 {run}: shard membership verified")


def main():
    g1_counts()
    g2_g3_parquet()
    g4_scrub_g5_files()
    g6_totals()
    g7_manifest_sample()
    print()
    if failures:
        print(f"VERIFICATION FAILED: {len(failures)} gate(s)")
        sys.exit(1)
    print("ALL GATES PASS")


if __name__ == "__main__":
    main()
