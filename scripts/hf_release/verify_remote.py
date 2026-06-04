#!/usr/bin/env python3
"""Remote verification gates against the uploaded (private) repo (task 3.3).

  venv python scripts/hf_release/verify_remote.py --repo-id <ns>/meaningful-moments

  R1  load_dataset round-trips every config; split sizes match Table 1
  R2  100 random sidecars downloaded from tars verify against sha256sums.txt
  R3  croissant.json retrievable and mlcroissant-valid
  R4  remote file listing has no .log / .human_annotated.json / _batch_state
"""

import argparse
import gzip
import hashlib
import io
import random
import sys
import tarfile

from huggingface_hub import HfApi, hf_hub_download

TABLE1 = {("ssv2", "train"): 168913, ("ssv2", "validation"): 24777,
          ("ssv2", "test"): 27157, ("k400", "train"): 239789,
          ("k400", "validation"): 19877, ("k400", "test"): 38671,
          ("diving48", "train"): 15027, ("diving48", "validation"): 1970}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    args = ap.parse_args()
    api = HfApi()

    # R4 first (cheap): listing
    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    bad = [f for f in files if f.endswith(".log") or ".human_annotated" in f
           or "_batch_state" in f]
    gate(not bad, f"R4 remote listing clean ({len(files)} files, {len(bad)} bad)")

    # R1: load_dataset split sizes
    from datasets import load_dataset
    for (config, split), n in TABLE1.items():
        ds = load_dataset(args.repo_id, config, split=split)
        gate(len(ds) == n, f"R1 {config}/{split}: {len(ds)} rows (expect {n})")

    # R2: sidecar hash spot-check — download 2 tars, verify 100 members total
    sha_path = hf_hub_download(args.repo_id, "sha256sums.txt", repo_type="dataset")
    expected = {}
    for line in open(sha_path):
        if line.startswith("#") or not line.strip():
            continue
        h, p = line.split(None, 1)
        expected[p.strip()] = h
    rng = random.Random(42)
    tars = [f for f in files if f.startswith("sidecars/") and f.endswith(".tar.gz")]
    checked = mismatched = 0
    for tar_rel in rng.sample(tars, 2):
        run = tar_rel.split("/")[1]
        local = hf_hub_download(args.repo_id, tar_rel, repo_type="dataset")
        with tarfile.open(local, "r:gz") as tf:
            members = tf.getmembers()
            for m in rng.sample(members, min(50, len(members))):
                data = tf.extractfile(m).read()
                h = hashlib.sha256(data).hexdigest()
                if expected.get(f"{run}/{m.name}") != h:
                    mismatched += 1
                checked += 1
    gate(mismatched == 0, f"R2 sidecar hashes: {checked} checked, {mismatched} mismatched")

    # R3: croissant valid
    cro = hf_hub_download(args.repo_id, "croissant.json", repo_type="dataset")
    import mlcroissant as mlc
    ds = mlc.Dataset(jsonld=cro)
    errs = list(ds.metadata.issues.errors or [])
    gate(not errs, f"R3 croissant.json mlcroissant-valid ({len(errs)} errors)")

    print()
    if failures:
        print(f"REMOTE VERIFICATION FAILED: {len(failures)} gate(s) — repo stays private")
        sys.exit(1)
    print("ALL REMOTE GATES PASS — ready to flip public (then check viewer, tag v1.0)")


if __name__ == "__main__":
    main()
