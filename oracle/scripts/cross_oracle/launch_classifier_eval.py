#!/usr/bin/env python
"""Stage 1 — fan out run_classifier_eval.py across N GPUs.

For the 9 (oracle × dataset) cells × 5 conditions = 45 jobs, dispatched to
GPUs 4-7 (default) via a simple work-queue: one subprocess per GPU at any
time, each pulling the next pending job. Each invocation of
run_classifier_eval.py is idempotent (skips already-done sidecars), so
re-running is safe.

Usage:
    python -m oracle.scripts.cross_oracle.launch_classifier_eval \\
        --gpus 4,5,6,7 --batch-size 4

Output: pseudo_labels/cross_oracle_eval/by_oracle/<oracle>_<dataset>/
        with one <video_id>__<condition>.json per (video, condition).
"""

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

from ._common import (
    CLASSIFIER_CONDITIONS,
    DATASETS,
    MSS_DIRS,
    ORACLES,
    OUT_ROOT,
    PILOT_CSVS,
    RECOGNIZERS,
    REPO_ROOT,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("cross_oracle.stage1")


def build_jobs() -> List[Tuple[str, str, str]]:
    """Return list of (oracle, dataset, condition) jobs to run."""
    jobs = []
    for oracle in ORACLES:
        for ds in DATASETS:
            for cond in CLASSIFIER_CONDITIONS:
                jobs.append((oracle, ds, cond))
    return jobs


def cell_outdir(oracle: str, dataset: str) -> Path:
    return OUT_ROOT / "by_oracle" / f"{oracle}_{dataset}"


def is_cell_complete(oracle: str, dataset: str, condition: str, n_pilot: int) -> bool:
    """Cheap pre-check: skip the subprocess if we already have N sidecars
    matching <vid>__<condition>.json. Lets us re-run the launcher without
    waiting for run_classifier_eval's own resume scan."""
    out = cell_outdir(oracle, dataset)
    if not out.exists():
        return False
    found = list(out.glob(f"*__{condition}.json"))
    return len(found) >= n_pilot


def make_command(
    oracle: str, dataset: str, condition: str, batch_size: int, log_level: str
) -> List[str]:
    out = cell_outdir(oracle, dataset)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "oracle.scripts.run_classifier_eval",
        "--recognizer", RECOGNIZERS[dataset],
        "--eval-csv", str(PILOT_CSVS[dataset]),
        "--output-dir", str(out),
        "--condition", condition,
        "--batch-size", str(batch_size),
        "--device", "cuda:0",   # per-process CUDA_VISIBLE_DEVICES makes index 0 = our chosen GPU
        "--log-level", log_level,
    ]
    if condition != "full":
        cmd += ["--mss-run-dir", str(MSS_DIRS[(oracle, dataset)])]
    return cmd


def run_one(
    gpu: int,
    job: Tuple[str, str, str],
    batch_size: int,
    log_dir: Path,
    log_level: str,
) -> Tuple[Tuple[str, str, str], int, float]:
    oracle, dataset, condition = job
    cmd = make_command(oracle, dataset, condition, batch_size, log_level)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH','')}"

    log_path = log_dir / f"{oracle}_{dataset}__{condition}.gpu{gpu}.log"
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    return job, proc.returncode, time.time() - t0


def worker(
    gpu: int,
    job_q: "queue.Queue[Tuple[str,str,str]]",
    results: list,
    batch_size: int,
    log_dir: Path,
    log_level: str,
    n_pilot_by_ds: dict,
) -> None:
    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return
        oracle, dataset, condition = job
        if is_cell_complete(oracle, dataset, condition, n_pilot_by_ds[dataset]):
            log.info(f"[gpu{gpu}] SKIP   {oracle:>6s} × {dataset:<8s} × {condition:<16s} (already complete)")
            results.append((job, 0, 0.0, "skipped"))
            job_q.task_done()
            continue
        log.info(f"[gpu{gpu}] START  {oracle:>6s} × {dataset:<8s} × {condition:<16s}")
        out_job, rc, elapsed = run_one(gpu, job, batch_size, log_dir, log_level)
        status = "ok" if rc == 0 else f"failed(rc={rc})"
        log.info(
            f"[gpu{gpu}] DONE   {oracle:>6s} × {dataset:<8s} × {condition:<16s}  "
            f"{elapsed:>6.1f}s  {status}"
        )
        results.append((out_job, rc, elapsed, status))
        job_q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus", default="4,5,6,7",
        help="Comma-separated GPU indices to use (default: 4,5,6,7).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "logs/cross_oracle_eval/stage1",
        help="Per-cell stdout/stderr captures.",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        help="--log-level passed to run_classifier_eval.py (default: WARNING to keep cell logs short).",
    )
    parser.add_argument(
        "--only-pending", action="store_true",
        help="Don't even queue cells that already have N pilot sidecars.",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Substring filter on '<oracle>_<dataset>' to limit cells (e.g. 'qwen' or 'k400').",
    )
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    log.info(f"Using GPUs: {gpus}")

    # Pilot sizes per dataset (200 in all cases, but lookup just in case).
    import csv
    n_pilot_by_ds = {}
    for ds in DATASETS:
        with open(PILOT_CSVS[ds]) as f:
            n_pilot_by_ds[ds] = sum(1 for _ in csv.DictReader(f))
    log.info(f"Pilot sizes: {n_pilot_by_ds}")

    jobs = build_jobs()
    if args.filter:
        f = args.filter
        jobs = [j for j in jobs if f in f"{j[0]}_{j[1]}" or f == j[2]]
    if args.only_pending:
        jobs = [
            j for j in jobs
            if not is_cell_complete(j[0], j[1], j[2], n_pilot_by_ds[j[1]])
        ]
    log.info(f"Queueing {len(jobs)} jobs across {len(gpus)} GPUs")

    job_q: "queue.Queue[Tuple[str,str,str]]" = queue.Queue()
    for j in jobs:
        job_q.put(j)
    results: list = []

    threads = []
    for gpu in gpus:
        t = threading.Thread(
            target=worker,
            args=(gpu, job_q, results, args.batch_size, args.log_dir, args.log_level, n_pilot_by_ds),
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Summary
    n_ok = sum(1 for _, rc, _, _ in results if rc == 0)
    n_skipped = sum(1 for _, _, _, s in results if s == "skipped")
    n_failed = sum(1 for _, rc, _, _ in results if rc != 0)
    log.info(f"=== summary === ok={n_ok} skipped={n_skipped} failed={n_failed}")
    if n_failed:
        log.error("Some jobs failed. See per-cell logs in {}".format(args.log_dir))
        for job, rc, _, status in results:
            if status != "ok" and status != "skipped":
                log.error(f"  failed: {job} -> {status}")
        sys.exit(1)

    summary_path = OUT_ROOT / "by_oracle" / "_stage1_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            [
                {
                    "oracle": j[0], "dataset": j[1], "condition": j[2],
                    "rc": rc, "elapsed_s": round(elapsed, 1), "status": status,
                }
                for j, rc, elapsed, status in results
            ],
            f, indent=2,
        )
    log.info(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
