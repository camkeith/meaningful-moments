"""Recompute EPS / TDS / Δ_shuf paired-bootstrap CIs for K400 and D48 from
the recognizer-eval sidecars referenced by exp5_reference_results.json.

Top-1 only (matches exp5 schema). Parallelized JSON reads across threads —
the bottleneck is filesystem latency on jumbo, not CPU. Writes results JSON
next to the script.
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle/scripts"))
from eval_paired_stats import paired_diff_bootstrap, mcnemar

REPO = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[1]))
RUNS = {
    "k400":     REPO / "pseudo_labels/classifier_eval/k400_videomae-k400-large_20260505_025747",
    "diving48": REPO / "pseudo_labels/classifier_eval/diving48_vjepa2-official_20260506_194702",
}
NEEDED = ("vlm-selected", "uniform", "lowest-evidence", "vlm-selected--shuffled", "uniform--shuffled")


def _read_top1(path: Path):
    """Return (video_id, condition, top1_correct, selector_failed) or None.

    Sidecars where selector_failed=True are returned with selector_failed=True
    so the matrix builder can drop the affected video from the paired subset.
    The exp5 convention is "n_succ paired" — exclude any video where the
    selector for that condition couldn't be applied.
    """
    stem = path.stem
    i = stem.rfind("__")
    if i < 0:
        return None
    vid, cond = stem[:i], stem[i + 2:]
    if cond not in NEEDED:
        return None
    try:
        with open(path) as f:
            r = json.load(f)
    except Exception:
        return None
    return vid, cond, r.get("top1_correct"), bool(r.get("selector_failed", False))


def load_top1_matrix(run_dir: Path, conditions, max_workers: int = 32):
    """Load per-video × per-condition top1_correct, applying the n_succ
    convention: any (video, condition) where selector_failed=True is dropped,
    and the final paired matrix only includes videos where ALL required
    conditions succeeded.
    """
    files = [p for p in run_dir.glob("*__*.json")]
    print(f"  scanning {len(files)} sidecars in {run_dir.name}", flush=True)

    by_video = defaultdict(dict)
    failures_by_cond: dict = defaultdict(int)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(_read_top1, files, chunksize=64)):
            if res is None:
                continue
            vid, cond, c, failed = res
            if c is None:
                continue
            if failed:
                failures_by_cond[cond] += 1
                continue  # exclude selector_failed from the paired subset
            by_video[vid][cond] = int(c)
            if (i + 1) % 5000 == 0:
                print(f"    read {i+1}/{len(files)}", flush=True)

    if failures_by_cond:
        print("  selector_failed counts (excluded from paired subset):", flush=True)
        for cond, n in sorted(failures_by_cond.items()):
            print(f"    {cond:<28} {n}", flush=True)

    keep = sorted(v for v in by_video if all(c in by_video[v] for c in conditions))
    M = np.array([[by_video[v][c] for c in conditions] for v in keep], dtype=np.int8)
    return keep, M


def main():
    out = {}
    for ds, run_dir in RUNS.items():
        print(f"\n=== {ds} ===", flush=True)
        vids, M = load_top1_matrix(run_dir, NEEDED)
        n = M.shape[0]
        print(f"  n_videos with all 4 conditions: {n}", flush=True)
        cell = {"n": n}
        # Conditions: 0=vlm-selected, 1=uniform, 2=lowest-evidence,
        #             3=vlm-selected--shuffled, 4=uniform--shuffled
        for key, vlm_i, base_i in [("eps", 0, 2), ("tds", 0, 1)]:
            b = paired_diff_bootstrap(M, vlm_i, base_i, replicates=10000, seed=42)
            mc = mcnemar(M, vlm_i, base_i)
            cell[key] = {
                "diff":   round(b["mean_diff"], 4),
                "ci_lo":  round(b["diff_ci_lo"], 4),
                "ci_hi":  round(b["diff_ci_hi"], 4),
                "p":      mc["p_value"],
            }
            print(f"  {key:>11}: {b['mean_diff']:+.4f}  CI=[{b['diff_ci_lo']:+.4f}, {b['diff_ci_hi']:+.4f}]  p_mcnemar={mc['p_value']:.4f}", flush=True)

        # delta_shuf := (acc_uniform - acc_uniform--shuffled)
        #             - (acc_vlm-selected - acc_vlm-selected--shuffled)
        #            ≡ damage(uniform) - damage(vlm-selected)
        per_video = (M[:, 1].astype(np.int32) - M[:, 4].astype(np.int32)) \
                  - (M[:, 0].astype(np.int32) - M[:, 3].astype(np.int32))
        rng = np.random.default_rng(42)
        replicates = 10000
        boot = np.empty(replicates)
        for r in range(replicates):
            idx = rng.integers(0, n, size=n)
            boot[r] = per_video[idx].mean()
        diff = float(per_video.mean())
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        # Two-sided p via fraction of bootstrap samples crossing zero on the
        # opposite side from the point estimate, doubled.
        if diff >= 0:
            p_boot = 2.0 * float((boot <= 0).mean())
        else:
            p_boot = 2.0 * float((boot >= 0).mean())
        p_boot = min(p_boot, 1.0)
        cell["delta_shuf"] = {
            "diff":  round(diff, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "p":     p_boot,
        }
        print(f"   delta_shuf: {diff:+.4f}  CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  p_boot={p_boot:.4f}", flush=True)
        out[ds] = cell
    out_path = Path(__file__).resolve().with_suffix(".out.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
