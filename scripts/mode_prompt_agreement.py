#!/usr/bin/env python3
"""Recompute the mode/prompt-ablation agreement tables (thesis App. A.2/A.3/A.5).

This is the ORIGINAL measure, recovered verbatim from the 2026-05-01 analysis
session that produced the thesis tables (reproduces every table value exactly):

  - kept set       = segments with label == "important" (no precheck filter)
  - Jaccard        = |A∩B| / |A∪B| over videos where at least one arm kept
                     anything (both-empty videos are EXCLUDED, not scored 1.0)
  - Spearman       = rank correlation over the weight values at segment indices
                     present in both arms (>=2 common indices), computed with
                     sort-rank ties (NO tie correction — ties get distinct ranks
                     by stable sort order). NOTE: on the greedy arms' coarse
                     weights this differs substantially from a tie-corrected
                     Spearman (e.g. scipy ~0.17 vs 0.71 here); the thesis values
                     use THIS definition. Reported per pair as the per-video mean.
  - n per pair     = videos with parseable sidecars in both arms (A.3 n=94-97)

Pinned arms (all on the 100-video SSv2 sample data/csvs/ssv2/val_sample.csv;
the arm dirs each carry ~101 sidecars):
  A.2  four independent greedy runs (Feb 03/10/11/12)
  A.3  each greedy arm vs the production direct run (ssv2_val, ssv2 prompt)
  A.5  mode-vs-prompt: all four greedy arms pooled per-video against F16
       direct-generic (pseudo_labels/f16_pilot_snapshot/run2_*, snapshot of
       /scratch2) and against the production direct-ssv2 run, plus
       F16 <-> production. A.5 pairs additionally require BOTH arms
       precheck-passed (unlike A.2/A.3), per the original 2026-05-20 analysis;
       pooled means are over videos, not over arm-means. (The snapshot's
       sidecars_20260520_122042 dir is a failed first attempt of the F16 run —
       empty mss_runs — and is not an arm.)

Usage:
  python scripts/mode_prompt_agreement.py --out <out-dir>
"""
import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

MM_ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[1]))

GREEDY_ARMS = {  # thesis App. A.2 arms A-D
    "greedyA_feb03": "pseudo_labels/mss/qwen3-vl-32b_20260203_161421",
    "greedyB_feb10": "pseudo_labels/mss/qwen3-vl-32b_20260210_145039",
    "greedyC_feb11": "pseudo_labels/mss/qwen3-vl-32b_20260211_231732",
    "greedyD_feb12": "pseudo_labels/mss/qwen3-vl-32b_20260212_115325",
}
DIRECT = "pseudo_labels/mss/qwen3-vl-32b_ssv2_val_20260430_134831"
F16_GENERIC = "pseudo_labels/f16_pilot_snapshot/run2_20260520_122951/qwen3-vl-32b_20260520_122955"


def vids_of(rel):
    d = MM_ROOT / rel
    return sorted(f.stem for f in d.iterdir()
                  if f.suffix == ".json" and "config" not in f.name
                  and "metadata" not in f.name and not f.name.startswith("."))


def loadv(rel, v):
    p = MM_ROOT / rel / f"{v}.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def extract(j):
    kept, weights = set(), {}
    for s in j.get("segments", []):
        idx = s.get("index", s.get("segment_id"))
        if idx is None:
            continue
        weights[idx] = s.get("weight", 0.0)
        if s.get("label") == "important":
            kept.add(idx)
    return kept, weights, j.get("mss_result", {}).get("total_oracle_calls", -1)


def jacc(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def spear(xs, ys):
    """Original sort-rank Spearman (NO tie correction) — see module docstring."""
    n = len(xs)
    if n < 2:
        return None
    rx, ry = [0] * n, [0] * n
    for ri, i in enumerate(sorted(range(n), key=lambda i: xs[i])):
        rx[i] = ri + 1
    for ri, i in enumerate(sorted(range(n), key=lambda i: ys[i])):
        ry[i] = ri + 1
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((r - mx) ** 2 for r in rx) ** 0.5
    dy = sum((r - my) ** 2 for r in ry) ** 0.5
    return num / (dx * dy) if dx * dy else None


def pair_lists(rel_a, rel_b, vids, require_both_passed=False):
    """Per-video jaccard/spearman lists (for pooling) + per-pair aggregates."""
    js, sps, ka_, kb_, ca_, cb_ = [], [], [], [], [], []
    n = 0
    for v in vids:
        a, b = loadv(rel_a, v), loadv(rel_b, v)
        if not a or not b:
            continue
        if require_both_passed:
            if not (a.get("mss_result", {}).get("precheck_passed")
                    and b.get("mss_result", {}).get("precheck_passed")):
                continue
        n += 1
        ka, wa, ca = extract(a)
        kb, wb, cb = extract(b)
        if ka or kb:
            js.append(jacc(ka, kb))
        ci = sorted(set(wa) & set(wb))
        if len(ci) >= 2:
            s = spear([wa[i] for i in ci], [wb[i] for i in ci])
            if s is not None:
                sps.append(s)
        ka_.append(len(ka)); kb_.append(len(kb)); ca_.append(ca); cb_.append(cb)
    return js, sps, {
        "n": n,
        "jaccard": statistics.mean(js) if js else None,
        "spearman": statistics.mean(sps) if sps else None,
        "kept_a": statistics.mean(ka_) if ka_ else None,
        "kept_b": statistics.mean(kb_) if kb_ else None,
        "calls_a": statistics.mean(ca_) if ca_ else None,
        "calls_b": statistics.mean(cb_) if cb_ else None,
    }


def pair_stats(rel_a, rel_b, vids, **kw):
    return pair_lists(rel_a, rel_b, vids, **kw)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    vids = vids_of(next(iter(GREEDY_ARMS.values())))

    report = {"a2_intra_greedy": {}, "a3_greedy_vs_direct": {}, "a5_mode_vs_prompt": {}}
    for (na, ra), (nb, rb) in itertools.combinations(GREEDY_ARMS.items(), 2):
        report["a2_intra_greedy"][f"{na}__{nb}"] = pair_stats(ra, rb, vids)
    a2 = list(report["a2_intra_greedy"].values())
    report["a2_summary"] = {
        "mean_jaccard": statistics.mean(p["jaccard"] for p in a2),
        "mean_spearman": statistics.mean(p["spearman"] for p in a2),
    }
    pooled_j, pooled_s = [], []
    for na, ra in GREEDY_ARMS.items():
        st = pair_stats(ra, DIRECT, vids)
        st["speedup"] = st["calls_a"] / st["calls_b"] if st["calls_b"] else None
        report["a3_greedy_vs_direct"][na] = st
        pooled_j.append(st["jaccard"]); pooled_s.append(st["spearman"])
    report["a3_summary"] = {"pooled_jaccard": statistics.mean(pooled_j),
                            "pooled_spearman": statistics.mean(pooled_s)}
    # A.5: pooled-over-videos across all four greedy arms, both-passed filter
    for direct_name, direct_rel in (("directGeneric", F16_GENERIC), ("directSSv2", DIRECT)):
        pooled_j, pooled_s = [], []
        for na, ra in GREEDY_ARMS.items():
            js, sps, st = pair_lists(ra, direct_rel, vids, require_both_passed=True)
            pooled_j.extend(js); pooled_s.extend(sps)
            report["a5_mode_vs_prompt"][f"{na}__{direct_name}"] = st
        report["a5_mode_vs_prompt"][f"POOLED_greedy__{direct_name}"] = {
            "n": len(pooled_j),
            "jaccard": statistics.mean(pooled_j) if pooled_j else None,
            "spearman": statistics.mean(pooled_s) if pooled_s else None,
        }
    report["a5_mode_vs_prompt"]["directGeneric__directSSv2"] = pair_stats(
        F16_GENERIC, DIRECT, vids, require_both_passed=True)

    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"a2_summary": report["a2_summary"],
                      "a3_summary": report["a3_summary"],
                      "a5": {k: {"n": v["n"], "jaccard": round(v["jaccard"], 3)}
                             for k, v in report["a5_mode_vs_prompt"].items()}}, indent=1))
    print(f"full report -> {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
