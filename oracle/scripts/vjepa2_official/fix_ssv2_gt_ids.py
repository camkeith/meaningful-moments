"""Post-fix SSv2 sidecars whose ground_truth_id is None due to runner missing
the text-label fallback. Resolves labels via data/SSv2/labels.json and
recomputes top1_correct / top5_correct in-place.

Idempotent: skips sidecars that already have a valid ground_truth_id.
"""
import argparse
import json
import os
from pathlib import Path

MM_ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[3]))

LABELS_JSON = MM_ROOT / "data" / "SSv2" / "labels.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    raw = json.loads(LABELS_JSON.read_text())
    text_to_id: dict[str, int] = {}
    for k, v in raw.items():
        text_to_id[k.strip().lower()] = int(v)

    fixed = skipped = unmapped = 0
    for p in args.run_dir.glob("*.json"):
        if p.name in ("config.json", "results.jsonl") or "summary_shard" in p.name:
            continue
        try:
            sc = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "ground_truth_id" not in sc:
            continue
        if sc.get("ground_truth_id") is not None:
            skipped += 1
            continue
        text = (sc.get("ground_truth") or "").strip()
        cleaned = text.replace("[", "").replace("]", "").strip().lower()
        gt = text_to_id.get(cleaned) or text_to_id.get(text.lower())
        if gt is None:
            unmapped += 1
            continue
        sc["ground_truth_id"] = int(gt)
        top5 = sc.get("top5_label_ids") or []
        sc["top1_correct"] = bool(top5 and top5[0] == gt)
        sc["top5_correct"] = bool(gt in top5)
        p.write_text(json.dumps(sc, sort_keys=True))
        fixed += 1

    print(f"fixed: {fixed}  skipped (already had gt_id): {skipped}  unmapped: {unmapped}")


if __name__ == "__main__":
    main()
