"""Render the per-class kept-fraction figure for the appendix:

  One panel per substrate: histogram of per-class MEDIAN kept-fraction over
  precheck-passed train-split videos, with the global median (of class
  medians) as a dashed line.

Reads the Meaningful Moments release manifests (kept_fraction per video), so
the figure is fully reproducible from released data:
  hf_release_staging/v1.0/manifests/{ssv2,k400,diving48}_train.csv locally,
  or the same files from the HF dataset repo.

Replaces a hand-era figure whose exact computation was not preserved
(2026-06-03 release revision); numbers shift slightly vs the original
(SSv2 median 0.58->0.57, K400 389->400 classes, D48 0.64->0.67).
"""
from __future__ import annotations
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
MM_ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[1]))
MANIFEST_DIR = Path(os.environ.get(
    "MM_MANIFEST_DIR", MM_ROOT / "hf_release_staging/v1.0/manifests"))

COLOR_BAR = "#4477aa"

SUBS = [
    ("SSv2", "ssv2_train.csv", "template"),
    ("K400", "k400_train.csv", "action_label"),
    ("Diving-48", "diving48_train.csv", "raw_class_name"),
]


def class_medians(manifest, label_col):
    by_class = defaultdict(list)
    with open(MANIFEST_DIR / manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r["precheck_passed"] != "True":
                continue
            cls = r.get(label_col) or r["action_label"]
            by_class[cls].append(float(r["kept_fraction"]))
    return [statistics.median(v) for v in by_class.values()]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    for ax, (sub, manifest, label_col) in zip(axes, SUBS):
        meds = class_medians(manifest, label_col)
        gmed = statistics.median(meds)
        ax.hist(meds, bins=np.arange(0, 1.05, 0.05), color=COLOR_BAR)
        ax.axvline(gmed, color="0.3", ls="--", lw=1,
                   label=f"global median = {gmed:.2f}")
        ax.set_title(f"{sub} ({len(meds)} classes)")
        ax.set_xlabel("Per-class median kept-fraction")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3, ls=":")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("# classes")
    fig.suptitle("Per-class kept-fraction medians (how aggressive is the oracle, by class)",
                 fontsize=9)
    fig.tight_layout()
    out_path = OUT / "fig_per_class_kept.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
