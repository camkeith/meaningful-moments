"""Render three dataset-composition figures for the appendix:

  Figure 1: Per-class MM-label-count histogram, log-x, one panel per substrate.
  Figure 2: Per-segment weight-distribution histogram, one panel per substrate.
  Figure 3: Source vs MM per-class scatter (parity), one panel per substrate.

Reads the committed pre-extracted data at scripts/distributions/dataset_figures_v2.json
(provenance: scripts/distributions/DATASET_FIGURES_V2_PROVENANCE.md).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
MM_ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[1]))
DATA = json.load(open(MM_ROOT / "scripts/distributions/dataset_figures_v2.json"))

SUBS = ["SSv2", "K400", "Diving-48"]
COLOR_BAR = "#4477aa"
COLOR_REF = "#bbbbbb"


def make_class_hist(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    for ax, name in zip(axes, SUBS):
        d = DATA[name]
        counts = sorted(d["mm_per_class"].values())
        n_classes = len(counts)
        bins = np.logspace(np.log10(max(1, min(counts))),
                            np.log10(max(counts) * 1.05),
                            20)
        ax.hist(counts, bins=bins, color=COLOR_BAR, edgecolor="white", linewidth=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("MM labels per class")
        ax.set_ylabel("# classes")
        ax.set_title(f"{name} ({n_classes} classes)")
        # Median line
        med = float(np.median(counts))
        ax.axvline(med, color="#cc4422", linestyle="--", linewidth=1.0,
                   label=f"median = {int(med)}")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.suptitle("Per-class MM-label coverage (train split)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_weight_hist(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    for ax, name in zip(axes, SUBS):
        d = DATA[name]
        weights = d["weight_values"]
        bins = np.linspace(0.0, 1.0, 21)  # 20 bins of 0.05
        ax.hist(weights, bins=bins, color=COLOR_BAR, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Importance weight")
        ax.set_ylabel("# segments")
        ax.set_xlim(0.0, 1.0)
        n_w = len(weights)
        med = float(np.median(weights))
        mean = float(np.mean(weights))
        ax.set_title(f"{name} ({n_w:,} segs sampled)")
        ax.axvline(med, color="#cc4422", linestyle="--", linewidth=1.0,
                   label=f"med={med:.2f}")
        ax.axvline(mean, color="#22aa44", linestyle=":", linewidth=1.0,
                   label=f"mean={mean:.2f}")
        ax.legend(loc="upper center", fontsize=7.5, framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.suptitle("Per-segment importance-weight distributions",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_parity_scatter(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, name in zip(axes, SUBS):
        d = DATA[name]
        keys = sorted(set(d["src_per_class"]) | set(d["mm_per_class"]))
        src = np.array([d["src_per_class"].get(k, 0) for k in keys])
        mm = np.array([d["mm_per_class"].get(k, 0) for k in keys])
        # Plot
        ax.scatter(src, mm, s=18, alpha=0.7, color=COLOR_BAR, edgecolor="white",
                   linewidth=0.4)
        # Diagonal parity
        lo = max(1, min(src.min(), mm.min()))
        hi = max(src.max(), mm.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color=COLOR_REF, linestyle="--", linewidth=0.9,
                label="parity (MM = src)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Source videos per class")
        ax.set_ylabel("MM labels per class")
        ratio = mm.sum() / src.sum() if src.sum() else 0.0
        ax.set_title(f"{name} (coverage = {100*ratio:.1f}%)")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    fig.suptitle("Per-class source $\\to$ MM coverage parity",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    make_class_hist(OUT / "fig_class_coverage_hist.pdf")
    make_weight_hist(OUT / "fig_weight_distribution.pdf")
    make_parity_scatter(OUT / "fig_source_mm_parity.pdf")


if __name__ == "__main__":
    main()
