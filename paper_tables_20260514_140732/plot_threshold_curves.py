"""Generate threshold + budget sweep curves for the α=0 sweep section.

Two side-by-side panels per dataset:
  Left:  Top-1 vs Score-Threshold T (vlm-score-threshold-tNN)
  Right: Top-1 vs Budget f% (id-vlm-fNN)

Reference lines on each panel: full, vlm-selected, lowest-evidence.

Numbers pasted from verified sidecar scans (see clean_story.tex tables).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

THRESH_COLOR = "#1f77b4"
BUDGET_COLOR = "#2ca02c"
FULL_COLOR = "#000000"
VLM_COLOR = "#d62728"
LE_COLOR = "#bbbbbb"


# ─── data ────────────────────────────────────────────────────────────────
D48 = {
    "title": r"Diving-48 ($4{\times}3$, paper checkpoint, $n=1970$)",
    "full": 87.82,
    "vlm_selected": 82.08,
    "lowest_evidence": 52.13,
    # Threshold sweep: keep segments with weight >= T/100
    "T": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    "T_top1": np.array([88.58, 88.38, 87.31, 86.95, 85.74, 85.03, 83.71, 82.23, 76.65, 64.77, 37.56]),
    # Budget sweep: keep top-f% by weight
    "f": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
    "f_top1": np.array([41.52, 57.11, 68.98, 76.60, 82.34, 84.77, 85.99, 87.66, 87.87]),
    "ylim": (32, 92),
}

SSV2 = {
    "title": r"SSv2 ($2{\times}3$ step=$2$, paper checkpoint, $n=2080$)",
    "full": 65.77,
    "vlm_selected": 62.45,
    "lowest_evidence": 53.41,
    "T": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    "T_top1": np.array([59.86, 59.90, 60.10, 62.45, 63.32, 63.37, 62.84, 62.36, 57.88, 43.99, 35.67]),
    "f": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
    "f_top1": np.array([37.31, 47.60, 55.24, 60.34, 62.69, 63.32, 63.65, 63.32, 60.77]),
    "ylim": (32, 70),
}

K400 = {
    "title": r"Kinetics-400 (VideoMAE-L single-clip, $n=2000$)",
    "full": 83.78,
    "vlm_selected": 85.40,
    "lowest_evidence": 80.15,
    "T": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    "T_top1": np.array([83.95, 84.00, 84.15, 84.40, 84.65, 84.30, 85.00, 86.00, 84.85, 82.20, 81.95]),
    "f": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
    "f_top1": np.array([83.55, 85.20, 85.75, 85.50, 85.55, 85.30, 85.00, 84.95, 84.85]),
    "ylim": (79, 88),
}


def make_plot(data: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5), sharey=True)

    # ── threshold panel ────────────────────────────────────────────────
    ax = axes[0]
    ax.axhline(data["full"], color=FULL_COLOR, linestyle="-", linewidth=1.8,
               label=f"full = {data['full']:.2f}%", zorder=2)
    ax.axhline(data["vlm_selected"], color=VLM_COLOR, linestyle="--", linewidth=1.0,
               label=f"vlm-selected = {data['vlm_selected']:.2f}%", zorder=1, alpha=0.75)
    ax.axhline(data["lowest_evidence"], color=LE_COLOR, linestyle="--", linewidth=1.0,
               label=f"lowest-evidence = {data['lowest_evidence']:.2f}%", zorder=1, alpha=0.75)
    ax.plot(data["T"], data["T_top1"], marker="o", markersize=5, linewidth=1.6,
            color=THRESH_COLOR, label="threshold sweep", zorder=3)

    # Mark T=0 peak
    i = int(np.argmax(data["T_top1"]))
    ax.annotate(
        f"T={data['T'][i]}: {data['T_top1'][i]:.2f}%",
        xy=(data["T"][i], data["T_top1"][i]),
        xytext=(data["T"][i] + 14, data["T_top1"][i] + 1.4),
        fontsize=8, color="black",
        arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.6),
    )

    ax.set_xlabel(r"Score-threshold $T$  (keep segments with weight$\geq T/100$)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Threshold axis")
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_ylim(*data["ylim"])
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.95)

    # ── budget panel ───────────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(data["full"], color=FULL_COLOR, linestyle="-", linewidth=1.8,
               label=f"full = {data['full']:.2f}%", zorder=2)
    ax.axhline(data["vlm_selected"], color=VLM_COLOR, linestyle="--", linewidth=1.0,
               label=f"vlm-selected = {data['vlm_selected']:.2f}%", zorder=1, alpha=0.75)
    ax.axhline(data["lowest_evidence"], color=LE_COLOR, linestyle="--", linewidth=1.0,
               label=f"lowest-evidence = {data['lowest_evidence']:.2f}%", zorder=1, alpha=0.75)
    ax.plot(data["f"], data["f_top1"], marker="^", markersize=5, linewidth=1.6,
            color=BUDGET_COLOR, label="budget sweep", zorder=3)

    # Mark f=90 peak
    i = int(np.argmax(data["f_top1"]))
    ax.annotate(
        f"f={data['f'][i]}%: {data['f_top1'][i]:.2f}%",
        xy=(data["f"][i], data["f_top1"][i]),
        xytext=(data["f"][i] - 28, data["f_top1"][i] - 7),
        fontsize=8, color="black",
        arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.6),
    )

    ax.set_xlabel(r"Budget $f$  (keep top-$f\%$ of segments by weight)")
    ax.set_title("Budget axis")
    ax.set_xlim(6, 94)
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95)

    fig.suptitle(data["title"], fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    make_plot(D48, OUT / "d48_threshold_budget.pdf")
    make_plot(SSV2, OUT / "ssv2_threshold_budget.pdf")
    make_plot(K400, OUT / "k400_threshold_budget.pdf")


if __name__ == "__main__":
    main()
