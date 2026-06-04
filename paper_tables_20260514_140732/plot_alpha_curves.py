"""Generate the three α-curve plots embedded in clean_story.tex.

Numbers are pasted from the verified sidecar scans (see the tables in
clean_story.tex). Single-file dependency: matplotlib + numpy.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

POS_COLOR = "#1f77b4"
NEG_COLOR = "#d62728"
FULL_COLOR = "#555555"


# ─── data ────────────────────────────────────────────────────────────────
# Top-1 accuracy at each α, for vlm-fastforward (pos) and lowest-evidence-fastforward (neg).
# α=0 = hard cut: pos = vlm-selected, neg = lowest-evidence.
# α=1 = full (shared endpoint).

D48 = {
    "title": r"Diving-48 ($4{\times}3$ paper protocol)",
    "full": 87.82,
    "alpha":  np.array([0.0,  0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8,   0.9,   1.0]),
    "vlm_ff": np.array([82.08, 83.60, 85.18, 85.08, 86.90, 88.07, 87.82, 88.38, 88.27, 88.27, 87.82]),
    "neg_ff": np.array([52.13, 58.32, 66.55, 66.80, 79.19, 84.01, 86.35, 88.17, 88.27, 88.38, 87.82]),
    "ylim": (48, 93),
}

SSV2 = {
    "title": r"SSv2 ($2{\times}3$, step=2 protocol)",
    "full": 65.77,
    "alpha":  np.array([0.0,   0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8,   0.9,   1.0]),
    "vlm_ff": np.array([62.45, 65.53, 65.53, 65.53, 64.04, 63.89, 63.89, 61.44, 61.15, 59.86, 65.77]),
    "neg_ff": np.array([53.41, 66.68, 66.68, 66.68, 64.66, 64.62, 64.62, 62.50, 62.60, 59.86, 65.77]),
    "ylim": (50, 70),
}

K400 = {
    "title": "Kinetics-400 (single-clip protocol)",
    "full": 83.95,
    "alpha":  np.array([0.0,   0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8,   0.9,   1.0]),
    "vlm_ff": np.array([85.40, 83.95, 83.90, 83.50, 83.35, 83.25, 83.30, 83.25, 83.30, 83.45, 83.95]),
    "neg_ff": np.array([80.15, 83.45, 83.55, 83.75, 83.80, 83.70, 83.55, 83.35, 83.45, 83.30, 83.95]),
    "ylim": (78, 87),
}


def make_plot(data: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.3))

    ax.axhline(data["full"], color=FULL_COLOR, linestyle=":", linewidth=1.2,
               label=f"full = {data['full']:.2f}%", zorder=1)
    ax.plot(data["alpha"], data["vlm_ff"], marker="o", markersize=5, linewidth=1.6,
            color=POS_COLOR, label="vlm-fastforward", zorder=3)
    ax.plot(data["alpha"], data["neg_ff"], marker="s", markersize=5, linewidth=1.6,
            color=NEG_COLOR, label="lowest-evidence-FF (neg ctrl)", zorder=3)

    # Endpoint gap annotation
    idx0 = np.where(data["alpha"] == 0.0)[0]
    if len(idx0):
        i = idx0[0]
        y_pos = data["vlm_ff"][i]
        y_neg = data["neg_ff"][i]
        gap = y_pos - y_neg
        ax.annotate(
            f"gap = {gap:+.2f} pp",
            xy=(0.0, (y_pos + y_neg) / 2),
            xytext=(0.12, (y_pos + y_neg) / 2),
            fontsize=8, va="center", color="black",
            arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.6),
        )

    ax.set_xlabel(r"$\alpha$  (density floor for unimportant segments)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(data["title"])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(*data["ylim"])
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    make_plot(D48, OUT / "d48_curve.pdf")
    make_plot(SSV2, OUT / "ssv2_curve.pdf")
    make_plot(K400, OUT / "k400_curve.pdf")


if __name__ == "__main__":
    main()
