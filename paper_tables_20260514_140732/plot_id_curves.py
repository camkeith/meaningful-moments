"""I/D-curve panels: top-1 vs keep-fraction f, 5 orderings per substrate.

Reads per_fraction.csv from each id_curve_analysis dir; produces one
two-panel figure (ssv2, k400). Mirrors plot_threshold_curves.py style.
"""
from __future__ import annotations
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SAGE = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[1]))

SOURCES = {
    "SSv2 (V-JEPA 2 $2{\\times}3$ step=2, $n=500$)": (
        SAGE / "pseudo_labels/classifier_eval/ssv2_vjepa2-ssv2_id_20260520_210920/id_curve_analysis/per_fraction.csv",
        OUT / "ssv2_id_curves.pdf",
    ),
    "Kinetics-400 (VideoMAE-L single clip, $n=500$)": (
        SAGE / "pseudo_labels/classifier_eval/k400_videomae-k400-large_id_20260520_210920/id_curve_analysis/per_fraction.csv",
        OUT / "k400_id_curves.pdf",
    ),
    "Diving-48 (V-JEPA 2 HF-port, $n=500$)": (
        SAGE / "pseudo_labels/classifier_eval/diving48_vjepa2-diving48_id_20260520_223500/id_curve_analysis/per_fraction.csv",
        OUT / "d48_id_curves.pdf",
    ),
}

ORDER_STYLE = {
    "vlm":      ("#1f77b4", "o", "-",  "vlm (Importance-Led)"),
    "anti-vlm": ("#d62728", "v", "-",  "anti-vlm (Anti-Importance)"),
    "random":   ("#7f7f7f", "s", "--", "random"),
    "temporal": ("#9467bd", "D", "-.", "temporal (chronological)"),
    "motion":   ("#2ca02c", "^", ":",  "motion (flow-magnitude)"),
}


def load_curves(path: Path):
    """Return {ordering: ([fractions], [top1_means], [top5_means])}"""
    curves = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            ord_ = row["ordering"]
            f = float(row["fraction"]) * 100
            m = float(row["mean"]) * 100
            metric = row["metric"]
            curves.setdefault(ord_, {"f": [], "top1": [], "top5": []})
            d = curves[ord_]
            if metric == "top1":
                d["f"].append(f); d["top1"].append(m)
            else:
                d["top5"].append(m)
    return curves


def make_plot(title: str, csv_path: Path, out_path: Path):
    curves = load_curves(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for ax, metric, label in zip(axes, ["top1", "top5"], ["Top-1", "Top-5"]):
        for ord_ in ["vlm", "motion", "random", "temporal", "anti-vlm"]:
            d = curves[ord_]
            color, marker, ls, legend = ORDER_STYLE[ord_]
            ax.plot(d["f"], d[metric], color=color, marker=marker, linestyle=ls,
                    linewidth=1.5, markersize=5, label=legend)
        ax.set_xlabel(r"Keep-fraction $f$ (\%)")
        ax.set_ylabel(f"{label} accuracy (\\%)")
        ax.set_title(f"{label} vs $f$")
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if metric == "top1":
            ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    for title, (csv_path, out_path) in SOURCES.items():
        make_plot(title, csv_path, out_path)


if __name__ == "__main__":
    main()
