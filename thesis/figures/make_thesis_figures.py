#!/usr/bin/env python3
"""Generate the five deck-ported thesis figures from first sources.

Run from the repository root:
    python3 thesis/figures/make_thesis_figures.py

Inputs
------
- thesis/Dartmouth Senior Thesis Presentation.pptx
    Position-ordered example stills (dataset montage, precheck exemplars,
    banana segment frames). Picture order within a slide is recovered from
    the slide XML x/y offsets, so left-to-right matches the slide's labels.
- pseudo_labels/mss/qwen3-vl-32b_k400_test_20260502_122107/
      ZtmchZLnVDo_000022_000032.json
    Teaser sidecar (K400 "barbequing"): per-segment weights + binary labels.
- data/k400/full/test/ZtmchZLnVDo_000022_000032.mp4
    Teaser source video; frames extracted at segment midpoints via
    /usr/bin/ffmpeg (the PATH ffmpeg on this box is ancient).
- pseudo_labels/mss/qwen3-vl-32b_20260203_161421/69417.json
    Greedy-removal trace for the walkthrough (February greedy pilot).
- pseudo_labels/thesis_candidates/PRESENTATION_VIDEOS/
      03_methodology_greedy_vs_direct/69417_DIRECT_metadata.json
    Direct-scoring labels for the same video (alpha diagram + captions).

Outputs (thesis/figures/)
-------------------------
fig_teaser.pdf, fig_dataset_examples.pdf, fig_greedy_walkthrough.pdf,
fig_alpha_sampling.pdf, fig_precheck_examples.pdf
"""
import json
import re
import subprocess
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis/figures"
WORK = Path("/tmp/thesis_assets")
PPTX = ROOT / "thesis/Dartmouth Senior Thesis Presentation.pptx"
FFMPEG = "/usr/bin/ffmpeg"
GREEN, RED = "#2ca02c", "#d62728"

TEASER_SIDECAR = ROOT / (
    "pseudo_labels/mss/qwen3-vl-32b_k400_test_20260502_122107/"
    "ZtmchZLnVDo_000022_000032.json"
)
TEASER_VIDEO = ROOT / "data/k400/full/test/ZtmchZLnVDo_000022_000032.mp4"
GREEDY_SIDECAR = ROOT / "pseudo_labels/mss/qwen3-vl-32b_20260203_161421/69417.json"
DIRECT_META = ROOT / (
    "pseudo_labels/thesis_candidates/PRESENTATION_VIDEOS/"
    "03_methodology_greedy_vs_direct/69417_DIRECT_metadata.json"
)

# Slide-text label order (matches left-to-right picture order on each slide).
MONTAGE_ROWS = [
    ("SSv2", "ssv2", [
        "Pretending to put something\nunderneath something",
        "Putting something onto a slanted\nsurface but it doesn't glide down",
        "Showing that something\nis empty",
        "Pushing something so that it\nalmost falls off but doesn't",
    ]),
    ("K400", "k400", ["Shooting goal", "Surfing", "Playing drums", "Barbecuing"]),
    ("Diving-48", "d48", [
        "Forward 1.5som NoTwis PIKE", "Back 2.5som NoTwis TUCK",
        "Reverse 1.5som 2.5Twis FREE", "Inward 2.5som NoTwis PIKE",
    ]),
]
PRECHECK_ROWS = [
    "driving tractor  (K400)",
    "trimming or shaving beard  (K400)",
    "Forward 2.5som 2Twis PIKE  (Diving-48)",
]
SLIDE_GROUPS = {18: "d48", 19: "ssv2", 20: "k400", 23: "banana", 31: "precheck"}


def pics_in_order(z, slide_no):
    xml = z.read(f"ppt/slides/slide{slide_no}.xml").decode("utf-8", "ignore")
    rels = z.read(f"ppt/slides/_rels/slide{slide_no}.xml.rels").decode()
    rid2file = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="\.\./media/([^"]+)"', rels))
    out = []
    for pic in re.findall(r"<p:pic>.*?</p:pic>", xml, re.S):
        m = re.search(r'r:embed="(rId\d+)"', pic)
        off = re.search(r'<a:off x="(\d+)" y="(\d+)"', pic)
        if m and off and m.group(1) in rid2file:
            out.append((int(off.group(1)), int(off.group(2)), rid2file[m.group(1)]))
    out.sort(key=lambda t: (t[1] // 400000, t[0]))  # row-major
    uniq, seen = [], set()
    for _, _, f in out:
        if f not in seen and f != "image1.png":  # skip shared background
            seen.add(f)
            uniq.append(f)
    return uniq


def extract_assets():
    z = zipfile.ZipFile(PPTX)
    for sn, name in SLIDE_GROUPS.items():
        d = WORK / name
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(pics_in_order(z, sn)):
            p = d / f"{i:02d}_{f}"
            if not p.exists():
                p.write_bytes(z.read(f"ppt/media/{f}"))
            if p.suffix == ".gif":  # three stills per example gif
                im = Image.open(p)
                for tag, idx in [("a", im.n_frames // 4), ("b", im.n_frames // 2),
                                 ("c", 3 * im.n_frames // 4)]:
                    sp = p.with_name(p.stem + f"_{tag}.png")
                    if not sp.exists():
                        im.seek(idx)
                        fr = im.convert("RGB")
                        fr.thumbnail((480, 480))
                        fr.save(sp)


def extract_teaser_frames(n_segments):
    d = WORK / "teaser_bbq"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n_segments):
        p = d / f"seg{i:02d}.png"
        if not p.exists():
            subprocess.run(
                [FFMPEG, "-loglevel", "error", "-ss", f"{i}.5", "-i",
                 str(TEASER_VIDEO), "-frames:v", "1", "-vf", "scale=320:-2",
                 "-y", str(p)], check=True)
    return d


def frame_ax(ax, img_path, edge, lw=3.0):
    ax.imshow(Image.open(img_path))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edge)
        s.set_linewidth(lw)


def build_teaser():
    d = json.load(open(TEASER_SIDECAR))
    weights = [s["weight"] for s in d["segments"]]
    labels = ["i" if s["label"] == "important" else "u" for s in d["segments"]]
    frames = extract_teaser_frames(len(weights))
    fig, axes = plt.subplots(1, len(weights), figsize=(13, 2.0))
    for i, ax in enumerate(axes):
        frame_ax(ax, frames / f"seg{i:02d}.png", GREEN if labels[i] == "i" else RED)
        ax.set_title(f"{i}–{i + 1}s", fontsize=8, pad=2)
        ax.set_xlabel(f"{weights[i]:.1f}", fontsize=10, fontweight="bold",
                      color=GREEN if labels[i] == "i" else RED, labelpad=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_teaser.pdf", bbox_inches="tight")
    plt.close(fig)


def allocate(labels, alpha, n_frames, durs):
    """Mirror of adapter.py:_decode_weighted_kept_frames_pyav allocation."""
    d = [1.0 if l == "i" else alpha for l in labels]
    w = [du * dk for du, dk in zip(durs, d)]
    raw = [n_frames * wk / sum(w) for wk in w]
    alloc = [round(r) for r in raw]
    diff = n_frames - sum(alloc)
    if diff:
        rem = sorted(range(len(raw)),
                     key=lambda k: -(raw[k] - alloc[k]) if diff > 0 else (raw[k] - alloc[k]))
        for i in range(abs(diff)):
            alloc[rem[i % len(rem)]] += 1 if diff > 0 else -1
    for k in range(len(alloc)):  # every segment keeps at least one frame
        if alloc[k] < 1:
            big = max(range(len(alloc)), key=lambda i: alloc[i])
            if alloc[big] > 1:
                alloc[big] -= 1
                alloc[k] = 1
    return d, alloc


def banana_frames():
    return sorted((WORK / "banana").glob("0[1-8]_*.png"))


def build_alpha(direct_labels):
    frames = banana_frames()
    durs = [0.5] * len(direct_labels)
    d_pos, a_pos = allocate(direct_labels, 0.25, 16, durs)
    inv = ["i" if l == "u" else "u" for l in direct_labels]
    d_neg, a_neg = allocate(inv, 0.25, 16, durs)
    fig = plt.figure(figsize=(12, 3.6))
    gs = gridspec.GridSpec(3, 8, height_ratios=[2.2, 1, 1], hspace=0.35, wspace=0.06)
    for i in range(8):
        ax = fig.add_subplot(gs[0, i])
        frame_ax(ax, frames[i], GREEN if direct_labels[i] == "i" else RED)
        ax.set_title(f"s{i}", fontsize=8, pad=2)
    rows = [(d_pos, a_pos, "Importance-Led FF"), (d_neg, a_neg, "Anti-Importance FF")]
    for row, (dd, aa, name) in enumerate(rows):
        for i in range(8):
            ax = fig.add_subplot(gs[row + 1, i])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#1f77b4",
                                            alpha=0.15 + 0.55 * dd[i]))
            for j in range(aa[i]):
                ax.plot((j + 1) / (aa[i] + 1), 0.5, "o", color="k", markersize=4)
            if i == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
            if i == 7:
                ax.annotate(f"{sum(aa)} frames", xy=(1.05, 0.5),
                            xycoords="axes fraction", fontsize=8, va="center")
    fig.savefig(OUT / "fig_alpha_sampling.pdf", bbox_inches="tight")
    plt.close(fig)


def build_walkthrough():
    g = json.load(open(GREEDY_SIDECAR))
    run = g["mss_result"]["mss_runs"][0]
    its = run["iteration_scores"]
    baselines = [its[0]["baseline"]] + [s["new_baseline"] for s in its[:-1]]
    removed = run["removal_order"]
    kept = set(run["kept_indices"])
    order_of = {seg: k + 1 for k, seg in enumerate(removed)}
    term = {c["segment_index"]: c["removal_score"] for c in its[-1]["candidates"]}
    frames = banana_frames()
    fig = plt.figure(figsize=(12, 5.2))
    gs = gridspec.GridSpec(2, 8, height_ratios=[1.15, 1.5], hspace=0.32, wspace=0.06)
    for i in range(8):
        ax = fig.add_subplot(gs[0, i])
        frame_ax(ax, frames[i], GREEN if i in kept else RED, lw=3.5)
        ax.set_title(f"s{i}", fontsize=9, pad=2)
        tag = "kept" if i in kept else f"#{order_of[i]} removed"
        ax.set_xlabel(tag, fontsize=8, color=GREEN if i in kept else RED, labelpad=2)
    ax = fig.add_subplot(gs[1, :])
    x = list(range(len(baselines)))
    ax.plot(x, baselines, "o-", color="#1f77b4", linewidth=1.8, markersize=6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.text(0.05, 0.515, "removability threshold  P(YES) = 0.5", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(["start"] + [f"remove s{s}" for s in removed], fontsize=9)
    for xi, b in zip(x, baselines):
        ax.annotate(f"{b:.3f}", (xi, b), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)
    k2, k6 = sorted(kept)
    ax.annotate(
        f"stop: removing s{k2} ({term[k2]:.2f}) or s{k6} ({term[k6]:.2f})\n"
        "would fall below threshold",
        xy=(x[-1], baselines[-1]), xytext=(4.15, 0.62), fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_ylabel("oracle  P(YES $\\mid$ kept set)", fontsize=9)
    ax.set_ylim(0.42, 1.0)
    fig.savefig(OUT / "fig_greedy_walkthrough.pdf", bbox_inches="tight")
    plt.close(fig)


def build_montage():
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.4))
    for r, (name, group, labels) in enumerate(MONTAGE_ROWS):
        files = sorted((WORK / group).glob("*_b.png"))
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(Image.open(files[c]))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(labels[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dataset_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def build_precheck():
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.2))
    for r, name in enumerate(PRECHECK_ROWS):
        files = sorted((WORK / "precheck").glob(f"{r:02d}_*_[abc].png"))
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(Image.open(files[c]))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 1:
                ax.set_xlabel(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_precheck_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    extract_assets()
    dm = json.load(open(DIRECT_META))
    direct_labels = ["i" if s["label"] == "important" else "u" for s in dm["segments"]]
    build_teaser()
    build_alpha(direct_labels)
    build_walkthrough()
    build_montage()
    build_precheck()
    print("wrote 5 figures to", OUT)


if __name__ == "__main__":
    main()
