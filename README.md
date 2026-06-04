# Meaningful Moments — code release

Annotation, evaluation, statistics, and plotting pipeline behind
**Meaningful Moments (MM)**: ~4.58M per-segment temporal-importance
pseudo-labels over 536,181 videos across SSv2, Kinetics-400, and Diving-48,
produced by a Qwen3-VL-32B oracle with single-call direct scoring.

- **Dataset:** https://huggingface.co/datasets/ckeith/meaningful-moments (CC-BY-4.0, tag `v1.0`)
- **Code license:** MIT (this repository)
- **Reproduction guide:** [`REPRODUCING.md`](REPRODUCING.md) — every thesis
  table/figure mapped to the command that regenerates it

## Install

```bash
python3 -m venv venv && source venv/bin/activate   # Python 3.12 used in production
pip install -r oracle/requirements.txt             # curated direct deps
# or, for the exact production environment:
pip install -r requirements-freeze.txt
```

Set `MM_ROOT` to this repository's root when running from elsewhere
(defaults resolve relative to each script's location).

## Quickstart (Tier 1 — no GPUs, released data only)

```python
from datasets import load_dataset
ds = load_dataset("ckeith/meaningful-moments", "ssv2", split="validation")
print(ds[0]["segments"])     # per-segment weight / phase / reason
```

Regenerate a thesis analysis from the released eval-sidecar supplement
(see `REPRODUCING.md` §Tier-1 for fetching it):

```bash
python scripts/mode_prompt_agreement.py --out /tmp/mm_check
# reproduces the mode/prompt-ablation agreement tables exactly
```

## Annotate your own video (Tier 2 — needs a GPU)

```bash
python oracle/scripts/mss_extract.py \
    --video your_video.mp4 --label "your action label" \
    --output out.json --model qwen3-vl-32b --direct-scoring
```

## Layout

```
oracle/scripts/mss/        oracle, segmentation, masking, prompts (the annotator)
oracle/scripts/            evaluation CLIs (recognizer eval, paired stats, I/D curves,
                           shuffle damage, temporal coverage, cross-oracle, heads)
scripts/                   analysis wrappers (exp5 CIs, mode/prompt agreement,
                           galleries, eval-pool samplers)
scripts/hf_release/        the pipeline that built the HF dataset release
scripts/distributions/     corpus statistics extractor
paper_tables_*/            figure plotters
data/csvs/                 pinned evaluation pools (seed=42)
```

## Citation

See [`CITATION.cff`](CITATION.cff).
