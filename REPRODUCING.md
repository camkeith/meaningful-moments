# Reproducing the thesis results

Two tiers. **Tier 1** regenerates every statistic, table, and figure from
*released data* on a laptop. **Tier 2** re-runs annotation/evaluation from
scratch (GPUs + substrate videos + third-party checkpoints).

Everything below assumes the repo root as cwd (or `MM_ROOT` pointing at it).

## Released data

| artifact | where |
|---|---|
| MM labels (parquet + canonical sidecars) | `hf.co/datasets/ckeith/meaningful-moments` (tag `v1.0`) |
| Eval-sidecar supplement (recognizer-eval records all stats read) | same dataset, `supplement/eval_sidecars/` |
| Pinned eval pools | `data/csvs/` here, mirrored in the dataset's `manifests/eval_pools/` |

Fetch and unpack the supplement so the stats scripts find their inputs
(default expected layout: `pseudo_labels/...` under the repo root —
`supplement/eval_sidecars/eval_sidecar_runs.json` maps each tar to its
original path):

```bash
hf download ckeith/meaningful-moments --repo-type dataset \
    --include 'supplement/eval_sidecars/*' --local-dir ./_supplement
python - <<'PY'
import json, tarfile, pathlib
root = pathlib.Path('.')
enum = json.load(open('_supplement/supplement/eval_sidecars/eval_sidecar_runs.json'))
for run in enum['runs']:
    name = run['path'].rstrip('/').split('/')[-1] if not run['path'].startswith('pseudo_labels/cross_oracle_eval') else 'cross_oracle_stage1'
    dest = root / run['path']; dest.mkdir(parents=True, exist_ok=True)
    for tar in sorted((root / f'_supplement/supplement/eval_sidecars/{name}').glob('sidecars-*.tar.gz')):
        with tarfile.open(tar) as tf: tf.extractall(dest)
PY
sha256sum -c <(grep -v '^#' _supplement/supplement/eval_sidecars/sha256sums-eval.txt)  # optional verify
```

## Tier 1 — thesis section → command

| thesis | command | output |
|---|---|---|
| Headline paired-bootstrap CIs (Results overview) | `python scripts/recompute_exp5_cis.py` | CI table values (B=10,000) |
| α-sweep tables + curves (Results per-substrate) | `python -m oracle.scripts.eval_paired_stats --run-dir pseudo_labels/classifier_eval/<run> --eval-csv data/csvs/<sub>/eval_*.csv --output-dir <out>` per run dir, then `python paper_tables_20260514_140732/plot_alpha_curves.py` | `{d48,ssv2,k400}_curve.pdf` |
| Threshold/budget sweeps | same sidecars; `python paper_tables_20260514_140732/plot_threshold_curves.py` | `*_threshold_budget.pdf` |
| I/D curves, 5 orderings | `python -m oracle.scripts.eval_id_curves --run-dir pseudo_labels/classifier_eval/<sub>_id_20260520_* --eval-csv data/csvs/<sub>/id_500_stratified.csv --recognizer <r>`, then `plot_id_curves.py` | AUC/ΔAUC tables, `*_id_curves.pdf` |
| Shuffle damage | `python -m oracle.scripts.shuffle_damage_analysis --run-dir pseudo_labels/classifier_eval/<base run> --output-dir <out>` | Δ_shuf, McNemar |
| Temporal coverage | `python -m oracle.scripts.temporal_coverage_analysis --run-dir <base run> --eval-csv ... --output-dir <out>` | coverage/Gini diagnostics |
| Mode/prompt ablations (App. A.2/A.3/A.5) | `python scripts/mode_prompt_agreement.py --out <out>` | reproduces every table value exactly |
| Cross-oracle agreement (App. B) | `python oracle/scripts/oracle_agreement_eval.py --runs ...` over the dataset's `supplement/cross_oracle/` arms | pairwise κ/Spearman/Jaccard |
| Cross-oracle recognition | `oracle/scripts/cross_oracle/{consistency_tables,agreement_vs_recognition,seed_robustness}.py` over `cross_oracle_stage1` | EPS/TDS tables |
| Corpus statistics + composition figures | `python scripts/distributions/extract_distributions.py`; `python paper_tables_20260514_140732/plot_class_coverage.py`; `python paper_tables_20260514_140732/plot_per_class_kept.py` | `distributions.json`, `fig_*.pdf` |
| Qualitative galleries | `python scripts/qualitative_gallery.py` | wins/losses montages (needs source videos) |

Conventions worth knowing before comparing numbers: the A.2 pilot Spearman
uses sort-order ranks without tie correction (documented in the thesis caption
and in `scripts/mode_prompt_agreement.py`); paired bootstrap is B=10,000 at
seed 42; "n_succ paired" excludes videos whose selector failed per condition.

## Tier 2 — from scratch

### Source videos (not redistributed)
SSv2 (Qualcomm, CC-BY-NC) · K400 (DeepMind, CC-BY-4.0) · Diving-48
(UW-Madison, research-only). Place/symlink under `data/` as
`data/SSv2/...`, `data/k400/...`, `data/diving48/...` to match the
`video_path` values in the release.

### Oracle annotation
```bash
python oracle/scripts/mss_extract.py --dataset data/csvs/<sub>/<split>.csv \
    --model qwen3-vl-32b --direct-scoring --prompt-template <sub> \
    --parallel --visible-gpus 0,1,2,3
```
Oracle: `Qwen/Qwen3-VL-32B-Instruct` revision
`0cfaf48183f594c314753d30a4c4974bc75f3ccb`, greedy decoding. Per-run
hyperparameters land in the output dir's `config.json`. API oracles
(Gemini / GPT-5.5 / Dartmouth) additionally need
`pip install -r oracle/requirements-optional.txt` and read
`DARTMOUTH_CHAT_API_KEY` / `AZURE_OPENAI_*` env vars.

### Recognizer evaluation
HF-port recognizers download automatically
(`MCG-NJU/videomae-large-finetuned-kinetics`, `facebook/vjepa2-vitl-*`).
The **paper-checkpoint** path additionally needs Meta's source tree and
checkpoints, which are not redistributed here:

```bash
git clone https://github.com/facebookresearch/vjepa2.git
git -C vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812   # pinned commit
export VJEPA2_REPO=$PWD/vjepa2
# checkpoints (from the V-JEPA 2 release): ssv2-vitl-16x2x3.pt,
# diving48-vitl-256.pt, vitl.pt
export VJEPA2_CKPT_DIR=/path/to/checkpoints
python -m oracle.scripts.vjepa2_official.run_eval --recognizer vjepa2-official-ssv2 ...
```

Then per-condition eval:
```bash
python -m oracle.scripts.run_classifier_eval \
    --recognizer vjepa2-ssv2 --condition vlm-selected \
    --mss-run-dir <MSS run> --eval-csv data/csvs/ssv2/eval_2k_stratified.csv \
    --output-dir <out> --device cuda:0
```
The `motion` ordering first needs `python -m oracle.scripts.vjepa2.compute_motion`.

### Selection-aware head finetuning
```bash
python -m oracle.scripts.vjepa2.cache_features ...   # Phase A (per GPU shard)
python -m oracle.scripts.train_head --head-arch vjepa2 ...   # Phase B
```

### Notes on what is deliberately absent
Site-specific fleet launchers (slurm arrays, shared-queue worker scripts) are
not released — they encode one workstation's GPU topology, not the method.
Every experiment above runs through the released CLIs directly. The thesis
figure generator `thesis/figures/make_thesis_figures.py` is included for
record, but its raster inputs come from the (unreleased) defense slide deck;
the five figures it produces ship pre-rendered in the thesis. `ffmpeg` is
resolved from `$FFMPEG` (default `/usr/bin/ffmpeg`).
