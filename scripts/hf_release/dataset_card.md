---
license: cc-by-4.0
pretty_name: Meaningful Moments
language:
- en
task_categories:
- video-classification
tags:
- video
- action-recognition
- temporal-saliency
- pseudo-labels
- vision-language-model
- segment-importance
size_categories:
- 100K<n<1M
configs:
- config_name: ssv2
  data_files:
  - split: train
    path: data/ssv2/train-*
  - split: validation
    path: data/ssv2/validation-*
  - split: test
    path: data/ssv2/test-*
- config_name: k400
  data_files:
  - split: train
    path: data/k400/train-*
  - split: validation
    path: data/k400/validation-*
  - split: test
    path: data/k400/test-*
- config_name: diving48
  data_files:
  - split: train
    path: data/diving48/train-*
  - split: validation
    path: data/diving48/validation-*
---

# Meaningful Moments (MM) — v1.0

Per-segment temporal-importance pseudo-labels for **536,181 videos** across
three canonical video action-recognition benchmarks, produced by querying a
vision-language oracle (Qwen3-VL-32B-Instruct) once per video with a
direct-scoring prompt. Each video is divided into fixed-duration segments;
the oracle scores every segment's importance for recognizing the video's
action, yielding **~4.58 million per-segment importance scores** over
**622 action classes**.

| substrate | train | validation | test | classes | Δt | prompt |
|---|---|---|---|---|---|---|
| Something-Something v2 | 168,913 | 24,777 | 27,157 | 174 | 0.5 s | `direct_scoring_ssv2` |
| Kinetics-400 | 239,789 | 19,877 | 38,671 | 400 | 1.0 s | `direct_scoring_k400` |
| Diving-48 | 15,027 | 1,970 | — | 48 | 0.5 s | `direct_scoring_diving48` |

Source videos are **not** redistributed (see Licensing below). Every released
label is keyed by the substrate's native video id.

## What's in each instance

One row = one video. The conceptual instance of the datasheet is one
*(video, segment)* pair — get that view with one explode:

```python
from datasets import load_dataset
ds = load_dataset("<namespace>/meaningful-moments", "ssv2", split="validation")
df = ds.to_pandas().explode("segments")          # one row per (video, segment)
```

Top-level fields (identical across configs; `template`/`placeholders` extra on
`ssv2`, `class_id`/`raw_class_name` extra on `diving48`, null where the source
lacks them):

| field | type | meaning |
|---|---|---|
| `video_id` | str | substrate-native video identifier |
| `action_label` | str | substrate class string (SSv2: filled caption; see `template`) |
| `video_path` | str | substrate-relative source path (`SSv2/...`, `k400/...`, `diving48/...`) |
| `timestamp` | str | ISO extraction time |
| `summary` | str | oracle's one-paragraph description of the video |
| `label_counts` | struct | per-video tallies of important/unimportant segments |
| `segments` | list&lt;struct&gt; | per-segment annotations (below) |
| `mss_result` | struct | extraction outcome (below) |
| `elapsed_s` | float | wall-clock for this video's annotation |

Per segment:

| field | type | meaning |
|---|---|---|
| `index` | int | 0-based segment index |
| `time_range` | str | `"t_start-t_end"` in seconds |
| `start_s`, `end_s` | float | segment boundaries in seconds |
| `label` | str | `important` (kept in the minimal sufficient subset) or `unimportant` |
| `weight` | float | continuous importance in [0, 1] (= oracle's 0–100 score / 100) |
| `phase` | str | `setup` / `contact` / `execution` / `continuation` / `result` / `disambiguation` (null where the oracle response was unparseable) |
| `reason` | str | oracle's one-line rationale for the score |

`mss_result`: `kept_indices` (the MSS as 0-based indices), `total_oracle_calls`
(1 for direct scoring), `precheck_passed` (did the oracle confirm the action is
present), `precheck_vote_yes` (vote fraction), `precheck_responses[]`
(decision / confidence / evidence / rationale / YES-NO-SKIP logits).

### Failure records are included

Videos where the oracle's precheck failed (action not confirmed: ~4.98% SSv2,
~7.80% K400, ~15.34% Diving-48) are **released, not removed** — they carry
`precheck_passed: false`, empty `kept_indices`, and the oracle's response
verbatim. Filter on `mss_result.precheck_passed` if you want clean labels only.

## Repository layout

```
data/<config>/<split>-*.parquet      # this card's configs (raw_output omitted)
sidecars/<run>/sidecars-NNN.tar.gz   # canonical per-video JSON sidecars
sidecars/<run>/config.json           # full extraction provenance per run
supplement/cross_oracle/<oracle>/    # 4-oracle agreement pilot (600 videos
                                     #   each: Qwen3-VL-32B, Gemini 3.1 Pro,
                                     #   InternVL3-38B, GPT-5.5) — see its README
supplement/eval_sidecars/<run>/      # recognizer-eval records behind every
                                     #   thesis statistic (22 runs; added v1.1).
                                     #   With these + the code repo, every
                                     #   table/figure regenerates on a laptop —
                                     #   see eval_sidecar_runs.json for the map
                                     #   and sha256sums-eval.txt for integrity
manifests/<run>.csv                  # per-video index incl. shard membership
manifests/eval_pools/                # pinned stratified eval pools (seed=42)
prompts/                             # the four scoring prompts, verbatim
distributions.json                   # full-corpus per-substrate distributions
sample_sidecar.json                  # one canonical sidecar, for reference
sha256sums.txt                       # hash of every sidecar + released file
croissant.json                       # Croissant metadata
```

The JSON sidecars are the ground-truth format; parquet mirrors them
field-for-field minus the verbatim `raw_output` string (recoverable from the
tars). Verify any extracted sidecar against `sha256sums.txt`
(`sha256sum -c`, paths are `<run>/<video_id>.json`).

## Provenance

- **Oracle:** `Qwen/Qwen3-VL-32B-Instruct`, revision
  `0cfaf48183f594c314753d30a4c4974bc75f3ccb`, greedy decoding (T=0).
- **Prompts:** SHA-256-pinned per run in `sidecars/<run>/config.json`; full
  text in `prompts/`.
- **Pipeline:** single-call direct scoring with a YES/NO/SKIP action-presence
  precheck; per-run hyperparameters (segment length Δt, mask operator) in each
  `config.json`.
- **Cross-oracle supplement:** the same pipeline run by three additional
  oracles over a shared 600-video pilot pool, for label-reliability analysis
  (`supplement/cross_oracle/README.md`).

## Licensing

The MM **labels** (sidecars, parquet, manifests) are released under
**CC-BY-4.0**. Source videos are governed by their own licenses and are not
redistributed; obtain them upstream:

- **Something-Something v2** — CC-BY-NC-4.0, from the Qualcomm AI Research
  distribution. (`action_label`/`template` strings on the `ssv2` config derive
  from SSv2's annotations and carry that license's terms.)
- **Kinetics-400** — CC-BY-4.0, from the DeepMind release.
- **Diving-48** — research-only, from the UW-Madison distribution.

## Versioning

- **v1.0** — the label corpus (parquet + sidecars + cross-oracle pilot). Immutable.
- **v1.1** — adds `supplement/eval_sidecars/` (recognizer-eval records enabling
  GPU-free reproduction of every thesis statistic). No v1.0 file changed.

Future re-extractions under different oracles or prompt revisions will be
released as sibling subsets under new semver tags — existing labels are never
overwritten.

## Code

The full annotation/evaluation/statistics pipeline:
https://github.com/camkeith/meaningful-moments (MIT, tag `v1.0`) — see its
`REPRODUCING.md` for the thesis-section → command map.

## Citation

```bibtex
@misc{meaningfulmoments2026,
  title  = {Meaningful Moments: VLM-Derived Per-Segment Importance Labels for
            Video Action Recognition},
  author = {Keith, Cameron},
  year   = {2026},
  note   = {Dataset v1.0},
}
```

Contact: cameron.s.keith.26@dartmouth.edu — issues affecting downstream
reproducibility answered within 7 days for the first 12 months post-release.
