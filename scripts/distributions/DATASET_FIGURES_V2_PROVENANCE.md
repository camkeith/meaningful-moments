# Provenance: dataset_figures_v2.json

Committed verified input for `paper_tables_20260514_140732/plot_class_coverage.py`,
which generates the thesis figures `fig_class_coverage_hist.pdf`,
`fig_weight_distribution.pdf`, and `fig_source_mm_parity.pdf`, and backs the
inline per-segment weight median/mean statistics in the Methodology.

- SHA-256: `eaaa3848190960466039dccddee159ce1f666e55902d22143ced350a94f991c8`
- Original location: `/tmp/dataset_figures_v2.json` (machine-local)
- Original mtime: 2026-05-19 23:22:53 EDT · 764,659 bytes
- Committed: 2026-06-03 (dataset release preparation)

The script that originally extracted this file from the production MSS sidecars
was **not preserved** — it predated the repo's `scripts/distributions/` extractor
family and was lost to a `/tmp` lifecycle. The file's contents were verified at
commit time by regenerating all three figures and matching them against the
thesis copies. The nearest committed relative is
`scripts/distributions/extract_distributions.py`, which produces the similar
(but not column-identical) `scripts/distributions/out/distributions.json`.
