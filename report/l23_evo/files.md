# Regulon → target-gene table — human (l23_evo script 27)

Clean regulon membership table derived from the Wang25 human SCENIC+ eRegulon
supplementary table, produced by `scripts/l23_evo/27.human_wang25_regulon_targets.py`.
This is the human analog of the mouse Yoo25 tables in `report/it/files.md`
(`scripts/it/40.yoo25_IT_regulon_targets.py`).

| Species | Output file | Regulons | +/+ | -/+ | direct | extended-only | Genes | Target pairs |
|---|---|---|---|---|---|---|---|---|
| Human (Wang25) | `local_data/res/l23_evo/27.human_wang25_regulon_targets.tsv` | 582 | 385 | 197 | 534 | 48 | 7,412 | 56,280 |

`Genes` = number of distinct target genes; `Target pairs` = rows (one per unique
regulon–gene pair). `+/+` and `-/+` are the regulon counts by sign pattern
(activating vs repressing TF→gene). Only regulons with a positive R2G (second) sign are kept
(`+/+`, `-/+`); `+/-` and `-/-` regulons are dropped — this dataset happens to
contain only `+/+` and `-/+` regulons, so none are removed. There is no
direct/extended overlap, so all 582 regulons are retained.

## How it's produced

The input is a single Excel file (`human_regulons_SuppTable13.xlsx`), read with the
Python standard library (`zipfile` + `ElementTree`) to avoid an openpyxl
dependency. Sheet **13b** (eRegulon target genes) is used; each row is one eRegulon
with a comma-separated `Target genes` cell. The script:

1. Parses the eRegulon name `<TF>[_extended]_<sign>_<sign>` (e.g. `ETV1_+_+` =
   direct, `RORB_extended_+_+` = extended, `NKX6-2_-_+`). There is no explicit
   "direct" tag — a regulon is direct unless it carries `_extended_`. Signs are
   joined with `/` (`+_+` → `+/+`) so the schema matches the mouse tables.
2. Normalizes each regulon to **TF + sign pattern** (e.g. `RORB_extended_+/+` →
   `RORB_+/+`). If a regulon exists in both direct and extended, keeps its
   **direct** targets; otherwise keeps the **extended** targets.
3. Explodes the comma-separated gene list to unique `(regulon, Gene)` pairs.
4. Keeps only regulons whose R2G (second) sign is positive (`+/+`, `-/+`),
   dropping `+/-` and `-/-`.

## File schema

Columns: `regulon, TF, regulation_direction, source, Gene`

One row per regulon–target-gene pair (long format), grouped by the `regulon`
column. `source` is `direct` or `extended`; `regulation_direction` is the sign
pattern (`+/+` or `-/+` — only positive-R2G regulons are retained). Identical
schema to the mouse Yoo25 tables.

# Archetype × regulon enrichment — l23_evo script 28

`scripts/l23_evo/28.human_l23_regulon_archetype_enrichment.py` is the human L2/3
analog of the mouse IT script 41 (`report/it/files.md`). For the single human L2/3 IT
population (jorstad23, NOC=4 archetypes A–D from script 25), it tests whether each
archetype's marker set is over-represented among each Wang25 regulon's target genes
(script 27) via a one-sided Fisher exact test, and renders activating/repressing
log2-OR heatmaps.

| Population | Universe (N) | Regulons tested | Significant (FDR<0.05 in ≥1 archetype) | Long-format TSV |
|---|---|---|---|---|
| Human L2/3 | 2,000 | 359 | 83 | `local_data/res/l23_evo/28.human_l23_regulon_archetype_enrichment.tsv` |

## How it's produced

1. **Background universe** = the gene set the human archetype marker Wilcoxon ran
   over, i.e. the 2,000 HVGs from PCA (index of
   `local_data/res/l23_evo/01.pca_loadings.tsv`). Unlike the mouse version this is a
   saved gene list, so no h5ad reconstruction is needed (single dataset, single
   population).
2. Marker sets (per archetype, from `25.human_archetype_markers.tsv`) and regulon
   target sets (from `27.human_wang25_regulon_targets.tsv`) are intersected with the
   universe; regulons with <5 in-universe targets are dropped (359 of 582 kept).
3. Per (archetype, regulon): 2×2 `[[x, M-x], [T-x, N-M-T+x]]` with `x`=overlap,
   `M`=markers, `T`=targets, `N`=universe → `scipy.stats.fisher_exact(...,
   alternative='greater')`. `log2 OR` uses a Haldane–Anscombe (+0.5) correction.
4. BH-FDR across all (archetype × regulon) pairs.

## Outputs

- `28.human_l23_regulon_archetype_enrichment.tsv` — long format: `layer, archetype,
  regulon, TF, regulation_direction, overlap, n_markers, n_targets, universe,
  log2_or, pval, fdr, neglog10_fdr, arch_letter` (`layer` is the constant `L2/3`).
- `28.human_l23_enrichment_neglog10fdr.tsv`, `28.human_l23_enrichment_log2or.tsv` —
  regulon × archetype matrices, columns = `arch_letter`.
- `28.human_l23_regulon_archetype_enrichment.html` — two-panel heatmap (activating
  `+/+`, repressing `-/+`), colored by log2 OR, `*` where FDR<1e-2; rows shown when
  log2 OR>2 AND FDR<1e-2 in ≥1 archetype.

`arch_letter` is the archetype display label, reversed to mirror the mouse L2/3
convention: `archetype_1`→`D'`, `archetype_2`→`C'`, `archetype_3`→`B'`,
`archetype_4`→`A'`. Columns then sort to `A' B' C' D'` in the matrices and heatmaps.

# Archetype markers over the full gene universe — l23_evo script 29

`scripts/l23_evo/29.human_archetype_markers_allgenes.py` re-derives the human L2/3
archetype markers over the **proper** gene universe instead of the 2000 HVGs used by
script 25. The archetype definition is unchanged — it reproduces script 25's PCHA fit
and top-300-cell-per-archetype assignment exactly (validated: each refit archetype
correlates > 0.99 with `25.human_pcha_aa.tsv`) — only the gene set the Wilcoxon runs
over is expanded.

| Output | What | Genes |
|---|---|---|
| `local_data/res/l23_evo/29.human_gene_universe.tsv` | expressed & non-uniform genes | 20,415 of 29,352 |
| `local_data/res/l23_evo/29.human_archetype_markers_allgenes.tsv` | NOC=4 markers (A–D) | 920 total (194/457/163/106) |

## How it's produced

1. **Gene universe** = all *expressed & non-uniform* genes: from the log-normalized
   sparse `adata.X` over all L2/3 cells, keep genes with nonzero total and nonzero
   variance — `(sum>0) & (var>0)`, the human analog of the mouse rule. 20,415 genes
   (`feature_name` symbols, unique). Written as a single `gene` column.
2. **Markers** = the same conservative one-vs-each Wilcoxon as script 25 (worst-case
   `log2FC=min`, `pval=max`, `frac_out=mean` across the NOC−1 pairwise tests), but
   BH-FDR is now computed over the full universe; filters `frac_in≥0.25`, `fdr<0.001`,
   `log2FC>log2(1.5)`. Same schema as `25.human_archetype_markers.tsv`:
   `gene, archetype, log2FC, pval, fdr, frac_in, frac_out`.

# Archetype × regulon enrichment over the full universe — l23_evo script 30

`scripts/l23_evo/30.human_l23_regulon_archetype_enrichment_allgenes.py` is script 28
re-run against the script-29 markers and the 20,415-gene universe (identical
statistics and heatmap code). **This supersedes script 28 for the universe-sensitive
results**: with the proper background, all 582 regulons become testable and the
Fisher/FDR numbers are no longer biased by the small HVG background.

| Population | Universe (N) | Regulons tested | Significant (FDR<0.05 in ≥1 archetype) | Long-format TSV |
|---|---|---|---|---|
| Human L2/3 (full universe) | 20,415 | 582 | 271 | `local_data/res/l23_evo/30.human_l23_regulon_archetype_enrichment_allgenes.tsv` |

(For comparison, the 2000-HVG script 28 tested only 359 regulons with N=2,000.)

## Outputs

- `30.human_l23_regulon_archetype_enrichment_allgenes.tsv` — long format, same columns
  as script 28 (`layer, archetype, regulon, TF, regulation_direction, overlap,
  n_markers, n_targets, universe, log2_or, pval, fdr, neglog10_fdr, arch_letter`).
- `30.human_l23_enrichment_allgenes_neglog10fdr.tsv`,
  `30.human_l23_enrichment_allgenes_log2or.tsv` — regulon × archetype matrices,
  columns = `arch_letter` (`A' B' C' D'`).
- `30.human_l23_regulon_archetype_enrichment_allgenes.html` — two-panel heatmap
  (activating `+/+`, repressing `-/+`), colored by log2 OR, `*` where FDR<1e-5; rows
  shown when log2 OR>2 AND FDR<1e-5 in ≥1 archetype (44 activating, 31 repressing).

Same archetype relabel as script 28 (`D' C' B' A'` → columns sort `A' B' C' D'`).
Selection thresholds: `MIN_REGULON_GENES=5`, `LOG2OR_SHOW=2`, `STAR_FDR=1e-5` (script
28 uses the looser `STAR_FDR=1e-2`).
