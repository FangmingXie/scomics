# Plan: Human L2/3 regulon–archetype enrichment over the full gene universe

## Context

Script 28 (`scripts/l23_evo/28.human_l23_regulon_archetype_enrichment.py`) ran the
Fisher-exact enrichment against a **2000-HVG** background, because the human archetype
markers (script 25) were only Wilcoxon-tested over those 2000 HVGs. That is a biased,
under-powered universe: regulon targets outside the HVG set are invisible, and the
2x2 background N is artificially small.

The proper universe is **all expressed & non-uniform genes** — genes with nonzero total
expression and nonzero variance across the L2/3 cells (the human analog of the mouse
`(X.sum>0) & (X.var>0)` rule). For jorstad23 this is **20,415 of 29,352 genes**.

Two things must change together: (1) the **markers** must be re-derived over this full
gene set (a marker can only be called if it was tested), and (2) the **enrichment** must
use the same expanded set as its background. This is why both a new marker script and a
new enrichment script are needed.

## Step 1 — Redefine archetype markers over the full universe (new script 29)

`scripts/l23_evo/29.human_archetype_markers_allgenes.py`, adapted from script 25.

- **Archetype/cell assignment — reproduce script 25 exactly.** Read
  `05.varimax_coords.tsv`, fit PCHA (`SCA`, `setup_feature_matrix('data')`,
  `proj_and_pcha(NDIM=5, NOC=4)`), back-project archetypes to VX space
  (`aa_vx = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_`), pick the
  `N_TOP_CELLS=300` nearest cells per archetype in VX space. Archetypes are ordered by
  PC1 (`argsort(XC[0])` inside `pcha`), so `archetype_1..4` match script 25.
  **Validation (fail-fast):** load `25.human_pcha_aa.tsv` and assert each archetype's
  refit coords correlate > 0.99 with the saved coords (same hull, same order).
- **Gene universe.** From the log-normalized sparse `adata.X` over **all** cells,
  compute per-gene total and variance; keep `(sum>0) & (var>0)` → ~20,415 genes
  (`feature_name` symbols, already unique). Write the list to
  `29.human_gene_universe.tsv` (single `gene` column).
- **Markers.** Densify `adata.X[:, universe_idx]` for the top cells only, then run the
  identical conservative one-vs-each Wilcoxon (worst-case `log2FC=min`, `pval=max`,
  `frac_out=mean` across the NOC−1 pairwise tests), BH-FDR **over the full universe**,
  filter `frac_in≥0.25`, `fdr<0.001`, `log2FC>log2(1.5)`. Same schema as script 25.
  Write `29.human_archetype_markers_allgenes.tsv`.
- Scores are not needed for enrichment and are out of scope (skip).

Reused params (identical to script 25): `VX_COLS`, `NOC=4`, `NDIM=5`,
`N_TOP_CELLS=300`, `FRAC_IN_THRESH=0.25`, `FDR_THRESH=0.001`, `LOG2FC_THRESH=log2(1.5)`.

## Step 2 — Redo the enrichment with the new markers + expanded universe (new script 30)

`scripts/l23_evo/30.human_l23_regulon_archetype_enrichment_allgenes.py`, a copy of
script 28 with three input changes; all statistics/heatmap code is unchanged:
- `MARKERS = 29.human_archetype_markers_allgenes.tsv`
- `UNIVERSE = 29.human_gene_universe.tsv` (read `set(...)`; replaces the HVG-loadings read)
- output prefix `30.human_l23_*_allgenes`
- keep the current script-28 conventions: relabel `archetype_1..4 → D',C',B',A'`
  (columns sort `A' B' C' D'`), `MIN_REGULON_GENES=5`, `LOG2OR_SHOW=2.0`,
  `STAR_FDR=1e-2`, `COLOR_ABS=5.0`.

Outputs:
- `local_data/res/l23_evo/30.human_l23_regulon_archetype_enrichment_allgenes.tsv` (long)
- `local_data/res/l23_evo/30.human_l23_enrichment_allgenes_neglog10fdr.tsv` (matrix)
- `local_data/res/l23_evo/30.human_l23_enrichment_allgenes_log2or.tsv` (matrix)
- `local_data/fig/l23_evo/30.human_l23_regulon_archetype_enrichment_allgenes.html`

## Step 3 — Register to report

Add to `report/l23_evo/files.md`: a section for script 29 (new markers + the
`29.human_gene_universe.tsv` universe, N=20,415, how it's produced) and a section for
script 30 (enrichment over the full universe; the regulons-kept / significant table),
noting these supersede the 2000-HVG script-28 results for the universe-sensitive stats.

## Verification

1. `conda run --no-capture-output -n archetype python -u scripts/l23_evo/29.human_archetype_markers_allgenes.py`
   → assert passes (PCHA matches script 25); universe ~20,415; per-archetype marker counts printed.
2. `conda run --no-capture-output -n archetype python -u scripts/l23_evo/30.human_l23_regulon_archetype_enrichment_allgenes.py`
   → universe N≈20,415; regulons kept of 582; per-sign shown rows.
3. Spot-check: more regulons kept than script 28 (larger universe → more in-universe
   targets); long TSV `universe` column == N; FDR ∈ [0,1]; known L2/3 TFs land sensibly.
4. Open the heatmap HTML; A'–D' columns, activating/repressing panels, `*` at FDR<1e-2.
