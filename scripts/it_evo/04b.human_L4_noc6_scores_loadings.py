"""Human L4 IT at NOC=6 — exploratory fork of script 04 (Gate B revisit).

Script 04 fits human L4 IT at `noc=3`, read off the ARV_mean elbow (0.029 at NOC=3 ->
0.405 at NOC=4). But 03.refine.human_L4_num_archetype_metrics.tsv is not monotone: NOC=6
has the best effEV_mean of the whole sweep (0.769 vs 0.630 at 3) and an ARV_rep (0.239)
indistinguishable from NOC=2/3 (0.227 / 0.203). This script fits the NOC=6 alternative so
it can be judged on the biology of its vertices.

Two caveats are accepted, not tested here. NOC=6 is exactly L4's structural ceiling
(NOC <= NDIM+1 = len(VX_COLS) = 6), where the fit is the minimal full-dimensional simplex
and EV is near-trivially high; and `get_relative_variation` aligns replicate vertices only
by `pcha()`'s PC1 sort, with a denominator that grows for a maximally-spread simplex.
Either could manufacture the NOC=6 ARV dip. Settling that would need a ceiling-control
sweep; the decision here rests on markers and donor composition instead.

Everything downstream of the fit is copied verbatim from script 04 so the two fits are
comparable — same PCHA_SEED, N_TOP_CELLS, marker thresholds and score percentiles.
Three deliberate deviations from 04:
  - no `pcha_gene_loadings` output: nothing in the repo consumes 04's copy (16 explicitly
    declines it in favour of 02's varimax loadings)
  - 04's two stdout WARNINGs are recorded as `*_marker_counts.tsv` — a vertex with 0
    markers is the primary kill signal for NOC=6, so it must land in a file
  - per-vertex `WithinArea_cluster` / `donor_id` composition of the top cells is written
    out: L4's ARV_rep is elevated at every NOC (0.20-0.24, vs 0.06-0.09 for L2/3), i.e.
    its three donors genuinely disagree more, so a vertex whose top cells are dominated by
    one donor is a donor artifact rather than a cell state

Self-check: the inner PCA is fitted by `proj(xf, ndim)` before PCHA runs and so does not
depend on `noc`. `pcha_xp`, `pcha_inner_components` and `pcha_inner_mean` must therefore
match script 04's canonical L4 outputs exactly; a mismatch means this fork drifted from 04
and nothing after it is trustworthy.

This is exploratory scaffolding, not a pipeline stage. If NOC=6 wins, the resolution is to
flip 04's L4 `noc` to 6, update 09 / 13 / 22 in lockstep, and delete this script.

Reads:
  local_data/res/it_evo/02.human_L4_varimax_{coords,loadings}.tsv
  local_data/res/it_evo/04.human_L4_pcha_{xp,inner_components,inner_mean}.tsv  (self-check)
  links/it_evo/jorstad23_human_WithinArea_L4IT.h5ad
Outputs:
  local_data/res/it_evo/04b.human_L4_noc6_pcha_xp.tsv
  local_data/res/it_evo/04b.human_L4_noc6_pcha_aa.tsv
  local_data/res/it_evo/04b.human_L4_noc6_pcha_inner_components.tsv
  local_data/res/it_evo/04b.human_L4_noc6_pcha_inner_mean.tsv
  local_data/res/it_evo/04b.human_L4_noc6_top_cells.tsv
  local_data/res/it_evo/04b.human_L4_noc6_archetype_markers.tsv
  local_data/res/it_evo/04b.human_L4_noc6_marker_counts.tsv
  local_data/res/it_evo/04b.human_L4_noc6_vertex_composition.tsv
  local_data/res/it_evo/04b.human_L4_noc6_archetype_scores.tsv
  local_data/fig/it_evo/04b.human_L4_noc6_archetype_scores.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import gene_expr_scatter_html
from scomics.main import SCA

# --- file paths ---
LINK_DIR    = os.path.join(PROJECT_ROOT, 'links', 'it_evo')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')

IN_H5AD           = os.path.join(LINK_DIR, 'jorstad23_human_WithinArea_L4IT.h5ad')
IN_VX_COORDS      = os.path.join(OUT_RES_DIR, '02.human_L4_varimax_coords.tsv')
IN_VX_LOADINGS    = os.path.join(OUT_RES_DIR, '02.human_L4_varimax_loadings.tsv')
# canonical noc=3 fit — used only for the noc-independent self-check
IN_CANON_XP       = os.path.join(OUT_RES_DIR, '04.human_L4_pcha_xp.tsv')
IN_CANON_INNER_CMP  = os.path.join(OUT_RES_DIR, '04.human_L4_pcha_inner_components.tsv')
IN_CANON_INNER_MEAN = os.path.join(OUT_RES_DIR, '04.human_L4_pcha_inner_mean.tsv')

OUT_PCHA_XP       = os.path.join(OUT_RES_DIR, '04b.human_L4_noc6_pcha_xp.tsv')
OUT_PCHA_AA       = os.path.join(OUT_RES_DIR, '04b.human_L4_noc6_pcha_aa.tsv')
OUT_INNER_CMP     = os.path.join(OUT_RES_DIR, '04b.human_L4_noc6_pcha_inner_components.tsv')
OUT_INNER_MEAN    = os.path.join(OUT_RES_DIR, '04b.human_L4_noc6_pcha_inner_mean.tsv')
OUT_TOP_CELLS     = os.path.join(OUT_RES_DIR, '04b.human_L4_noc6_top_cells.tsv')
OUT_MARKERS       = os.path.join(OUT_RES_DIR, '04b.human_L4_noc6_archetype_markers.tsv')
OUT_MARKER_COUNTS = os.path.join(OUT_RES_DIR, '04b.human_L4_noc6_marker_counts.tsv')
OUT_COMPOSITION   = os.path.join(OUT_RES_DIR, '04b.human_L4_noc6_vertex_composition.tsv')
OUT_SCORES        = os.path.join(OUT_RES_DIR, '04b.human_L4_noc6_archetype_scores.tsv')
OUT_SCORE_HTML    = os.path.join(OUT_FIG_DIR, '04b.human_L4_noc6_archetype_scores.html')

# --- parameters (identical to script 04 except NOC) ---
SUBCLASS         = 'L4 IT'
VX_COLS          = ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10']   # Gate A, see script 03
NOC              = 6
NDIM             = len(VX_COLS) - 1
CLUSTER_COL      = 'WithinArea_cluster'
DONOR_COL        = 'donor_id'
GENE_NAME_COL    = 'feature_name'
PCHA_SEED        = 0
N_TOP_CELLS      = 300
FRAC_IN_THRESH   = 0.25
FDR_THRESH       = 0.001
LOG2FC_THRESH    = np.log2(1.5)
SCORE_PCTILE_LO  = 2
SCORE_PCTILE_HI  = 98
TOP_CELL_FRAC_MAX = 0.25             # G3 — NOC * N_TOP_CELLS must stay under this
SELFCHECK_ATOL   = 1e-8
ALPHABET         = ['A', 'B', 'C', 'D', 'E', 'F']

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

archetype_names = ALPHABET[:NOC]

print(f'\n{"=" * 70}\nL4 — human {SUBCLASS}  '
      f'(NOC={NOC}, NDIM={NDIM}, VX={VX_COLS})\n{"=" * 70}')

# --- load varimax coords ---
vx_df   = pd.read_csv(IN_VX_COORDS, sep='\t', index_col=0)
xn      = vx_df[VX_COLS].values
types   = vx_df[CLUSTER_COL].values
n_cells = len(vx_df)

if NOC * N_TOP_CELLS > TOP_CELL_FRAC_MAX * n_cells:
    raise ValueError(
        f'NOC*N_TOP_CELLS = {NOC * N_TOP_CELLS} exceeds {TOP_CELL_FRAC_MAX:.0%} of '
        f'{n_cells} cells — the "top" sets stop being archetype-pure (G3)'
    )

# --- fit PCHA (seeded immediately before the fit, G8) ---
print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')
# py_pcha draws its furthest-sum start index and S init from the global NumPy RNG and takes
# no seed argument, so the seed must be set here rather than once at import.
np.random.seed(PCHA_SEED)
sca.proj_and_pcha(NDIM, NOC)

pc_names = [f'PC{i+1}' for i in range(NDIM)]
xp_df = pd.DataFrame(sca.xp, index=vx_df.index, columns=pc_names)
xp_df.to_csv(OUT_PCHA_XP, sep='\t')
pd.DataFrame(sca.aa.T, columns=pc_names,
             index=[f'archetype_{k+1}' for k in range(NOC)]).to_csv(OUT_PCHA_AA, sep='\t')
print(f'Saved {OUT_PCHA_XP} and {OUT_PCHA_AA}')

# --- inner PCA parameters (VX axes -> PCHA space) ---
inner_comps = sca.pca_.components_[:NDIM]   # (NDIM, len(VX_COLS))
inner_mean  = sca.pca_.mean_                # (len(VX_COLS),)
inner_comps_df = pd.DataFrame(inner_comps, columns=VX_COLS, index=pc_names)
inner_mean_df  = pd.DataFrame(inner_mean.reshape(1, -1), columns=VX_COLS)
inner_comps_df.to_csv(OUT_INNER_CMP, sep='\t')
inner_mean_df.to_csv(OUT_INNER_MEAN, sep='\t')
print(f'Saved {OUT_INNER_CMP} and {OUT_INNER_MEAN}')

# --- self-check against script 04's canonical noc=3 fit ---
# proj() fits the inner PCA before PCHA runs, so xp / inner_components / inner_mean are
# noc-independent and must reproduce 04's L4 outputs exactly. A mismatch means this fork
# has drifted from 04 and everything below it is void.
print('Self-check against canonical 04.human_L4_* (noc-independent quantities)...')
for out_path, canon_path, label in [
    (OUT_PCHA_XP,    IN_CANON_XP,          'pcha_xp'),
    (OUT_INNER_CMP,  IN_CANON_INNER_CMP,   'pcha_inner_components'),
    (OUT_INNER_MEAN, IN_CANON_INNER_MEAN,  'pcha_inner_mean'),
]:
    new_df   = pd.read_csv(out_path,   sep='\t', index_col=0)
    canon_df = pd.read_csv(canon_path, sep='\t', index_col=0)
    if new_df.shape != canon_df.shape:
        raise ValueError(
            f'self-check failed: {label} shape {new_df.shape} != canonical '
            f'{canon_df.shape} — this fork is not reproducing script 04'
        )
    max_dev = np.abs(new_df.values - canon_df.values).max()
    if max_dev > SELFCHECK_ATOL:
        raise ValueError(
            f'self-check failed: {label} differs from {canon_path} by up to {max_dev:.3g} '
            f'(> {SELFCHECK_ATOL:g}). The inner PCA does not depend on noc, so this fork '
            f'has drifted from script 04 and its NOC=6 fit is not comparable.'
        )
    print(f'  {label}: matches canonical (max |dev| = {max_dev:.3g})')

# --- top cells per archetype, by distance in VX space ---
aa_vx     = sca.aa.T @ inner_comps + inner_mean            # (NOC, len(VX_COLS))
dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(NOC)], axis=1)
top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(NOC)]

for k in range(NOC):
    for j in range(k + 1, NOC):
        shared = np.intersect1d(top_cells[k], top_cells[j])
        if len(shared):
            raise ValueError(
                f'top-cell sets for archetype_{k+1} and archetype_{j+1} share {len(shared)} '
                f'cells — at NOC={NOC} the vertices are too close to define distinct cell '
                f'sets at N_TOP_CELLS={N_TOP_CELLS} (G3). This is itself a Gate-B result.'
            )
print(f'Top {N_TOP_CELLS} cells per archetype (pairwise disjoint).')

cell_index = vx_df.index.values
pd.DataFrame([
    {'archetype': f'archetype_{k+1}', 'rank': r, 'cell': cell_index[i]}
    for k in range(NOC) for r, i in enumerate(top_cells[k])
]).to_csv(OUT_TOP_CELLS, sep='\t', index=False)
print(f'Saved {OUT_TOP_CELLS}')

# --- per-vertex composition of the top cells ---
# L4's ARV_rep is elevated at every NOC (0.20-0.24 vs 0.06-0.09 for L2/3): its three donors
# disagree more than any other subclass's. A vertex whose top cells come mostly from one
# donor is a donor artifact, so the background split is written alongside for comparison.
print('Tabulating per-vertex cluster / donor composition...')
comp_rows = []
for field, col in [('cluster', CLUSTER_COL), ('donor', DONOR_COL)]:
    background = vx_df[col].value_counts(normalize=True)
    for level, frac in background.items():
        comp_rows.append({'archetype': 'background', 'field': field, 'level': level,
                          'n': int(round(frac * n_cells)), 'frac': frac})
    for k in range(NOC):
        counts = vx_df[col].iloc[top_cells[k]].value_counts()
        for level, n in counts.items():
            comp_rows.append({'archetype': f'archetype_{k+1}', 'field': field,
                              'level': level, 'n': int(n), 'frac': n / N_TOP_CELLS})
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(OUT_COMPOSITION, sep='\t', index=False)
print(f'Saved {OUT_COMPOSITION}')

donor_comp = comp_df[(comp_df['field'] == 'donor') & (comp_df['archetype'] != 'background')]
bg_max_donor = comp_df[(comp_df['field'] == 'donor')
                       & (comp_df['archetype'] == 'background')]['frac'].max()
print(f'  background max donor fraction: {bg_max_donor:.3f}')
for k in range(NOC):
    sub = donor_comp[donor_comp['archetype'] == f'archetype_{k+1}']
    top = sub.loc[sub['frac'].idxmax()]
    flag = '  <-- donor-dominated' if top['frac'] > 2 * bg_max_donor else ''
    print(f"  archetype_{k+1} ({archetype_names[k]}): max donor {top['level']} "
          f"{top['frac']:.3f}{flag}")
n_clusters_covered = comp_df[(comp_df['field'] == 'cluster')
                             & (comp_df['archetype'] != 'background')] \
    .sort_values('frac').groupby('archetype').tail(1)['level'].nunique()
print(f'  {n_clusters_covered} distinct {CLUSTER_COL} levels are the dominant type of some '
      f'vertex (of {vx_df[CLUSTER_COL].nunique()} present)')

# --- load gene expression (HVGs only; .X is already log-normalized) ---
print('Loading h5ad and filtering to HVGs...')
vx_load    = pd.read_csv(IN_VX_LOADINGS, sep='\t', index_col=0)   # genes x VX1..VX10
adata      = ad.read_h5ad(IN_H5AD)
hvg_genes  = vx_load.index.values
gene_names = adata.var[GENE_NAME_COL].values
hvg_idx    = np.where(np.isin(gene_names, hvg_genes))[0]
# slice the sparse matrix to the HVG columns before densifying (G7)
X_norm     = adata.X[:, hvg_idx].toarray().astype(np.float32)
hvg_names  = gene_names[hvg_idx]
cell_barcodes = adata.obs_names.values
n_hvg = X_norm.shape[1]
print(f'X_norm shape: {X_norm.shape}')

if not np.array_equal(cell_barcodes, cell_index):
    raise ValueError(
        f'h5ad cell order differs from {IN_VX_COORDS} — top-cell indices would point at '
        f'the wrong cells'
    )

# --- conservative one-vs-each Wilcoxon per archetype ---
print('Running conservative one-vs-each Wilcoxon per archetype...')
all_markers, count_rows = [], []
for k in range(NOC):
    archetype_label = f'archetype_{k + 1}'
    print(f'  {archetype_label}...')
    X_in    = X_norm[top_cells[k]]
    frac_in = (X_in > 0).mean(axis=0)

    log2fc_mat, pval_mat, frac_out_list = [], [], []
    for j in range(NOC):
        if j == k:
            continue
        X_out = X_norm[top_cells[j]]
        # human .X is ln(CPM+1) — the /ln(2) converts the difference to log2
        log2fc_j = (X_in.mean(axis=0) - X_out.mean(axis=0)) / np.log(2)
        pvals_j  = np.array([scipy.stats.ranksums(X_in[:, g], X_out[:, g])[1]
                             for g in range(n_hvg)])
        log2fc_mat.append(log2fc_j)
        pval_mat.append(pvals_j)
        frac_out_list.append((X_out > 0).mean(axis=0))

    # worst-case across pairwise comparisons
    log2fc   = np.stack(log2fc_mat).min(axis=0)
    pvals    = np.stack(pval_mat).max(axis=0)
    frac_out = np.stack(frac_out_list).mean(axis=0)

    _, fdr, _, _ = multipletests(pvals, method='fdr_bh')

    df = pd.DataFrame({
        'gene':      hvg_names,
        'archetype': archetype_label,
        'log2FC':    log2fc,
        'pval':      pvals,
        'fdr':       fdr,
        'frac_in':   frac_in,
        'frac_out':  frac_out,
    })
    df = df[(df['frac_in'] >= FRAC_IN_THRESH) & (df['fdr'] < FDR_THRESH)
            & (df['log2FC'] > LOG2FC_THRESH)]
    df = df.sort_values('log2FC', ascending=False)
    all_markers.append(df)
    count_rows.append({'archetype': archetype_label, 'name': archetype_names[k],
                       'n_markers': len(df)})
    print(f'    {len(df)} markers after filtering')
    if len(df) == 0:
        print(f'    WARNING: {archetype_label} has 0 markers — NOC={NOC} is too high '
              f'(return to Gate B)')

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(OUT_MARKERS, sep='\t', index=False)
print(f'Saved {OUT_MARKERS}  ({len(markers_df)} total markers)')

# 04 only prints the 0-marker warning; here it is the primary kill signal for NOC=6, so the
# per-vertex counts are written out rather than left in stdout.
counts_df = pd.DataFrame(count_rows)
counts_df.to_csv(OUT_MARKER_COUNTS, sep='\t', index=False)
print(f'Saved {OUT_MARKER_COUNTS}  (min {counts_df["n_markers"].min()}, '
      f'max {counts_df["n_markers"].max()} markers per vertex)')

# --- archetype score computation ---
print('Computing archetype scores...')
scores = np.zeros((n_cells, NOC), dtype=np.float32)
for k, name in enumerate(archetype_names):
    top_genes = markers_df[markers_df['archetype'] == f'archetype_{k+1}']['gene'].values
    cols = [np.where(hvg_names == g)[0][0] for g in top_genes if g in hvg_names]
    if not cols:
        print(f'  WARNING: no genes found for archetype {name}')
        continue
    mat = X_norm[:, cols]
    lo  = np.percentile(mat, SCORE_PCTILE_LO, axis=0)
    hi  = np.percentile(mat, SCORE_PCTILE_HI, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    scores[:, k] = np.clip((mat - lo) / rng, 0, 1).mean(axis=1)
    print(f'  Score {name}: {len(cols)} genes used')

pd.DataFrame(scores, index=cell_barcodes,
             columns=[f'score_{n}' for n in archetype_names]).to_csv(OUT_SCORES, sep='\t')
print(f'Saved {OUT_SCORES}')

# --- archetype score scatter ---
print('Generating archetype score scatter...')
panels = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')][:max(NDIM - 1, 1)]
panel_3d = (0, 1, 2, 'PC1', 'PC2', 'PC3') if NDIM >= 3 else None
gene_vals = {f'Score {n}': scores[:, k] for k, n in enumerate(archetype_names)}
gene_expr_scatter_html(
    gene_vals=gene_vals,
    x=sca.xp[:, 0], y=sca.xp[:, 1],
    title=f'Jorstad23 human {SUBCLASS} — archetype scores (NOC={NOC})',
    out_path=OUT_SCORE_HTML,
    xp=sca.xp,
    panels=panels,
    panel_3d=panel_3d,
    aa=sca.aa,
    pctile_low=5, pctile_high=95,
    colorbar_title='archetype score [0–1]',
)

print('\nDone.')
