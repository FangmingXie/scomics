"""Archetype scores for Jorstad23 human L2/3 IT — seeded re-fit that also saves loadings.

Copy of script 25 with two changes: the PCHA fit is seeded (py_pcha draws its
furthest-sum start index and its S initialization from the global NumPy RNG, and takes
no seed argument), and the inner PCA that maps the 6 varimax axes into the 5-D PCHA
space is written out — components, mean, and gene-level loadings composed with the
script-05 varimax loadings. Script 25 and every 25.* output are kept as the original
record; because 25.* was fit unseeded, the vertices here will not match it exactly and
the downstream top-300 cell sets, markers, and scores differ numerically (qualitatively
the same). From this script on, the 56.* record is reproducible.

Reads:
  local_data/res/l23_evo/05.varimax_coords.tsv
  local_data/res/l23_evo/05.varimax_loadings.tsv
  local_data/res/l23_evo/01.pca_loadings.tsv
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  local_data/res/l23_evo/56.human_archetype_markers.tsv
  local_data/res/l23_evo/56.human_pcha_xp.tsv
  local_data/res/l23_evo/56.human_pcha_aa.tsv
  local_data/res/l23_evo/56.human_pcha_inner_components.tsv
  local_data/res/l23_evo/56.human_pcha_inner_mean.tsv
  local_data/res/l23_evo/56.human_pcha_gene_loadings.tsv
  local_data/res/l23_evo/56.human_archetype_scores.tsv
  local_data/fig/l23_evo/56.human_archetype_scatter.html
  local_data/fig/l23_evo/56.human_archetype_scores.html
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

from viz import scatter_categorical_html, gene_expr_scatter_html
from scomics.main import SCA

# --- file paths ---
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_VARIMAX     = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
IN_VARIMAX_LOADINGS = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
IN_PCA_LOADINGS= os.path.join(OUT_RES_DIR, '01.pca_loadings.tsv')
INPUT_HUMAN    = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_MARKERS    = os.path.join(OUT_RES_DIR, '56.human_archetype_markers.tsv')
OUT_PCHA_XP    = os.path.join(OUT_RES_DIR, '56.human_pcha_xp.tsv')
OUT_PCHA_AA    = os.path.join(OUT_RES_DIR, '56.human_pcha_aa.tsv')
OUT_INNER_COMPS   = os.path.join(OUT_RES_DIR, '56.human_pcha_inner_components.tsv')
OUT_INNER_MEAN    = os.path.join(OUT_RES_DIR, '56.human_pcha_inner_mean.tsv')
OUT_GENE_LOADINGS = os.path.join(OUT_RES_DIR, '56.human_pcha_gene_loadings.tsv')
OUT_SCORES     = os.path.join(OUT_RES_DIR, '56.human_archetype_scores.tsv')
OUT_ARCH_HTML  = os.path.join(OUT_FIG_DIR, '56.human_archetype_scatter.html')
OUT_SCORE_HTML = os.path.join(OUT_FIG_DIR, '56.human_archetype_scores.html')

# --- parameters ---
CLUSTER_COL     = 'WithinArea_cluster'
VX_COLS         = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
NOC             = 4
NDIM            = len(VX_COLS) - 1   # 5
PCHA_SEED       = 0
N_TOP_CELLS     = 300
FRAC_IN_THRESH  = 0.25
FDR_THRESH      = 0.001
LOG2FC_THRESH   = np.log2(1.5)
ARCHETYPE_NAMES = ['A', 'B', 'C', 'D']
SCORE_PCTILE_LO = 2
SCORE_PCTILE_HI = 98

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load varimax coords ---
vx_df  = pd.read_csv(IN_VARIMAX, sep='\t', index_col=0)
xn     = vx_df[VX_COLS].values
types  = vx_df[CLUSTER_COL].values

# --- fit PCHA ---
print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')
# py_pcha seeds itself from the global NumPy RNG and takes no seed argument; fixing the
# global seed here is what makes the 56.* record reproducible (and why it will not
# reproduce the unseeded 25.* vertices).
np.random.seed(PCHA_SEED)
sca.proj_and_pcha(NDIM, NOC)

pd.DataFrame(sca.xp, index=vx_df.index,
             columns=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_PCHA_XP, sep='\t')
pd.DataFrame(sca.aa.T, columns=[f'PC{i+1}' for i in range(NDIM)],
             index=[f'archetype_{k+1}' for k in range(NOC)]).to_csv(OUT_PCHA_AA, sep='\t')
print(f'Saved {OUT_PCHA_XP} and {OUT_PCHA_AA}')

# --- save inner PCA parameters (VX axes -> PCHA space) ---
# PCs as rows, VX axes as columns — matches the `sca.aa.T @ components_ + mean_` math
# used below, no transpose needed.
inner_comps = sca.pca_.components_[:NDIM]   # (NDIM, len(VX_COLS))
inner_mean  = sca.pca_.mean_                # (len(VX_COLS),)

pd.DataFrame(inner_comps, columns=VX_COLS,
             index=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_INNER_COMPS, sep='\t')
pd.DataFrame(inner_mean.reshape(1, -1), columns=VX_COLS).to_csv(OUT_INNER_MEAN, sep='\t')
print(f'Saved {OUT_INNER_COMPS} and {OUT_INNER_MEAN}')

# gene-level loadings: compose genes->VX (script 05) with VX->PC (inner PCA).
# Units are inherited from scripts 01/05 — weights on StandardScaler-z-scored HVG
# expression, not raw log-CP10k.
vx_load = pd.read_csv(IN_VARIMAX_LOADINGS, sep='\t', index_col=0)   # genes x VX1..VX10
gene_ld = vx_load[VX_COLS].values @ inner_comps.T                   # (n_genes, NDIM)
pd.DataFrame(gene_ld, index=vx_load.index,
             columns=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_GENE_LOADINGS, sep='\t')
print(f'Saved {OUT_GENE_LOADINGS}  ({gene_ld.shape[0]} genes x {gene_ld.shape[1]} PCs)')

# back-project archetype coords from PCHA space to VX space
aa_vx = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_   # (NOC, len(VX_COLS))

# --- cell-to-archetype distances in VX space ---
dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(NOC)], axis=1)
top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(NOC)]
print(f'Top {N_TOP_CELLS} cells selected per archetype.')

# --- load gene expression (HVGs only) ---
print('Loading h5ad and filtering to HVGs...')
adata      = ad.read_h5ad(INPUT_HUMAN)
hvg_genes  = pd.read_csv(IN_PCA_LOADINGS, sep='\t', index_col=0).index.values
gene_names = (adata.var['feature_name'].values
              if 'feature_name' in adata.var.columns
              else adata.var_names.values)
hvg_idx    = np.where(np.isin(gene_names, hvg_genes))[0]
X_norm     = adata.X[:, hvg_idx].toarray().astype(np.float32)  # already log-normalized
hvg_names  = gene_names[hvg_idx]
cell_barcodes = adata.obs_names.values
n_cells = X_norm.shape[0]
n_hvg   = X_norm.shape[1]
print(f'X_norm shape: {X_norm.shape}')

# --- conservative one-vs-each Wilcoxon per archetype ---
print('Running conservative one-vs-each Wilcoxon per archetype...')
all_markers = []

for k in range(NOC):
    archetype_label = f'archetype_{k + 1}'
    print(f'  {archetype_label}...')
    in_idx  = top_cells[k]
    X_in    = X_norm[in_idx]
    frac_in = (X_in > 0).mean(axis=0)

    log2fc_mat    = []
    pval_mat      = []
    frac_out_list = []

    for j in range(NOC):
        if j == k:
            continue
        X_out = X_norm[top_cells[j]]
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
    df = df[(df['frac_in'] >= FRAC_IN_THRESH) & (df['fdr'] < FDR_THRESH) & (df['log2FC'] > LOG2FC_THRESH)]
    df = df.sort_values('log2FC', ascending=False)
    all_markers.append(df)
    print(f'    {len(df)} markers after filtering')

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(OUT_MARKERS, sep='\t', index=False)
print(f'Saved {OUT_MARKERS}  ({len(markers_df)} total markers)')

# --- archetype score computation ---
print('Computing archetype scores...')
scores = np.zeros((n_cells, NOC), dtype=np.float32)

for k, name in enumerate(ARCHETYPE_NAMES):
    top_genes = markers_df[markers_df['archetype'] == f'archetype_{k+1}']['gene'].values
    cols = [np.where(hvg_names == g)[0][0] for g in top_genes if g in hvg_names]
    if not cols:
        print(f'  WARNING: no genes found for archetype {name}')
        continue
    mat = X_norm[:, cols]
    lo  = np.percentile(mat, SCORE_PCTILE_LO, axis=0)
    hi  = np.percentile(mat, SCORE_PCTILE_HI, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    mat_norm     = np.clip((mat - lo) / rng, 0, 1)
    scores[:, k] = mat_norm.mean(axis=1)
    print(f'  Score {name}: {len(cols)} genes used')

scores_df = pd.DataFrame(scores, index=cell_barcodes,
                          columns=[f'score_{n}' for n in ARCHETYPE_NAMES])
scores_df.to_csv(OUT_SCORES, sep='\t')
print(f'Saved {OUT_SCORES}')

# --- PCHA scatter ---
print('\nGenerating PCHA scatter...')
panels = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')]
scatter_categorical_html(
    xp_grid=[sca.xp],
    cell_metadata={CLUSTER_COL: types},
    title=f'Jorstad23 human L2/3 IT — varimax PCHA space (NOC={NOC})',
    out_path=OUT_ARCH_HTML,
    panels=panels,
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    arch_vis=sca.aa,
)

# --- archetype score scatter ---
print('Generating archetype score scatter...')
gene_vals = {f'Score {n}': scores[:, k] for k, n in enumerate(ARCHETYPE_NAMES)}
gene_expr_scatter_html(
    gene_vals=gene_vals,
    x=sca.xp[:, 0], y=sca.xp[:, 1],
    title=f'Jorstad23 human L2/3 IT — archetype scores (NOC={NOC})',
    out_path=OUT_SCORE_HTML,
    xp=sca.xp,
    panels=panels,
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    aa=sca.aa,
    pctile_low=5, pctile_high=95,
    colorbar_title='archetype score [0–1]',
)
print('Done.')
