"""Archetype markers for Jorstad23 L2/3 IT — fixed NOC=4.

Fits PCHA at NOC=4 on the VX2/VX6–VX10 subspace, assigns each archetype the
300 nearest cells (Euclidean in VX space), then runs Wilcoxon one-vs-rest
enrichment to find marker genes per archetype.

Reads:
  local_data/res/l23_evo/05.varimax_coords.tsv
  local_data/res/l23_evo/01.pca_loadings.tsv   (HVG gene list)
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Output:
  local_data/res/l23_evo/09.archetype_markers.tsv
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

from scomics.main import SCA

# --- file paths ---
INPUT_FILE      = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
IN_VARIMAX      = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
IN_PCA_LOADINGS = os.path.join(OUT_RES_DIR, '01.pca_loadings.tsv')
OUT_MARKERS     = os.path.join(OUT_RES_DIR, '09.archetype_markers.tsv')
OUT_XP          = os.path.join(OUT_RES_DIR, '09.pcha_xp.tsv')
OUT_AA          = os.path.join(OUT_RES_DIR, '09.pcha_aa.tsv')

# --- parameters ---
VX_COLS         = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
NOC             = 4
NDIM            = 5       # n_fit = NDIM+1 = 6 = len(VX_COLS); drops last PC
N_TOP_CELLS     = 300
FRAC_IN_THRESH  = 0.25
FDR_THRESH      = 0.001
CLUSTER_COL     = 'WithinArea_cluster'

# --- load varimax coords ---
vx_df = pd.read_csv(IN_VARIMAX, sep='\t', index_col=0)
xn    = vx_df[VX_COLS].values   # (n_cells, 6)
types = vx_df[CLUSTER_COL].values

# --- fit PCHA at NOC=4 ---
print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)

pd.DataFrame(sca.xp, index=vx_df.index,
             columns=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_XP, sep='\t')
pd.DataFrame(sca.aa.T, columns=[f'PC{i+1}' for i in range(NDIM)],
             index=[f'archetype_{k+1}' for k in range(NOC)]).to_csv(OUT_AA, sep='\t')
print(f'Saved {OUT_XP} and {OUT_AA}')

# back-project archetype coords from PCA space to VX space.
# sca.pca_ fits NDIM+1 components but drops the last; sca.aa uses the first NDIM only.
# inverse_transform uses all components, so reconstruct manually.
aa_vx = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_   # (NOC, len(VX_COLS))
print(f'Archetype VX coords shape: {aa_vx.shape}')

# --- cell-to-archetype distances in VX space ---
dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(NOC)], axis=1)
top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(NOC)]
print(f'Top {N_TOP_CELLS} cells selected per archetype.')

# --- load gene expression (HVGs only) ---
print('Loading h5ad and filtering to HVGs...')
adata      = ad.read_h5ad(INPUT_FILE)
hvg_genes  = pd.read_csv(IN_PCA_LOADINGS, sep='\t', index_col=0).index.values
gene_names = (adata.var['feature_name'].values
              if 'feature_name' in adata.var.columns
              else adata.var_names.values)
hvg_idx   = np.where(np.isin(gene_names, hvg_genes))[0]
X_norm    = adata.X[:, hvg_idx].toarray().astype(np.float32)  # (n_cells, n_hvg)
hvg_names = gene_names[hvg_idx]
print(f'X_norm shape: {X_norm.shape}')

# --- Wilcoxon one-vs-rest enrichment per archetype ---
print('Running Wilcoxon enrichment per archetype...')
n_cells = X_norm.shape[0]
n_hvg   = X_norm.shape[1]
all_markers = []

for k in range(NOC):
    archetype_label = f'archetype_{k + 1}'
    print(f'  {archetype_label}...')
    in_idx   = top_cells[k]
    out_mask = np.ones(n_cells, dtype=bool)
    out_mask[in_idx] = False

    X_in  = X_norm[in_idx]
    X_out = X_norm[out_mask]

    log2fc   = (X_in.mean(axis=0) - X_out.mean(axis=0)) / np.log(2)
    frac_in  = (X_in  > 0).mean(axis=0)
    frac_out = (X_out > 0).mean(axis=0)

    pvals = np.empty(n_hvg)
    for g in range(n_hvg):
        _, pvals[g] = scipy.stats.ranksums(X_in[:, g], X_out[:, g])

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
    df = df[(df['frac_in'] >= FRAC_IN_THRESH) & (df['fdr'] < FDR_THRESH)]
    df = df.sort_values('log2FC', ascending=False)
    all_markers.append(df)
    print(f'    {len(df)} markers after filtering')

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(OUT_MARKERS, sep='\t', index=False)
print(f'Saved {OUT_MARKERS}  ({len(markers_df)} total markers)')
print('Done.')
