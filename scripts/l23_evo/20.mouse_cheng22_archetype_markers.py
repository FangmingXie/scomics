"""Archetype markers for Cheng22 mouse L2/3 IT — fixed NOC.

Fits PCHA at the selected NOC on the VX subspace, assigns each archetype the
300 nearest cells (Euclidean in VX space), then runs Wilcoxon one-vs-rest
enrichment to find marker genes per archetype.

Update VX_COLS after inspecting 18.mouse_vx_variance_partition.tsv.
Update NOC after inspecting 19.mouse_num_archetype_metrics.png.

Reads:
  local_data/res/l23_evo/18.mouse_varimax_coords.tsv
  local_data/res/l23_evo/18.mouse_varimax_loadings.tsv
  links/l23_evo/cheng22_mouse_IT_P28.h5ad
Outputs:
  local_data/res/l23_evo/20.mouse_archetype_markers.tsv
  local_data/res/l23_evo/20.mouse_pcha_xp.tsv
  local_data/res/l23_evo/20.mouse_pcha_aa.tsv
  local_data/fig/l23_evo/20.mouse_archetype_scatter.html
  local_data/fig/l23_evo/20.mouse_pc_scatter.html
  local_data/fig/l23_evo/20.mouse_gene_expr.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html, gene_expr_scatter_html
from scomics.main import SCA

# --- file paths ---
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_VX_COORDS   = os.path.join(OUT_RES_DIR, '18.mouse_varimax_coords.tsv')
IN_VX_LOADINGS = os.path.join(OUT_RES_DIR, '18.mouse_varimax_loadings.tsv')
INPUT_MOUSE    = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'cheng22_mouse_IT_P28.h5ad')
OUT_MARKERS    = os.path.join(OUT_RES_DIR, '20.mouse_archetype_markers.tsv')
OUT_PCHA_XP    = os.path.join(OUT_RES_DIR, '20.mouse_pcha_xp.tsv')
OUT_PCHA_AA    = os.path.join(OUT_RES_DIR, '20.mouse_pcha_aa.tsv')
OUT_HTML       = os.path.join(OUT_FIG_DIR, '20.mouse_archetype_scatter.html')
OUT_PC_HTML    = os.path.join(OUT_FIG_DIR, '20.mouse_pc_scatter.html')
OUT_EXPR_HTML  = os.path.join(OUT_FIG_DIR, '20.mouse_gene_expr.html')

# --- parameters ---
MOUSE_SUBCLASS = 'L2/3'
CLUSTER_COL    = 'Type'
SAMPLE_COL     = 'sample'
# Update after inspecting 18.mouse_vx_variance_partition.tsv:
VX_COLS        = ['VX1', 'VX2', 'VX6', 'VX7', 'VX8', 'VX10']
# Update after inspecting 19.mouse_num_archetype_metrics.png:
NOC            = 3
NDIM           = len(VX_COLS) - 1   # 5
N_TOP_CELLS    = 300
FRAC_IN_THRESH = 0.25
FDR_THRESH     = 0.001
N_TOP_GENES    = 5

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load varimax coords ---
vx_df   = pd.read_csv(IN_VX_COORDS, sep='\t', index_col=0)
xn      = vx_df[VX_COLS].values
types   = vx_df[CLUSTER_COL].values
samples = vx_df[SAMPLE_COL].values

# --- fit PCHA ---
print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)

pd.DataFrame(sca.xp, index=vx_df.index,
             columns=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_PCHA_XP, sep='\t')
pd.DataFrame(sca.aa.T, columns=[f'PC{i+1}' for i in range(NDIM)],
             index=[f'archetype_{k+1}' for k in range(NOC)]).to_csv(OUT_PCHA_AA, sep='\t')
print(f'Saved {OUT_PCHA_XP} and {OUT_PCHA_AA}')

# back-project archetype coords from PCHA space to VX space
aa_vx = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_   # (NOC, len(VX_COLS))
print(f'Archetype VX coords shape: {aa_vx.shape}')

# --- cell-to-archetype distances in VX space ---
dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(NOC)], axis=1)
top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(NOC)]
print(f'Top {N_TOP_CELLS} cells selected per archetype.')

# --- load normalized gene expression (HVGs only) ---
print('Loading h5ad and filtering to HVGs...')
m_adata   = ad.read_h5ad(INPUT_MOUSE)
m_adata   = m_adata[m_adata.obs['Subclass'] == MOUSE_SUBCLASS]
hvg_genes = pd.read_csv(IN_VX_LOADINGS, sep='\t', index_col=0).index.values

X_raw  = m_adata.X.toarray().astype(np.float32)
depths = X_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_norm_full = np.log2(X_raw / depths * 1e4 + 1)

gene_names = m_adata.var_names.values
hvg_idx    = np.where(np.isin(gene_names, hvg_genes))[0]
X_norm     = X_norm_full[:, hvg_idx]
hvg_names  = gene_names[hvg_idx]
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

# --- PCHA scatter: archetypes + Type/sample dropdown ---
print('\nGenerating PCHA scatter...')
panels = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')]
scatter_categorical_html(
    xp_grid=[sca.xp],
    cell_metadata={CLUSTER_COL: types, SAMPLE_COL: samples},
    title=f'Cheng22 mouse L2/3 IT — varimax PCHA space (NOC={NOC})',
    out_path=OUT_HTML,
    panels=panels,
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    arch_vis=sca.aa,
)

# --- Raw PCA scatter (pre-varimax, pre-VX selection) ---
print('Generating raw PCA scatter...')
X_scaled_pc = StandardScaler().fit_transform(X_norm_full[:, hvg_idx])
pc_scores   = PCA(n_components=3, random_state=0).fit_transform(X_scaled_pc)
scatter_categorical_html(
    xp_grid=[pc_scores],
    cell_metadata={CLUSTER_COL: types, SAMPLE_COL: samples},
    title='Cheng22 mouse L2/3 IT — raw PCA space (PC1/PC2/PC3, pre-varimax)',
    out_path=OUT_PC_HTML,
    panels=panels,
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
)

# --- Gene expression scatter: top N genes per archetype ---
print('Generating gene expression scatter...')
top_genes_ordered = []
seen = set()
for k in range(NOC):
    arch_markers = markers_df[markers_df['archetype'] == f'archetype_{k+1}'].head(N_TOP_GENES)
    for gene in arch_markers['gene'].values:
        if gene not in seen:
            seen.add(gene)
            top_genes_ordered.append(gene)

gene_vals = {
    gene: X_norm[:, np.where(hvg_names == gene)[0][0]]
    for gene in top_genes_ordered
    if len(np.where(hvg_names == gene)[0]) > 0
}
print(f'  {len(gene_vals)} genes selected (top {N_TOP_GENES} per archetype, deduplicated)')

gene_expr_scatter_html(
    gene_vals=gene_vals,
    x=sca.xp[:, 0], y=sca.xp[:, 1],
    title=f'Cheng22 mouse L2/3 IT — top archetype marker genes (NOC={NOC})',
    out_path=OUT_EXPR_HTML,
    xp=sca.xp,
    panels=panels,
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    aa=sca.aa,
    pctile_low=5, pctile_high=95,
    colorbar_title='log2(CP10k+1)',
)
print('Done.')
