"""Archetype markers — P56 gao25 Arch1+Arch2 cells, VX4/VX6/VX7 subspace.

Based on script 35.2; two additional filters:
  - Only Arch1 and Arch2 cells (Arch3 excluded alongside the prior Arch4 exclusion).
  - VX subspace restricted to VX4, VX6, VX7.
Produces only the gene-expression VX panel HTML.

Reads:
  local_data/res/astro/33.2.varimax_coords.tsv
  local_data/res/astro/33.2.varimax_loadings.tsv
  links/astro/gao25_scrna_astro.h5ad
Outputs:
  local_data/res/astro/35.3.pcha_xp.tsv
  local_data/res/astro/35.3.pcha_aa.tsv
  local_data/res/astro/35.3.inner_pca_components.tsv
  local_data/res/astro/35.3.inner_pca_mean.tsv
  local_data/res/astro/35.3.archetype_markers.tsv
  local_data/fig/astro/35.3.gene_expr_vx_panels.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import gene_expr_scatter_html
from scomics.main import SCA

# --- file paths ---
INPUT_H5AD          = os.path.join(PROJECT_ROOT, 'links', 'astro', 'gao25_scrna_astro.h5ad')
OUT_RES_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_FIG_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
IN_VARIMAX_COORDS   = os.path.join(OUT_RES_DIR, '33.2.varimax_coords.tsv')
IN_VARIMAX_LOADINGS = os.path.join(OUT_RES_DIR, '33.2.varimax_loadings.tsv')
OUT_XP              = os.path.join(OUT_RES_DIR, '35.3.pcha_xp.tsv')
OUT_AA              = os.path.join(OUT_RES_DIR, '35.3.pcha_aa.tsv')
OUT_INNER_COMPS     = os.path.join(OUT_RES_DIR, '35.3.inner_pca_components.tsv')
OUT_INNER_MEAN      = os.path.join(OUT_RES_DIR, '35.3.inner_pca_mean.tsv')
OUT_MARKERS         = os.path.join(OUT_RES_DIR, '35.3.archetype_markers.tsv')
OUT_GENE_EXPR_HTML  = os.path.join(OUT_FIG_DIR, '35.3.gene_expr_vx_panels.html')

VX_COLS        = ['VX4', 'VX6', 'VX7']
NOC            = 4
NDIM           = 3
N_TOP_CELLS    = 300
FRAC_IN_THRESH = 0.25
FDR_THRESH     = 0.001
P56_AGE        = 'P56'

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load varimax coords, filter to Arch1 + Arch2 ---
vx_df_all   = pd.read_csv(IN_VARIMAX_COORDS, sep='\t', index_col=0)
vx_df       = vx_df_all[vx_df_all['archetype'].isin(['Arch1', 'Arch2'])].copy()
xn          = vx_df[VX_COLS].values
arch_labels = vx_df['archetype'].values
donors      = vx_df['donor_name'].values
print(f'Arch1+Arch2 cells: {len(vx_df)}  '
      f'({pd.Series(arch_labels).value_counts().to_dict()})')

# --- fit PCHA ---
print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(xn, arch_labels)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC, drop_pcs=[])

pd.DataFrame(sca.xp, index=vx_df.index,
             columns=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_XP, sep='\t')
pd.DataFrame(sca.aa.T, columns=[f'PC{i+1}' for i in range(NDIM)],
             index=[f'archetype_{k+1}' for k in range(NOC)]).to_csv(OUT_AA, sep='\t')
print(f'Saved {OUT_XP} and {OUT_AA}')

inner_comps = sca.pca_.components_[:NDIM]
inner_mean  = sca.pca_.mean_
pd.DataFrame(inner_comps, columns=VX_COLS,
             index=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_INNER_COMPS, sep='\t')
pd.DataFrame(inner_mean.reshape(1, -1), columns=VX_COLS).to_csv(OUT_INNER_MEAN, sep='\t')
print(f'Saved {OUT_INNER_COMPS} and {OUT_INNER_MEAN}')

# --- nearest cells per archetype ---
aa_vx     = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_
dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(NOC)], axis=1)
top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(NOC)]
print(f'Top {N_TOP_CELLS} cells selected per archetype.')

# --- load raw counts (Arch1+Arch2 cells only) and normalize ---
print('Loading h5ad for marker gene analysis...')
adata     = ad.read_h5ad(INPUT_H5AD)
adata_p56 = adata[adata.obs['Age'] == P56_AGE].copy()
adata_p56 = adata_p56[vx_df.index].copy()
print(f'  Arch1+Arch2 P56 cells: {adata_p56.shape[0]}')

hvg_genes  = pd.read_csv(IN_VARIMAX_LOADINGS, sep='\t', index_col=0).index.values
gene_names = adata_p56.var_names.values
hvg_idx    = np.where(np.isin(gene_names, hvg_genes))[0]
hvg_names  = gene_names[hvg_idx]

x_raw = adata_p56.X[:, hvg_idx]
if sp.issparse(x_raw):
    x_raw = x_raw.toarray()
x_raw  = x_raw.astype(np.float32)
depths = x_raw.sum(axis=1, keepdims=True)
depths = np.where(depths == 0, 1, depths)
X_norm = np.log2(x_raw / depths * 1e4 + 1)
print(f'X_norm shape: {X_norm.shape}')

# --- Wilcoxon one-vs-rest per archetype ---
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

    log2fc   = X_in.mean(axis=0) - X_out.mean(axis=0)
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

# --- build gene_vals (z-scored log2CP10k) ---
vx_load_df = pd.read_csv(IN_VARIMAX_LOADINGS, sep='\t', index_col=0)
top_genes_ordered = list(dict.fromkeys(
    gene
    for vx_col in VX_COLS
    for gene in vx_load_df[vx_col].abs().nlargest(5).index.tolist()
))

gene_to_hidx = {g: i for i, g in enumerate(hvg_names)}
from scipy.stats import zscore as _zscore
gene_vals = {}
for gene in top_genes_ordered:
    if gene in gene_to_hidx:
        gene_vals[gene] = _zscore(X_norm[:, gene_to_hidx[gene]])

for arch_name in [f'archetype_{k+1}' for k in range(NOC)]:
    for gene in markers_df[markers_df['archetype'] == arch_name]['gene'].head(5):
        if gene not in gene_vals and gene in gene_to_hidx:
            gene_vals[gene] = _zscore(X_norm[:, gene_to_hidx[gene]])

# --- gene expression VX panels ---
# VX_COLS indices: VX4=0, VX6=1, VX7=2
# row 1: all pairwise combinations; row 2: 3D
panels_vx = [
    (0, 1, 'VX4', 'VX6'),
    (0, 2, 'VX4', 'VX7'),
    (1, 2, 'VX6', 'VX7'),
]

gene_expr_scatter_html(
    x=xn[:, 0], y=xn[:, 1],
    gene_vals=gene_vals,
    title='Top genes — P56 Arch1+Arch2 VX4/VX6/VX7',
    out_path=OUT_GENE_EXPR_HTML,
    xp=xn,
    panels=panels_vx,
    panel_3d=(0, 1, 2, 'VX4', 'VX6', 'VX7'),
)
print('Done.')
