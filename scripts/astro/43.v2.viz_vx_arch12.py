"""VX1-3 pairwise and 3D visualizations for cheng22 all P28+P38 Arch1+Arch2 cells.

Same as script 43 but reads from the all-samples embedding (script 41.v2).
Gene expression loaded from the original h5ad (all genes, log2 CP10k).

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/41.v2.cheng22_arch12_harmony.h5ad
Outputs:
  local_data/fig/astro/43.v2.vx_gene_expr.html   -- 2D pairs + 3D colored by top-VX gene expression
  local_data/fig/astro/43.v2.vx_categorical.html  -- same layout colored by categorical labels
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import gene_expr_scatter_html, scatter_categorical_html

IN_H5AD     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.v2.cheng22_arch12_harmony.h5ad')
IN_RAW_H5AD = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_GENE    = os.path.join(OUT_FIG_DIR, '43.v2.vx_gene_expr.html')
OUT_CAT     = os.path.join(OUT_FIG_DIR, '43.v2.vx_categorical.html')

N_TOP = 5
EXTRA_GENES  = ['Chrdl1', 'Igfbp2', 'Cdh13', 'Cdh19', 'Gria1', 'Il33']
AVG_GENE_SET = ['Chrdl1', 'Igfbp2', 'Lef1']

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load embedding h5ad ---
print(f'Loading {IN_H5AD}')
adata = ad.read_h5ad(IN_H5AD)
print(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

vx_scores     = np.array(adata.obsm['X_vx'],       dtype=np.float64)
vx_load       = np.array(adata.varm['VX_loadings'], dtype=np.float64)
hvg_names     = adata.var_names.values
cell_barcodes = adata.obs_names.values

# --- top N_TOP genes by |loading| for VX1, VX2, VX3 ---
load_df = pd.DataFrame(vx_load, index=hvg_names, columns=[f'VX{i+1}' for i in range(vx_load.shape[1])])
top_genes_per_comp = {}
for comp in ['VX1', 'VX2', 'VX3']:
    top_genes_per_comp[comp] = load_df[comp].abs().nlargest(N_TOP).index.tolist()
    print(f'  Top {N_TOP} genes for {comp}: {top_genes_per_comp[comp]}')

# deduplicate while preserving per-component order, then append extras
seen = set()
ordered_genes = []
for comp in ['VX1', 'VX2', 'VX3']:
    for g in top_genes_per_comp[comp]:
        if g not in seen:
            ordered_genes.append(g)
            seen.add(g)
for g in EXTRA_GENES + AVG_GENE_SET:
    if g not in seen:
        ordered_genes.append(g)
        seen.add(g)
print(f'  Genes for visualization ({len(ordered_genes)}): {ordered_genes}')

# --- load raw counts and compute log2(CP10k) ---
print(f'Loading raw counts from {IN_RAW_H5AD}')
adata_raw = ad.read_h5ad(IN_RAW_H5AD)

raw_idx = pd.Index(adata_raw.obs_names).get_indexer(cell_barcodes)
assert (raw_idx >= 0).all(), 'Some barcodes not found in raw h5ad'

missing = [g for g in ordered_genes if g not in adata_raw.var_names]
if missing:
    raise ValueError(f'Genes not found in raw h5ad: {missing}')

gene_col_idx = [adata_raw.var_names.get_loc(g) for g in ordered_genes]
x_raw = adata_raw.X[raw_idx]
x_raw = x_raw.toarray() if sp.issparse(x_raw) else np.array(x_raw, dtype=np.float64)
depths = x_raw.sum(axis=1)
x_genes = x_raw[:, gene_col_idx]
x_lognorm = np.log2(1 + x_genes / depths.reshape(-1, 1) * 1e4)
gene_vals = {g: x_lognorm[:, i] for i, g in enumerate(ordered_genes)}

# averaged gene set
avg_label = f'avg({",".join(AVG_GENE_SET)})'
avg_cols = [ordered_genes.index(g) for g in AVG_GENE_SET]
gene_vals[avg_label] = x_lognorm[:, avg_cols].mean(axis=1)

print(f'  Extracted log2(CP10k) for {len(ordered_genes)} genes + average "{avg_label}"')

# --- panel layout: VX1-2, VX1-3, VX2-3 + 3D VX1-2-3 ---
panels   = [(0, 1, 'VX1', 'VX2'), (0, 2, 'VX1', 'VX3'), (1, 2, 'VX2', 'VX3')]
panel_3d = (0, 1, 2, 'VX1', 'VX2', 'VX3')

# --- gene expression HTML ---
print('Building gene expression HTML...')
gene_expr_scatter_html(
    x=None, y=None,
    gene_vals=gene_vals,
    title='cheng22 all P28+P38 Arch1+Arch2 — VX gene expression',
    out_path=OUT_GENE,
    xp=vx_scores,
    panels=panels,
    panel_3d=panel_3d,
    colorbar_title='log2(CP10k)',
    marker_size=3,
    marker_opacity=0.6,
)

# --- categorical HTML ---
print('Building categorical HTML...')
cell_metadata = {
    'archetype': adata.obs['archetype'].values,
    'Age':       adata.obs['Age'].values,
    'leiden':    adata.obs['leiden'].values,
    'Sample':    adata.obs['Sample'].values,
    'Type':      adata.obs['Type'].values,
}
scatter_categorical_html(
    xp_grid=[vx_scores],
    cell_metadata=cell_metadata,
    title='cheng22 all P28+P38 Arch1+Arch2 — VX categorical labels',
    out_path=OUT_CAT,
    panels=panels,
    panel_3d=panel_3d,
    ordered_labels=['Age'],
)

print('Done.')
