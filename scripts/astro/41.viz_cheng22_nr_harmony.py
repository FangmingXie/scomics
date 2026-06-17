"""UMAP, PC1-2, and VX1-2 visualizations for cheng22 NR Harmony-corrected embedding.

Reads:
  local_data/res/astro/39.cheng22_nr_harmony.h5ad
Outputs:
  local_data/fig/astro/41.umap.html
  local_data/fig/astro/41.pc12.html
  local_data/fig/astro/41.vx12.html
"""

import os
import sys
import numpy as np
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_2d_categorical_html, gene_expr_scatter_html

IN_H5AD    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '39.cheng22_nr_harmony.h5ad')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_UMAP   = os.path.join(OUT_FIG_DIR, '41.umap.html')
OUT_PC12   = os.path.join(OUT_FIG_DIR, '41.pc12.html')
OUT_VX12   = os.path.join(OUT_FIG_DIR, '41.vx12.html')

GENES = ['Gfap', 'Apoe', 'Mfge8', 'Id3', 'Lama3', 'Trpm3', 'Il33']

os.makedirs(OUT_FIG_DIR, exist_ok=True)


def make_html(coords, xlabel, ylabel, cell_metadata, gene_vals, title_cat, title_gene):
    html_cat = scatter_2d_categorical_html(
        xp_grid=[coords],
        cell_metadata=cell_metadata,
        title=title_cat,
        out_path=None,
        xlabel=xlabel, ylabel=ylabel,
        return_html=True,
    )
    html_gene = gene_expr_scatter_html(
        x=coords[:, 0], y=coords[:, 1],
        gene_vals=gene_vals,
        title=title_gene,
        out_path=None,
        xlabel=xlabel, ylabel=ylabel,
        return_html=True,
    )
    return f'<html><body>{html_cat}{html_gene}</body></html>'


# --- load ---
print(f'Loading {IN_H5AD}')
adata = ad.read_h5ad(IN_H5AD)
print(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

umap_coords = np.array(adata.obsm['X_umap'])
pca_coords  = np.array(adata.obsm['X_pca'])[:, :2]
vx_coords   = np.array(adata.obsm['X_vx'])[:, :2]

cell_metadata = {
    'Type':      adata.obs['Type'].values,
    'archetype': adata.obs['archetype'].values,
    'leiden':    adata.obs['leiden'].values,
    'Sample':    adata.obs['Sample'].values,
}

# gene expression: adata.X is already CP10k → log2(1+x) → z-score per gene
present = [g for g in GENES if g in adata.var_names]
missing = [g for g in GENES if g not in adata.var_names]
if missing:
    print(f'  Warning: genes not in HVG set, skipping: {missing}')
gene_idx = {g: adata.var_names.get_loc(g) for g in present}
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
gene_vals = {g: X[:, gene_idx[g]] for g in present}
gene_vals['library_size'] = adata.obs['depth'].values

# --- UMAP ---
print('Building UMAP HTML...')
html = make_html(
    umap_coords, 'UMAP1', 'UMAP2', cell_metadata, gene_vals,
    title_cat='cheng22 NR astrocytes — UMAP (Harmony corrected)',
    title_gene='cheng22 NR astrocytes — gene expression on UMAP',
)
with open(OUT_UMAP, 'w') as f:
    f.write(html)
print(f'Saved {OUT_UMAP}')

# --- PC1-2 ---
print('Building PC1-2 HTML...')
html = make_html(
    pca_coords, 'PC1', 'PC2', cell_metadata, gene_vals,
    title_cat='cheng22 NR astrocytes — PC1 vs PC2 (raw PCA)',
    title_gene='cheng22 NR astrocytes — gene expression on PC1-2',
)
with open(OUT_PC12, 'w') as f:
    f.write(html)
print(f'Saved {OUT_PC12}')

# --- VX1-2 ---
print('Building VX1-2 HTML...')
html = make_html(
    vx_coords, 'VX1', 'VX2', cell_metadata, gene_vals,
    title_cat='cheng22 NR astrocytes — VX1 vs VX2 (varimax on Harmony)',
    title_gene='cheng22 NR astrocytes — gene expression on VX1-2',
)
with open(OUT_VX12, 'w') as f:
    f.write(html)
print(f'Saved {OUT_VX12}')
print('Done.')
