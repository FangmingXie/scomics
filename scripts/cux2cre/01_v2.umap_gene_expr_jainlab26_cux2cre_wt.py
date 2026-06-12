# UMAP visualization of jainlab26_cux2cre + jainlab26_wt colored by marker gene expression.
# Loads both labeled h5ads from 00_v2, stacks cells, and writes a single HTML with
# toggle buttons per gene plus source label.

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import scatter_2d_categorical_html

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

INPUT_CUX2CRE_H5AD = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_H5AD      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_wt_labeled.h5ad')
OUT_FIG            = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '01_v2.umap_gene_expr.html')

GENES = ['Slc17a7', 'Gad1', 'Cux2', 'Rorb', 'Foxp2']

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

print('Loading labeled h5ads...')
cux2cre = ad.read_h5ad(INPUT_CUX2CRE_H5AD)
wt      = ad.read_h5ad(INPUT_WT_H5AD)
print(f'  cux2cre: {cux2cre.n_obs} cells  |  wt: {wt.n_obs} cells')

umap_coords = np.vstack([cux2cre.obsm['X_umap'], wt.obsm['X_umap']])

def _to_norm(adata):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    depths = X.sum(axis=1, keepdims=True)
    return np.log1p(X / depths * 1e4)

X_norm_cux2cre = _to_norm(cux2cre)
X_norm_wt      = _to_norm(wt)

missing = [g for g in GENES if g not in cux2cre.var_names]
if missing:
    raise ValueError(f'Genes not found: {missing}')

cell_metadata = {'source': np.array(['jainlab26_cux2cre'] * cux2cre.n_obs + ['jainlab26_wt'] * wt.n_obs)}
for gene in GENES:
    idx = cux2cre.var_names.get_loc(gene)
    cell_metadata[gene] = np.concatenate([X_norm_cux2cre[:, idx], X_norm_wt[:, idx]])

scatter_2d_categorical_html(
    xp_grid=[umap_coords],
    cell_metadata=cell_metadata,
    title='UMAP — marker gene expression (jainlab26_cux2cre + jainlab26_wt)',
    out_path=OUT_FIG,
    xlabel='UMAP1', ylabel='UMAP2',
)
print(f'Saved → {OUT_FIG}')
