# UMAP visualization of jainlab26_cux2cre colored by marker gene expression.
# Loads labeled h5ad, extracts CP10k log1p expression for selected genes,
# and writes a single HTML with toggle buttons per gene.

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import scatter_2d_categorical_html

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

INPUT_H5AD  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00.jainlab26_cux2cre_labeled.h5ad')
OUT_FIG     = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '01.umap_gene_expr.html')

GENES = ['Slc17a7', 'Gad1', 'Cux2', 'Rorb', 'Foxp2']

print('Loading labeled h5ad...')
adata = ad.read_h5ad(INPUT_H5AD)
print(f'  {adata.n_obs} cells, {adata.n_vars} genes')

umap_coords = adata.obsm['X_umap']

X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
depths = X.sum(axis=1, keepdims=True)
X_norm = np.log1p(X / depths * 1e4)

missing = [g for g in GENES if g not in adata.var_names]
if missing:
    raise ValueError(f'Genes not found in adata: {missing}')

gene_metadata = {}
for gene in GENES:
    idx = adata.var_names.get_loc(gene)
    gene_metadata[gene] = X_norm[:, idx]

scatter_2d_categorical_html(
    xp_grid=[umap_coords],
    cell_metadata=gene_metadata,
    title='UMAP — marker gene expression (jainlab26_cux2cre)',
    out_path=OUT_FIG,
    xlabel='UMAP1', ylabel='UMAP2',
)
print(f'Saved → {OUT_FIG}')
