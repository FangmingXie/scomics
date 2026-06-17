# Visualize Xenium UMAP (from 00.v2) with two side-by-side panels:
#   Left:  gene expression dropdown (Slc17a6, Slc17a7) — log1p CP10k
#   Right: categorical label dropdown (leiden, tran19_subtype)

import os
import sys
import numpy as np
import scipy.sparse as sp
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_2d_categorical_html

# --- File paths ---
IN_XENIUM = os.path.join(PROJECT_ROOT, 'links', 'retina', 'trial', 'retina_xenium.h5ad')

OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'retina')
OUT_FIG_UMAP = os.path.join(OUT_FIG_DIR, '02.xenium_umap.html')

SELECTED_GENES = ['Slc17a6', 'Slc17a7']
SUBSAMPLE_FRAC = 0.20
SUBSAMPLE_SEED = 0

os.makedirs(OUT_FIG_DIR, exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


# --- Load ---
print('Loading xenium...')
adata = ad.read_h5ad(IN_XENIUM)
print(f'  {adata.n_obs} cells × {adata.n_vars} genes')

# --- Subsample ---
rng     = np.random.default_rng(SUBSAMPLE_SEED)
n_sub   = int(adata.n_obs * SUBSAMPLE_FRAC)
sub_idx = rng.choice(adata.n_obs, size=n_sub, replace=False)
sub_idx = np.sort(sub_idx)
print(f'  Subsampled to {n_sub} cells ({SUBSAMPLE_FRAC*100:.0f}%)')

umap_coords = adata.obsm['X_umap'][sub_idx]

# --- Gene expression: log1p CP10k ---
depths = np.asarray(adata.X.sum(axis=1)).ravel().astype(float)
gene_expr = {}
for gene in SELECTED_GENES:
    if gene not in adata.var_names:
        print(f'  WARNING: {gene} not in dataset — skipping.')
        continue
    gi   = adata.var_names.get_loc(gene)
    col  = _to_array(adata.X[:, gi]).ravel().astype(float)
    gene_expr[gene] = np.log1p(col / depths * 1e4)[sub_idx]

# --- Leiden labels with "l" prefix ---
leiden_labels = np.array(['l' + v for v in adata.obs['leiden'].values[sub_idx]])

# --- Left panel: gene expression ---
print('Building gene expression panel...')
html_genes = scatter_2d_categorical_html(
    xp_grid=[umap_coords],
    cell_metadata=gene_expr,
    title='Xenium UMAP — gene expression',
    out_path=None,
    xlabel='UMAP1', ylabel='UMAP2',
    return_html=True,
)

# --- Right panel: categorical labels ---
print('Building categorical labels panel...')
html_labels = scatter_2d_categorical_html(
    xp_grid=[umap_coords],
    cell_metadata={
        'leiden':         leiden_labels,
        'tran19_subtype': adata.obs['tran19_subtype'].values[sub_idx],
    },
    title='Xenium UMAP — cell labels',
    out_path=None,
    xlabel='UMAP1', ylabel='UMAP2',
    return_html=True,
)

# --- Combine side by side ---
with open(OUT_FIG_UMAP, 'w') as fh:
    fh.write(
        '<html><body style="display:flex;gap:10px;flex-wrap:wrap;">'
        + html_genes
        + html_labels
        + '</body></html>'
    )
print(f'Saved → {OUT_FIG_UMAP}')
print('Done.')
