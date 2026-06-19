# PCA analysis of LPLC2 cells — Dombrovski 2025 fly dataset
# Separates the three time points (APF_48h, APF_72h, APF_96h).
# Mitochondrial genes removed; nCount_RNA regressed out before PCA.
# For each time point:
#   - PC1-PC2 scatter colored by sample metadata (interactive dropdown)
#   - PC1-PC2 scatter colored by top PC1 gene expression (interactive gene dropdown)

import os
import sys
import numpy as np
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg
from viz import scatter_2d_categorical_html, gene_expr_scatter_html

from scomics.utils import norm

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, 'links', 'fly', 'dombrovski25_fly.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'dombrovski25_fly')

CELLTYPE = 'LPLC2'
AGES = ['APF_48h', 'APF_72h', 'APF_96h']
N_TOP_GENES = 2000
NDIM = 20
N_PC1_GENES = 20

os.makedirs(FIG_DIR, exist_ok=True)

# --- load and filter to LPLC2 ---
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

lplc2_mask = adata.obs['type1'] == CELLTYPE
adata = adata[lplc2_mask].copy()
print(f'\nLPLC2 cells: {adata.shape[0]}')

# remove mitochondrial genes (fly prefix: mt:)
mt_mask = np.array([g.lower().startswith('mt:') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
print(f'  Removed {mt_mask.sum()} mitochondrial genes; {(~mt_mask).sum()} remaining')

x_raw = np.array(adata.X)
depths = adata.obs['nCount_RNA'].values

# HVG selection on all LPLC2 cells (consistent gene set across time points)
hvg_mask = select_hvg(x_raw, depths, N_TOP_GENES)
gene_names = np.array(adata.var_names)[hvg_mask]
x_hvg = x_raw[:, hvg_mask]

# --- per time point ---
for age in AGES:
    age_mask = (adata.obs['orig.ident'] == age).values
    n_cells = age_mask.sum()
    print(f'\n=== {age}  n_cells={n_cells} ===')

    x = x_hvg[age_mask]
    d = depths[age_mask]
    obs = adata.obs[age_mask]

    # drop constant genes within this time point
    var_mask = x.var(axis=0) > 0
    x = x[:, var_mask]
    genes = gene_names[var_mask]

    xn = norm(x, d)

    # regress out log(nCount_RNA) from normalized expression
    log_depth = np.log(d).reshape(-1, 1)
    reg = LinearRegression().fit(log_depth, xn)
    xn = xn - reg.predict(log_depth)

    pca = PCA(n_components=NDIM, random_state=0)
    xp = pca.fit_transform(xn)
    ev = pca.explained_variance_ratio_
    print(f'  PC1 EV={ev[0]:.3f}  PC2 EV={ev[1]:.3f}')

    cell_meta = {col: obs[col].values for col in obs.columns}

    # metadata scatter
    out_meta = os.path.join(FIG_DIR, f'05.lplc2_{age}_pca_metadata.html')
    scatter_2d_categorical_html(
        [xp], cell_meta,
        f'LPLC2 {age} — PC1 vs PC2  (EV: {ev[0]:.3f}, {ev[1]:.3f})',
        out_meta,
        xlabel='PC1', ylabel='PC2',
    )

    # top PC1 genes by absolute loading
    pc1_loadings = pca.components_[0]
    top_idx = np.argsort(np.abs(pc1_loadings))[::-1][:N_PC1_GENES]
    top_genes = genes[top_idx]
    print(f'  Top PC1 genes: {list(top_genes[:5])}')

    gene_vals = {g: xn[:, i] for g, i in zip(top_genes, top_idx)}

    out_genes = os.path.join(FIG_DIR, f'05.lplc2_{age}_pca_genes.html')
    gene_expr_scatter_html(
        xp[:, 0], xp[:, 1],
        gene_vals,
        f'LPLC2 {age} — top PC1 genes  (EV: {ev[0]:.3f}, {ev[1]:.3f})',
        out_genes,
        xlabel='PC1', ylabel='PC2',
        colorbar_title='z-score',
    )
