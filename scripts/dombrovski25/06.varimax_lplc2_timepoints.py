# Varimax rotation of PCA — LPLC2 cells, Dombrovski 2025 fly dataset
# Applies Kaiser varimax to the first 5 PCs for each time point separately.
# Mitochondrial genes removed; nCount_RNA regressed out before PCA.
# Produces per-time-point VX1-VX2 scatter plots:
#   - colored by sample metadata (interactive dropdown)
#   - colored by top VX1 gene expression (interactive gene dropdown)

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
N_PCS = 5       # PCs to rotate
N_VX1_GENES = 20

os.makedirs(FIG_DIR, exist_ok=True)


def varimax(L, gamma=1.0, max_iter=1000, tol=1e-6):
    """Kaiser varimax rotation of loading matrix L (n_vars × n_factors)."""
    n, p = L.shape
    R = np.eye(p)
    for _ in range(max_iter):
        R_old = R.copy()
        for i in range(p - 1):
            for j in range(i + 1, p):
                Lr = L @ R
                u = Lr[:, i] ** 2 - Lr[:, j] ** 2
                v = 2 * Lr[:, i] * Lr[:, j]
                A = u.sum()
                B = v.sum()
                C = (u ** 2 - v ** 2).sum()
                D = 2 * (u * v).sum()
                theta = 0.25 * np.arctan2(
                    D - gamma * 2 * A * B / n,
                    C - gamma * (A ** 2 - B ** 2) / n,
                )
                c, s = np.cos(theta), np.sin(theta)
                Rij = np.eye(p)
                Rij[i, i] = Rij[j, j] = c
                Rij[i, j] = -s
                Rij[j, i] = s
                R = R @ Rij
        if np.max(np.abs(R - R_old)) < tol:
            break
    return R


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

    # PCA then varimax
    pca = PCA(n_components=N_PCS, random_state=0)
    pca_scores = pca.fit_transform(xn)          # (n_cells, N_PCS)
    L = pca.components_.T                        # (n_genes, N_PCS)

    R = varimax(L)
    vx_scores = pca_scores @ R                   # (n_cells, N_PCS)
    vx_loadings = L @ R                          # (n_genes, N_PCS)

    # reorder axes by descending variance
    order = np.argsort(vx_scores.var(axis=0))[::-1]
    vx_scores = vx_scores[:, order]
    vx_loadings = vx_loadings[:, order]

    ev = pca.explained_variance_ratio_
    print(f'  PCA: PC1 EV={ev[0]:.3f}  PC2 EV={ev[1]:.3f}  (total {N_PCS} PCs={ev.sum():.3f})')
    vx_var = vx_scores.var(axis=0)
    vx_frac = vx_var / vx_var.sum() * ev.sum()
    print(f'  VX:  VX1 EV={vx_frac[0]:.3f}  VX2 EV={vx_frac[1]:.3f}')

    cell_meta = {col: obs[col].values for col in obs.columns}

    # metadata scatter
    out_meta = os.path.join(FIG_DIR, f'06.lplc2_{age}_varimax_metadata.html')
    scatter_2d_categorical_html(
        [vx_scores], cell_meta,
        f'LPLC2 {age} — VX1 vs VX2  (EV: {vx_frac[0]:.3f}, {vx_frac[1]:.3f})',
        out_meta,
        xlabel='VX1', ylabel='VX2',
    )

    # top VX1 genes by absolute loading
    top_idx = np.argsort(np.abs(vx_loadings[:, 0]))[::-1][:N_VX1_GENES]
    top_genes = genes[top_idx]
    print(f'  Top VX1 genes: {list(top_genes[:5])}')

    gene_vals = {g: xn[:, i] for g, i in zip(top_genes, top_idx)}

    out_genes = os.path.join(FIG_DIR, f'06.lplc2_{age}_varimax_genes.html')
    gene_expr_scatter_html(
        vx_scores[:, 0], vx_scores[:, 1],
        gene_vals,
        f'LPLC2 {age} — top VX1 genes  (EV: {vx_frac[0]:.3f}, {vx_frac[1]:.3f})',
        out_genes,
        xlabel='VX1', ylabel='VX2',
        colorbar_title='z-score',
    )
