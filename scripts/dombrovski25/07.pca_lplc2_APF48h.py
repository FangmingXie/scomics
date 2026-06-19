# PCA analysis of LPLC2 cells at APF_48h — Dombrovski 2025 fly dataset
# Mitochondrial genes removed; nCount_RNA regressed out before PCA.
# Cells binned into 10 equal-sized quantile bins along PC2.
# Outputs:
#   - local_data/res/dombrovski25_fly/07.lplc2_APF48h_pca.parquet  (PC coords + PC2 bin label)
#   - PC1-PC2 scatter colored by sample metadata + PC2 bin (interactive dropdown)
#   - PC1-PC2 scatter colored by top PC1 gene expression (interactive gene dropdown)
#   - PC1-PC2 scatter colored by top PC2 gene expression (interactive gene dropdown)

import os
import sys
import numpy as np
import pandas as pd
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
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'dombrovski25_fly')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'dombrovski25_fly')
OUT_PCA = os.path.join(RES_DIR, '07.lplc2_APF48h_pca.parquet')
OUT_METADATA = os.path.join(FIG_DIR, '07.lplc2_APF48h_pca_metadata.html')
OUT_PC1_GENES = os.path.join(FIG_DIR, '07.lplc2_APF48h_pca_pc1_genes.html')
OUT_PC2_GENES = os.path.join(FIG_DIR, '07.lplc2_APF48h_pca_pc2_genes.html')

CELLTYPE = 'LPLC2'
AGE = 'APF_48h'
N_TOP_GENES = 2000
NDIM = 20
N_LOADING_GENES = 20
N_PC2_BINS = 10

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# --- load, filter to LPLC2, remove mt genes ---
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

adata = adata[adata.obs['type1'] == CELLTYPE].copy()
print(f'\nLPLC2 cells: {adata.shape[0]}')

mt_mask = np.array([g.lower().startswith('mt:') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
print(f'  Removed {mt_mask.sum()} mitochondrial genes; {(~mt_mask).sum()} remaining')

adata = adata[adata.obs['orig.ident'] == AGE].copy()
print(f'  {AGE} cells: {adata.shape[0]}')

x_raw = np.array(adata.X)
depths = adata.obs['nCount_RNA'].values

# HVG selection
hvg_mask = select_hvg(x_raw, depths, N_TOP_GENES)
gene_names = np.array(adata.var_names)[hvg_mask]
x_hvg = x_raw[:, hvg_mask]

# drop constant genes
var_mask = x_hvg.var(axis=0) > 0
x_hvg = x_hvg[:, var_mask]
gene_names = gene_names[var_mask]

# normalize then regress out log(nCount_RNA)
xn = norm(x_hvg, depths)
log_depth = np.log(depths).reshape(-1, 1)
reg = LinearRegression().fit(log_depth, xn)
xn = xn - reg.predict(log_depth)

# PCA
pca = PCA(n_components=NDIM, random_state=0)
xp = pca.fit_transform(xn)
ev = pca.explained_variance_ratio_
print(f'  PC1 EV={ev[0]:.3f}  PC2 EV={ev[1]:.3f}')

title_suffix = f'EV: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}'

# bin cells into N_PC2_BINS equal-sized quantile bins along PC2
pc2_vals = xp[:, 1]
quantiles = np.linspace(0, 100, N_PC2_BINS + 1)
bin_edges = np.percentile(pc2_vals, quantiles)
bin_edges[-1] += 1e-10  # include the max value in the last bin
pc2_bin = np.digitize(pc2_vals, bin_edges[1:])  # labels 0..N_PC2_BINS-1
pc2_bin_label = np.array([f'bin{b:02d}' for b in pc2_bin])
print(f'  PC2 bin sizes: {np.unique(pc2_bin, return_counts=True)[1].tolist()}')

# save PCA coords + bin label
pc_cols = [f'PC{i+1}' for i in range(NDIM)]
res_df = pd.DataFrame(xp, index=adata.obs_names, columns=pc_cols)
res_df.index.name = 'cell_id'
for col in adata.obs.columns:
    res_df[col] = adata.obs[col].values
res_df['pc2_bin'] = pc2_bin_label
res_df.to_parquet(OUT_PCA)
print(f'  Saved {OUT_PCA}')

cell_meta = {col: adata.obs[col].values for col in adata.obs.columns}
cell_meta['pc2_bin'] = pc2_bin_label

# metadata scatter
scatter_2d_categorical_html(
    [xp], cell_meta,
    f'LPLC2 {AGE} — PC1 vs PC2  ({title_suffix})',
    OUT_METADATA,
    xlabel='PC1', ylabel='PC2',
)

# top PC1 genes
pc1_idx = np.argsort(np.abs(pca.components_[0]))[::-1][:N_LOADING_GENES]
top_pc1_genes = gene_names[pc1_idx]
print(f'  Top PC1 genes: {list(top_pc1_genes[:5])}')

gene_expr_scatter_html(
    xp[:, 0], xp[:, 1],
    {g: xn[:, i] for g, i in zip(top_pc1_genes, pc1_idx)},
    f'LPLC2 {AGE} — top PC1 genes  ({title_suffix})',
    OUT_PC1_GENES,
    xlabel='PC1', ylabel='PC2',
    colorbar_title='z-score',
)

# top PC2 genes
pc2_idx = np.argsort(np.abs(pca.components_[1]))[::-1][:N_LOADING_GENES]
top_pc2_genes = gene_names[pc2_idx]
print(f'  Top PC2 genes: {list(top_pc2_genes[:5])}')

gene_expr_scatter_html(
    xp[:, 0], xp[:, 1],
    {g: xn[:, i] for g, i in zip(top_pc2_genes, pc2_idx)},
    f'LPLC2 {AGE} — top PC2 genes  ({title_suffix})',
    OUT_PC2_GENES,
    xlabel='PC1', ylabel='PC2',
    colorbar_title='z-score',
)
