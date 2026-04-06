import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import umap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

# --- file paths ---
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'raw', 'morcom26_cux2mice', 'P26_ENs.h5ad')
OUT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26_cux2mice')
OUT_FIG_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_PCA_FILE  = os.path.join(OUT_RES_DIR, '07.pca_coords.tsv')
OUT_UMAP_FILE = os.path.join(OUT_RES_DIR, '07.umap_coords.tsv')
OUT_PCA_HTML  = os.path.join(OUT_FIG_DIR, '07.pca_celltype.html')
OUT_UMAP_HTML = os.path.join(OUT_FIG_DIR, '07.umap_celltype.html')

# --- config ---
CELLTYPE_COL   = 'celltype'
N_HVG          = 2000
N_PCS          = 50
N_NEIGHBORS    = 15
MARKER_SIZE    = 3
MARKER_OPACITY = 0.6

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load ---
print(f'Loading {INPUT_FILE}')
adata = ad.read_h5ad(INPUT_FILE)
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes')

# --- filter out Other-mix cell types ---
mask = ~adata.obs[CELLTYPE_COL].astype(str).str.contains('Other-mix')
adata = adata[mask].copy()
print(f'  {adata.shape[0]} cells after removing Other-mix types')
print(f'  Cell types: {sorted(adata.obs[CELLTYPE_COL].unique())}')

# --- normalize: CP10k + log1p ---
X = adata.X
if sp.issparse(X):
    X = X.toarray()
depths = X.sum(axis=1, keepdims=True)
X_norm = np.log1p(X / depths * 1e4)

# --- select highly variable genes by variance ---
gene_var = X_norm.var(axis=0)
hvg_idx = np.argsort(gene_var)[::-1][:N_HVG]
X_hvg = X_norm[:, hvg_idx]
print(f'  HVGs selected: {N_HVG}')

# --- scale (zero-mean, unit-variance per gene) ---
X_scaled = scale(X_hvg)

# --- PCA ---
print(f'Running PCA (n_comps={N_PCS})...')
pca_model = PCA(n_components=N_PCS, random_state=0)
X_pca = pca_model.fit_transform(X_scaled)

# --- UMAP ---
print(f'Running UMAP (n_neighbors={N_NEIGHBORS})...')
reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, n_components=2, random_state=0)
X_umap = reducer.fit_transform(X_pca)

# --- save coordinates ---
pca_cols = [f'PC{i+1}' for i in range(N_PCS)]
pca_df = pd.DataFrame(X_pca, index=adata.obs_names, columns=pca_cols)
pca_df.index.name = 'cell'
pca_df.to_csv(OUT_PCA_FILE, sep='\t')
print(f'Saved PCA coords -> {OUT_PCA_FILE}')

umap_df = pd.DataFrame(X_umap, index=adata.obs_names, columns=['UMAP1', 'UMAP2'])
umap_df.index.name = 'cell'
umap_df.to_csv(OUT_UMAP_FILE, sep='\t')
print(f'Saved UMAP coords -> {OUT_UMAP_FILE}')

# --- build cell type color palette ---
celltypes = sorted(adata.obs[CELLTYPE_COL].unique())
cmap = plt.get_cmap('tab20', len(celltypes))
celltype_colors = {ct: mcolors.to_hex(cmap(i)) for i, ct in enumerate(celltypes)}


def make_scatter_html(x_vals, y_vals, cell_labels, xlab, ylab, title, out_path):
    """Create a plotly HTML scatterplot with one trace per cell type."""
    fig = go.Figure()
    for ct in celltypes:
        mask = cell_labels == ct
        fig.add_trace(go.Scatter(
            x=x_vals[mask],
            y=y_vals[mask],
            mode='markers',
            name=ct,
            marker=dict(
                size=MARKER_SIZE,
                color=celltype_colors[ct],
                opacity=MARKER_OPACITY,
            ),
        ))
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        legend=dict(itemsizing='constant'),
        width=800,
        height=700,
    )
    fig.write_html(out_path)
    print(f'Saved {out_path}')


cell_labels = adata.obs[CELLTYPE_COL].values

# PCA plot (PC1 vs PC2)
make_scatter_html(
    x_vals=X_pca[:, 0],
    y_vals=X_pca[:, 1],
    cell_labels=cell_labels,
    xlab='PC1', ylab='PC2',
    title='morcom26 CUX2 mice EN (Other-mix excluded) — PCA (colored by cell type)',
    out_path=OUT_PCA_HTML,
)

# UMAP plot
make_scatter_html(
    x_vals=X_umap[:, 0],
    y_vals=X_umap[:, 1],
    cell_labels=cell_labels,
    xlab='UMAP1', ylab='UMAP2',
    title='morcom26 CUX2 mice EN (Other-mix excluded) — UMAP (colored by cell type)',
    out_path=OUT_UMAP_HTML,
)

print('Done.')
