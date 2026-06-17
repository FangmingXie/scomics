# First-pass analysis of Xenium mouse retina trial dataset (v2).
# Loads cell_feature_matrix.h5 (Gene Expression only) and cells.parquet,
# filters to the upper-right tissue section, computes QC metrics + figures,
# filters low-quality cells, runs PCA + UMAP + Leiden, saves h5ad.
# Normalization: raw counts / cell_area * median_cell_area → log1p → zscore.

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import h5py
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg
from viz import scatter_2d_categorical_html

# --- File paths ---
IN_H5    = os.path.join(PROJECT_ROOT, 'links', 'retina', 'trial', 'cell_feature_matrix.h5')
IN_CELLS = os.path.join(PROJECT_ROOT, 'links', 'retina', 'trial', 'cells.parquet')

OUT_RES_DIR         = os.path.join(PROJECT_ROOT, 'links', 'retina', 'trial')
OUT_FIG_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'retina')
OUT_H5AD            = os.path.join(OUT_RES_DIR, 'retina_xenium.h5ad')
OUT_FIG_SPATIAL_QC  = os.path.join(OUT_FIG_DIR, '00.v2.spatial_section_qc.html')
OUT_FIG_VIOLIN      = os.path.join(OUT_FIG_DIR, '00.v2.qc_violin.html')
OUT_FIG_SCATTER     = os.path.join(OUT_FIG_DIR, '00.v2.qc_scatter.html')
OUT_FIG_FDR         = os.path.join(OUT_FIG_DIR, '00.v2.fdr_histogram.html')
OUT_FIG_UMAP_LEIDEN = os.path.join(OUT_FIG_DIR, '00.v2.umap_leiden.html')
OUT_FIG_UMAP_QC     = os.path.join(OUT_FIG_DIR, '00.v2.umap_qc.html')

# --- Section filter (adjust based on spatial overview) ---
SECTION_X_MIN = 5000
SECTION_X_MAX = 10000
SECTION_Y_MIN = 0
SECTION_Y_MAX = 4500

# --- QC thresholds ---
MIN_COUNTS = 100
MIN_GENES  = 50
MAX_FDR    = 0.05

# --- Analysis parameters ---
N_HVG       = 2000
N_PCS       = 30
N_NEIGHBORS = 15
LEIDEN_RES  = 0.5
UMAP_SEED   = 42

os.makedirs(OUT_FIG_DIR, exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


def _read_xenium_h5(path):
    """Read Xenium cell_feature_matrix.h5; return AnnData with Gene Expression features only."""
    with h5py.File(path, 'r') as f:
        m          = f['matrix']
        data       = m['data'][:]
        indices    = m['indices'][:]
        indptr     = m['indptr'][:]
        shape      = m['shape'][:]         # (n_features, n_barcodes)
        barcodes   = m['barcodes'][:].astype(str)
        feat_names = m['features']['name'][:].astype(str)
        feat_ids   = m['features']['id'][:].astype(str)
        feat_types = m['features']['feature_type'][:].astype(str)
    # CSC stored as (features × barcodes); transpose to (cells × features) CSR
    X   = sp.csc_matrix((data, indices, indptr), shape=(shape[0], shape[1])).T.tocsr()
    obs = pd.DataFrame(index=barcodes)
    obs.index.name = None
    var = pd.DataFrame({'gene_id': feat_ids, 'feature_type': feat_types}, index=feat_names)
    var.index.name = None
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata = adata[:, adata.var['feature_type'] == 'Gene Expression'].copy()
    return adata


# --- Load data ---
print('Loading cell_feature_matrix.h5...')
adata = _read_xenium_h5(IN_H5)
print(f'  {adata.n_obs} cells × {adata.n_vars} genes (Gene Expression only)')

print('Loading cells.parquet...')
cells = pd.read_parquet(IN_CELLS).set_index('cell_id')
adata.obs = adata.obs.join(cells, how='left')
total  = adata.obs['total_counts'].values.astype(float)
gene   = adata.obs['transcript_counts'].values.astype(float)
adata.obs['fdr'] = np.where(total > 0, 1.0 - gene / total, 1.0)
adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
print('  Metadata joined; FDR computed.')

# --- Section filter ---
mask = (
    (adata.obs['x_centroid'] > SECTION_X_MIN) &
    (adata.obs['x_centroid'] < SECTION_X_MAX) &
    (adata.obs['y_centroid'] > SECTION_Y_MIN) &
    (adata.obs['y_centroid'] < SECTION_Y_MAX)
)
adata = adata[mask].copy()
print(f'Section filter: {adata.n_obs} cells '
      f'(x ∈ ({SECTION_X_MIN}, {SECTION_X_MAX}), y ∈ ({SECTION_Y_MIN}, {SECTION_Y_MAX}))')

# --- QC metrics ---
x_raw = _to_array(adata.X)
adata.obs['n_genes_by_counts'] = (x_raw > 0).sum(axis=1).astype(int)

# --- Spatial QC: 3 side-by-side figures ---
print('Plotting spatial QC (section)...')
qc_cols   = ['total_counts', 'n_genes_by_counts', 'fdr']
qc_titles = ['Total counts', 'Genes detected', 'FDR']
spatial_htmls = []
for i, (col, title) in enumerate(zip(qc_cols, qc_titles)):
    f = go.Figure(go.Scattergl(
        x=adata.obs['x_centroid'], y=adata.obs['y_centroid'],
        mode='markers',
        marker=dict(size=2, color=adata.obs[col], colorscale='Viridis',
                    colorbar=dict(title=col), showscale=True),
        showlegend=False,
    ))
    f.update_yaxes(autorange='reversed')
    f.update_layout(title=title, xaxis_title='x (µm)', yaxis_title='y (µm)',
                    height=500, width=500)
    spatial_htmls.append(f.to_html(full_html=False,
                                   include_plotlyjs='cdn' if i == 0 else False))
with open(OUT_FIG_SPATIAL_QC, 'w') as fh:
    fh.write('<html><body style="display:flex;gap:10px;flex-wrap:wrap;">'
             + ''.join(spatial_htmls) + '</body></html>')
print(f'  Saved → {OUT_FIG_SPATIAL_QC}')

# --- Violin plots ---
print('Plotting QC violins...')
fig = make_subplots(rows=1, cols=3, subplot_titles=['Genes detected', 'Total counts', 'FDR'])
for ci, col in enumerate(['n_genes_by_counts', 'total_counts', 'fdr'], 1):
    fig.add_trace(go.Violin(
        y=adata.obs[col], name=col,
        box_visible=True, meanline_visible=True,
        fillcolor='steelblue', line_color='navy', opacity=0.7,
    ), row=1, col=ci)
fig.update_layout(title='QC distributions', showlegend=False, height=500, width=950)
fig.write_html(OUT_FIG_VIOLIN)
print(f'  Saved → {OUT_FIG_VIOLIN}')

# --- Scatter panels ---
print('Plotting QC scatter panels...')
seg_methods = list(adata.obs['segmentation_method'].unique())
colors_seg  = ['steelblue', 'tomato', 'seagreen', 'gold']
valid       = adata.obs['nucleus_area'].notna()

def _seg_traces(y_col):
    traces = []
    for sm, color in zip(seg_methods, colors_seg):
        m = adata.obs['segmentation_method'] == sm
        traces.append(go.Scattergl(
            x=adata.obs.loc[m, 'cell_area'], y=adata.obs.loc[m, y_col],
            mode='markers', marker=dict(size=2, color=color, opacity=0.4), name=sm,
        ))
    return traces

fig1 = go.Figure(_seg_traces('n_genes_by_counts'))
fig1.update_layout(title='Cell area vs genes detected',
                   xaxis_title='Cell area (µm²)', yaxis_title='Genes detected',
                   height=480, width=560)

fig2 = go.Figure(_seg_traces('transcript_counts'))
fig2.update_layout(title='Cell area vs total gene counts',
                   xaxis_title='Cell area (µm²)', yaxis_title='Gene counts (transcripts)',
                   showlegend=False, height=480, width=490)

fig3 = go.Figure(go.Scattergl(
    x=adata.obs.loc[valid, 'cell_area'],
    y=adata.obs.loc[valid, 'nucleus_area'],
    mode='markers',
    marker=dict(size=2, color=adata.obs.loc[valid, 'nucleus_count'],
                colorscale='RdYlBu_r', colorbar=dict(title='nucleus_count'),
                showscale=True, opacity=0.5),
))
fig3.update_layout(title='Nucleus area vs cell area',
                   xaxis_title='Cell area (µm²)', yaxis_title='Nucleus area (µm²)',
                   height=480, width=530)

scatter_htmls = [
    fig1.to_html(full_html=False, include_plotlyjs='cdn'),
    fig2.to_html(full_html=False, include_plotlyjs=False),
    fig3.to_html(full_html=False, include_plotlyjs=False),
]
with open(OUT_FIG_SCATTER, 'w') as fh:
    fh.write('<html><body style="display:flex;gap:10px;flex-wrap:wrap;">'
             + ''.join(scatter_htmls) + '</body></html>')
print(f'  Saved → {OUT_FIG_SCATTER}')

# --- FDR histogram ---
print('Plotting FDR histogram...')
fig = go.Figure(go.Histogram(
    x=adata.obs['fdr'], nbinsx=100,
    marker_color='steelblue', opacity=0.8,
))
fig.add_vline(x=MAX_FDR, line_dash='dash', line_color='red',
              annotation_text=f'MAX_FDR={MAX_FDR}', annotation_position='top right')
fig.update_layout(title='FDR distribution (1 − gene_counts / total_counts)',
                  xaxis_title='FDR', yaxis_title='Cells', height=450, width=650)
fig.write_html(OUT_FIG_FDR)
print(f'  Saved → {OUT_FIG_FDR}')

# --- Cell filtering ---
n_before = adata.n_obs
keep = (
    (adata.obs['total_counts'] >= MIN_COUNTS) &
    (adata.obs['n_genes_by_counts'] >= MIN_GENES) &
    (adata.obs['fdr'] <= MAX_FDR)
)
adata = adata[keep].copy()
print(f'QC filter: {n_before} → {adata.n_obs} cells (removed {n_before - adata.n_obs})')

# --- HVG selection ---
x_raw  = _to_array(adata.X)
depths = x_raw.sum(axis=1)
hvg_mask = select_hvg(x_raw, depths, N_HVG)
print(f'HVGs selected: {hvg_mask.sum()}')

# --- Normalize: raw counts / cell_area * median_area → log1p → zscore ---
cell_areas  = adata.obs['cell_area'].values.astype(float)
median_area = np.median(cell_areas)
x_area_norm = x_raw[:, hvg_mask] / cell_areas[:, None] * median_area
x_log1p     = np.log1p(x_area_norm)
mu  = x_log1p.mean(axis=0)
std = x_log1p.std(axis=0)
std[std == 0] = 1.0
xn  = (x_log1p - mu) / std
xn  = np.nan_to_num(xn, nan=0.0)

# --- PCA ---
print('Running PCA...')
pca = PCA(n_components=N_PCS, random_state=0)
xp  = pca.fit_transform(xn)
adata.obsm['X_pca'] = xp
print(f'  Cumulative variance ({N_PCS} PCs): {np.cumsum(pca.explained_variance_ratio_)[-1]:.3f}')

# --- UMAP ---
print(f'Running UMAP on {adata.n_obs} cells...')
reducer     = umap.UMAP(n_neighbors=N_NEIGHBORS, n_components=2, random_state=UMAP_SEED)
umap_coords = reducer.fit_transform(xp)
adata.obsm['X_umap'] = umap_coords
print('  UMAP done.')

# --- Leiden clustering ---
print(f'Running Leiden (resolution={LEIDEN_RES})...')
nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric='euclidean', n_jobs=-1)
nn.fit(xp)
_, knn_indices = nn.kneighbors(xp)
edges = [(i, int(j)) for i, nbrs in enumerate(knn_indices) for j in nbrs[1:]]
g = ig.Graph(n=adata.n_obs, edges=edges, directed=False)
g = g.simplify()
partition = leidenalg.find_partition(
    g, leidenalg.RBConfigurationVertexPartition,
    resolution_parameter=LEIDEN_RES, seed=0,
)
adata.obs['leiden'] = [str(m) for m in partition.membership]
n_clusters = len(set(partition.membership))
print(f'  {n_clusters} clusters found.')
print(f'  Cluster sizes:\n{adata.obs["leiden"].value_counts().sort_index().to_string()}')

# --- Save h5ad ---
adata.write_h5ad(OUT_H5AD)
print(f'Saved h5ad → {OUT_H5AD}')

# --- UMAP: Leiden clusters ---
print('Plotting UMAP (Leiden clusters)...')
scatter_2d_categorical_html(
    xp_grid=[umap_coords],
    cell_metadata={'Leiden cluster': adata.obs['leiden'].values},
    title='UMAP — Leiden clusters (retina Xenium trial v2)',
    out_path=OUT_FIG_UMAP_LEIDEN,
    xlabel='UMAP1', ylabel='UMAP2',
)
print(f'  Saved → {OUT_FIG_UMAP_LEIDEN}')

# --- UMAP: QC metrics ---
print('Plotting UMAP (QC metrics)...')
scatter_2d_categorical_html(
    xp_grid=[umap_coords],
    cell_metadata={
        'total_counts':      adata.obs['total_counts'].values.astype(float),
        'n_genes_by_counts': adata.obs['n_genes_by_counts'].values.astype(float),
        'fdr':               adata.obs['fdr'].values.astype(float),
    },
    title='UMAP — QC metrics (retina Xenium trial v2)',
    out_path=OUT_FIG_UMAP_QC,
    xlabel='UMAP1', ylabel='UMAP2',
)
print(f'  Saved → {OUT_FIG_UMAP_QC}')

print('Done.')
