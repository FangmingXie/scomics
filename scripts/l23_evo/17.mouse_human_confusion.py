"""Confusion matrix between mouse and human L2/3 IT types via 1-NN in PCHA space.

Reads 16.3 mouse embeddings (PC1-PC5) and human PCHA coordinates, assigns each
mouse cell a human cluster label via 1-nearest-neighbor lookup, then builds a
row-normalized confusion matrix (mouse Type_leiden vs. human WithinArea_cluster).
Also plots PC1 distribution histograms for each mouse and human type.

Reads:
  local_data/res/l23_evo/16.3.mouse_sk_tau_lower_coords.tsv
  local_data/res/l23_evo/09.pcha_xp.tsv
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Output:
  local_data/res/l23_evo/17.mouse_human_confusion.tsv
  local_data/fig/l23_evo/17.mouse_human_confusion.html
  local_data/fig/l23_evo/17.mouse_human_pc1_hist.html
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_MOUSE_COORDS  = os.path.join(OUT_RES_DIR, '16.3.mouse_sk_tau_lower_coords.tsv')
IN_HUMAN_XP      = os.path.join(OUT_RES_DIR, '09.pcha_xp.tsv')
IN_HUMAN_ADATA   = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_COUNTS       = os.path.join(OUT_RES_DIR, '17.mouse_human_confusion.tsv')
OUT_HTML         = os.path.join(OUT_FIG_DIR, '17.mouse_human_confusion.html')
OUT_HIST_HTML    = os.path.join(OUT_FIG_DIR, '17.mouse_human_pc1_hist.html')

# --- parameters ---
PC_COLS        = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
CLUSTER_COL    = 'WithinArea_cluster'
MOUSE_TYPE_COL = 'Type_leiden'

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- 1. Load mouse embeddings ---
print('Loading mouse embeddings...')
mouse_df = pd.read_csv(IN_MOUSE_COORDS, sep='\t', index_col=0)
X_mouse  = mouse_df[PC_COLS].values
mouse_types = mouse_df[MOUSE_TYPE_COL].values
print(f'  {len(mouse_df)} mouse cells, types: {sorted(set(mouse_types))}')

# --- 2. Load human PCHA coordinates ---
print('Loading human PCHA coordinates...')
human_xp = pd.read_csv(IN_HUMAN_XP, sep='\t', index_col=0)
X_human  = human_xp[PC_COLS].values
print(f'  {len(human_xp)} human cells')

# --- 3. Load human cluster labels aligned to xp index ---
print('Loading human cluster labels...')
h_adata       = ad.read_h5ad(IN_HUMAN_ADATA)
h_cluster_map = h_adata.obs[CLUSTER_COL]
human_clusters = h_cluster_map.reindex(human_xp.index).values
if pd.isnull(human_clusters).any():
    raise ValueError(f'Some human xp cells have no cluster label — index mismatch between xp and h5ad.')
print(f'  Human cluster labels: {sorted(set(human_clusters))}')

# --- 4. 1-NN assignment ---
print('Running 1-NN assignment...')
nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
nn.fit(X_human)
_, indices = nn.kneighbors(X_mouse)
assigned_human_labels = human_clusters[indices[:, 0]]
print(f'  Assigned {len(assigned_human_labels)} mouse cells to human clusters')

# --- 5. Confusion matrix ---
print('Building confusion matrix...')
counts = pd.crosstab(
    pd.Series(mouse_types, name='mouse_type'),
    pd.Series(assigned_human_labels, name='human_cluster'),
)
# ensure consistent ordering
counts = counts.sort_index(axis=0).sort_index(axis=1)
counts.to_csv(OUT_COUNTS, sep='\t')
print(f'Saved counts: {OUT_COUNTS}  (shape {counts.shape})')

# row-normalize
norm = counts.div(counts.sum(axis=1), axis=0)

# --- 6. Plotly heatmap ---
print('Generating heatmap...')
z      = norm.values
x_labs = list(norm.columns)
y_labs = list(norm.index)

text = [[f'{v:.2f}' for v in row] for row in z]

fig = go.Figure(go.Heatmap(
    z=z,
    x=x_labs,
    y=y_labs,
    text=text,
    texttemplate='%{text}',
    colorscale='Blues',
    zmin=0, zmax=1,
    colorbar=dict(title='Fraction'),
))
fig.update_layout(
    title='Mouse Type_leiden vs. Human WithinArea_cluster (1-NN, τ=0.005 embedding)',
    xaxis_title='Human cluster (WithinArea_cluster)',
    yaxis_title='Mouse type (Type_leiden)',
    width=900,
    height=600,
)
fig.write_html(OUT_HTML)
print(f'Saved {OUT_HTML}')

# --- 7. PC1 distribution histograms ---
print('Generating PC1 histograms...')
mouse_pc1 = mouse_df['PC1'].values
human_pc1 = human_xp['PC1'].values

mouse_type_labels = sorted(set(mouse_types))
human_cluster_labels = sorted(set(human_clusters))

fig_hist = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Mouse types (Type_leiden) — PC1', 'Human clusters (WithinArea_cluster) — PC1'),
)

for label in mouse_type_labels:
    mask = mouse_types == label
    fig_hist.add_trace(go.Histogram(
        x=mouse_pc1[mask],
        name=label,
        opacity=0.6,
        histnorm='probability density',
        nbinsx=40,
        legendgroup=f'mouse_{label}',
    ), row=1, col=1)

for label in human_cluster_labels:
    mask = human_clusters == label
    fig_hist.add_trace(go.Histogram(
        x=human_pc1[mask],
        name=label,
        opacity=0.6,
        histnorm='probability density',
        nbinsx=40,
        legendgroup=f'human_{label}',
        showlegend=True,
    ), row=1, col=2)

fig_hist.update_layout(
    title='PC1 distributions by type (τ=0.005 embedding)',
    barmode='overlay',
    width=1200,
    height=500,
    xaxis_title='PC1',
    xaxis2_title='PC1',
    yaxis_title='Density',
    yaxis2_title='Density',
)
fig_hist.write_html(OUT_HIST_HTML)
print(f'Saved {OUT_HIST_HTML}')
print('Done.')
