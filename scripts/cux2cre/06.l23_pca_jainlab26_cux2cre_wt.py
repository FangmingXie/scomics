# L2/3 PCA density comparison: jainlab26_cux2cre vs jainlab26_wt (no yoo25, no Harmony).
# Loads L2/3 cells from 04_v2, applies 00_v2 UMAP spatial filter,
# fits PCA on wt (reference), projects cux2cre, and produces a 3-panel
# density figure (wt, cux2cre, cux2cre − wt) in PC2 vs PC3 space.

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg
from viz import scatter_categorical_html
from scomics.utils import norm

INPUT_CUX2CRE_L23  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_L23       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_jainlab26_wt_labeled.h5ad')
INPUT_CUX2CRE_00   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_00        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_wt_labeled.h5ad')
OUT_FIG_SCATTER    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '06.l23_pca_pc_scatter.html')
OUT_FIG_VX_SCATTER = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '06.l23_pca_vx_scatter.html')
OUT_FIG_DENSITY    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '06.l23_pca_pc13_density.html')
OUT_FIG_VX_DENSITY = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '06.l23_pca_vx13_density.html')
OUT_FIG_CELLTYPE   = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '06.l23_celltype_proportion.html')

UMAP_X_MAX  = -4
UMAP_Y_MAX  = 10
N_HVG       = 3000
N_PCS       = 30
N_VX        = 5
N_BINS      = 20

os.makedirs(os.path.dirname(OUT_FIG_DENSITY), exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


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
                A, B = u.sum(), v.sum()
                C = (u ** 2 - v ** 2).sum()
                D = 2 * (u * v).sum()
                theta = 0.25 * np.arctan2(
                    D - gamma * 2 * A * B / n,
                    C - gamma * (A ** 2 - B ** 2) / n,
                )
                c, s = np.cos(theta), np.sin(theta)
                Rij = np.eye(p)
                Rij[i, i] = Rij[j, j] = c
                Rij[i, j] = -s; Rij[j, i] = s
                R = R @ Rij
        if np.max(np.abs(R - R_old)) < tol:
            break
    return R


# --- Load L2/3 cells ---
print('Loading L2/3 h5ads...')
cux2cre = ad.read_h5ad(INPUT_CUX2CRE_L23)
wt      = ad.read_h5ad(INPUT_WT_L23)
print(f'  cux2cre: {cux2cre.n_obs} cells  |  wt: {wt.n_obs} cells')

# --- Apply 00_v2 UMAP spatial filter ---
print('Applying 00_v2 UMAP filter (x < {}, y < {})...'.format(UMAP_X_MAX, UMAP_Y_MAX))
cux2cre_00 = ad.read_h5ad(INPUT_CUX2CRE_00)
wt_00      = ad.read_h5ad(INPUT_WT_00)

def _umap_mask(adata_l23, adata_00):
    umap = pd.DataFrame(adata_00.obsm['X_umap'], index=adata_00.obs_names,
                        columns=['UMAP1', 'UMAP2'])
    u = umap.reindex(adata_l23.obs_names)
    return (u['UMAP1'] < UMAP_X_MAX) & (u['UMAP2'] < UMAP_Y_MAX)

mask_cux2cre = _umap_mask(cux2cre, cux2cre_00)
mask_wt      = _umap_mask(wt,      wt_00)
cux2cre = cux2cre[mask_cux2cre].copy()
wt      = wt[mask_wt].copy()
print(f'  After filter — cux2cre: {cux2cre.n_obs}  |  wt: {wt.n_obs}')

# --- HVG selection from wt (reference) ---
x_wt  = _to_array(wt.X)
depths_wt = x_wt.sum(axis=1)
hvg_mask = select_hvg(x_wt, depths_wt, N_HVG)
print(f'  HVGs selected: {hvg_mask.sum()}')

# --- Normalize independently ---
x_cux2cre  = _to_array(cux2cre.X)
depths_cux2cre = x_cux2cre.sum(axis=1)

xn_wt      = np.nan_to_num(norm(x_wt[:,      hvg_mask], depths_wt),      nan=0.0)
xn_cux2cre = np.nan_to_num(norm(x_cux2cre[:, hvg_mask], depths_cux2cre), nan=0.0)

# --- PCA: fit on wt, project both ---
print('Running PCA (fit on wt)...')
pca = PCA(n_components=N_PCS, random_state=0)
pca.fit(xn_wt)
xp_wt      = pca.transform(xn_wt)
xp_cux2cre = pca.transform(xn_cux2cre)
print(f'  wt: {xp_wt.shape}  |  cux2cre: {xp_cux2cre.shape}')

# --- Scatter plot: PC1/2 and PC2/3 colored by source and Type ---
xp_all     = np.vstack([xp_wt, xp_cux2cre])
source_all = np.array(['jainlab26_wt'] * len(xp_wt) + ['jainlab26_cux2cre'] * len(xp_cux2cre))
type_all   = np.concatenate([wt.obs['Type_transferred'].values, cux2cre.obs['Type_transferred'].values])

scatter_categorical_html(
    xp_grid=[xp_all],
    cell_metadata={'source': source_all, 'Type': type_all},
    title='PC scatter (wt-PCA, no Harmony) — jainlab26 L2/3',
    out_path=OUT_FIG_SCATTER,
    panels=[
        (0, 1, 'PC1', 'PC2'),
        (1, 2, 'PC2', 'PC3'),
        (0, 2, 'PC1', 'PC3'),
    ],
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
)
print(f'Saved → {OUT_FIG_SCATTER}')

# --- Varimax rotation on first N_VX PCs (fit on wt loadings) ---
print(f'Running varimax on first {N_VX} PCs...')
L   = pca.components_.T          # (n_hvg, N_PCS)
R   = varimax(L[:, :N_VX])       # rotation matrix (N_VX, N_VX)
vx_wt      = xp_wt[:,      :N_VX] @ R
vx_cux2cre = xp_cux2cre[:, :N_VX] @ R

# reorder by descending variance (fit on wt)
vx_order   = np.argsort(vx_wt.var(axis=0))[::-1]
vx_wt      = vx_wt[:,      vx_order]
vx_cux2cre = vx_cux2cre[:, vx_order]
print(f'  wt: {vx_wt.shape}  |  cux2cre: {vx_cux2cre.shape}')

vx_all = np.vstack([vx_wt, vx_cux2cre])
scatter_categorical_html(
    xp_grid=[vx_all],
    cell_metadata={'source': source_all, 'Type': type_all},
    title='VX scatter (varimax on first 5 PCs, wt-PCA) — jainlab26 L2/3',
    out_path=OUT_FIG_VX_SCATTER,
    panels=[
        (0, 1, 'VX1', 'VX2'),
        (1, 2, 'VX2', 'VX3'),
        (0, 2, 'VX1', 'VX3'),
    ],
    panel_3d=(0, 1, 2, 'VX1', 'VX2', 'VX3'),
)
print(f'Saved → {OUT_FIG_VX_SCATTER}')

def _density_fig(scores_wt, scores_cux2cre, col_a, col_b, xlabel, ylabel, title, out_path):
    a_all = np.concatenate([scores_wt[:, col_a], scores_cux2cre[:, col_a]])
    b_all = np.concatenate([scores_wt[:, col_b], scores_cux2cre[:, col_b]])
    x_edges   = np.linspace(a_all.min(), a_all.max(), N_BINS + 1)
    y_edges   = np.linspace(b_all.min(), b_all.max(), N_BINS + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    def norm_hist2d(a, b):
        h, _, _ = np.histogram2d(a, b, bins=[x_edges, y_edges])
        h = h / h.sum()
        return h

    h_wt_      = norm_hist2d(scores_wt[:, col_a],      scores_wt[:, col_b])
    h_cux2cre_ = norm_hist2d(scores_cux2cre[:, col_a], scores_cux2cre[:, col_b])
    h_diff_    = h_cux2cre_ - h_wt_

    density_max  = max(h_wt_.max(), h_cux2cre_.max())
    diff_abs_max = max(abs(h_diff_.min()), abs(h_diff_.max()))

    panels = [
        ('jainlab26_wt',           h_wt_,      'Viridis', 0,             density_max),
        ('jainlab26_cux2cre',      h_cux2cre_, 'Viridis', 0,             density_max),
        ('jainlab26_cux2cre − wt', h_diff_,    'RdBu_r',  -diff_abs_max, diff_abs_max),
    ]
    PANEL_CENTERS = [0.14, 0.50, 0.86]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[t for t, *_ in panels],
                        shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.08)
    for col_idx, ((t, h, colorscale, zmin, zmax), cx) in enumerate(
            zip(panels, PANEL_CENTERS), start=1):
        fig.add_trace(go.Heatmap(
            x=x_centers, y=y_centers, z=h.T,
            colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=True,
            colorbar=dict(orientation='h', x=cx, xanchor='center',
                          y=-0.18, yanchor='top', len=0.28, thickness=12,
                          title=dict(text='density' if col_idx < 3 else 'Δ density',
                                     side='bottom')),
        ), row=1, col=col_idx)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel, col=1)
    fig.update_layout(title=title, height=500, width=1100, margin=dict(b=100))
    fig.write_html(out_path)
    print(f'Saved → {out_path}')


# --- 2D density: PC1 vs PC3 ---
_density_fig(xp_wt, xp_cux2cre, 0, 2, 'PC1', 'PC3',
             'PC1 vs PC3 density (wt-PCA, no Harmony) — jainlab26 L2/3',
             OUT_FIG_DENSITY)

# --- 2D density: VX1 vs VX3 ---
_density_fig(vx_wt, vx_cux2cre, 0, 2, 'VX1', 'VX3',
             'VX1 vs VX3 density (varimax on 5 PCs, wt-PCA) — jainlab26 L2/3',
             OUT_FIG_VX_DENSITY)

# --- Cell type proportion stacked 100% barplot ---
ct_wt      = pd.Series(wt.obs['Type_transferred'].values).value_counts(normalize=True).rename('wt')
ct_cux2cre = pd.Series(cux2cre.obs['Type_transferred'].values).value_counts(normalize=True).rename('cux2cre')
ct_prop    = pd.concat([ct_wt, ct_cux2cre], axis=1).fillna(0).sort_index()

import matplotlib as mpl
colors = [mpl.colors.to_hex(f'C{i}') for i in range(len(ct_prop))]

# Panel 1: stacked 100% (x = group, color = cell type)
fig_stacked = go.Figure()
for ct, color in zip(ct_prop.index, colors):
    fig_stacked.add_trace(go.Bar(
        name=ct, x=['wt', 'cux2cre'],
        y=[ct_prop.loc[ct, 'wt'] * 100, ct_prop.loc[ct, 'cux2cre'] * 100],
        marker_color=color,
    ))
fig_stacked.update_layout(
    barmode='stack',
    title='Cell type proportion — jainlab26 L2/3',
    xaxis_title='Group', yaxis_title='Proportion (%)',
    height=500, width=500, legend=dict(x=1.02, y=1.0),
)

# Panel 2: grouped (x = cell type, two bars per type)
fig_grouped = go.Figure()
fig_grouped.add_trace(go.Bar(name='wt',      x=ct_prop.index,
                             y=ct_prop['wt']      * 100, marker_color='steelblue'))
fig_grouped.add_trace(go.Bar(name='cux2cre', x=ct_prop.index,
                             y=ct_prop['cux2cre'] * 100, marker_color='tomato'))
fig_grouped.update_layout(
    barmode='group',
    title='Cell type proportion — jainlab26 L2/3',
    xaxis_title='Cell type', yaxis_title='Proportion (%)',
    height=500, width=700, legend=dict(x=0.85, y=0.95),
)

html_stacked = fig_stacked.to_html(full_html=False, include_plotlyjs='cdn')
html_grouped = fig_grouped.to_html(full_html=False, include_plotlyjs=False)
with open(OUT_FIG_CELLTYPE, 'w') as f:
    f.write('<html><body style="display:flex;gap:20px;">'
            + html_stacked + html_grouped + '</body></html>')
print(f'Saved → {OUT_FIG_CELLTYPE}')

print('Done.')
