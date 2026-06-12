# Scatter plot of L2/3 cells in harmonized PC space, colored by Type and source.
# Also produces a 2D density (histogram) of PC2 vs PC3 for each dataset
# with difference panels (cux2cre − yoo25 and wt − yoo25).

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html

INPUT_HARMONY        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_harmony_coords.tsv')
INPUT_CUX2CRE_H5AD   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_H5AD        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_wt_labeled.h5ad')
OUT_FIG              = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '05_v2.l23_harmony_pc_scatter.html')
OUT_FIG_DENSITY      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '05_v2.l23_harmony_pc23_density.html')

UMAP_X_MAX = -4
UMAP_Y_MAX = 10

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

print('Loading harmony coords...')
df = pd.read_csv(INPUT_HARMONY, sep='\t', index_col=0)
print(f'  {len(df)} cells (before UMAP filter)')

# --- Filter target cells by 00_v2 UMAP coords; ref cells (no saved UMAP) are kept ---
# Build per-dataset barcode→UMAP lookup (barcodes are unique within each dataset).
cux2cre_ad = ad.read_h5ad(INPUT_CUX2CRE_H5AD)
wt_ad      = ad.read_h5ad(INPUT_WT_H5AD)

def _umap_series(adata, col):
    return pd.Series(adata.obsm['X_umap'][:, col], index=adata.obs_names)

umap_x_cux2cre = _umap_series(cux2cre_ad, 0)
umap_y_cux2cre = _umap_series(cux2cre_ad, 1)
umap_x_wt      = _umap_series(wt_ad, 0)
umap_y_wt      = _umap_series(wt_ad, 1)

cells_cux2cre = df.index[df['source'] == 'jainlab26_cux2cre']
cells_wt      = df.index[df['source'] == 'jainlab26_wt']

mask_cux2cre = ((umap_x_cux2cre.reindex(cells_cux2cre) < UMAP_X_MAX) &
                (umap_y_cux2cre.reindex(cells_cux2cre) < UMAP_Y_MAX)).values
mask_wt      = ((umap_x_wt.reindex(cells_wt) < UMAP_X_MAX) &
                (umap_y_wt.reindex(cells_wt) < UMAP_Y_MAX)).values

keep_ref     = np.ones((df['source'] == 'yoo25').sum(), dtype=bool)
keep = np.concatenate([keep_ref, mask_cux2cre, mask_wt])
df = df[keep]
print(f'  {len(df)} cells after 00_v2 UMAP filter (x < {UMAP_X_MAX}, y < {UMAP_Y_MAX})')

xp = df[['PC1', 'PC2', 'PC3']].values

scatter_categorical_html(
    xp_grid=[xp],
    cell_metadata={'Type': df['Type'].values, 'source': df['source'].values},
    title='Harmonized PC scatter — jainlab26 L2/3 (Type and source)',
    out_path=OUT_FIG,
    panels=[
        (0, 1, 'Harmony PC1', 'Harmony PC2'),
        (1, 2, 'Harmony PC2', 'Harmony PC3'),
    ],
)
print(f'Saved → {OUT_FIG}')

# --- 2D density: PC2 vs PC3, normalized histograms + difference panels ---
N_BINS = 20

pc2_all = df['PC2'].values
pc3_all = df['PC3'].values
x_edges   = np.linspace(pc2_all.min(), pc2_all.max(), N_BINS + 1)
y_edges   = np.linspace(pc3_all.min(), pc3_all.max(), N_BINS + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

def norm_hist2d(pc2, pc3):
    h, _, _ = np.histogram2d(pc2, pc3, bins=[x_edges, y_edges])
    h = h / h.sum()
    return h  # shape (N_BINS, N_BINS), axis0=PC2, axis1=PC3

h_yoo25    = norm_hist2d(df.loc[df['source'] == 'yoo25',             'PC2'].values,
                         df.loc[df['source'] == 'yoo25',             'PC3'].values)
h_cux2cre  = norm_hist2d(df.loc[df['source'] == 'jainlab26_cux2cre', 'PC2'].values,
                         df.loc[df['source'] == 'jainlab26_cux2cre', 'PC3'].values)
h_wt       = norm_hist2d(df.loc[df['source'] == 'jainlab26_wt',      'PC2'].values,
                         df.loc[df['source'] == 'jainlab26_wt',      'PC3'].values)
h_diff_cux2cre = h_cux2cre - h_yoo25
h_diff_wt      = h_wt      - h_yoo25
h_diff_cux2cre_wt = h_cux2cre - h_wt

density_max  = max(h_yoo25.max(), h_cux2cre.max(), h_wt.max())
diff_abs_max = max(abs(h_diff_cux2cre.min()),    abs(h_diff_cux2cre.max()),
                   abs(h_diff_wt.min()),          abs(h_diff_wt.max()),
                   abs(h_diff_cux2cre_wt.min()), abs(h_diff_cux2cre_wt.max()))

panels = [
    ('yoo25',                     h_yoo25,          'Viridis', 0,             density_max),
    ('jainlab26_cux2cre',         h_cux2cre,        'Viridis', 0,             density_max),
    ('jainlab26_wt',              h_wt,             'Viridis', 0,             density_max),
    ('jainlab26_cux2cre − yoo25', h_diff_cux2cre,   'RdBu_r', -diff_abs_max, diff_abs_max),
    ('jainlab26_wt − yoo25',      h_diff_wt,        'RdBu_r', -diff_abs_max, diff_abs_max),
    ('jainlab26_cux2cre − wt',    h_diff_cux2cre_wt,'RdBu_r', -diff_abs_max, diff_abs_max),
]

PANEL_CENTERS = [0.08, 0.24, 0.41, 0.59, 0.75, 0.92]

fig = make_subplots(
    rows=1, cols=6,
    subplot_titles=[title for title, *_ in panels],
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0.04,
)

for col_idx, ((title, h, colorscale, zmin, zmax), cx) in enumerate(
        zip(panels, PANEL_CENTERS), start=1):
    fig.add_trace(
        go.Heatmap(
            x=x_centers, y=y_centers,
            z=h.T,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            showscale=True,
            colorbar=dict(
                orientation='h',
                x=cx, xanchor='center',
                y=-0.18, yanchor='top',
                len=0.13, thickness=12,
                title=dict(text='density' if col_idx <= 3 else 'Δ density', side='bottom'),
            ),
        ),
        row=1, col=col_idx,
    )

fig.update_xaxes(title_text='Harmony PC2')
fig.update_yaxes(title_text='Harmony PC3', col=1)
fig.update_layout(
    title='PC2 vs PC3 density (normalized) — jainlab26_cux2cre & jainlab26_wt vs yoo25 L2/3',
    height=500, width=1900,
    margin=dict(b=100),
)
fig.write_html(OUT_FIG_DENSITY)
print(f'Saved → {OUT_FIG_DENSITY}')
