# Scatter plot of L2/3 cells in harmonized PC space, colored by Type and source.
# Also produces a 2D density (histogram) of PC2 vs PC3 separately for each dataset.

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html

INPUT_HARMONY  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04.l23_harmony_coords.tsv')
OUT_FIG        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '05.l23_harmony_pc_scatter.html')
OUT_FIG_DENSITY = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '05.l23_harmony_pc23_density.html')

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

print('Loading harmony coords...')
df = pd.read_csv(INPUT_HARMONY, sep='\t', index_col=0)
print(f'  {len(df)} cells')

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

# --- 2D density: PC2 vs PC3, normalized histograms + difference panel ---
N_BINS = 20

pc2_all = df['PC2'].values
pc3_all = df['PC3'].values
x_edges = np.linspace(pc2_all.min(), pc2_all.max(), N_BINS + 1)
y_edges = np.linspace(pc3_all.min(), pc3_all.max(), N_BINS + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

def norm_hist2d(pc2, pc3):
    h, _, _ = np.histogram2d(pc2, pc3, bins=[x_edges, y_edges])
    h = h / h.sum()
    return h  # shape (N_BINS, N_BINS), axis0=PC2, axis1=PC3

h_yoo25    = norm_hist2d(df.loc[df['source'] == 'yoo25',    'PC2'].values,
                         df.loc[df['source'] == 'yoo25',    'PC3'].values)
h_jainlab26 = norm_hist2d(df.loc[df['source'] == 'jainlab26', 'PC2'].values,
                          df.loc[df['source'] == 'jainlab26', 'PC3'].values)
h_diff = h_jainlab26 - h_yoo25

density_max = max(h_yoo25.max(), h_jainlab26.max())
diff_abs_max = max(abs(h_diff.min()), abs(h_diff.max()))

panels = [
    ('yoo25',             h_yoo25,     'Viridis', 0,             density_max),
    ('jainlab26',         h_jainlab26, 'Viridis', 0,             density_max),
    ('jainlab26 − yoo25', h_diff,      'RdBu_r',  -diff_abs_max, diff_abs_max),
]

# panel centers in paper coords (3 equal panels, spacing=0.08)
PANEL_CENTERS = [0.14, 0.50, 0.86]

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[title for title, *_ in panels],
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0.08,
)

for col_idx, ((title, h, colorscale, zmin, zmax), cx) in enumerate(
        zip(panels, PANEL_CENTERS), start=1):
    fig.add_trace(
        go.Heatmap(
            x=x_centers, y=y_centers,
            z=h.T,  # transpose: rows=PC3, cols=PC2
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            showscale=True,
            colorbar=dict(
                orientation='h',
                x=cx, xanchor='center',
                y=-0.18, yanchor='top',
                len=0.28, thickness=12,
                title=dict(text='density' if col_idx < 3 else 'Δ density', side='bottom'),
            ),
        ),
        row=1, col=col_idx,
    )

fig.update_xaxes(title_text='Harmony PC2')
fig.update_yaxes(title_text='Harmony PC3', col=1)
fig.update_layout(
    title='PC2 vs PC3 density (normalized) — jainlab26 vs yoo25 L2/3',
    height=500, width=1100,
    margin=dict(b=100),
)
fig.write_html(OUT_FIG_DENSITY)
print(f'Saved → {OUT_FIG_DENSITY}')
