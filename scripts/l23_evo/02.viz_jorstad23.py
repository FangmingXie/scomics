"""Visualization for Jorstad23 human L2/3 IT first-pass analysis.

Reads precomputed TSV results from 01.compute_jorstad23.py and produces:
  - Interactive PC-covariate correlation heatmap (HTML)
  - UMAP with toggle buttons for cluster/donor/source/age coloring (HTML)
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from viz import scatter_categorical_html
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_UMAP      = os.path.join(OUT_RES_DIR, '01.umap_coords.tsv')
IN_PC_CORR   = os.path.join(OUT_RES_DIR, '01.pc_covariate_corr.tsv')
OUT_HEATMAP  = os.path.join(OUT_FIG_DIR, '02.pc_covariate_heatmap.html')
OUT_UMAP_FIG = os.path.join(OUT_FIG_DIR, '02.umap_panels.html')

CLUSTER_COL    = 'WithinArea_cluster'
COVARIATE_COLS = ['donor_id', 'Source', 'development_stage']  # development_stage kept for UMAP coloring only

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- Figure A: PC-covariate correlation heatmap ---
print('Plotting PC-covariate heatmap...')
corr_df = pd.read_csv(IN_PC_CORR, sep='\t', index_col=0)

z = corr_df.values
text = [[f'{v:.2f}' for v in row] for row in z]
abs_max = np.abs(z).max()

fig = go.Figure(go.Heatmap(
    z=z,
    x=corr_df.columns.tolist(),
    y=corr_df.index.tolist(),
    text=text,
    texttemplate='%{text}',
    colorscale='RdBu_r',
    zmid=0,
    zmin=-abs_max,
    zmax=abs_max,
    colorbar=dict(title='Pearson r'),
))
fig.update_layout(
    title='PC–covariate Pearson correlation (Jorstad23 L2/3 IT)',
    xaxis=dict(tickangle=45),
    width=800,
    height=500,
)
fig.write_html(OUT_HEATMAP)
print(f'  Saved {OUT_HEATMAP}')

# --- Figure B: UMAP with toggle buttons ---
print('Plotting UMAP panels...')
umap_df = pd.read_csv(IN_UMAP, sep='\t', index_col=0)

umap_coords = umap_df[['UMAP1', 'UMAP2']].values
cell_metadata = {col: umap_df[col].values for col in [CLUSTER_COL] + COVARIATE_COLS}

scatter_categorical_html(
    xp_grid=[umap_coords],
    cell_metadata=cell_metadata,
    title='Jorstad23 L2/3 IT — UMAP',
    out_path=OUT_UMAP_FIG,
    panels=[(0, 1, 'UMAP1', 'UMAP2')],
)

print('\nDone.')
