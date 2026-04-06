import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5AD_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'raw', 'morcom26_cux2mice', 'P26_ENs.h5ad')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '05.boxplot_metadata.html')

# --- config ---
CELLTYPE_COL = 'celltype'

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load ---
print(f'Loading {H5AD_FILE}')
adata = ad.read_h5ad(H5AD_FILE)
obs = adata.obs.copy()
print(f'  {obs.shape[0]} cells, columns: {list(obs.columns)}')

# --- identify continuous columns ---
continuous_cols = [c for c in obs.columns
                   if c != CELLTYPE_COL and obs[c].dtype.kind in ('f', 'i', 'u')]
print(f'  Continuous columns: {continuous_cols}')

# --- cell type color palette (tab20, sorted) ---
celltypes = sorted(obs[CELLTYPE_COL].astype(str).unique())
cmap = plt.get_cmap('tab20', max(len(celltypes), 1))
ct_colors = {ct: mcolors.to_hex(cmap(i)) for i, ct in enumerate(celltypes)}

# --- build one Box trace group per continuous column ---
# Each group = one go.Box per cell type (for per-celltype coloring)
trace_groups = []   # list of (col, [traces])
for col in continuous_cols:
    traces = []
    for ct in celltypes:
        mask = obs[CELLTYPE_COL].astype(str) == ct
        traces.append(go.Box(
            x=[ct] * mask.sum(),
            y=obs.loc[mask, col].astype(float).values,
            name=ct,
            marker_color=ct_colors[ct],
            width=0.8,
            boxpoints=False,
            visible=False,
            showlegend=True,
        ))
    trace_groups.append((col, traces))

# --- assemble figure ---
fig = go.Figure()
trace_counts = []
for col, traces in trace_groups:
    for t in traces:
        fig.add_trace(t)
    trace_counts.append(len(traces))

total_traces = sum(trace_counts)

# make first column visible
for j in range(trace_counts[0]):
    fig.data[j].visible = True

# --- build dropdown ---
buttons = []
offset = 0
for i, (col, _) in enumerate(trace_groups):
    n = trace_counts[i]
    vis = [False] * total_traces
    for j in range(offset, offset + n):
        vis[j] = True
    buttons.append(dict(label=col, method='update',
                        args=[{'visible': vis}, {'yaxis.title.text': col, 'title': col}]))
    offset += n

fig.update_layout(
    title=continuous_cols[0],
    xaxis_title='Cell type',
    yaxis_title=continuous_cols[0],
    xaxis=dict(tickangle=45),
    boxmode='group',
    boxgap=0.0,
    boxgroupgap=0.0,
    width=1400,
    height=650,
    legend=dict(itemsizing='constant'),
    updatemenus=[dict(
        type='dropdown',
        buttons=buttons,
        x=0.0,
        xanchor='left',
        y=1.07,
        yanchor='top',
        bgcolor='white',
        bordercolor='grey',
        font=dict(size=12),
    )],
)

fig.write_html(OUT_HTML)
print(f'Saved {OUT_HTML}')
print('Done.')
