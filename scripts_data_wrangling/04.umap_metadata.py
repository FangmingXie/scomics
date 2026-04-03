import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UMAP_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26_cux2mice', '03.umap_coords.tsv')
H5AD_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'raw', 'morcom26_cux2mice', 'P26_ENs.h5ad')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '04.umap_metadata.html')

# --- config ---
MARKER_SIZE    = 3
MARKER_OPACITY = 0.6
PCTILE_LOW     = 5
PCTILE_HIGH    = 95

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load ---
print(f'Loading UMAP coords from {UMAP_FILE}')
umap_df = pd.read_csv(UMAP_FILE, sep='\t', index_col=0)

print(f'Loading h5ad from {H5AD_FILE}')
adata = ad.read_h5ad(H5AD_FILE)
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes')

# align cells
common_cells = umap_df.index.intersection(adata.obs_names)
umap_df = umap_df.loc[common_cells]
obs = adata.obs.loc[common_cells]
x = umap_df['UMAP1'].values
y = umap_df['UMAP2'].values
print(f'  {len(common_cells)} cells after alignment')
print(f'  Metadata columns: {list(obs.columns)}')


def is_categorical(series):
    return series.dtype.name in ('object', 'category', 'bool')


def build_categorical_traces(series, x, y, visible):
    """One trace per unique category, tab20 colors."""
    cats = sorted(series.astype(str).unique())
    cmap = plt.get_cmap('tab20', max(len(cats), 1))
    colors = {c: mcolors.to_hex(cmap(i)) for i, c in enumerate(cats)}
    traces = []
    for cat in cats:
        mask = series.astype(str) == cat
        traces.append(go.Scatter(
            x=x[mask],
            y=y[mask],
            mode='markers',
            name=cat,
            marker=dict(size=MARKER_SIZE, color=colors[cat], opacity=MARKER_OPACITY),
            visible=visible,
        ))
    return traces


def build_continuous_traces(series, x, y, col, visible):
    """Single trace with Viridis colorscale, cmin/cmax at 5th/95th percentile."""
    vals = series.astype(float).values
    cmin = np.nanpercentile(vals, PCTILE_LOW)
    cmax = np.nanpercentile(vals, PCTILE_HIGH)
    traces = [go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name=col,
        marker=dict(
            size=MARKER_SIZE,
            color=vals,
            colorscale='Viridis',
            cmin=cmin,
            cmax=cmax,
            opacity=MARKER_OPACITY,
            showscale=True,
            colorbar=dict(title=col),
        ),
        visible=visible,
        showlegend=False,
    )]
    return traces


# --- build all trace groups ---
cols = list(obs.columns)
trace_groups = []   # list of (col, [traces])
for i, col in enumerate(cols):
    visible = (i == 0)
    series = obs[col]
    if is_categorical(series):
        traces = build_categorical_traces(series, x, y, visible)
    else:
        traces = build_continuous_traces(series, x, y, col, visible)
    trace_groups.append((col, traces))

# --- assemble figure ---
fig = go.Figure()
trace_counts = []
for col, traces in trace_groups:
    for t in traces:
        fig.add_trace(t)
    trace_counts.append(len(traces))

total_traces = sum(trace_counts)

# --- build dropdown buttons ---
buttons = []
offset = 0
for i, (col, traces) in enumerate(trace_groups):
    n = trace_counts[i]
    vis = [False] * total_traces
    for j in range(offset, offset + n):
        vis[j] = True
    buttons.append(dict(label=col, method='update', args=[{'visible': vis}, {'title': col}]))
    offset += n

fig.update_layout(
    title=cols[0],
    xaxis_title='UMAP1',
    yaxis_title='UMAP2',
    width=850,
    height=700,
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
