import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5AD_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'raw', 'morcom26_cux2mice', 'P26_ENs.h5ad')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '06.barplot_composition.html')

# --- config ---
CELLTYPE_COL  = 'celltype'
SAMPLE_COL    = 'samples'
CONDITION_COL = 'Condition'
OTHER_MIX_LABEL = 'Other-mix'

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load ---
print(f'Loading {H5AD_FILE}')
adata = ad.read_h5ad(H5AD_FILE)
obs = adata.obs[[CELLTYPE_COL, SAMPLE_COL, CONDITION_COL]].copy()
print(f'  {len(obs)} cells')

# --- merge Other-mix* cell types ---
obs[CELLTYPE_COL] = obs[CELLTYPE_COL].astype(str).apply(
    lambda ct: OTHER_MIX_LABEL if 'Other-mix' in ct else ct
)
print(f'  Cell types after merge: {sorted(obs[CELLTYPE_COL].unique())}')

# --- ordered cell types: alphabetical, Other-mix last ---
celltypes = sorted(ct for ct in obs[CELLTYPE_COL].unique() if ct != OTHER_MIX_LABEL)
celltypes.append(OTHER_MIX_LABEL)

# --- color palette ---
cmap = plt.get_cmap('tab20', len(celltypes))
ct_colors = {ct: mcolors.to_hex(cmap(i)) for i, ct in enumerate(celltypes)}

# --- ordered samples and conditions ---
sample_order    = sorted(obs[SAMPLE_COL].unique())
condition_order = sorted(obs[CONDITION_COL].unique())


def compute_fractions(obs, group_col, group_order):
    """Return DataFrame (group x celltype) of cell type fractions."""
    counts = (obs.groupby([group_col, CELLTYPE_COL], observed=True)
                 .size()
                 .unstack(fill_value=0)
                 .reindex(index=group_order, columns=celltypes, fill_value=0))
    return counts.div(counts.sum(axis=1), axis=0)


sample_frac    = compute_fractions(obs, SAMPLE_COL, sample_order)
condition_frac = compute_fractions(obs, CONDITION_COL, condition_order)

# --- build figure ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Per sample', 'Per condition'],
    shared_yaxes=True,
)

for i, ct in enumerate(celltypes):
    color = ct_colors[ct]
    # panel 1: per sample
    fig.add_trace(go.Bar(
        name=ct,
        x=sample_order,
        y=sample_frac[ct].values,
        marker_color=color,
        legendgroup=ct,
        showlegend=True,
    ), row=1, col=1)
    # panel 2: per condition
    fig.add_trace(go.Bar(
        name=ct,
        x=condition_order,
        y=condition_frac[ct].values,
        marker_color=color,
        legendgroup=ct,
        showlegend=False,
    ), row=1, col=2)

fig.update_layout(
    barmode='stack',
    title='Cell type composition',
    yaxis_title='Fraction of cells',
    yaxis=dict(range=[0, 1]),
    legend=dict(itemsizing='constant', traceorder='normal'),
    width=1000,
    height=600,
)
fig.update_xaxes(tickangle=45)

fig.write_html(OUT_HTML)
print(f'Saved {OUT_HTML}')
print('Done.')
