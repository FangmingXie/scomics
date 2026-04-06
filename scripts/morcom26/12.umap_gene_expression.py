import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import plotly.graph_objects as go

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UMAP_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26_cux2mice', '09.umap_coords.tsv')
H5AD_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'raw', 'morcom26_cux2mice', 'P26_ENs.h5ad')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '12.umap_gene_expression.html')

# --- config ---
GENES          = ['Cdh13', 'Trpc6', 'Sorcs3', 'Pcdh15', 'Chrm2',
                  'Cux1', 'Cux2', 'Rorb', 'Meis2', 'Foxp1']
MARKER_SIZE    = 3
MARKER_OPACITY = 0.6
PCTILE_LOW     = 9
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
adata = adata[common_cells]
x = umap_df['UMAP1'].values
y = umap_df['UMAP2'].values
print(f'  {len(common_cells)} cells after alignment')

# --- check genes exist ---
missing = [g for g in GENES if g not in adata.var_names]
if missing:
    raise ValueError(f'Genes not found in h5ad: {missing}')

# --- normalize: CP10k + log1p ---
X = adata.X
if sp.issparse(X):
    X = X.toarray()
depths = X.sum(axis=1, keepdims=True)
X_norm = np.log1p(X / depths * 1e4)

# --- z-score per gene across cells ---
gene_means = X_norm.mean(axis=0)
gene_stds  = X_norm.std(axis=0)
gene_stds[gene_stds == 0] = 1
X_zscore = (X_norm - gene_means) / gene_stds

gene_idx = {g: list(adata.var_names).index(g) for g in GENES}

# --- build one trace per gene ---
traces = []
for i, gene in enumerate(GENES):
    vals = X_zscore[:, gene_idx[gene]]
    cmin = np.nanpercentile(vals, PCTILE_LOW)
    cmax = np.nanpercentile(vals, PCTILE_HIGH)
    traces.append(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name=gene,
        marker=dict(
            size=MARKER_SIZE,
            color=vals,
            colorscale='RdBu_r',
            cmin=cmin,
            cmax=cmax,
            opacity=MARKER_OPACITY,
            showscale=True,
            colorbar=dict(title='z-score'),
        ),
        visible=(i == 0),
        showlegend=False,
    ))

# --- build dropdown buttons ---
buttons = []
for i, gene in enumerate(GENES):
    vis = [j == i for j in range(len(GENES))]
    buttons.append(dict(
        label=gene,
        method='update',
        args=[{'visible': vis}, {'title': f'{gene} — z-score(log1p(CP10k))'}],
    ))

fig = go.Figure(data=traces)
fig.update_layout(
    title=f'{GENES[0]} — z-score(log1p(CP10k))',
    xaxis_title='UMAP1',
    yaxis_title='UMAP2',
    width=850,
    height=700,
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
