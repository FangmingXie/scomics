# Rfx3, Rfx4, Rfx7 gene expression visualization for P56 gao25 astrocytes.
# Scatter plots in PCA space (reuses 18.3 pattern) + boxplots by archetype.
# Extends 31 with additional genes: Igfbp2, Chrdl1, Trpm3.

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pandas as pd
import plotly.graph_objects as go
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import gene_expr_scatter_html
from scomics.utils import norm

# --- file paths ---
SCRIPTS_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT          = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE            = os.path.join(PROJECT_ROOT, 'links', 'astro', 'gao25_scrna_astro.h5ad')
PARQUET_ALL_IN        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '17.labels_all_ages.parquet')
ARCHETYPE_VERTICES_IN = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '17.archetype_vertices_knn.parquet')
PARQUET_COMBINED      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
FIG_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
HTML_SCATTER          = os.path.join(FIG_DIR, '31.2.rfx_gene_scatter.html')
HTML_BOXPLOT          = os.path.join(FIG_DIR, '31.2.rfx_gene_boxplot.html')

# --- config ---
GENES    = ['Rfx3', 'Rfx4', 'Rfx7', 'Igfbp2', 'Chrdl1', 'Trpm3', 'Id3', 'Gfap']
PANELS   = [(0, 1, 'PC1', 'PC3'), (0, 2, 'PC1', 'PC4'), (1, 2, 'PC3', 'PC4')]
PANEL_3D = (0, 1, 2, 'PC1', 'PC3', 'PC4')
ARCHETYPE_COLORS = {
    'Arch1': '#1f77b4',
    'Arch2': '#ff7f0e',
    'Arch3': '#2ca02c',
    'Arch4': '#d62728',
}

os.makedirs(FIG_DIR, exist_ok=True)

# --- load data ---
print('Loading parquets...')
df_all      = pd.read_parquet(PARQUET_ALL_IN)
pc_cols     = [c for c in df_all.columns if c.startswith('PC')]
df_vertices = pd.read_parquet(ARCHETYPE_VERTICES_IN)
aa_vis      = df_vertices.values.T[:3, :]  # (3, NOC)

df_combined = pd.read_parquet(PARQUET_COMBINED)
df_p56_meta = df_combined[
    (df_combined['dataset'] == 'gao25') & (df_combined['age'] == 'P56')
].reset_index(drop=True)

print(f'Loading {INPUT_FILE}...')
adata    = ad.read_h5ad(INPUT_FILE)
adata_p56 = adata[adata.obs['Age'] == 'P56'].copy()

if len(df_p56_meta) != adata_p56.shape[0]:
    raise ValueError(
        f'Cell count mismatch: parquet {len(df_p56_meta)} vs h5ad {adata_p56.shape[0]}'
    )

X_p56     = adata_p56.X.toarray() if sp.issparse(adata_p56.X) else np.array(adata_p56.X)
depths    = X_p56.sum(axis=1)
var_names = np.array(adata_p56.var_names)

# z-scored (for scatter) and log-CPM (for boxplot)
xn_p56 = norm(X_p56, depths)
logcpm = np.log1p(X_p56 / depths[:, None] * 1e4)

# --- extract per-gene expression values ---
gene_vals  = {}  # z-score, for scatter
gene_logcpm = {}  # log(CP10k+1), for boxplot
for gene in GENES:
    matches = np.where(var_names == gene)[0]
    if len(matches) == 0:
        print(f'  WARNING: {gene!r} not found in var_names, skipping')
        continue
    gene_vals[gene]   = xn_p56[:, matches[0]]
    gene_logcpm[gene] = logcpm[:, matches[0]]
    print(f'  Found {gene!r} at index {matches[0]}')

found_genes = list(gene_vals.keys())

# --- scatter plot ---
print('Building scatter plot...')
df_p56_pca = df_all[df_all['age'] == 'P56'].reset_index(drop=True)
xp_p56     = df_p56_pca[pc_cols].values

gene_expr_scatter_html(
    x=None, y=None,
    xp=xp_p56,
    gene_vals=gene_vals,
    aa=aa_vis,
    title='P56 gao25 astrocytes — Rfx gene expression',
    out_path=HTML_SCATTER,
    panels=PANELS,
    panel_3d=PANEL_3D,
    marker_size=5,
    bg_color='white',
)
print(f'Saved {HTML_SCATTER}')

# --- boxplot by archetype ---
print('Building boxplot...')
archetypes = natsorted(df_p56_meta['archetype'].unique())

for gene in found_genes:
    df_p56_meta[gene] = gene_logcpm[gene]

all_traces = []
gene_trace_ranges = {}

for gene in found_genes:
    start = len(all_traces)
    for arch in archetypes:
        mask = df_p56_meta['archetype'] == arch
        all_traces.append(go.Box(
            x=[arch] * mask.sum(),
            y=df_p56_meta.loc[mask, gene].values,
            name=arch,
            legendgroup=arch,
            showlegend=(gene == found_genes[0]),
            marker_color=ARCHETYPE_COLORS.get(arch, '#888888'),
            boxpoints='outliers',
            visible=False,
        ))
    gene_trace_ranges[gene] = (start, len(all_traces))

first_start, first_end = gene_trace_ranges[found_genes[0]]
for i in range(first_start, first_end):
    all_traces[i].visible = True

fig = go.Figure(data=all_traces)
n_total = len(all_traces)
buttons = []
for gene in found_genes:
    start, end = gene_trace_ranges[gene]
    vis = [start <= i < end for i in range(n_total)]
    buttons.append(dict(
        label=gene, method='update',
        args=[{'visible': vis}, {'title': f'{gene} — P56 gao25 astrocytes by archetype'}],
    ))

fig.update_layout(
    title=f'{found_genes[0]} — P56 gao25 astrocytes by archetype',
    xaxis_title='Archetype',
    yaxis_title='log(CP10k + 1)',
    width=650, height=500,
    legend_title='Archetype',
    updatemenus=[dict(
        type='dropdown',
        buttons=buttons,
        x=0.0, xanchor='left', y=1.07, yanchor='top',
        bgcolor='white', bordercolor='grey', font=dict(size=12),
    )],
)
fig.write_html(HTML_BOXPLOT)
print(f'Saved {HTML_BOXPLOT}')
print('Done.')
