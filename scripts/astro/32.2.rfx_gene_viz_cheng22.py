# Rfx gene expression for cheng22 astrocytes, P28 and P38 separately.
# Scatter: cells in combined PCA space, dropdown for condition/archetype/gene.
# Boxplot: archetype on x-axis, NR/DR/DL side-by-side within each archetype group.
# Extends 32 with additional genes: Igfbp2, Chrdl1, Trpm3.

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pandas as pd
import plotly.graph_objects as go
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import scatter_categorical_html
from scomics.utils import norm

# --- file paths ---
SCRIPTS_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT          = os.path.dirname(SCRIPTS_DIR)
INPUT_CHENG22         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
PARQUET_COMBINED      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
ARCHETYPE_VERTICES_IN = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.archetype_vertices.parquet')
FIG_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
HTML_SCATTER_P28      = os.path.join(FIG_DIR, '32.2.rfx_scatter_cheng22_P28.html')
HTML_BOXPLOT_P28      = os.path.join(FIG_DIR, '32.2.rfx_gene_boxplot_cheng22_P28.html')
HTML_SCATTER_P38      = os.path.join(FIG_DIR, '32.2.rfx_scatter_cheng22_P38.html')
HTML_BOXPLOT_P38      = os.path.join(FIG_DIR, '32.2.rfx_gene_boxplot_cheng22_P38.html')

# --- config ---
GENES            = ['Rfx3', 'Rfx4', 'Rfx7', 'Igfbp2', 'Chrdl1', 'Trpm3', 'Id3', 'Gfap']
CHENG22_AGES     = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
P28_AGE_TO_COND  = {'P28': 'NR', 'P28_dr': 'DR', 'P28_dl': 'DL'}
P38_AGE_TO_COND  = {'P38': 'NR', 'P38_dr': 'DR'}
CONDITION_COLORS = {'NR': '#4C72B0', 'DR': '#DD8452', 'DL': '#2ca02c'}
PANELS           = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')]
PANEL_3D         = (0, 1, 2, 'PC1', 'PC2', 'PC3')

os.makedirs(FIG_DIR, exist_ok=True)

# --- load shared data ---
print('Loading parquets...')
df_combined = pd.read_parquet(PARQUET_COMBINED)
df_cheng22  = df_combined[df_combined['dataset'] == 'cheng22'].reset_index(drop=True)
pc_cols     = [c for c in df_combined.columns if c.startswith('PC')]

df_vertices = pd.read_parquet(ARCHETYPE_VERTICES_IN)
aa_vis      = df_vertices.values.T[:3, :]  # (3, NOC)

print(f'Loading {INPUT_CHENG22}...')
adata_full    = ad.read_h5ad(INPUT_CHENG22)
adata_cheng22 = adata_full[adata_full.obs['Age'].isin(CHENG22_AGES)].copy()

if len(df_cheng22) != adata_cheng22.shape[0]:
    raise ValueError(
        f'Cell count mismatch: parquet {len(df_cheng22)} vs h5ad {adata_cheng22.shape[0]}'
    )
df_cheng22 = df_cheng22.copy()
df_cheng22['cell_barcode'] = adata_cheng22.obs_names.values

archetypes = natsorted(df_cheng22['archetype'].unique())
var_names  = np.array(adata_cheng22.var_names)


def _build_plots(age_to_cond, html_scatter, html_boxplot, label):
    ages       = list(age_to_cond.keys())
    conditions = list(age_to_cond.values())  # ordered, unique (NR/DR/DL)

    df_sub = df_cheng22[df_cheng22['age'].isin(ages)].copy().reset_index(drop=True)
    df_sub['condition'] = df_sub['age'].map(age_to_cond)
    adata_sub = adata_cheng22[df_sub['cell_barcode'].values].copy()

    X = adata_sub.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    depths = X.sum(axis=1)
    xn     = norm(X, depths)                          # z-score (for scatter)
    logcpm = np.log1p(X / depths[:, None] * 1e4)     # log(CP10k+1) for boxplot

    found_genes = []
    gene_zscore = {}
    gene_logcpm = {}
    for gene in GENES:
        matches = np.where(var_names == gene)[0]
        if len(matches) == 0:
            print(f'  WARNING: {gene!r} not found, skipping')
            continue
        gene_zscore[gene] = xn[:, matches[0]]
        gene_logcpm[gene] = logcpm[:, matches[0]]
        found_genes.append(gene)
        print(f'  Found {gene!r} at index {matches[0]}')

    # --- scatter: condition + archetype + gene expression dropdown ---
    print(f'  Building scatter for {label}...')
    xp = df_sub[pc_cols].values
    meta = {
        'condition': df_sub['condition'].values,
        'archetype': df_sub['archetype'].values,
        'age':       df_sub['age'].values,
    }
    for gene in found_genes:
        meta[gene] = gene_zscore[gene]

    scatter_categorical_html(
        xp_grid=[xp],
        cell_metadata=meta,
        title=f'cheng22 {label} — Rfx gene expression (z-score) and condition',
        out_path=html_scatter,
        panels=PANELS,
        panel_3d=PANEL_3D,
        arch_vis=aa_vis,
        ordered_labels=['age'],
    )
    print(f'  Saved {html_scatter}')

    # --- boxplot: archetype × condition, spacers between archetype groups ---
    print(f'  Building boxplot for {label}...')
    for gene in found_genes:
        df_sub[gene] = gene_logcpm[gene]

    active_colors = {c: CONDITION_COLORS[c] for c in conditions}

    # x-axis: Arch1 NR, Arch1 DR, Arch1 DL, <spacer>, Arch2 NR, ...
    x_order = []
    for arch in archetypes:
        for cond in conditions:
            x_order.append(f'{arch} {cond}')
        x_order.append(f'_{arch}')  # invisible spacer between archetype groups
    x_order = x_order[:-1]  # drop trailing spacer

    all_traces = []
    gene_trace_ranges = {}

    for gene in found_genes:
        start = len(all_traces)
        for i, arch in enumerate(archetypes):
            for cond in conditions:
                mask = (df_sub['archetype'] == arch) & (df_sub['condition'] == cond)
                all_traces.append(go.Box(
                    x=[f'{arch} {cond}'] * int(mask.sum()),
                    y=df_sub.loc[mask, gene].values,
                    name=cond,
                    legendgroup=cond,
                    showlegend=(i == 0),
                    marker_color=active_colors[cond],
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
        s, e = gene_trace_ranges[gene]
        vis = [s <= i < e for i in range(n_total)]
        buttons.append(dict(
            label=gene, method='update',
            args=[{'visible': vis}, {'title': f'{gene} — cheng22 {label} astrocytes by archetype'}],
        ))

    fig.update_layout(
        title=f'{found_genes[0]} — cheng22 {label} astrocytes by archetype',
        xaxis=dict(
            title='',
            categoryorder='array', categoryarray=x_order,
            tickmode='array',
            tickvals=x_order,
            ticktext=['' if v.startswith('_') else v for v in x_order],
            tickangle=45,
        ),
        yaxis_title='log(CP10k + 1)',
        boxmode='overlay',
        width=850, height=500,
        legend_title='Condition',
        updatemenus=[dict(
            type='dropdown',
            buttons=buttons,
            x=0.0, xanchor='left', y=1.07, yanchor='top',
            bgcolor='white', bordercolor='grey', font=dict(size=12),
        )],
    )
    fig.write_html(html_boxplot)
    print(f'  Saved {html_boxplot}')


print('--- P28 ---')
_build_plots(P28_AGE_TO_COND, HTML_SCATTER_P28, HTML_BOXPLOT_P28, 'P28')

print('--- P38 ---')
_build_plots(P38_AGE_TO_COND, HTML_SCATTER_P38, HTML_BOXPLOT_P38, 'P38')

print('Done.')
