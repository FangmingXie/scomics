# Visualization of DEG results — LPLC2 APF_48h PC2 extreme bins
# 1. Volcano plot: Log2FC vs -log10(FDR)
# 2. PC2 vs expression scatter+line for top significant DEGs (gene dropdown)
#    Scatter: all cells; line: per-bin mean
# Outputs:
#   local_data/fig/dombrovski25_fly/09.volcano_pc2_bins_APF48h.html
#   local_data/fig/dombrovski25_fly/09.pc2_vs_deg_expr_APF48h.html

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import plotly.graph_objects as go
from matplotlib import cm, colors as mcolors

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE  = os.path.join(PROJECT_ROOT, 'links', 'fly', 'dombrovski25_fly.h5ad')
IN_DEG      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'dombrovski25_fly', '08.deg_pc2_bins_APF48h.parquet')
IN_PCA      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'dombrovski25_fly', '07.lplc2_APF48h_pca.parquet')
FIG_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'dombrovski25_fly')
OUT_VOLCANO = os.path.join(FIG_DIR, '09.volcano_pc2_bins_APF48h.html')
OUT_EXPR    = os.path.join(FIG_DIR, '09.pc2_vs_deg_expr_APF48h.html')

CELLTYPE      = 'LPLC2'
AGE           = 'APF_48h'
FDR_THRESH    = 0.05
LOG2FC_THRESH = 1.0
N_LABEL       = 15   # genes labeled on volcano
N_TOP_DEG     = 20   # top significant DEGs shown in PC2 vs expr plot

os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# 1. Volcano plot
# =============================================================================
deg_df = pd.read_parquet(IN_DEG)
deg_df['neg_log10_fdr'] = -np.log10(deg_df['fdr'].clip(lower=1e-300))

sig_up = (deg_df['fdr'] < FDR_THRESH) & (deg_df['log2fc'] >  LOG2FC_THRESH)
sig_dn = (deg_df['fdr'] < FDR_THRESH) & (deg_df['log2fc'] < -LOG2FC_THRESH)
ns     = ~(sig_up | sig_dn)

COLOR_UP = '#d62728'
COLOR_DN = '#1f77b4'
COLOR_NS = '#aaaaaa'

fig_v = go.Figure()
fig_v.add_trace(go.Scatter(
    x=deg_df.loc[ns, 'log2fc'], y=deg_df.loc[ns, 'neg_log10_fdr'],
    mode='markers', name='n.s.',
    marker=dict(color=COLOR_NS, size=5, opacity=0.5),
    text=deg_df.loc[ns, 'gene'],
    hovertemplate='%{text}<br>Log2FC=%{x:.2f}<br>-log10(FDR)=%{y:.1f}<extra></extra>',
))
fig_v.add_trace(go.Scatter(
    x=deg_df.loc[sig_dn, 'log2fc'], y=deg_df.loc[sig_dn, 'neg_log10_fdr'],
    mode='markers', name='up in low PC2',
    marker=dict(color=COLOR_DN, size=6, opacity=0.8),
    text=deg_df.loc[sig_dn, 'gene'],
    hovertemplate='%{text}<br>Log2FC=%{x:.2f}<br>-log10(FDR)=%{y:.1f}<extra></extra>',
))
fig_v.add_trace(go.Scatter(
    x=deg_df.loc[sig_up, 'log2fc'], y=deg_df.loc[sig_up, 'neg_log10_fdr'],
    mode='markers', name='up in high PC2',
    marker=dict(color=COLOR_UP, size=6, opacity=0.8),
    text=deg_df.loc[sig_up, 'gene'],
    hovertemplate='%{text}<br>Log2FC=%{x:.2f}<br>-log10(FDR)=%{y:.1f}<extra></extra>',
))

fdr_line = -np.log10(FDR_THRESH)
x_range  = [deg_df['log2fc'].min() - 0.3, deg_df['log2fc'].max() + 0.3]
fig_v.add_shape(type='line', x0=x_range[0], x1=x_range[1],
                y0=fdr_line, y1=fdr_line,
                line=dict(color='black', width=1, dash='dash'))
fig_v.add_shape(type='line', x0=-LOG2FC_THRESH, x1=-LOG2FC_THRESH,
                y0=0, y1=deg_df['neg_log10_fdr'].max() * 1.05,
                line=dict(color='black', width=1, dash='dash'))
fig_v.add_shape(type='line', x0=LOG2FC_THRESH, x1=LOG2FC_THRESH,
                y0=0, y1=deg_df['neg_log10_fdr'].max() * 1.05,
                line=dict(color='black', width=1, dash='dash'))

for _, row in deg_df[deg_df['significant']].nsmallest(N_LABEL, 'fdr').iterrows():
    fig_v.add_annotation(
        x=row['log2fc'], y=row['neg_log10_fdr'],
        text=row['gene'], showarrow=False,
        font=dict(size=10), xshift=8, yshift=4,
    )

n_up = sig_up.sum()
n_dn = sig_dn.sum()
fig_v.update_layout(
    title=(f'LPLC2 APF_48h — DEGs: PC2 high (bin08+09) vs low (bin00+01)<br>'
           f'<sup>up in high PC2: {n_up}  |  up in low PC2: {n_dn}  '
           f'(FDR<{FDR_THRESH}, |Log2FC|>{LOG2FC_THRESH})</sup>'),
    xaxis_title='Log2 fold change (high PC2 / low PC2)',
    yaxis_title='-log10(FDR)',
    width=800, height=650,
    legend=dict(itemsizing='constant'),
)
fig_v.write_html(OUT_VOLCANO)
print(f'Saved {OUT_VOLCANO}')

# =============================================================================
# 2. PC2 vs expression scatter + per-bin mean line (gene dropdown)
# =============================================================================
# load PC2 values and bin labels
pca_df  = pd.read_parquet(IN_PCA)
pc2_vals = pca_df['PC2'].values
pc2_bin  = pca_df['pc2_bin'].values

# load expression for top DEGs from h5ad
adata = ad.read_h5ad(INPUT_FILE)
adata = adata[adata.obs['type1'] == CELLTYPE].copy()
mt_mask = np.array([g.lower().startswith('mt:') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
adata = adata[adata.obs['orig.ident'] == AGE].copy()

x_raw    = np.array(adata.X)
depths   = adata.obs['nCount_RNA'].values
xlog     = np.log2(x_raw / depths[:, None] * 1e4 + 1)
all_genes = np.array(adata.var_names)

top_deg_genes = deg_df[deg_df['significant']].nsmallest(N_TOP_DEG, 'fdr')['gene'].tolist()
print(f'Building PC2 vs expression plot for {len(top_deg_genes)} top DEGs...')

# bin order and per-bin mean PC2 (x-position for the mean line)
bin_order    = [f'bin{i:02d}' for i in range(10)]
bin_mean_pc2 = np.array([pc2_vals[pc2_bin == b].mean() for b in bin_order])

# bin colors (turbo colormap, ordered low→high PC2)
cmap       = cm.get_cmap('turbo', 10)
bin_colors = [mcolors.to_hex(cmap(i / 9)) for i in range(10)]
bin_color_map = dict(zip(bin_order, bin_colors))
cell_colors   = np.array([bin_color_map[b] for b in pc2_bin])

fig_e = go.Figure()
traces_per_gene = 2   # scatter (all cells) + line (bin means)

for gi, gene in enumerate(top_deg_genes):
    gene_idx = np.where(all_genes == gene)[0][0]
    expr     = xlog[:, gene_idx]
    visible  = (gi == 0)

    bin_mean_expr = np.array([expr[pc2_bin == b].mean() for b in bin_order])

    # scatter: all cells
    fig_e.add_trace(go.Scatter(
        x=pc2_vals, y=expr,
        mode='markers',
        marker=dict(color=cell_colors, size=3, opacity=0.4),
        name=gene, showlegend=False, visible=visible,
        hovertemplate=f'{gene}<br>PC2=%{{x:.2f}}<br>expr=%{{y:.2f}}<extra></extra>',
    ))
    # line: per-bin means
    fig_e.add_trace(go.Scatter(
        x=bin_mean_pc2, y=bin_mean_expr,
        mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(color=bin_colors, size=8, line=dict(color='black', width=1)),
        name=f'{gene} mean', showlegend=False, visible=visible,
        hovertemplate=f'{gene} mean<br>PC2=%{{x:.2f}}<br>expr=%{{y:.2f}}<extra></extra>',
    ))

# dropdown
n_total = len(fig_e.data)
buttons = []
for gi, gene in enumerate(top_deg_genes):
    vis = [False] * n_total
    vis[gi * traces_per_gene]     = True
    vis[gi * traces_per_gene + 1] = True
    buttons.append(dict(label=gene, method='update',
                        args=[{'visible': vis}, {'title': f'PC2 vs {gene} expression — LPLC2 {AGE}'}]))

fig_e.update_layout(
    title=f'PC2 vs {top_deg_genes[0]} expression — LPLC2 {AGE}',
    xaxis_title='PC2',
    yaxis_title='log2(CP10k + 1)',
    width=750, height=600,
    updatemenus=[dict(
        type='dropdown',
        x=0.0, xanchor='left', y=1.05, yanchor='bottom',
        buttons=buttons,
    )],
)
fig_e.write_html(OUT_EXPR)
print(f'Saved {OUT_EXPR}')
