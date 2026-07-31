"""Volcano plots of NR-vs-DR pseudobulk DESeq2 DEGs across 4 VX1 bins (v6).

Same as the v4 volcano viz but for the all-archetype (Arch1-4) VX1-binned DEG
tables from script 45.v6. Standalone: reads only the v6 DESeq2 tables (no per-cell
pipeline needed for a volcano). x-axis = shrunk log2FC (DESeq2 apeglm), y-axis =
-log10(FDR / padj). Genes with FDR = NaN (DESeq2 independent filtering / Cook's
outliers) are dropped.

Reads:
  local_data/res/astro/45.v6.deg_vx1_bin{1..4}_NR_vs_DR_all.tsv
Outputs:
  local_data/fig/astro/45.v6.volcano_vx1_bins_NR_vs_DR.html
  local_data/fig/astro/45.v6.deg_counts_vx1_bins_NR_vs_DR.html
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_VOLCANO  = os.path.join(FIG_DIR, '45.v6.volcano_vx1_bins_NR_vs_DR.html')
OUT_COUNTS   = os.path.join(FIG_DIR, '45.v6.deg_counts_vx1_bins_NR_vs_DR.html')
DEG_ALL_TMPL = os.path.join(DEG_DIR, '45.v6.deg_vx1_bin{b}_NR_vs_DR_all.tsv')

N_BINS        = 4
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(1.5)  # threshold used for the volcano panels
N_LABEL       = 10  # top genes labeled per panel

# log2FC thresholds for the DEG-count bar panels (all with FDR < FDR_THRESH)
COUNT_PANELS = [(np.log2(1.5), '|log2FC_shrink| > log2(1.5)'),
                (1.0,          '|log2FC_shrink| > 1'),
                (2.0,          '|log2FC_shrink| > 2')]

COLOR_UP = '#d62728'  # up in DR (positive log2FC)
COLOR_DN = '#1f77b4'  # up in NR (negative log2FC)
COLOR_NS = '#aaaaaa'

os.makedirs(FIG_DIR, exist_ok=True)

deg_counts = []  # (bin_label, n_up_DR, n_up_NR) collected per bin for the count plot

print('Building DESeq2 volcano plots...')
fig_v = make_subplots(
    rows=1, cols=N_BINS,
    subplot_titles=[f'Bin {b + 1}' for b in range(N_BINS)],
    horizontal_spacing=0.06,
)

for b in range(N_BINS):
    col = b + 1
    deg_df = pd.read_csv(DEG_ALL_TMPL.format(b=b + 1), sep='\t')
    deg_df = deg_df[deg_df['fdr'].notna()].copy()           # drop independently-filtered genes
    deg_df['neg_log10_fdr'] = -np.log10(deg_df['fdr'].clip(lower=1e-300))

    sig_up = (deg_df['fdr'] < FDR_THRESH) & (deg_df['log2FC'] >  LOG2FC_THRESH)
    sig_dn = (deg_df['fdr'] < FDR_THRESH) & (deg_df['log2FC'] < -LOG2FC_THRESH)
    ns     = ~(sig_up | sig_dn)

    for sub, color, name in [(ns, COLOR_NS, 'n.s.'),
                             (sig_dn, COLOR_DN, 'up in NR'),
                             (sig_up, COLOR_UP, 'up in DR')]:
        fig_v.add_trace(go.Scatter(
            x=deg_df.loc[sub, 'log2FC'], y=deg_df.loc[sub, 'neg_log10_fdr'],
            mode='markers', name=name, legendgroup=name,
            showlegend=(b == 0),
            marker=dict(color=color, size=4 if name == 'n.s.' else 6,
                        opacity=0.4 if name == 'n.s.' else 0.8),
            text=deg_df.loc[sub, 'gene'],
            hovertemplate='%{text}<br>log2FC=%{x:.2f}<br>-log10(FDR)=%{y:.1f}<extra></extra>',
        ), row=1, col=col)

    xref = 'x' if col == 1 else f'x{col}'
    yref = 'y' if col == 1 else f'y{col}'
    x_lo = deg_df['log2FC'].min() - 0.3
    x_hi = deg_df['log2FC'].max() + 0.3
    y_hi = deg_df['neg_log10_fdr'].max() * 1.05
    fig_v.add_shape(type='line', x0=x_lo, x1=x_hi,
                    y0=-np.log10(FDR_THRESH), y1=-np.log10(FDR_THRESH),
                    line=dict(color='black', width=1, dash='dash'), xref=xref, yref=yref)
    for xthr in (-LOG2FC_THRESH, LOG2FC_THRESH):
        fig_v.add_shape(type='line', x0=xthr, x1=xthr, y0=0, y1=y_hi,
                        line=dict(color='black', width=1, dash='dash'), xref=xref, yref=yref)

    sig_df = deg_df[sig_up | sig_dn]
    for _, r in sig_df.nsmallest(N_LABEL, 'fdr').iterrows():
        fig_v.add_annotation(x=r['log2FC'], y=r['neg_log10_fdr'], text=r['gene'],
                             showarrow=False, font=dict(size=9), xshift=6, yshift=4,
                             xref=xref, yref=yref)

    up_by_thr = [int(((deg_df['fdr'] < FDR_THRESH) & (deg_df['log2FC'] >  thr)).sum())
                 for thr, _ in COUNT_PANELS]
    dn_by_thr = [int(((deg_df['fdr'] < FDR_THRESH) & (deg_df['log2FC'] < -thr)).sum())
                 for thr, _ in COUNT_PANELS]
    deg_counts.append((f'Bin {b + 1}', up_by_thr, dn_by_thr))
    fig_v.layout.annotations[b].text = f'Bin {b + 1} (up DR: {int(sig_up.sum())}, up NR: {int(sig_dn.sum())})'
    fig_v.update_xaxes(title_text='log2FC (DR / NR), shrunk', row=1, col=col)
    if col == 1:
        fig_v.update_yaxes(title_text='-log10(FDR)', row=1, col=col)

for ann in fig_v.layout.annotations[:N_BINS]:
    ann.font.size = 12

fig_v.update_layout(
    title=(f'NR vs DR pseudobulk DESeq2 DEGs across VX1 bins '
           f'(FDR<{FDR_THRESH}, |log2FC_shrink|>log2(1.5))'),
    width=2000, height=470, legend=dict(itemsizing='constant'),
)
fig_v.write_html(OUT_VOLCANO)
print(f'  Saved {OUT_VOLCANO}')

print('Building DESeq2 DEG-count bar plot...')
n_panels = len(COUNT_PANELS)
bins     = [c[0] for c in deg_counts][::-1]  # reverse: Bin 4, 3, 2, 1

fig_c = make_subplots(
    rows=n_panels, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=[label for _, label in COUNT_PANELS],
)
for p in range(n_panels):
    row = p + 1
    n_up_dr = [c[1][p] for c in deg_counts][::-1]
    n_up_nr = [c[2][p] for c in deg_counts][::-1]
    fig_c.add_trace(go.Bar(x=bins, y=n_up_dr, name='up in DR', legendgroup='up in DR',
                           marker_color=COLOR_UP, text=n_up_dr, textposition='outside',
                           showlegend=(p == 0)), row=row, col=1)
    fig_c.add_trace(go.Bar(x=bins, y=n_up_nr, name='up in NR', legendgroup='up in NR',
                           marker_color=COLOR_DN, text=n_up_nr, textposition='outside',
                           showlegend=(p == 0)), row=row, col=1)
    fig_c.update_yaxes(title_text='number of DEGs', row=row, col=1)
fig_c.update_xaxes(title_text='VX1 bin', row=n_panels, col=1)
fig_c.update_layout(
    barmode='group',
    title=(f'NR vs DR pseudobulk DESeq2 DEG counts across VX1 bins (FDR<{FDR_THRESH})'),
    width=700, height=360 * n_panels, legend=dict(itemsizing='constant'),
)
fig_c.write_html(OUT_COUNTS)
print(f'  Saved {OUT_COUNTS}')
print('Done.')
