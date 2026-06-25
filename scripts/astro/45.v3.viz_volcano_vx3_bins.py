"""Volcano plots of NR-vs-DR pseudobulk t-test DEGs across 4 VX3 bins (v3).

Standalone: reads only the v3 pseudobulk tables. x-axis = log2FC (mean log2CPM
difference, DR − NR), y-axis = -log10(FDR) from the Welch t-test (`fdr_t`, the
primary test; Wilcoxon has ~0 power at n=7/8 and is not plotted).

Reads:
  local_data/res/astro/45.v3.deg_vx3_bin{1..4}_NR_vs_DR_all.tsv
Outputs:
  local_data/fig/astro/45.v3.volcano_vx3_bins_NR_vs_DR.html
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_VOLCANO  = os.path.join(FIG_DIR, '45.v3.volcano_vx3_bins_NR_vs_DR.html')
DEG_ALL_TMPL = os.path.join(DEG_DIR, '45.v3.deg_vx3_bin{b}_NR_vs_DR_all.tsv')

N_BINS        = 4
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(1.5)
N_LABEL       = 10  # top genes labeled per panel

COLOR_UP = '#d62728'  # up in DR (positive log2FC)
COLOR_DN = '#1f77b4'  # up in NR (negative log2FC)
COLOR_NS = '#aaaaaa'

os.makedirs(FIG_DIR, exist_ok=True)

print('Building pseudobulk t-test volcano plots...')
fig_v = make_subplots(
    rows=1, cols=N_BINS,
    subplot_titles=[f'Bin {b + 1}' for b in range(N_BINS)],
    horizontal_spacing=0.06,
)

for b in range(N_BINS):
    col = b + 1
    deg_df = pd.read_csv(DEG_ALL_TMPL.format(b=b + 1), sep='\t')
    deg_df = deg_df[deg_df['fdr_t'].notna()].copy()
    deg_df['neg_log10_fdr'] = -np.log10(deg_df['fdr_t'].clip(lower=1e-300))

    sig_up = (deg_df['fdr_t'] < FDR_THRESH) & (deg_df['log2FC'] >  LOG2FC_THRESH)
    sig_dn = (deg_df['fdr_t'] < FDR_THRESH) & (deg_df['log2FC'] < -LOG2FC_THRESH)
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
    for _, r in sig_df.nsmallest(N_LABEL, 'fdr_t').iterrows():
        fig_v.add_annotation(x=r['log2FC'], y=r['neg_log10_fdr'], text=r['gene'],
                             showarrow=False, font=dict(size=9), xshift=6, yshift=4,
                             xref=xref, yref=yref)

    fig_v.layout.annotations[b].text = f'Bin {b + 1} (up DR: {int(sig_up.sum())}, up NR: {int(sig_dn.sum())})'
    fig_v.update_xaxes(title_text='log2FC (DR / NR)', row=1, col=col)
    if col == 1:
        fig_v.update_yaxes(title_text='-log10(FDR, t-test)', row=1, col=col)

for ann in fig_v.layout.annotations[:N_BINS]:
    ann.font.size = 12

fig_v.update_layout(
    title=(f'NR vs DR pseudobulk Welch t-test DEGs across VX3 bins '
           f'(FDR<{FDR_THRESH}, |log2FC|>log2(1.5))'),
    width=2000, height=470, legend=dict(itemsizing='constant'),
)
fig_v.write_html(OUT_VOLCANO)
print(f'  Saved {OUT_VOLCANO}')
print('Done.')
