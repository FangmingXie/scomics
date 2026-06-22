# Volcano plots at two Log2FC thresholds — Pvalb Gaba vs Sst Gaba, P56 mouse VIS cortex
# Threshold 1: |Log2FC| > 1.0
# Threshold 2: |Log2FC| > 0.5
# Outputs:
#   local_data/fig/dombrovski25_fly/11.volcano_lfc1_pv_sst_P56_gao25.html
#   local_data/fig/dombrovski25_fly/11.volcano_lfc05_pv_sst_P56_gao25.html

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
IN_DEG          = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'dombrovski25_fly', '10.deg_pv_sst_P56_gao25.parquet')
FIG_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'dombrovski25_fly')
OUT_LFC1        = os.path.join(FIG_DIR, '11.volcano_lfc1_pv_sst_P56_gao25.html')
OUT_LFC05       = os.path.join(FIG_DIR, '11.volcano_lfc05_pv_sst_P56_gao25.html')

GROUP_A_LABEL = 'Pvalb Gaba'
GROUP_B_LABEL = 'Sst Gaba'
FDR_THRESH    = 0.05
N_LABEL       = 15
COLOR_UP      = '#d62728'   # up in Group B (Sst)
COLOR_DN      = '#1f77b4'   # up in Group A (Pvalb)
COLOR_NS      = '#aaaaaa'

os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_parquet(IN_DEG)
df['neg_log10_fdr'] = -np.log10(df['fdr'].clip(lower=1e-300))


def make_volcano(df, lfc_thresh, out_path):
    sig_up = (df['fdr'] < FDR_THRESH) & (df['log2fc'] >  lfc_thresh)
    sig_dn = (df['fdr'] < FDR_THRESH) & (df['log2fc'] < -lfc_thresh)
    ns     = ~(sig_up | sig_dn)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.loc[ns, 'log2fc'], y=df.loc[ns, 'neg_log10_fdr'],
        mode='markers', name='n.s.',
        marker=dict(color=COLOR_NS, size=5, opacity=0.5),
        text=df.loc[ns, 'gene'],
        hovertemplate='%{text}<br>Log2FC=%{x:.2f}<br>-log10(FDR)=%{y:.1f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=df.loc[sig_dn, 'log2fc'], y=df.loc[sig_dn, 'neg_log10_fdr'],
        mode='markers', name=f'up in {GROUP_A_LABEL}',
        marker=dict(color=COLOR_DN, size=6, opacity=0.8),
        text=df.loc[sig_dn, 'gene'],
        hovertemplate='%{text}<br>Log2FC=%{x:.2f}<br>-log10(FDR)=%{y:.1f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=df.loc[sig_up, 'log2fc'], y=df.loc[sig_up, 'neg_log10_fdr'],
        mode='markers', name=f'up in {GROUP_B_LABEL}',
        marker=dict(color=COLOR_UP, size=6, opacity=0.8),
        text=df.loc[sig_up, 'gene'],
        hovertemplate='%{text}<br>Log2FC=%{x:.2f}<br>-log10(FDR)=%{y:.1f}<extra></extra>',
    ))

    fdr_line = -np.log10(FDR_THRESH)
    x_range  = [df['log2fc'].min() - 0.3, df['log2fc'].max() + 0.3]
    y_max    = df['neg_log10_fdr'].max() * 1.05
    fig.add_shape(type='line', x0=x_range[0], x1=x_range[1],
                  y0=fdr_line, y1=fdr_line,
                  line=dict(color='black', width=1, dash='dash'))
    fig.add_shape(type='line', x0=-lfc_thresh, x1=-lfc_thresh,
                  y0=0, y1=y_max, line=dict(color='black', width=1, dash='dash'))
    fig.add_shape(type='line', x0=lfc_thresh, x1=lfc_thresh,
                  y0=0, y1=y_max, line=dict(color='black', width=1, dash='dash'))

    sig_df = df[sig_up | sig_dn]
    for _, row in sig_df.nsmallest(N_LABEL, 'fdr').iterrows():
        fig.add_annotation(
            x=row['log2fc'], y=row['neg_log10_fdr'],
            text=row['gene'], showarrow=False,
            font=dict(size=10), xshift=8, yshift=4,
        )

    n_up = sig_up.sum()
    n_dn = sig_dn.sum()
    lfc_label = f'{lfc_thresh:.3f}' if lfc_thresh != round(lfc_thresh, 1) else f'{lfc_thresh:.1f}'
    fig.update_layout(
        title=(f'{GROUP_A_LABEL} vs {GROUP_B_LABEL} — P56 mouse VIS cortex (Gao 2025)<br>'
               f'<sup>|Log2FC| > {lfc_label}  |  up in {GROUP_B_LABEL}: {n_up}  |  '
               f'up in {GROUP_A_LABEL}: {n_dn}  (FDR < {FDR_THRESH})</sup>'),
        xaxis_title=f'Log2 fold change ({GROUP_B_LABEL} / {GROUP_A_LABEL})',
        yaxis_title='-log10(FDR)',
        width=800, height=650,
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(out_path)
    print(f'Saved {out_path}')


make_volcano(df, lfc_thresh=1.0, out_path=OUT_LFC1)
make_volcano(df, lfc_thresh=0.5, out_path=OUT_LFC05)
