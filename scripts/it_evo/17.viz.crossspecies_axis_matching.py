"""Visualize cross-species IT axis matching (reads script 16's outputs).

Per diagonal token (L23/L4/L5IT/L6IT) writes one HTML with four panels:
  1. Canonical spectrum — observed r per component (bars) with the permutation null band and
     the component-wise blocked-CV value overlaid, and the per-component z printed above each
     bar. The gap between the observed bar and the CV marker is the generalization story; the
     fact that the tallest bar is often not the highest-z bar is the other.
  2. Σcos²θ subspace overlap — observed vs its permutation null band, plus blocked and
     random-fold CV *excess* (held-out Σcos²θ minus its permuted-pairing baseline). The
     random-vs-blocked gap is the co-expression leakage diagnostic.
  3. Pairwise Pearson baseline (|r|) of the raw VX loading columns — the component-by-
     component matching CCA improves on by allowing mixtures.
  4. Leading canonical pair gene loadings (mouse vs human score), top genes labelled;
     greyed with a "weights unstable" annotation when the bootstrap gate (step 3) fails.

Plus one specificity HTML: six 4×4 heatmaps (Σcos²θ frac, its z, blocked-CV excess, BH q,
CCA1 z, CCA1) with the diagonal outlined. The Σcos²θ panels lead.

Reads (local_data/res/it_evo/):
  16.<TOKEN>_axis_cca_spectrum.tsv
  16.<TOKEN>_axis_cca_weights_human.tsv
  16.<TOKEN>_axis_pairwise_corr.tsv
  16.<TOKEN>_axis_cca_top_genes.tsv
  16.crossspecies_axis_specificity_{cos2,cos2z,cos2cv,cca1z,cca1,cos2q}.tsv
Outputs (local_data/fig/it_evo/):
  17.<TOKEN>_axis_matching.html
  17.crossspecies_axis_specificity.html
"""

import os
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')

TOKENS      = ['L23', 'L4', 'L5IT', 'L6IT']
GRID_ORDER  = ['L23', 'L4', 'L5IT', 'L6IT']
OBS_COLOR   = '#1f77b4'
NULL_COLOR  = '#999999'
CV_COLOR    = '#ff7f0e'
RND_COLOR   = '#d62728'

os.makedirs(FIG_DIR, exist_ok=True)


def parse_subspace_cv(folds_str):
    """Pull blk_raw/blk_null/rnd_raw/rnd_null/p_emp out of the subspace row's r_cv_folds."""
    d = {}
    for kv in str(folds_str).split(';'):
        if '=' in kv:
            k, v = kv.split('=')
            d[k] = float(v)
    return d


def token_figure(token):
    spec = pd.read_csv(os.path.join(RES_DIR, f'16.{token}_axis_cca_spectrum.tsv'), sep='\t')
    comp = spec[spec['component'] != 'subspace'].reset_index(drop=True)
    sub = spec[spec['component'] == 'subspace'].iloc[0]
    weights = pd.read_csv(os.path.join(RES_DIR, f'16.{token}_axis_cca_weights_human.tsv'), sep='\t')
    pairwise = pd.read_csv(os.path.join(RES_DIR, f'16.{token}_axis_pairwise_corr.tsv'), sep='\t', index_col=0)
    top = pd.read_csv(os.path.join(RES_DIR, f'16.{token}_axis_cca_top_genes.tsv'), sep='\t')
    sub_cv = parse_subspace_cv(sub['r_cv_folds'])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{token}: canonical spectrum (z above bars)',
            f'{token}: Σcos²θ subspace overlap',
            f'{token}: pairwise |r| baseline (max {pairwise.abs().values.max():.2f})',
            f'{token}: leading canonical pair gene loadings'],
        horizontal_spacing=0.13, vertical_spacing=0.14,
        specs=[[{'type': 'xy'}, {'type': 'xy'}], [{'type': 'heatmap'}, {'type': 'xy'}]])

    # --- panel 1: spectrum ---
    comp_names = comp['component'].values
    fig.add_trace(go.Bar(
        x=comp_names, y=comp['r'], name='observed r', marker_color=OBS_COLOR,
        showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=comp_names, y=comp['null_mean'], name='perm null (±sd)', mode='markers',
        marker=dict(color=NULL_COLOR, symbol='line-ew-open', size=10),
        error_y=dict(type='data', array=comp['null_sd'], color=NULL_COLOR),
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=comp_names, y=comp['r_cv_blocked'], name='blocked-CV r (signed)', mode='markers',
        marker=dict(color=CV_COLOR, symbol='diamond', size=9)), row=1, col=1)
    for i, row in comp.iterrows():
        fig.add_annotation(x=row['component'], y=max(row['r'], row['null_mean']) + 0.04,
                           text=f"z={row['z']:.1f}", showarrow=False, font=dict(size=10),
                           row=1, col=1)
    fig.update_yaxes(title_text='canonical correlation', range=[min(0, comp['r_cv_blocked'].min()) - 0.05, 1.0],
                     row=1, col=1)

    # --- panel 2: Σcos²θ ---
    labels = ['observed', 'perm null', 'blocked-CV excess', 'random-CV excess']
    vals = [sub['r'], sub['null_mean'], sub['r_cv_blocked'], sub['r_cv_random']]
    errs = [0, sub['null_sd'], 0, 0]
    colors = [OBS_COLOR, NULL_COLOR, CV_COLOR, RND_COLOR]
    fig.add_trace(go.Bar(
        x=labels, y=vals, marker_color=colors, showlegend=False,
        error_y=dict(type='data', array=errs, color=NULL_COLOR)), row=1, col=2)
    fig.add_annotation(x='observed', y=sub['r'] + 0.05, text=f"z={sub['z']:.1f}",
                       showarrow=False, font=dict(size=10), row=1, col=2)
    fig.add_annotation(
        x='blocked-CV excess', y=sub['r_cv_blocked'] + 0.05,
        text=f"raw {sub_cv.get('blk_raw', np.nan):.2f}<br>−null {sub_cv.get('blk_null', np.nan):.2f}",
        showarrow=False, font=dict(size=8), row=1, col=2)
    fig.update_yaxes(title_text='Σcos²θ', row=1, col=2)

    # --- panel 3: pairwise heatmap ---
    fig.add_trace(go.Heatmap(
        z=pairwise.abs().values, x=list(pairwise.columns), y=list(pairwise.index),
        colorscale='Blues', zmin=0, zmax=1,
        text=[[f'{v:.2f}' for v in r] for r in pairwise.abs().values],
        texttemplate='%{text}', textfont=dict(size=9),
        colorbar=dict(title='|r|', x=0.42, y=0.21, len=0.42), showscale=True), row=2, col=1)
    fig.update_xaxes(title_text='mouse VX', row=2, col=1)
    fig.update_yaxes(title_text='human VX', autorange='reversed', row=2, col=1)

    # --- panel 4: leading canonical pair gene scatter ---
    stable1 = bool(weights.iloc[0]['stable']) if len(weights) else False
    cca1_top = top[top['component'] == 'CCA1'] if len(top) else top
    if stable1 and len(cca1_top):
        fig.add_trace(go.Scatter(
            x=cca1_top['mouse_score'], y=cca1_top['human_score'], mode='markers+text',
            marker=dict(color=OBS_COLOR, size=8), text=cca1_top['human_gene'],
            textposition='top center', textfont=dict(size=8), showlegend=False,
            hovertemplate='%{text}<br>mouse=%{x:.2f} human=%{y:.2f}<extra></extra>'),
            row=2, col=2)
    else:
        # greyed placeholder with the top genes if present, plus an instability note
        if len(cca1_top):
            fig.add_trace(go.Scatter(
                x=cca1_top['mouse_score'], y=cca1_top['human_score'], mode='markers',
                marker=dict(color='#cccccc', size=7), showlegend=False), row=2, col=2)
        med = weights.iloc[0]['boot_cos_median'] if len(weights) else np.nan
        fig.add_annotation(x=0.5, y=0.5, xref='x domain', yref='y domain',
                           text=f'weights unstable<br>(bootstrap median |cos| = {med:.2f} < 0.9)',
                           showarrow=False, font=dict(size=11, color='#d62728'), row=2, col=2)
    fig.update_xaxes(title_text='mouse canonical-1 gene score', row=2, col=2)
    fig.update_yaxes(title_text='human canonical-1 gene score', row=2, col=2)

    fig.update_layout(height=850, width=1250, barmode='group',
                      title=f'{token}: cross-species IT axis matching',
                      legend=dict(orientation='h', y=1.08, x=0.0))
    out = os.path.join(FIG_DIR, f'17.{token}_axis_matching.html')
    fig.write_html(out)
    print(f'saved {out}')


def specificity_figure():
    panels = [
        ('cos2',  'Σcos²θ (frac of min k)', 'Blues', None),
        ('cos2cv', 'blocked-CV Σcos²θ excess', 'Blues', None),
        ('cos2z', 'z of Σcos²θ', 'Blues', None),
        ('cos2q', 'BH q of Σcos²θ', 'Blues_r', None),
        ('cca1z', 'z of CCA1', 'Blues', None),
        ('cca1',  'CCA1 (raw)', 'Blues', None),
    ]
    mats = {name: pd.read_csv(
        os.path.join(RES_DIR, f'16.crossspecies_axis_specificity_{name}.tsv'), sep='\t', index_col=0)
        for name, *_ in panels}

    fig = make_subplots(rows=2, cols=3, subplot_titles=[p[1] for p in panels],
                        horizontal_spacing=0.09, vertical_spacing=0.14)
    xlab = [f'M {t}' for t in GRID_ORDER]
    ylab = [f'H {t}' for t in GRID_ORDER]
    for k, (name, title, cscale, _) in enumerate(panels):
        r, c = divmod(k, 3)
        df = mats[name].loc[[f'human_{t}' for t in GRID_ORDER], [f'mouse_{t}' for t in GRID_ORDER]]
        z = df.values
        fig.add_trace(go.Heatmap(
            z=z, x=xlab, y=ylab, colorscale=cscale,
            text=[[f'{v:.3g}' for v in row] for row in z], texttemplate='%{text}',
            textfont=dict(size=9),
            colorbar=dict(len=0.42, x=0.30 + 0.365 * c, y=0.79 - 0.58 * r, thickness=10)),
            row=r + 1, col=c + 1)
        fig.update_yaxes(autorange='reversed', row=r + 1, col=c + 1)
        # outline the diagonal cells
        for d in range(len(GRID_ORDER)):
            fig.add_shape(type='rect', x0=d - 0.5, x1=d + 0.5, y0=d - 0.5, y1=d + 0.5,
                          line=dict(color='black', width=2), row=r + 1, col=c + 1)

    fig.update_layout(height=780, width=1350,
                      title='Cross-species IT axis specificity grid (human rows × mouse cols; diagonal outlined)')
    out = os.path.join(FIG_DIR, '17.crossspecies_axis_specificity.html')
    fig.write_html(out)
    print(f'saved {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='*', default=None, help='subset of tokens (default: all four)')
    parser.add_argument('--no-grid', action='store_true', help='skip the specificity grid figure')
    args = parser.parse_args()
    for token in (args.tokens or TOKENS):
        token_figure(token)
    if not args.no_grid:
        specificity_figure()
    print('Done.')


if __name__ == '__main__':
    main()
