"""Fisher-exact enrichment of archetype markers in SCENIC+ regulons (human L2/3).

Human L2/3 analog of the mouse IT script scripts/it/41.regulon_archetype_enrichment.py.
For the single human L2/3 IT population (jorstad23, NOC=4 archetypes A-D), test whether
each archetype's marker gene set is over-represented among each Wang25 regulon's target
genes (one-sided Fisher exact), then render activating/repressing log2-OR heatmaps.

Unlike the mouse version there is a single population (not 4 layers) and a single
dataset, so the gene universe is NOT reconstructed from h5ad: the human marker Wilcoxon
ran over the 2000 HVGs, so the universe is simply the index of 01.pca_loadings.tsv.

Two gene-set resources are combined (see report/l23_evo/files.md):
  - archetype markers: local_data/res/l23_evo/25.human_archetype_markers.tsv (NOC=4)
  - regulon targets  : local_data/res/l23_evo/27.human_wang25_regulon_targets.tsv

Background universe: the tested gene set the marker Wilcoxon ran over (2000 HVGs), read
from the index of local_data/res/l23_evo/01.pca_loadings.tsv.

Per (archetype a, regulon r) over universe U (|U| = N):
  x = |M_a & T_r|, M = |M_a|, T = |T_r|; 2x2 = [[x, M-x], [T-x, N-M-T+x]]
  p   = fisher_exact(2x2, alternative='greater')
  OR  = Haldane-Anscombe corrected odds ratio (+0.5 per cell); log2_or = log2(OR)
BH-FDR is applied across all (archetype, regulon) pairs.

Reads:
  local_data/res/l23_evo/25.human_archetype_markers.tsv
  local_data/res/l23_evo/27.human_wang25_regulon_targets.tsv
  local_data/res/l23_evo/01.pca_loadings.tsv
Outputs:
  local_data/res/l23_evo/28.human_l23_regulon_archetype_enrichment.tsv   (long format)
  local_data/res/l23_evo/28.human_l23_enrichment_neglog10fdr.tsv         (matrix)
  local_data/res/l23_evo/28.human_l23_enrichment_log2or.tsv              (matrix)
  local_data/fig/l23_evo/28.human_l23_regulon_archetype_enrichment.html  (heatmap)
"""

import os

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import _write_fig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')

MARKERS = os.path.join(RES_DIR, '25.human_archetype_markers.tsv')
REGULONS = os.path.join(RES_DIR, '27.human_wang25_regulon_targets.tsv')
HVG_LOADINGS = os.path.join(RES_DIR, '01.pca_loadings.tsv')  # index = 2000 HVG universe

NOC = 4                        # human L2/3 archetypes (script 25)
# display relabel (mirrors mouse L2/3 reversal): archetype_1..4 -> D',C',B',A'.
# Columns then sort to A',B',C',D' in pivots/heatmaps.
ARCH_RELABEL = {'archetype_1': "D'", 'archetype_2': "C'", 'archetype_3': "B'", 'archetype_4': "A'"}
MIN_REGULON_GENES = 5    # drop regulons with fewer than this many targets in-universe
LOG2OR_SHOW = 2.0        # a regulon row is shown if log2 OR > this in >= 1 archetype
STAR_FDR = 1e-2          # heatmap cells marked '*' where FDR < this
COLOR_ABS = 5.0          # fixed log2 OR colorbar range [-COLOR_ABS, COLOR_ABS]

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def log2_odds_ratio(x, M, T, N):
    """Haldane-Anscombe corrected log2 odds ratio for the 2x2 [[x,M-x],[T-x,N-M-T+x]]."""
    a, b, c, d = x + 0.5, (M - x) + 0.5, (T - x) + 0.5, (N - M - T + x) + 0.5
    return float(np.log2((a * d) / (b * c)))


def enrich():
    print('=== human L2/3 ===')

    # gene universe: the 2000 HVGs the human marker Wilcoxon ran over (script 01 / 25)
    universe = set(pd.read_csv(HVG_LOADINGS, sep='\t', index_col=0).index)
    N = len(universe)

    # archetype marker sets (restricted to universe), ordered archetype_1..NOC
    markers = pd.read_csv(MARKERS, sep='\t')
    arch_labels = [f'archetype_{k + 1}' for k in range(NOC)]
    M_sets = {a: set(markers.loc[markers['archetype'] == a, 'gene']) & universe for a in arch_labels}

    # regulon target sets (restricted to universe), keep TF + sign metadata
    reg = pd.read_csv(REGULONS, sep='\t')
    reg_meta = reg.drop_duplicates('regulon').set_index('regulon')[['TF', 'regulation_direction']]
    T_sets = {r: set(g['Gene']) & universe for r, g in reg.groupby('regulon')}
    n_total = len(T_sets)
    T_sets = {r: t for r, t in T_sets.items() if len(t) >= MIN_REGULON_GENES}
    print(f'    regulons: {len(T_sets)} kept (>= {MIN_REGULON_GENES} in-universe targets) '
          f'of {n_total}; universe N={N}')

    rows = []
    for r, T in T_sets.items():
        for a in arch_labels:
            M = M_sets[a]
            x = len(M & T)
            table = [[x, len(M) - x], [len(T) - x, N - len(M) - len(T) + x]]
            _, pval = scipy.stats.fisher_exact(table, alternative='greater')
            rows.append(dict(
                layer='L2/3', archetype=a, regulon=r,
                TF=reg_meta.loc[r, 'TF'], regulation_direction=reg_meta.loc[r, 'regulation_direction'],
                overlap=x, n_markers=len(M), n_targets=len(T), universe=N,
                log2_or=log2_odds_ratio(x, len(M), len(T), N), pval=pval,
            ))

    long = pd.DataFrame(rows)
    long['fdr'] = multipletests(long['pval'].values, method='fdr_bh')[1]
    long['neglog10_fdr'] = -np.log10(long['fdr'].clip(lower=np.nextafter(0, 1)))

    # archetype display labels: archetype_1->D', ... archetype_4->A' (cols sort A'..D')
    long['arch_letter'] = long['archetype'].map(ARCH_RELABEL)

    out_long = os.path.join(RES_DIR, '28.human_l23_regulon_archetype_enrichment.tsv')
    long.to_csv(out_long, sep='\t', index=False)
    print(f'    wrote -> {out_long} ({len(long)} pairs)')

    fdr_mat = long.pivot(index='regulon', columns='arch_letter', values='neglog10_fdr')
    or_mat = long.pivot(index='regulon', columns='arch_letter', values='log2_or')
    fdr_mat.to_csv(os.path.join(RES_DIR, '28.human_l23_enrichment_neglog10fdr.tsv'), sep='\t')
    or_mat.to_csv(os.path.join(RES_DIR, '28.human_l23_enrichment_log2or.tsv'), sep='\t')

    # ---- two heatmaps (+/+ and -/+), colored by log2 OR, '*' where FDR < STAR_FDR ----
    panels = [('+/+', 'activating'), ('-/+', 'repressing')]

    # collect each panel's shown matrices (regulons with log2 OR > LOG2OR_SHOW AND
    # FDR < STAR_FDR in >= 1 archetype within that sign)
    panel_show = []
    for sign, _lbl in panels:
        sub = long[long['regulation_direction'] == sign]
        orm = sub.pivot(index='regulon', columns='arch_letter', values='log2_or')
        fdm = sub.pivot(index='regulon', columns='arch_letter', values='fdr')
        keep = orm.index[((orm > LOG2OR_SHOW) & (fdm < STAR_FDR)).any(axis=1)]
        if len(keep) == 0:
            panel_show.append(None)
            continue
        # group rows by dominant archetype (peak log2 OR): A-block first, then B, C, ...
        # within a block, sort by descending peak log2 OR
        sub_or = orm.loc[keep]
        letter_rank = {c: i for i, c in enumerate(sorted(sub_or.columns))}
        ordering = pd.DataFrame({
            'prank': sub_or.idxmax(axis=1).map(letter_rank),
            'peak': sub_or.max(axis=1),
        })
        order = ordering.sort_values(['prank', 'peak'], ascending=[True, False]).index
        panel_show.append(dict(
            sign=sign, order=order,
            log2or=orm.loc[order],
            fdr=fdm.loc[order],
            overlap=sub.pivot(index='regulon', columns='arch_letter', values='overlap').loc[order],
            ntarg=sub.pivot(index='regulon', columns='arch_letter', values='n_targets').loc[order],
        ))
        print(f'    {sign}: {len(keep)} regulons '
              f'(log2 OR>{LOG2OR_SHOW} AND FDR<{STAR_FDR:g} in >=1 archetype) shown')

    if all(p is None for p in panel_show):
        print('    [skip heatmap] no regulons meeting both criteria')
        return long

    max_rows = max(len(p['order']) for p in panel_show if p is not None)

    # one independent color axis per panel; place its colorbar beside its subplot
    cax_names = ['coloraxis', 'coloraxis2']
    cbar_x = [0.46, 1.0]  # colorbar x positions (subplot1 gap, subplot2 right edge)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.24,
        subplot_titles=[f'{lbl} ({sign})  n={len(p["order"]) if p else 0}'
                        for (sign, lbl), p in zip(panels, panel_show)],
    )
    layout_caxes = {}
    for col, p in enumerate(panel_show, start=1):
        if p is None:
            continue
        cax = cax_names[col - 1]
        stars = np.where(p['fdr'].values < STAR_FDR, '*', '')
        customdata = np.dstack([p['overlap'].values, p['ntarg'].values, p['fdr'].values])
        fig.add_trace(go.Heatmap(
            z=p['log2or'].values, x=list(p['log2or'].columns), y=list(p['log2or'].index),
            coloraxis=cax, text=stars, texttemplate='%{text}',
            textfont=dict(size=14, color='black'),
            customdata=customdata,
            hovertemplate=('regulon=%{y}<br>archetype=%{x}<br>log2 OR=%{z:.2f}'
                           '<br>overlap=%{customdata[0]}<br>n_targets=%{customdata[1]}'
                           '<br>FDR=%{customdata[2]:.2e}<extra></extra>'),
        ), row=1, col=col)
        fig.update_yaxes(autorange='reversed', row=1, col=col)
        fig.update_xaxes(title_text='archetype', row=1, col=col)
        layout_caxes[cax] = dict(
            colorscale='RdBu_r', cmid=0, cmin=-COLOR_ABS, cmax=COLOR_ABS,
            colorbar=dict(title=f'log2 OR<br>({p["sign"]})', x=cbar_x[col - 1],
                          xanchor='left', len=0.9, thickness=14),
        )

    fig.update_layout(
        title=f'Human L2/3 IT — archetype marker enrichment in regulons '
              f'(log2 OR; * FDR<{STAR_FDR:g}; N={N} genes)',
        height=max(400, 18 * max_rows + 180), width=1000,
        **layout_caxes,
    )
    out_html = os.path.join(FIG_DIR, '28.human_l23_regulon_archetype_enrichment.html')
    _write_fig(fig, out_html)  # HTML whose screenshot button exports SVG by default
    return long


def main():
    enrich()


if __name__ == '__main__':
    main()
