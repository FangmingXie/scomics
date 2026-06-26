"""AP-1 regulon × archetype enrichment heatmap (human L2/3) — ALL AP-1 regulons shown.

Focused companion to scripts/l23_evo/30.human_l23_regulon_archetype_enrichment_allgenes.py.
Script 30's heatmap only shows regulons passing log2 OR>2 AND FDR<1e-5, which drops the
small, high-OR/low-power classic AP-1 regulons (Jun/Fos/ATF/BATF/JDP). This script reads
the same precomputed enrichment table and plots EVERY AP-1 regulon regardless of
significance; '*' still marks FDR<STAR_FDR cells so significance remains visible.

"AP-1" here = the classic bZIP AP-1 dimer components (Jun/Fos/ATF/BATF/JDP). The bZIP
relatives BACH1/2, MAF/MAFB/MAFG/MAFK and NFE2L2 (which bind overlapping MARE/TRE-like
motifs) are intentionally excluded.

Reads:
  local_data/res/l23_evo/30.human_l23_regulon_archetype_enrichment_allgenes.tsv
Outputs:
  local_data/res/l23_evo/31.ap1_regulon_enrichment.tsv            (AP-1 subset, long format)
  local_data/fig/l23_evo/31.ap1_regulon_archetype_heatmap.html    (heatmap, all AP-1 regulons)
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import _write_fig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')

IN_LONG = os.path.join(RES_DIR, '30.human_l23_regulon_archetype_enrichment_allgenes.tsv')
OUT_TSV = os.path.join(RES_DIR, '31.ap1_regulon_enrichment.tsv')
OUT_HTML = os.path.join(FIG_DIR, '31.ap1_regulon_archetype_heatmap.html')

# classic AP-1 dimer components (Jun / Fos / ATF / BATF / JDP); BACH/MAF/NFE2L2 excluded
AP1_TFS = [
    'JUN', 'JUNB', 'JUND',
    'FOS', 'FOSB', 'FOSL1', 'FOSL2',
    'ATF2', 'ATF3', 'ATF4', 'ATF5', 'ATF7',
    'BATF', 'BATF2', 'BATF3',
    'JDP2',
]
STAR_FDR = 1e-5          # heatmap cells marked '*' where FDR < this
COLOR_PCTILE = (5, 95)   # colorbar range = these percentiles of shown cells (white pinned at 0)

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def zero_anchored_colorscale(vmin, vmax, base='RdBu_r', n=33):
    """Diverging colorscale spanning [vmin, vmax] with the neutral midpoint pinned to
    data value 0 (so an asymmetric percentile range keeps white at 0, not at the center)."""
    neg = (0 - vmin) if vmin < 0 else None
    pos = (vmax - 0) if vmax > 0 else None
    cs = []
    for u in np.linspace(0, 1, n):
        v = vmin + u * (vmax - vmin)
        if v <= 0 and neg:
            t = 0.5 * (v - vmin) / neg
        elif v >= 0 and pos:
            t = 0.5 + 0.5 * v / pos
        else:
            t = 0.5
        cs.append([float(u), pc.sample_colorscale(base, [min(max(t, 0.0), 1.0)])[0]])
    return cs


def main():
    long = pd.read_csv(IN_LONG, sep='\t')
    N = int(long['universe'].iloc[0])
    ap1 = long[long['TF'].isin(AP1_TFS)].copy()
    present = sorted(ap1['TF'].unique())
    missing = [tf for tf in AP1_TFS if tf not in present]
    print(f'AP-1 TFs present: {present}')
    print(f'AP-1 TFs absent from regulon table: {missing}')
    print(f'AP-1 regulons: {ap1["regulon"].nunique()} ({len(ap1)} archetype pairs)')
    ap1.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'wrote -> {OUT_TSV}')

    # ---- two panels by sign; show ALL AP-1 regulons (no significance/OR filter) ----
    panels = [('+/+', 'activating'), ('-/+', 'repressing')]
    panel_show = []
    for sign, _lbl in panels:
        sub = ap1[ap1['regulation_direction'] == sign]
        if sub.empty:
            panel_show.append(None)
            continue
        orm = sub.pivot(index='regulon', columns='arch_letter', values='log2_or')
        fdm = sub.pivot(index='regulon', columns='arch_letter', values='fdr')
        # order rows by dominant archetype (peak log2 OR), then descending peak
        letter_rank = {c: i for i, c in enumerate(sorted(orm.columns))}
        ordering = pd.DataFrame({'prank': orm.idxmax(axis=1).map(letter_rank), 'peak': orm.max(axis=1)})
        order = ordering.sort_values(['prank', 'peak'], ascending=[True, False]).index
        panel_show.append(dict(
            sign=sign, order=order,
            log2or=orm.loc[order], fdr=fdm.loc[order],
            overlap=sub.pivot(index='regulon', columns='arch_letter', values='overlap').loc[order],
            ntarg=sub.pivot(index='regulon', columns='arch_letter', values='n_targets').loc[order],
        ))
        print(f'    {sign}: {len(order)} AP-1 regulons shown')

    if all(p is None for p in panel_show):
        print('    [skip heatmap] no AP-1 regulons found')
        return

    max_rows = max(len(p['order']) for p in panel_show if p is not None)

    # colorbar range = 5/95 percentile of shown AP-1 cells, zero-anchored colorscale
    shown_vals = np.concatenate([p['log2or'].values.ravel() for p in panel_show if p is not None])
    shown_vals = shown_vals[np.isfinite(shown_vals)]
    cmin, cmax = np.nanpercentile(shown_vals, COLOR_PCTILE)
    colorscale0 = zero_anchored_colorscale(cmin, cmax)
    print(f'    colorbar range (p{COLOR_PCTILE[0]}-p{COLOR_PCTILE[1]}): [{cmin:.2f}, {cmax:.2f}] (0 = white)')

    cax_names = ['coloraxis', 'coloraxis2']
    cbar_x = [0.46, 1.0]
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
            colorscale=colorscale0, cmin=cmin, cmax=cmax,
            colorbar=dict(title=f'log2 OR<br>({p["sign"]})', x=cbar_x[col - 1],
                          xanchor='left', len=0.9, thickness=14),
        )

    fig.update_layout(
        title=f'Human L2/3 IT — AP-1 regulon enrichment (all AP-1 regulons; '
              f'log2 OR; * FDR<{STAR_FDR:g}; N={N} genes)',
        height=max(400, 22 * max_rows + 180), width=1000,
        **layout_caxes,
    )
    _write_fig(fig, OUT_HTML)


if __name__ == '__main__':
    main()
