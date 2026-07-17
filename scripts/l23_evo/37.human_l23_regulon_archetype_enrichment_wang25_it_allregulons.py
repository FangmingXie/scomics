"""Fisher-exact enrichment of archetype markers in SCENIC+ regulons (human L2/3),
over the FULL expressed & non-uniform gene universe.

Same method and inputs as scripts/l23_evo/36.human_l23_regulon_archetype_enrichment_mtg5_allregulons.py,
but the regulon targets come from the wang25 IT eRegulon table instead of the mtg5 table.
Two heatmap HTMLs are written from the same enrichment table:
  - *_allregulons.html : EVERY regulon with >= MIN_REGULON_GENES in-universe targets (script 36 style)
  - *_significant.html : only regulons with FDR < STAR_FDR in >= 1 archetype (script 35 style)
In both, cells are starred where FDR < STAR_FDR.

  - archetype markers: 29.human_archetype_markers_allgenes.tsv (script 29)
  - gene universe    : 29.human_gene_universe.tsv (script 29, ~20.4k expressed/non-uniform genes)
  - regulon targets  : links/l23_evo/regulon_gene_table_wang25_it.csv

The wang25 IT table is already long-format (one row per regulon-gene pair; tab-separated
despite the .csv extension), so no Excel parsing (script 27) is needed. The 4 columns this
analysis consumes (regulon, TF, regulation_direction, Gene) are derived inline:
  regulation_direction = TF2G_sign/R2G_sign ; regulon = TF_direction.
NOTE: the IT table contains both +/+ (activating) and -/+ (repressing) regulons, so both
heatmap panels render.

Per (archetype a, regulon r) over universe U (|U| = N):
  x = |M_a & T_r|, M = |M_a|, T = |T_r|; 2x2 = [[x, M-x], [T-x, N-M-T+x]]
  p   = fisher_exact(2x2, alternative='greater')
  OR  = Haldane-Anscombe corrected odds ratio (+0.5 per cell); log2_or = log2(OR)
BH-FDR is applied across all (archetype, regulon) pairs.

Reads:
  local_data/res/l23_evo/29.human_archetype_markers_allgenes.tsv
  local_data/res/l23_evo/29.human_gene_universe.tsv
  links/l23_evo/regulon_gene_table_wang25_it.csv
Outputs:
  local_data/res/l23_evo/37.human_l23_regulon_archetype_enrichment_wang25_it_allregulons.tsv  (long format)
  local_data/res/l23_evo/37.human_l23_enrichment_wang25_it_allregulons_neglog10fdr.tsv          (matrix)
  local_data/res/l23_evo/37.human_l23_enrichment_wang25_it_allregulons_log2or.tsv               (matrix)
  local_data/fig/l23_evo/37.human_l23_regulon_archetype_enrichment_wang25_it_allregulons.html   (heatmap, all regulons)
  local_data/fig/l23_evo/37.human_l23_regulon_archetype_enrichment_wang25_it_significant.html   (heatmap, FDR<STAR_FDR only)
"""

import os

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import _write_fig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')

MARKERS = os.path.join(RES_DIR, '29.human_archetype_markers_allgenes.tsv')
UNIVERSE = os.path.join(RES_DIR, '29.human_gene_universe.tsv')  # expressed & non-uniform genes
REGULONS = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'regulon_gene_table_wang25_it.csv')

NOC = 4                        # human L2/3 archetypes (script 25/29)
# display relabel (mirrors mouse L2/3 reversal): archetype_1..4 -> D',C',B',A'.
# Columns then sort to A',B',C',D' in pivots/heatmaps.
ARCH_RELABEL = {'archetype_1': "D'", 'archetype_2': "C'", 'archetype_3': "B'", 'archetype_4': "A'"}
KEEP_DIRECTIONS = {'+/+', '-/+'}  # keep only regulons with a positive R2G (second) sign
MIN_REGULON_GENES = 5    # drop regulons with fewer than this many targets in-universe
STAR_FDR = 0.05          # heatmap cells marked '*' where FDR < this; also the row-keep
                         # threshold for the significant-only heatmap (FDR < this in >=1 archetype)
COLOR_PCTILE = (5, 95)   # log2 OR colorbar range = these percentiles of shown cells (white pinned at 0)

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def log2_odds_ratio(x, M, T, N):
    """Haldane-Anscombe corrected log2 odds ratio for the 2x2 [[x,M-x],[T-x,N-M-T+x]]."""
    a, b, c, d = x + 0.5, (M - x) + 0.5, (T - x) + 0.5, (N - M - T + x) + 0.5
    return float(np.log2((a * d) / (b * c)))


def zero_anchored_colorscale(vmin, vmax, base='RdBu_r', n=33):
    """Diverging colorscale spanning [vmin, vmax] with the neutral midpoint pinned to
    data value 0 (so an asymmetric percentile range keeps white at 0, not at the center).
    Value v maps to base-scale position: 0.5 at v=0, ->0 toward vmin(<0), ->1 toward vmax(>0).
    """
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


def draw_heatmap(long, N, significant_only, out_html, mode_label):
    """Two-panel (+/+, -/+) log2-OR heatmap, '*' where FDR < STAR_FDR.

    significant_only=False -> every regulon in each sign is drawn (script 36 style).
    significant_only=True  -> only regulons with FDR < STAR_FDR in >= 1 archetype
                              within that sign are drawn (script 35 style).
    """
    panels = [('+/+', 'activating'), ('-/+', 'repressing')]

    panel_show = []
    for sign, _lbl in panels:
        sub = long[long['regulation_direction'] == sign]
        if len(sub) == 0:
            panel_show.append(None)
            continue
        orm = sub.pivot(index='regulon', columns='arch_letter', values='log2_or')
        fdm = sub.pivot(index='regulon', columns='arch_letter', values='fdr')
        if significant_only:
            keep = orm.index[(fdm < STAR_FDR).any(axis=1)]
            if len(keep) == 0:
                panel_show.append(None)
                continue
            orm = orm.loc[keep]
            fdm = fdm.loc[keep]
        # group rows by dominant archetype (peak log2 OR): A-block first, then B, C, ...
        # within a block, sort by descending peak log2 OR
        letter_rank = {c: i for i, c in enumerate(sorted(orm.columns))}
        ordering = pd.DataFrame({
            'prank': orm.idxmax(axis=1).map(letter_rank),
            'peak': orm.max(axis=1),
        })
        order = ordering.sort_values(['prank', 'peak'], ascending=[True, False]).index
        panel_show.append(dict(
            sign=sign, order=order,
            log2or=orm.loc[order],
            fdr=fdm.loc[order],
            overlap=sub.pivot(index='regulon', columns='arch_letter', values='overlap').loc[order],
            ntarg=sub.pivot(index='regulon', columns='arch_letter', values='n_targets').loc[order],
        ))
        crit = f'FDR<{STAR_FDR:g} in >=1 archetype' if significant_only else 'all'
        print(f'    [{mode_label}] {sign}: {len(order)} regulons ({crit}) shown')

    if all(p is None for p in panel_show):
        print(f'    [{mode_label}] [skip heatmap] no regulons')
        return

    max_rows = max(len(p['order']) for p in panel_show if p is not None)

    # colorbar range = 5th/95th percentile of all shown log2 OR cells (both panels),
    # with a zero-anchored diverging colorscale so the midpoint stays at 0
    shown_vals = np.concatenate([p['log2or'].values.ravel() for p in panel_show if p is not None])
    shown_vals = shown_vals[np.isfinite(shown_vals)]
    cmin, cmax = np.nanpercentile(shown_vals, COLOR_PCTILE)
    colorscale0 = zero_anchored_colorscale(cmin, cmax)
    print(f'    [{mode_label}] colorbar range (p{COLOR_PCTILE[0]}-p{COLOR_PCTILE[1]}): '
          f'[{cmin:.2f}, {cmax:.2f}] (0 = white)')

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
            colorscale=colorscale0, cmin=cmin, cmax=cmax,
            colorbar=dict(title=f'log2 OR<br>({p["sign"]})', x=cbar_x[col - 1],
                          xanchor='left', len=0.9, thickness=14),
        )

    fig.update_layout(
        title=f'Human L2/3 IT — archetype marker enrichment in regulons (wang25 IT, {mode_label}) '
              f'(full universe; log2 OR; * FDR<{STAR_FDR:g}; N={N} genes)',
        height=max(400, 18 * max_rows + 180), width=1000,
        **layout_caxes,
    )
    _write_fig(fig, out_html)  # HTML whose screenshot button exports SVG by default


def enrich():
    print('=== human L2/3 (full universe; wang25 IT regulons; all regulons shown) ===')

    # gene universe: all expressed & non-uniform genes the marker Wilcoxon ran over (script 29)
    universe = set(pd.read_csv(UNIVERSE, sep='\t')['gene'])
    N = len(universe)

    # archetype marker sets (restricted to universe), ordered archetype_1..NOC
    markers = pd.read_csv(MARKERS, sep='\t')
    arch_labels = [f'archetype_{k + 1}' for k in range(NOC)]
    M_sets = {a: set(markers.loc[markers['archetype'] == a, 'gene']) & universe for a in arch_labels}

    # wang25 IT regulon targets (already long-format): derive the (regulon, TF,
    # regulation_direction, Gene) columns this analysis consumes.
    reg = pd.read_csv(REGULONS, sep='\t')
    reg['regulation_direction'] = reg['TF2G_sign'] + '/' + reg['R2G_sign']
    reg = reg[reg['regulation_direction'].isin(KEEP_DIRECTIONS)].copy()
    reg['regulon'] = reg['TF'] + '_' + reg['regulation_direction']
    reg = reg.drop_duplicates(subset=['regulon', 'Gene'])

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

    out_long = os.path.join(RES_DIR, '37.human_l23_regulon_archetype_enrichment_wang25_it_allregulons.tsv')
    long.to_csv(out_long, sep='\t', index=False)
    print(f'    wrote -> {out_long} ({len(long)} pairs)')

    fdr_mat = long.pivot(index='regulon', columns='arch_letter', values='neglog10_fdr')
    or_mat = long.pivot(index='regulon', columns='arch_letter', values='log2_or')
    fdr_mat.to_csv(os.path.join(RES_DIR, '37.human_l23_enrichment_wang25_it_allregulons_neglog10fdr.tsv'), sep='\t')
    or_mat.to_csv(os.path.join(RES_DIR, '37.human_l23_enrichment_wang25_it_allregulons_log2or.tsv'), sep='\t')

    # two heatmaps from the same enrichment: every regulon, then significant-only
    draw_heatmap(
        long, N, significant_only=False, mode_label='all regulons',
        out_html=os.path.join(FIG_DIR, '37.human_l23_regulon_archetype_enrichment_wang25_it_allregulons.html'),
    )
    draw_heatmap(
        long, N, significant_only=True, mode_label='significant regulons',
        out_html=os.path.join(FIG_DIR, '37.human_l23_regulon_archetype_enrichment_wang25_it_significant.html'),
    )
    return long


def main():
    enrich()


if __name__ == '__main__':
    main()
