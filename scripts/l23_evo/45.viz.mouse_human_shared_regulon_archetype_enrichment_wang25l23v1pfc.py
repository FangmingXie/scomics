"""Side-by-side mouse vs human archetype-marker enrichment for SHARED (+/+) regulons.

Identical to scripts 43/44, but the HUMAN panel uses the wang25 L2/3 (V1+PFC) eRegulon
source: `links/l23_evo/regulon_gene_table_wang25_l23_v1pfc.tsv`, read from script 38's
enrichment output. This differs from the `wang25 IT` table (script 37 / script 43) and from
the Wang25 SuppTable13 table (script 27, used by scripts 28/30/42 and script 44).

Human enrichment is read from script 38's output (full expressed & non-uniform gene
universe, ~20.4k genes) so its background matches the mouse panel (script 41 reconstructs
the analogous expressed/nonzero-variance universe).

Two heatmaps (mouse left, human right), same layout as script 37, showing the SAME set of
activating (+/+) regulons shared between the mouse and human L2/3 enrichment analyses
(matched by TF name, case-insensitively: mouse `Meis2_+/+` <-> human `MEIS2_+/+`).

Cells are colored by log2 odds ratio and starred where FDR < STAR_FDR (each table carries
its own BH-FDR from its source script). Design choices for this comparison figure:
  - Rows: shared TFs, ALIGNED (same row order in both panels) and ordered by the MOUSE
    dominant archetype (peak log2 OR), then descending mouse peak, so rows line up across
    species for reading.
  - Row labels: species-native regulon ids (mouse `Meis2_+/+`, human `MEIS2_+/+`).
  - Color: INDEPENDENT per panel — each panel's zero-anchored diverging scale spans the
    COLOR_PCTILE percentiles of its own shown cells (white pinned at log2 OR = 0).

Two figures are written from the same shared set (script-37 style, all vs significant):
  - *_allregulons.html : all shared (+/+) regulons.
  - *_significant.html : only regulons clearing (log2 OR > LOG2OR_MIN AND FDR < STAR_FDR)
    in >= 1 archetype in BOTH species. Same mouse-driven row order.

Reads:
  local_data/res/it/41.L2_3_regulon_archetype_enrichment.tsv
  local_data/res/l23_evo/38.human_l23_regulon_archetype_enrichment_wang25_l23_v1pfc_allregulons.tsv
Outputs:
  local_data/fig/l23_evo/45.mouse_human_shared_regulon_archetype_enrichment_wang25l23v1pfc_allregulons.html
  local_data/fig/l23_evo/45.mouse_human_shared_regulon_archetype_enrichment_wang25l23v1pfc_significant.html
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import _write_fig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_IT_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
RES_L23_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')

IN_MOUSE = os.path.join(RES_IT_DIR, '41.L2_3_regulon_archetype_enrichment.tsv')
IN_HUMAN = os.path.join(RES_L23_DIR, '38.human_l23_regulon_archetype_enrichment_wang25_l23_v1pfc_allregulons.tsv')
OUT_HTML_ALL = os.path.join(FIG_DIR, '45.mouse_human_shared_regulon_archetype_enrichment_wang25l23v1pfc_allregulons.html')
OUT_HTML_SIG = os.path.join(FIG_DIR, '45.mouse_human_shared_regulon_archetype_enrichment_wang25l23v1pfc_significant.html')

# --- parameters ---
DIRECTION = '+/+'        # activating regulons only
STAR_FDR = 0.05          # heatmap cells marked '*' where FDR < this (each table's own BH-FDR)
LOG2OR_MIN = 2.0         # significant-in-both filter also requires log2 OR > this (in >=1 archetype)
COLOR_PCTILE = (5, 95)   # per-panel log2 OR colorbar range (percentiles of that panel's shown cells)
MOUSE_TITLE = 'Cheng22 mouse L2/3 IT (yoo25 regulons)'
HUMAN_TITLE = 'Jorstad23 human L2/3 IT (wang25 L2/3 V1+PFC regulons)'

os.makedirs(FIG_DIR, exist_ok=True)


def zero_anchored_colorscale(vmin, vmax, base='RdBu_r', n=33):
    """Diverging colorscale over [vmin, vmax] with the neutral midpoint pinned to 0
    (asymmetric percentile ranges keep white at log2 OR = 0, not at the center)."""
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


def load_direction(path, direction):
    """Load one enrichment long-table, keep one direction, index each TF's single regulon.

    Returns (sub, tf2regulon): sub is the filtered long-frame with an uppercase 'TF_U'
    column; tf2regulon maps TF_U -> the species-native regulon id (fail-fast if a TF has
    more than one regulon for this direction).
    """
    df = pd.read_csv(path, sep='\t')
    sub = df[df['regulation_direction'] == direction].copy()
    sub['TF_U'] = sub['TF'].str.upper()
    reg_per_tf = sub.groupby('TF_U')['regulon'].nunique()
    bad = reg_per_tf[reg_per_tf > 1]
    if len(bad):
        raise ValueError(f'{os.path.basename(path)}: TFs with >1 {direction} regulon: {list(bad.index)}')
    tf2regulon = sub.drop_duplicates('TF_U').set_index('TF_U')['regulon'].to_dict()
    return sub, tf2regulon


def panel_matrices(sub, tf_order):
    """Pivot one species' long-frame to TF_U x arch_letter matrices, row-ordered by tf_order."""
    def piv(val):
        return sub.pivot(index='TF_U', columns='arch_letter', values=val).reindex(tf_order)
    cols = sorted(sub['arch_letter'].unique())
    return {v: piv(v)[cols] for v in ('log2_or', 'fdr', 'overlap', 'n_targets')}


def add_panel(fig, col, mats, ylabels, title, cax):
    """Add one species' heatmap (log2 OR, '*' where FDR<STAR_FDR) with its own color axis."""
    orm, fdm = mats['log2_or'], mats['fdr']
    stars = np.where(fdm.values < STAR_FDR, '*', '')
    customdata = np.dstack([mats['overlap'].values, mats['n_targets'].values, fdm.values])
    fig.add_trace(go.Heatmap(
        z=orm.values, x=list(orm.columns), y=ylabels,
        coloraxis=cax, text=stars, texttemplate='%{text}',
        textfont=dict(size=14, color='black'),
        customdata=customdata,
        hovertemplate=('regulon=%{y}<br>archetype=%{x}<br>log2 OR=%{z:.2f}'
                       '<br>overlap=%{customdata[0]}<br>n_targets=%{customdata[1]}'
                       '<br>FDR=%{customdata[2]:.2e}<extra></extra>'),
    ), row=1, col=col)
    fig.update_yaxes(autorange='reversed', row=1, col=col)
    fig.update_xaxes(title_text='archetype', row=1, col=col)

    shown = orm.values.ravel()
    shown = shown[np.isfinite(shown)]
    cmin, cmax = np.nanpercentile(shown, COLOR_PCTILE)
    print(f'    {title}: colorbar p{COLOR_PCTILE[0]}-p{COLOR_PCTILE[1]} = [{cmin:.2f}, {cmax:.2f}] (0=white)')
    return {cax: dict(
        colorscale=zero_anchored_colorscale(cmin, cmax), cmin=cmin, cmax=cmax,
        colorbar=dict(title='log2 OR', x=CBAR_X[col - 1], xanchor='left', len=0.9, thickness=14),
    )}


CBAR_X = [0.46, 1.0]  # colorbar x positions (subplot1 gap, subplot2 right edge)


def significant_tfs(sub):
    """TFs with (log2 OR > LOG2OR_MIN AND FDR < STAR_FDR) in >= 1 archetype (this species)."""
    orm = sub.pivot(index='TF_U', columns='arch_letter', values='log2_or')
    fdm = sub.pivot(index='TF_U', columns='arch_letter', values='fdr')
    return set(fdm.index[((orm > LOG2OR_MIN) & (fdm < STAR_FDR)).any(axis=1)])


def build_figure(mouse_sub, human_sub, mouse_reg, human_reg, tf_order, out_html, mode_label):
    """Two-panel (mouse, human) log2 OR heatmap over tf_order rows; '*' where FDR<STAR_FDR."""
    if not tf_order:
        print(f'    [{mode_label}] no regulons to show — skipping {os.path.basename(out_html)}')
        return
    mouse_mats = panel_matrices(mouse_sub, tf_order)
    human_mats = panel_matrices(human_sub, tf_order)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.24,
        subplot_titles=[f'{MOUSE_TITLE}  n={len(tf_order)}',
                        f'{HUMAN_TITLE}  n={len(tf_order)}'],
    )
    layout_caxes = {}
    layout_caxes.update(add_panel(fig, 1, mouse_mats,
                                  [mouse_reg[tf] for tf in tf_order], MOUSE_TITLE, 'coloraxis'))
    layout_caxes.update(add_panel(fig, 2, human_mats,
                                  [human_reg[tf] for tf in tf_order], HUMAN_TITLE, 'coloraxis2'))

    fig.update_layout(
        title=f'Shared activating ({DIRECTION}) regulons ({mode_label}) — archetype marker '
              f'enrichment, mouse vs human L2/3 IT '
              f'(log2 OR; * FDR<{STAR_FDR:g}; rows aligned by mouse peak)',
        height=max(400, 18 * len(tf_order) + 180), width=1000,
        **layout_caxes,
    )
    _write_fig(fig, out_html)


def main():
    mouse_sub, mouse_reg = load_direction(IN_MOUSE, DIRECTION)
    human_sub, human_reg = load_direction(IN_HUMAN, DIRECTION)

    shared = sorted(set(mouse_reg) & set(human_reg))
    print(f'Shared {DIRECTION} TFs present in both enrichment tables: {len(shared)}')
    print(f'  {shared}')

    mouse_sub = mouse_sub[mouse_sub['TF_U'].isin(shared)]
    human_sub = human_sub[human_sub['TF_U'].isin(shared)]

    # row order: MOUSE dominant archetype (peak log2 OR), then descending mouse peak
    mouse_or = mouse_sub.pivot(index='TF_U', columns='arch_letter', values='log2_or')
    letter_rank = {c: i for i, c in enumerate(sorted(mouse_or.columns))}
    ordering = pd.DataFrame({
        'prank': mouse_or.idxmax(axis=1).map(letter_rank),
        'peak': mouse_or.max(axis=1),
    })
    tf_order = list(ordering.sort_values(['prank', 'peak'], ascending=[True, False]).index)

    # rows significant (FDR<STAR_FDR in >=1 archetype) in BOTH species, mouse-driven order
    sig_both = significant_tfs(mouse_sub) & significant_tfs(human_sub)
    tf_order_sig = [tf for tf in tf_order if tf in sig_both]
    print(f'Significant (log2 OR>{LOG2OR_MIN:g} & FDR<{STAR_FDR:g}) in both species: '
          f'{len(tf_order_sig)} of {len(tf_order)}')
    print(f'  {tf_order_sig}')

    build_figure(mouse_sub, human_sub, mouse_reg, human_reg,
                 tf_order, OUT_HTML_ALL, 'all shared')
    build_figure(mouse_sub, human_sub, mouse_reg, human_reg,
                 tf_order_sig, OUT_HTML_SIG, 'significant in both')
    print('Done.')


if __name__ == '__main__':
    main()
