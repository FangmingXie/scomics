"""41c's figure, redrawn on 54's expression-stratified statistics.

41c selects and colours every enriched regulon by 41's Fisher log2 odds ratio. That statistic
carries two systematic inflations (see 54): the single-stratum null ignores that markers and
regulon targets are both biased toward highly expressed genes, and the Haldane-Anscombe odds
ratio overstates the enrichment ratio at small counts. On the cells this figure actually
colours (overlap >= MASK_MIN_OVERLAP) the median moves

    log2_or +3.47  ->  log2(obs/exp), single stratum +2.58  ->  log2_enr +0.94

This script keeps 41c's design exactly -- same columns, same two panels, same masking, same
row-selection logic, same drawing code where it can be shared -- and changes only the
statistic: colour is 54's `log2_enr` (log2 of observed overlap over the expression-matched
expectation) and significance is `fdr_strat` (BH over the exact stratified mid-p).

Nothing is recomputed here. Both panels are reshaped from the tables 54 wrote.

ONE VISUAL DEPARTURE FROM 41c, forced by the data: its colour ramp is sequential and floored
at zero, which 41b justifies by noting that after masking, every remaining cell is enriched.
That is no longer true -- 16% of this figure's coloured native cells and 20% of its L2/3-set
cells fall below zero (to -1.41), i.e. the regulon covers fewer of the archetype's markers
than an expression-matched set would. The ramp therefore extends to -1.5 with a blue tail
below zero; 41b's YlOrRd is kept unchanged above it, so enriched cells look as they do in 41c.

THRESHOLDS ARE NOT INHERITED FROM 41b. `log2_enr` lives on a different scale from `log2_or`
(max 3.46 vs 6.4 over the activating cells), so 41b's STAR_LOG2OR = 3.0 and its [0, 6.5] ramp
are meaningless here and are re-derived below. They are provisional: 54's plan leaves the
final cutoffs to be agreed rather than assumed, and this figure is the artefact that decision
should be made from.

Reads:
  local_data/res/it/54.<layer>_stratified_enrichment.tsv    (panel 1, native regulons)
  local_data/res/it/54.l23set_stratified_enrichment.tsv     (panel 2, L2/3 set everywhere)
Outputs:
  local_data/res/it/54b.enriched_regulon_selection.tsv       (chosen regulons + peak cell)
  local_data/res/it/54b.enriched_native_log2enr.tsv / _fdr.tsv
  local_data/res/it/54b.enriched_l23set_log2enr.tsv / _fdr.tsv
  local_data/fig/it/54b.stratified_enriched_regulon_heatmap.html
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)
from viz import _write_fig

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

SCRIPT_41B = os.path.join(SCRIPTS_DIR, 'it', '41b.selected_regulon_archetype_enrichment.py')
INPUT_NATIVE_TMPL = os.path.join(RES_DIR, '54.{layer}_stratified_enrichment.tsv')
INPUT_L23SET = os.path.join(RES_DIR, '54.l23set_stratified_enrichment.tsv')

OUT_SELECTION = os.path.join(RES_DIR, '54b.enriched_regulon_selection.tsv')
OUT_NATIVE = {'log2_enr': os.path.join(RES_DIR, '54b.enriched_native_log2enr.tsv'),
              'fdr_strat': os.path.join(RES_DIR, '54b.enriched_native_fdr.tsv')}
OUT_L23SET = {'log2_enr': os.path.join(RES_DIR, '54b.enriched_l23set_log2enr.tsv'),
              'fdr_strat': os.path.join(RES_DIR, '54b.enriched_l23set_fdr.tsv')}
OUT_HTML = os.path.join(FIG_DIR, '54b.stratified_enriched_regulon_heatmap.html')

# --- the only constants that differ from 41b, and why ------------------------------------
# BH-FDR over the stratified mid-p. Same 0.05 as 41b: the evidence bar is unchanged, only the
# null it is measured against.
STAR_FDR = 0.05
# log2_enr > 1 == "the regulon covers at least twice as many of this archetype's markers as an
# expression-matched gene set would". Round, interpretable, and on the plateau of the
# selection curve -- over the activating unmasked cells, thresholds of 0.5 / 1.0 / 1.5 / 2.0
# keep 63 / 60 / 48 / 29 cells, so 1.0 sits just past the shoulder where near-null cells stop
# entering and before real signal starts being cut.
STAR_LOG2ENR = 1.0
# The ramp stops at the observed maximum over activating unmasked cells (3.46) rather than
# 41b's 6.5, which was set for the inflated odds-ratio scale and would leave this washed out.
# It also has to reach BELOW zero, which 41b's does not. 41b says why it is purely sequential:
# "once thin cells are masked, every remaining cell is enriched (min log2 OR = +0.21), so a
# diverging scale would waste half its range on unused blue". Under the stratified statistic
# that premise fails -- 17 of the native panel's 107 coloured cells (16%) and 27 of the
# L2/3-set panel's 138 (20%) sit below zero, down to -1.41, meaning the regulon covers FEWER
# of the archetype's markers than an expression-matched gene set would. A zero-floored ramp
# would clamp all of them to the palest yellow and read as weak enrichment.
COLOR_MIN, COLOR_MAX = -1.5, 3.5
# mirrors 41b.MASK_MIN_OVERLAP; asserted against it in main() rather than imported at module
# scope so the star criterion is readable next to the other two cutoffs
MASK_MIN_OVERLAP = 5

# Everything that depends on WHICH regulon catalogue is being drawn lives here, so 55b reuses
# this module instead of copying it. The thresholds above do NOT vary: gao25's unmasked
# activating log2_enr spans -1.495..3.197, inside this ramp, and its selection curve is flat
# from 0.5 to 1.25 (58/58/57/55 cells), so STAR_LOG2ENR is nearly non-binding there. Keeping
# both identical is what lets the two families be compared by colour.
CFG_YOO25 = dict(
    tag='yoo25',
    label='mouse IT subclasses',
    native_tmpl=INPUT_NATIVE_TMPL,
    l23set=INPUT_L23SET,
    out_selection=OUT_SELECTION,
    out_native=OUT_NATIVE,
    out_l23set=OUT_L23SET,
    out_html=OUT_HTML,
)

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def build_colorscale():
    """41b's YlOrRd ramp above zero, a blue tail below it, with the break exactly at zero.

    Keeps 41b's positive half unchanged -- so cells that were enriched look as they do in 41c
    -- and adds only the negative range the stratified statistic actually occupies. The hard
    step at zero is deliberate: zero is "observed overlap equals the expression-matched
    expectation", the one value in this figure that means something on its own.
    """
    zero = (0.0 - COLOR_MIN) / (COLOR_MAX - COLOR_MIN)
    neg = sample_colorscale('Blues', np.linspace(0.62, 0.04, 12))
    pos = sample_colorscale('YlOrRd', np.linspace(0, 0.74, 21))
    return ([[zero * t, c] for t, c in zip(np.linspace(0, 1, 12), neg)]
            + [[zero + (1 - zero) * t, c] for t, c in zip(np.linspace(0, 1, 21), pos)])


def load_41b():
    """Import 41b for the shared visual language (columns, palette, masking, block rules)."""
    spec = importlib.util.spec_from_file_location('script41b', SCRIPT_41B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_native(primed, m41b, cfg=None):
    cfg = cfg or CFG_YOO25
    frames = []
    for layer, _token, _label in m41b.LAYER_TOKEN:
        path = cfg['native_tmpl'].format(layer=layer)
        assert os.path.exists(path), \
            f'missing {path}; run the {cfg["tag"]} enrichment script first'
        frames.append(pd.read_csv(path, sep='\t'))
    return m41b.to_col(pd.concat(frames, ignore_index=True), primed)


def starred(df):
    """41c's star criterion with the stratified statistic substituted in."""
    return ((df['fdr_strat'] < STAR_FDR)
            & (df['log2_enr'] > STAR_LOG2ENR)
            & (df['overlap'] >= MASK_MIN_OVERLAP))


def select_rows(native, l23set, cols, cfg):
    """Regulons enriched in >=1 archetype of >=1 subclass, grouped by their peak column."""
    nat_sig = native[starred(native)]
    l23_sig = l23set[starred(l23set)]
    keep = set(nat_sig['TF']) | set(l23_sig['TF'])
    print(f'  {native["TF"].nunique()} regulons tested in the native panel; '
          f'{len(keep)} enriched in >=1 (subclass, archetype) cell')
    print(f'    native hits {nat_sig["TF"].nunique()}, L2/3-set hits {l23_sig["TF"].nunique()}, '
          f'union {len(keep)}')
    assert keep, 'no regulon passes the star criterion'

    # peak = the strongest starred cell, preferring the native panel (it covers every
    # subclass); regulons starred only in the L2/3-set panel fall back to that panel
    peaks = []
    col_rank = {c: i for i, c in enumerate(cols)}
    for tf in keep:
        hits = nat_sig[nat_sig['TF'] == tf]
        panel = 'native'
        if hits.empty:
            hits, panel = l23_sig[l23_sig['TF'] == tf], 'l23set'
        top = hits.loc[hits['log2_enr'].idxmax()]
        peaks.append(dict(TF=tf, peak_panel=panel, peak_col=top['col'],
                          peak_log2_enr=top['log2_enr'], peak_log2_or=top['log2_or'],
                          peak_overlap=top['overlap'], peak_exp=top['exp'],
                          peak_fdr_strat=top['fdr_strat'],
                          n_starred_native=int((nat_sig['TF'] == tf).sum()),
                          n_starred_l23set=int((l23_sig['TF'] == tf).sum()),
                          col_rank=col_rank[top['col']]))

    sel = pd.DataFrame(peaks).sort_values(['col_rank', 'peak_log2_enr'], ascending=[True, False])
    sel.to_csv(cfg['out_selection'], sep='\t', index=False)
    print(f"  wrote -> {cfg['out_selection']}")
    for col, grp in sel.groupby('peak_col', sort=False):
        print(f'    peak at {col:10s}: {len(grp):2d} regulons  ({", ".join(grp["TF"][:8])}'
              f'{", ..." if len(grp) > 8 else ""})')
    return list(sel['TF']), sel


def to_matrices(long, rows, cols, sign):
    """{key: TF x column matrix} for the activating regulons of the selected TFs."""
    sub = long[(long['regulation_direction'] == sign) & (long['TF'].isin(rows))]
    return {key: sub.pivot_table(index='TF', columns='col', values=key, aggfunc='first')
                    .reindex(index=rows, columns=cols)
            for key in ['log2_enr', 'fdr_strat', 'overlap', 'n_markers', 'n_targets',
                        'exp', 'log2_or']}


def add_panel(fig, row, mats, rows, cols, primed, m41b):
    """41b.add_panel with log2_enr/fdr_strat in place of log2_or/fdr.

    Kept as a separate copy rather than parameterising 41b's: the hover card gains the matched
    expectation and 41's log2 OR (the whole point of this figure is the contrast), and 41b's
    IEG/control horizontal rule has no meaning for a data-selected row set.
    """
    m = mats
    tested = np.isfinite(m['log2_enr'].values)   # False where the regulon does not exist here
    sig = (tested
           & (m['fdr_strat'].values < STAR_FDR)
           & (m['log2_enr'].values > STAR_LOG2ENR)
           & (m['overlap'].values >= MASK_MIN_OVERLAP))
    thin = tested & (m['overlap'].values < MASK_MIN_OVERLAP)   # too few shared genes to trust
    customdata = np.dstack([m['overlap'].values, m['exp'].values, m['n_targets'].values,
                            m['fdr_strat'].values, m['log2_or'].values])
    hover = ('TF=%{y}<br>%{x}<br>overlap=%{customdata[0]}'
             '<br>expected (matched)=%{customdata[1]:.2f}'
             '<br>n_targets=%{customdata[2]}<br>FDR=%{customdata[3]:.2e}'
             '<br><i>41 log2 OR=%{customdata[4]:.2f}</i>')

    # gray underlay for the thin cells; the colored trace leaves them NaN so it shows through
    if thin.any():
        fig.add_trace(go.Heatmap(
            z=np.where(thin, 1.0, np.nan), x=cols, y=rows,
            colorscale=[[0, m41b.MASK_COLOR], [1, m41b.MASK_COLOR]], showscale=False,
            zmin=0, zmax=1, xgap=1, ygap=1, customdata=customdata,
            hovertemplate=hover + f'<br><i>masked: overlap < {MASK_MIN_OVERLAP}</i><extra></extra>',
        ), row=row, col=1)

    fig.add_trace(go.Heatmap(
        z=np.where(thin, np.nan, m['log2_enr'].values), x=cols, y=rows, coloraxis='coloraxis',
        text=m41b.cell_text(m, tested), texttemplate='%{text}',
        textfont=dict(size=11, color=m41b.TEXT_COLOR),
        xgap=1, ygap=1, customdata=customdata,
        hovertemplate=hover + '<br>log2 enrichment=%{z:.2f}<extra></extra>',
    ), row=row, col=1)
    fig.update_yaxes(autorange='reversed', row=row, col=1)
    # every panel keeps its own tick labels despite shared_xaxes
    fig.update_xaxes(tickangle=-45, showticklabels=True, row=row, col=1)

    for i, j in zip(*np.nonzero(sig)):
        fig.add_shape(type='rect', x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=m41b.BOX_LINE, fillcolor='rgba(0,0,0,0)', row=row, col=1)

    start = 0
    for layer, _token, _label in m41b.LAYER_TOKEN[:-1]:
        start += len(primed[layer])
        fig.add_vline(x=start - 0.5, line=dict(color='black', width=1.5), row=row, col=1)


def main(cfg=CFG_YOO25):
    m41b = load_41b()
    print(f"Regulon catalogue: {cfg['tag']}")
    # the one criterion no universe or null choice can inflate, so it is shared verbatim
    assert MASK_MIN_OVERLAP == m41b.MASK_MIN_OVERLAP, \
        f'overlap floor drifted from 41b: {MASK_MIN_OVERLAP} vs {m41b.MASK_MIN_OVERLAP}'

    primed = m41b.load_primed_labels()
    cols = m41b.column_keys(primed)

    native = load_native(primed, m41b, cfg)
    assert os.path.exists(cfg['l23set']), \
        f'missing {cfg["l23set"]}; run the {cfg["tag"]} enrichment script first'
    l23set = m41b.to_col(pd.read_csv(cfg['l23set'], sep='\t'), primed)

    native = native[native['regulation_direction'] == m41b.SIGN]
    l23set = l23set[l23set['regulation_direction'] == m41b.SIGN]

    rows, _sel = select_rows(native, l23set, cols, cfg)

    panels = [("each subclass's own regulons", to_matrices(native, rows, cols, m41b.SIGN),
               cfg['out_native']),
              ('L2/3 regulons applied to every subclass',
               to_matrices(l23set, rows, cols, m41b.SIGN), cfg['out_l23set'])]
    print(f'\n  {len(rows)} regulons x {len(cols)} subclass-archetype columns')
    for title, mats, outs in panels:
        tested = mats['log2_enr'].notna()
        thin = int((tested & (mats['overlap'] < MASK_MIN_OVERLAP)).sum().sum())
        print(f'  {title}: {int(tested.sum().sum())} of {len(rows) * len(cols)} cells populated, '
              f'{thin} masked (overlap<{MASK_MIN_OVERLAP})')
        for key, path in outs.items():
            mats[key].to_csv(path, sep='\t')
            print(f'    wrote -> {path}')

    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[t for t, _m, _o in panels])
    for i, (_title, mats, _outs) in enumerate(panels, start=1):
        add_panel(fig, i, mats, rows, cols, primed, m41b)

    panel1 = fig.layout.yaxis.domain   # row 1 y-domain; colorbar is sized to match
    fig.update_layout(
        title=f'All enriched regulons ({m41b.SIGN}) — archetype marker enrichment across '
              f'{cfg["label"]} (expression-stratified, {cfg["tag"]} regulons)<br>'
              f'<sub>colour = log2(observed overlap / expression-matched expectation), 54; '
              f'rows = regulons starred in >=1 cell, grouped by peak column; '
              f'cell label = overlap gene count; boxed = FDR<{STAR_FDR:g} AND '
              f'log2 enr>{STAR_LOG2ENR:g} AND overlap>={MASK_MIN_OVERLAP}; '
              f'gray = overlap<{MASK_MIN_OVERLAP}, too few shared genes to trust; '
              f'blue = below the matched expectation</sub>',
        coloraxis=dict(colorscale=build_colorscale(), cmin=COLOR_MIN, cmax=COLOR_MAX,
                       colorbar=dict(title='log2 enr', thickness=14, x=1.01, xanchor='left',
                                     len=panel1[1] - panel1[0], y=panel1[1], yanchor='top')),
        height=160 + len(panels) * (22 * len(rows) + 70),
        width=max(760, 62 * len(cols) + 340),
        plot_bgcolor='white', margin=dict(t=120), showlegend=False,
    )
    _write_fig(fig, cfg['out_html'])


if __name__ == '__main__':
    main()
