"""Every enriched regulon across all mouse IT subclasses — same two views as 41b.

41b shows a hand-picked TF set. This script keeps that figure's design exactly (same
columns, same two panels, same statistics, same masking) but selects its rows by data:
every regulon that is *significantly enriched in at least one archetype of at least one
subclass*, using 41b's star criterion (FDR < STAR_FDR AND log2 OR > STAR_LOG2OR AND
overlap >= STAR_MIN_OVERLAP genes).

Rows are grouped by the column where the regulon peaks, so the figure reads as blocks of
regulons marking the same laminar pole; within a block they are sorted by descending peak
log2 OR. Selection uses the native panel, which covers regulons from all four subclasses
(the L2/3-set panel only ever contains L2/3 regulons); the union with the L2/3-set panel's
hits is taken so nothing significant in either view is dropped.

Nothing is recomputed here — both panels are reshaped from tables 41 and 41b already
wrote, and the helpers, constants and drawing code are imported from 41b so the two figures
cannot drift apart.

Reads:
  local_data/res/it/41.<layer>_regulon_archetype_enrichment.tsv     (native panel)
  local_data/res/it/41b.l23_regulon_all_subclass_enrichment.tsv     (L2/3-set panel)
Outputs:
  local_data/res/it/41c.enriched_regulon_selection.tsv        (chosen regulons + peak cell)
  local_data/res/it/41c.enriched_native_log2or.tsv / _fdr.tsv
  local_data/res/it/41c.enriched_l23set_log2or.tsv / _fdr.tsv
  local_data/fig/it/41c.enriched_regulon_archetype_enrichment_masked.html
"""

import os
import importlib.util

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)
from viz import _write_fig

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

SCRIPT_41B = os.path.join(SCRIPTS_DIR, 'it', '41b.selected_regulon_archetype_enrichment.py')
INPUT_L23SET_LONG = os.path.join(RES_DIR, '41b.l23_regulon_all_subclass_enrichment.tsv')

OUT_SELECTION = os.path.join(RES_DIR, '41c.enriched_regulon_selection.tsv')
OUT_NATIVE = {'log2_or': os.path.join(RES_DIR, '41c.enriched_native_log2or.tsv'),
              'fdr': os.path.join(RES_DIR, '41c.enriched_native_fdr.tsv')}
OUT_L23SET = {'log2_or': os.path.join(RES_DIR, '41c.enriched_l23set_log2or.tsv'),
              'fdr': os.path.join(RES_DIR, '41c.enriched_l23set_fdr.tsv')}
OUT_HTML = os.path.join(FIG_DIR, '41c.enriched_regulon_archetype_enrichment_masked.html')

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_41b():
    """Import 41b (its filename is not a valid identifier) for its helpers and constants."""
    spec = importlib.util.spec_from_file_location('script41b', SCRIPT_41B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def starred(df, m41b):
    """41b's star criterion, as a boolean Series over a long-format table."""
    return ((df['fdr'] < m41b.STAR_FDR)
            & (df['log2_or'] > m41b.STAR_LOG2OR)
            & (df['overlap'] >= m41b.STAR_MIN_OVERLAP))


def select_rows(native, l23set, cols, m41b):
    """Regulons enriched in >=1 archetype of >=1 subclass, grouped by their peak column.

    Returns (ordered TF list, per-regulon selection table).
    """
    nat_sig = native[starred(native, m41b)]
    l23_sig = l23set[starred(l23set, m41b)]
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
        top = hits.loc[hits['log2_or'].idxmax()]
        peaks.append(dict(TF=tf, peak_panel=panel, peak_col=top['col'],
                          peak_log2_or=top['log2_or'], peak_overlap=top['overlap'],
                          peak_fdr=top['fdr'], n_starred_native=int((nat_sig['TF'] == tf).sum()),
                          n_starred_l23set=int((l23_sig['TF'] == tf).sum()),
                          col_rank=col_rank[top['col']]))

    sel = pd.DataFrame(peaks).sort_values(['col_rank', 'peak_log2_or'], ascending=[True, False])
    sel.to_csv(OUT_SELECTION, sep='\t', index=False)
    print(f'  wrote -> {OUT_SELECTION}')
    for col, grp in sel.groupby('peak_col', sort=False):
        print(f'    peak at {col:10s}: {len(grp):2d} regulons  ({", ".join(grp["TF"][:8])}'
              f'{", ..." if len(grp) > 8 else ""})')
    return list(sel['TF']), sel


def main():
    m41b = load_41b()
    primed = m41b.load_primed_labels()
    cols = m41b.column_keys(primed)

    native = m41b.load_native(primed)
    assert os.path.exists(INPUT_L23SET_LONG), \
        f'missing {INPUT_L23SET_LONG}; run 41b.selected_regulon_archetype_enrichment.py first'
    l23set = pd.read_csv(INPUT_L23SET_LONG, sep='\t')

    native = native[native['regulation_direction'] == m41b.SIGN]
    l23set = l23set[l23set['regulation_direction'] == m41b.SIGN]

    rows, _sel = select_rows(native, l23set, cols, m41b)

    panels = [("each subclass's own regulons", m41b.to_matrices(native, rows, cols), OUT_NATIVE),
              ('L2/3 regulons applied to every subclass',
               m41b.to_matrices(l23set, rows, cols), OUT_L23SET)]
    print(f'\n  {len(rows)} regulons x {len(cols)} subclass-archetype columns')
    for title, mats, outs in panels:
        tested = mats['log2_or'].notna()
        thin = int((tested & (mats['overlap'] < m41b.MASK_MIN_OVERLAP)).sum().sum())
        print(f'  {title}: {int(tested.sum().sum())} of {len(rows) * len(cols)} cells populated, '
              f'{thin} masked (overlap<{m41b.MASK_MIN_OVERLAP})')
        for key, path in outs.items():
            mats[key].to_csv(path, sep='\t')
            print(f'    wrote -> {path}')

    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[t for t, _m, _o in panels])
    for i, (_title, mats, _outs) in enumerate(panels, start=1):
        # n_ieg = len(rows) suppresses 41b's IEG/control rule; there are no blocks here
        m41b.add_panel(fig, i, mats, rows, cols, primed, len(rows))

    fig.update_layout(
        title=f'All enriched regulons ({m41b.SIGN}) — archetype marker enrichment across mouse '
              f'IT subclasses (log2 OR)<br>'
              f'<sub>rows = regulons starred in >=1 cell, grouped by peak column; '
              f'cell label = overlap gene count; * FDR<{m41b.STAR_FDR:g} AND '
              f'log2 OR>{m41b.STAR_LOG2OR:g} AND overlap>={m41b.STAR_MIN_OVERLAP}; '
              f'gray = overlap<{m41b.MASK_MIN_OVERLAP}, too few shared genes to trust</sub>',
        coloraxis=dict(colorscale='RdBu_r', cmid=0, cmin=-m41b.COLOR_ABS, cmax=m41b.COLOR_ABS,
                       colorbar=dict(title='log2 OR', len=0.9, thickness=14)),
        height=160 + len(panels) * (22 * len(rows) + 70),
        width=max(760, 62 * len(cols) + 340),
        plot_bgcolor='white', margin=dict(t=120), showlegend=False,
    )
    _write_fig(fig, OUT_HTML)


if __name__ == '__main__':
    main()
