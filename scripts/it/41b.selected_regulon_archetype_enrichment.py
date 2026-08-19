"""Selected IEG/stress regulons across all mouse IT subclasses — two views.

Companion to scripts/it/41.regulon_archetype_enrichment.py, which renders one heatmap per
subclass from that subclass's *own* SCENIC+ regulons. Here a hand-picked TF set is shown in
heatmaps whose columns concatenate every (subclass, archetype) pair, so one regulon can be
read across L2/3, L4, L5IT and L6IT:

  panel 1 "native"    each subclass's own regulons vs its own archetype markers
                      (reshaped from 41's long outputs; no recomputation)
  panel 2 "L2/3 set"  the *L2/3* regulon target sets tested against every subclass's
                      archetype markers (new Fisher tests, run here)

Panel 2 asks whether a regulon defined in L2/3 still marks the same laminar-depth pole in
the other subclasses; its L2/3 columns reproduce panel 1's L2/3 columns by construction.
Rows below the horizontal rule (CONTROL_TFS) are non-IEG regulons carried as controls.
Only activating (+/+) regulons are shown. Cell color is the Haldane-Anscombe log2 odds
ratio; a black box marks cells clearing all three of FDR < STAR_FDR, log2 OR > STAR_LOG2OR,
and overlap >= STAR_MIN_OVERLAP genes.

Cells resting on fewer than MASK_MIN_OVERLAP shared genes are grayed rather than colored
(see that constant for why), and every cell is labelled with its overlap gene count.

Statistics for panel 2 replicate 41 exactly (same universe reconstruction, same
Haldane-Anscombe log2 OR, one-sided Fisher, BH-FDR across all (archetype, regulon) pairs
within a subclass) by importing 41's helpers.

Archetype letters are the published primed labels from scripts/ARCHETYPE_MAPPING.md, read
from the persisted mouse depth-arc table (41 itself primes L2/3 only, so its `arch_letter`
column is bypassed and letters are re-derived from `archetype_k` here). Columns run in
laminar-depth order: L2/3 A'B'C' | L4 A'B'C' | L5IT A'B' | L6IT A'B'C'.

Reads:
  local_data/res/it/41.<layer>_regulon_archetype_enrichment.tsv     (panel 1)
  local_data/res/it/40.yoo25_L2_3_regulon_targets.tsv               (panel 2 regulon sets)
  local_data/res/it/<markers>.tsv, <coords>.tsv                     (panel 2, via 41's LAYERS)
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv   (primed letters)
  links/it/superdupermegaRNA_{cheng22,yoo25}_IT_*.h5ad              (panel 2 universes)
Outputs:
  local_data/res/it/41b.l23_regulon_all_subclass_enrichment.tsv  (panel 2 long, all regulons)
  local_data/res/it/41b.selected_native_log2or.tsv   / _fdr.tsv   (panel 1 matrices)
  local_data/res/it/41b.selected_l23set_log2or.tsv   / _fdr.tsv   (panel 2 matrices)
  local_data/fig/it/41b.selected_regulon_archetype_enrichment.html
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import anndata as ad
import scipy.stats
from statsmodels.stats.multitest import multipletests
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

SCRIPT_41 = os.path.join(SCRIPTS_DIR, 'it', '41.regulon_archetype_enrichment.py')
INPUT_ARC_ORDER = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo',
                               '15.mouse_IT_joint_archetype_arc_order.tsv')
INPUT_L23_REGULONS = os.path.join(RES_DIR, '40.yoo25_L2_3_regulon_targets.tsv')

OUT_L23SET_LONG = os.path.join(RES_DIR, '41b.l23_regulon_all_subclass_enrichment.tsv')
OUT_NATIVE = {'log2_or': os.path.join(RES_DIR, '41b.selected_native_log2or.tsv'),
              'fdr': os.path.join(RES_DIR, '41b.selected_native_fdr.tsv')}
OUT_L23SET = {'log2_or': os.path.join(RES_DIR, '41b.selected_l23set_log2or.tsv'),
              'fdr': os.path.join(RES_DIR, '41b.selected_l23set_fdr.tsv')}
OUT_HTML = os.path.join(FIG_DIR, '41b.selected_regulon_archetype_enrichment.html')

SELECTED_TFS = ['Fos', 'Fosb', 'Fosl2', 'Junb', 'Egr1', 'Egr2', 'Egr3', 'Egr4', 'Atf6', 'Smad3']
# non-IEG regulons carried as controls; drawn below a horizontal rule
CONTROL_TFS = ['Rfx3', 'Nfib', 'Tcf4', 'Jdp2', 'Satb1', 'Tcf12', 'Bach2']
SIGN = '+/+'        # activating regulons only
# In the masked figure, cells resting on fewer than this many shared genes are drawn gray
# instead of colored: at a median expected overlap of ~1 gene the Haldane-Anscombe
# correction gives sizeable positive log2 OR to cells with little or no actual overlap
# (e.g. L4 Bach2 A': overlap=0, FDR=1.0, log2 OR=+2.28), which reads as enrichment.
MASK_MIN_OVERLAP = 5

# a cell is starred only if it clears all three criteria (significant, strong, and not
# carried by a handful of genes)
STAR_FDR = 0.05         # BH-FDR below this
STAR_LOG2OR = 2.0       # log2 odds ratio above this
# Kept equal to MASK_MIN_OVERLAP: the two encode one decision (how many shared genes are
# enough to trust a cell), and Fisher already penalizes thin overlap on its own. At 10 this
# floor acted as a regulon-size filter -- 0% of regulons with <=25 targets could ever pass
# it against 93% of those with >120 -- discarding 26 regulons that cleared both other
# criteria, Meis2 among them (9/23 targets are L2/3 A' markers, FDR 6e-11).
STAR_MIN_OVERLAP = MASK_MIN_OVERLAP
# Once thin cells are masked, every remaining cell is enriched (min log2 OR = +0.21 across
# both panels), so a diverging scale would waste half its range on unused blue. A
# sequential ramp is used instead, over a fixed [COLOR_MIN, COLOR_MAX] so panels and
# scripts stay comparable.
COLOR_MIN, COLOR_MAX = 0.0, 6.5
MASK_COLOR = '#d9d9d9'
# Significance is drawn as a box around the cell rather than an asterisk, and the overlap
# count is printed in a single color -- so the ramp is YlOrRd stopped at 0.74, the point
# past which black text drops below the 4.5:1 WCAG AA contrast ratio. Its pale-yellow floor
# also stays distinguishable from MASK_COLOR.
COLORSCALE = [[t, c] for t, c in zip(
    np.linspace(0, 1, 21), sample_colorscale('YlOrRd', np.linspace(0, 0.74, 21)))]
TEXT_COLOR = 'black'
BOX_LINE = dict(color='black', width=2)

# layer key (41's `layer`) -> depth-arc table token / display label, in laminar order
LAYER_TOKEN = [('L2_3', 'L23', 'L2/3'), ('L4', 'L4', 'L4'),
               ('L5IT', 'L5IT', 'L5IT'), ('L6IT', 'L6IT', 'L6IT')]
ARCHETYPE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']  # archetype_1 -> A (41's fit-order letters)

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_41():
    """Import 41 as a module (its filename is not a valid identifier) for its helpers."""
    spec = importlib.util.spec_from_file_location('script41', SCRIPT_41)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_primed_labels():
    """{layer: {archetype_k: primed letter}} from the persisted mouse depth-arc table.

    The table keys archetypes by their fit-order letter (`old_letter`, = archetype_1 -> A);
    see scripts/ARCHETYPE_MAPPING.md.
    """
    arc = pd.read_csv(INPUT_ARC_ORDER, sep='\t')
    old2arch = {ltr: f'archetype_{i + 1}' for i, ltr in enumerate(ARCHETYPE_LETTERS)}
    out = {}
    for layer, token, _label in LAYER_TOKEN:
        sub = arc[arc['token'] == token]
        assert len(sub) > 0, f'token {token} absent from {INPUT_ARC_ORDER}'
        out[layer] = {old2arch[o]: n for o, n in zip(sub['old_letter'], sub['new_letter'])}
    return out


def column_keys(primed):
    """(subclass, archetype) column labels in laminar-depth order."""
    return [f'{label} {primed[layer][a]}'
            for layer, _token, label in LAYER_TOKEN
            for a in sorted(primed[layer], key=lambda k: primed[layer][k])]


def to_col(long, primed):
    """Add the concatenated column key, re-deriving primed letters from `archetype`."""
    label = {layer: lbl for layer, _t, lbl in LAYER_TOKEN}
    long = long.copy()
    long['arch_primed'] = [primed[l][a] for l, a in zip(long['layer'], long['archetype'])]
    long['col'] = long['layer'].map(label) + ' ' + long['arch_primed']
    return long


# ---------------------------------------------------------------------------
# panel 1 — each subclass's own regulons (reshape of 41's outputs)
# ---------------------------------------------------------------------------

def load_native(primed):
    frames = []
    for layer, _token, _label in LAYER_TOKEN:
        path = os.path.join(RES_DIR, f'41.{layer}_regulon_archetype_enrichment.tsv')
        assert os.path.exists(path), f'missing {path}; run 41.regulon_archetype_enrichment.py first'
        frames.append(pd.read_csv(path, sep='\t'))
    return to_col(pd.concat(frames, ignore_index=True), primed)


# ---------------------------------------------------------------------------
# panel 2 — L2/3 regulon sets tested against every subclass (new Fisher tests)
# ---------------------------------------------------------------------------

def enrich_l23_set_in_layer(cfg, T_sets_all, reg_meta, m41, adatas):
    """One subclass: L2/3 regulon target sets vs this subclass's archetype markers.

    Mirrors 41.enrich_layer's statistics; only the regulon source differs.
    """
    layer, noc = cfg['layer'], cfg['noc']
    print(f'\n=== L2/3 regulons in {layer} ===')

    coords = pd.read_csv(os.path.join(RES_DIR, cfg['coords']), sep='\t', index_col=0)
    universe = m41.reconstruct_gene_universe(coords.index.values, cfg['subclass_val'], adatas)
    N = len(universe)

    markers = pd.read_csv(os.path.join(RES_DIR, cfg['markers']), sep='\t')
    arch_labels = [f'archetype_{k + 1}' for k in range(noc)]
    M_sets = {a: set(markers.loc[markers['archetype'] == a, 'gene']) & universe for a in arch_labels}

    T_sets = {r: t & universe for r, t in T_sets_all.items()}
    T_sets = {r: t for r, t in T_sets.items() if len(t) >= m41.MIN_REGULON_GENES}
    print(f'    L2/3 regulons: {len(T_sets)} kept (>= {m41.MIN_REGULON_GENES} targets in the '
          f'{layer} universe) of {len(T_sets_all)}; universe N={N}')

    rows = []
    for r, T in T_sets.items():
        for a in arch_labels:
            M = M_sets[a]
            x = len(M & T)
            table = [[x, len(M) - x], [len(T) - x, N - len(M) - len(T) + x]]
            _, pval = scipy.stats.fisher_exact(table, alternative='greater')
            rows.append(dict(
                layer=layer, archetype=a, regulon=r,
                TF=reg_meta.loc[r, 'TF'], regulation_direction=reg_meta.loc[r, 'regulation_direction'],
                overlap=x, n_markers=len(M), n_targets=len(T), universe=N,
                log2_or=m41.log2_odds_ratio(x, len(M), len(T), N), pval=pval,
            ))

    long = pd.DataFrame(rows)
    long['fdr'] = multipletests(long['pval'].values, method='fdr_bh')[1]
    long['neglog10_fdr'] = -np.log10(long['fdr'].clip(lower=np.nextafter(0, 1)))
    return long


def build_l23_set(primed, m41):
    reg = pd.read_csv(INPUT_L23_REGULONS, sep='\t')
    reg_meta = reg.drop_duplicates('regulon').set_index('regulon')[['TF', 'regulation_direction']]
    T_sets_all = {r: set(g['Gene']) for r, g in reg.groupby('regulon')}
    print(f'  L2/3 regulon source: {len(T_sets_all)} regulons from {INPUT_L23_REGULONS}')

    print('  Loading h5ad inputs once...')
    adatas = {d['tag']: ad.read_h5ad(d['path']) for d in m41.DATASETS}

    longs = [enrich_l23_set_in_layer(cfg, T_sets_all, reg_meta, m41, adatas) for cfg in m41.LAYERS]
    long = pd.concat(longs, ignore_index=True)
    long = to_col(long, primed)
    long.to_csv(OUT_L23SET_LONG, sep='\t', index=False)
    print(f'\n  wrote -> {OUT_L23SET_LONG} ({len(long)} pairs)')
    return long


# ---------------------------------------------------------------------------

def to_matrices(long, rows, cols):
    """{key: TF x column matrix} for the activating regulons of the selected TFs."""
    sub = long[(long['regulation_direction'] == SIGN) & (long['TF'].isin(rows))]
    return {key: sub.pivot_table(index='TF', columns='col', values=key, aggfunc='first')
                    .reindex(index=rows, columns=cols)
            for key in ['log2_or', 'fdr', 'overlap', 'n_targets']}


def cell_text(m, tested):
    """Per-cell label: the overlap gene count (significance is the box, not a suffix)."""
    ov, out = m['overlap'].values, []
    for i in range(ov.shape[0]):
        out.append([f'{int(ov[i, j])}' if tested[i, j] else '' for j in range(ov.shape[1])])
    return out


def add_panel(fig, row, mats, rows, cols, primed, n_ieg):
    m = mats
    tested = np.isfinite(m['log2_or'].values)   # False where the regulon does not exist here
    sig = (tested
           & (m['fdr'].values < STAR_FDR)
           & (m['log2_or'].values > STAR_LOG2OR)
           & (m['overlap'].values >= STAR_MIN_OVERLAP))
    thin = tested & (m['overlap'].values < MASK_MIN_OVERLAP)   # too few shared genes to trust
    customdata = np.dstack([m['overlap'].values, m['n_targets'].values, m['fdr'].values])
    hover = ('TF=%{y}<br>%{x}<br>overlap=%{customdata[0]}'
             '<br>n_targets=%{customdata[1]}<br>FDR=%{customdata[2]:.2e}')

    # gray underlay for the thin cells; the colored trace leaves them NaN so it shows through
    if thin.any():
        fig.add_trace(go.Heatmap(
            z=np.where(thin, 1.0, np.nan), x=cols, y=rows,
            colorscale=[[0, MASK_COLOR], [1, MASK_COLOR]], showscale=False,
            zmin=0, zmax=1, xgap=1, ygap=1, customdata=customdata,
            hovertemplate=hover + f'<br><i>masked: overlap < {MASK_MIN_OVERLAP}</i><extra></extra>',
        ), row=row, col=1)

    fig.add_trace(go.Heatmap(
        z=np.where(thin, np.nan, m['log2_or'].values), x=cols, y=rows, coloraxis='coloraxis',
        text=cell_text(m, tested), texttemplate='%{text}',
        textfont=dict(size=11, color=TEXT_COLOR),
        xgap=1, ygap=1, customdata=customdata,
        hovertemplate=hover + '<br>log2 OR=%{z:.2f}<extra></extra>',
    ), row=row, col=1)
    fig.update_yaxes(autorange='reversed', row=row, col=1)
    # every panel keeps its own tick labels despite shared_xaxes
    fig.update_xaxes(tickangle=-45, showticklabels=True, row=row, col=1)

    for i, j in zip(*np.nonzero(sig)):
        fig.add_shape(type='rect', x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=BOX_LINE, fillcolor='rgba(0,0,0,0)', row=row, col=1)

    start = 0
    for layer, _token, _label in LAYER_TOKEN[:-1]:
        start += len(primed[layer])
        fig.add_vline(x=start - 0.5, line=dict(color='black', width=1.5), row=row, col=1)

    # rule between the IEG block and the control block
    if 0 < n_ieg < len(rows):
        fig.add_hline(y=n_ieg - 0.5, line=dict(color='black', width=2.5), row=row, col=1)


def main():
    m41 = load_41()
    primed = load_primed_labels()
    cols = column_keys(primed)

    native = load_native(primed)
    l23set = build_l23_set(primed, m41)

    have = (set(native.loc[native['regulation_direction'] == SIGN, 'TF'])
            | set(l23set.loc[l23set['regulation_direction'] == SIGN, 'TF']))
    ieg_rows = [t for t in SELECTED_TFS if t in have]
    ctrl_rows = [t for t in CONTROL_TFS if t in have]
    rows = ieg_rows + ctrl_rows
    n_ieg = len(ieg_rows)   # horizontal rule goes below this many rows
    missing = [t for t in SELECTED_TFS + CONTROL_TFS if t not in have]
    assert rows, f'none of {SELECTED_TFS + CONTROL_TFS} has an activating regulon'
    if missing:
        print(f'  [note] TFs with no {SIGN} regulon anywhere: {missing}')

    panels = [
        ("each subclass's own regulons", to_matrices(native, rows, cols), OUT_NATIVE),
        ('L2/3 regulons applied to every subclass', to_matrices(l23set, rows, cols), OUT_L23SET),
    ]
    print(f'\n  {len(rows)} TFs ({n_ieg} IEG + {len(ctrl_rows)} control) '
          f'x {len(cols)} subclass-archetype columns')
    outs_all = []
    for title, mats, outs in panels:
        tested = mats['log2_or'].notna()
        thin = int((tested & (mats['overlap'] < MASK_MIN_OVERLAP)).sum().sum())
        print(f'  {title}: {int(tested.sum().sum())} of {len(rows) * len(cols)} cells populated, '
              f'{thin} masked (overlap<{MASK_MIN_OVERLAP})')
        for key, path in outs.items():
            mats[key].to_csv(path, sep='\t')
            outs_all.append((key, path))

    for key, path in outs_all:
        print(f'    wrote -> {path}')

    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=[t for t, _m, _o in panels])
    for i, (title, mats, _outs) in enumerate(panels, start=1):
        add_panel(fig, i, mats, rows, cols, primed, n_ieg)

    panel1 = fig.layout.yaxis.domain   # row 1 y-domain; colorbar is sized to match
    fig.update_layout(
        title=f'Selected regulons ({SIGN}) — archetype marker enrichment across mouse IT '
              f'subclasses (log2 OR)<br>'
              f'<sub>cell label = overlap gene count; boxed = FDR<{STAR_FDR:g} AND '
              f'log2 OR>{STAR_LOG2OR:g} AND overlap>={STAR_MIN_OVERLAP}; '
              f'gray = overlap<{MASK_MIN_OVERLAP}, too few shared genes to trust</sub>',
        coloraxis=dict(colorscale=COLORSCALE, cmin=COLOR_MIN, cmax=COLOR_MAX,
                       colorbar=dict(title='log2 OR', thickness=14, x=1.01, xanchor='left',
                                     len=panel1[1] - panel1[0], y=panel1[1], yanchor='top')),
        height=160 + len(panels) * (26 * len(rows) + 70),
        width=max(760, 62 * len(cols) + 340),
        plot_bgcolor='white', margin=dict(t=110), showlegend=False,
    )
    _write_fig(fig, OUT_HTML)


if __name__ == '__main__':
    main()
