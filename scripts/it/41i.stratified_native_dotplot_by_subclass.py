"""41h's native panel, split per subclass with rows clustered inside each (PDF).

41h draws both of 41f's panels as dot matrices over one shared row order and all 11
subclass-archetype columns. That shared order is set by each regulon's global peak column, so
within any single subclass the rows arrive in an order chosen by *other* subclasses, and the
structure that subclass actually has is scattered down the figure.

This script keeps only the first panel -- each subclass's own regulons against its own
archetype markers -- and splits it into four independent matrices, one per subclass. Each gets
its own rows, in its own order, from a hierarchical clustering of that subclass's log2_enr
profile. Reading a row across subclasses is therefore NOT possible here, by design; 41h is the
figure for that. What this one shows is the within-subclass pattern: which regulons behave
alike across the archetypes of one laminar sheet.

Encoding is 41h's, unchanged:

  colour = log2_enr                    (41f's ramp, blue below the matched expectation)
  area   = overlap / n_markers         (fraction of the archetype marker set covered)
  gray   = overlap < MASK_MIN_OVERLAP  (too few shared genes to trust)
  outline = FDR < STAR_FDR AND log2_enr > STAR_LOG2ENR AND overlap >= MASK_MIN_OVERLAP

Rows per panel are those of 41f's selection that have at least one UNMASKED cell in that
subclass (see REQUIRE_UNMASKED_ROW). An all-gray row carries no trustworthy signal to cluster
on, and would otherwise dominate the small panels; the count dropped is printed per subclass.

Reads:
  local_data/res/it/41e.<layer>_stratified_enrichment.tsv    (via 41f.load_native)
  local_data/res/it/41f.enriched_regulon_selection.tsv       (row set; order is re-derived)
Outputs:
  local_data/res/it/41i.clustered_row_order.tsv
  local_data/fig/it/41i.stratified_native_dotplot_by_subclass.pdf
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# fonttype 42 embeds TrueType so the PDF keeps real text objects; the default (3) writes
# Type 3 glyph procedures, which Illustrator/Inkscape open as uneditable outlines
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, leaves_list

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

SCRIPT_41B = os.path.join(SCRIPTS_DIR, 'it', '41b.selected_regulon_archetype_enrichment.py')
SCRIPT_41F = os.path.join(SCRIPTS_DIR, 'it', '41f.stratified_enriched_regulon_heatmap.py')
SCRIPT_41G = os.path.join(SCRIPTS_DIR, 'it', '41g.stratified_enriched_regulon_heatmap_pdf.py')
INPUT_SELECTION = os.path.join(RES_DIR, '41f.enriched_regulon_selection.tsv')
OUT_ORDER = os.path.join(RES_DIR, '41i.clustered_row_order.tsv')
OUT_PDF = os.path.join(FIG_DIR, '41i.stratified_native_dotplot_by_subclass.pdf')

# area encoding, as 41h: a dot at FRAC_REF covers SIZE_REF points^2, area linear in the
# fraction. FRAC_REF sits just above the native panel's observed maximum (0.353).
FRAC_REF = 0.40
SIZE_REF = 260.0
SIZE_LEGEND = [0.05, 0.15, 0.25, 0.35]
BOX_LW = 1.4

# drop rows with no unmasked cell in a subclass; see the module docstring
REQUIRE_UNMASKED_ROW = True
# average linkage on the euclidean distance between log2_enr profiles, with masked and absent
# cells read as 0 (no trustworthy enrichment). optimal_ordering flips branches to minimise the
# distance between adjacent leaves, which is what makes the blocks read as blocks.
CLUSTER_METHOD = 'average'
CLUSTER_METRIC = 'euclidean'

# inches per matrix cell; wider than 41h because each panel is only 2-3 columns
CELL_W, CELL_H = 0.62, 0.21
LEFT_PAD = 0.75          # room for the leftmost panel's TF labels
PANEL_GAP = 0.80         # room for each following panel's TF labels
RIGHT_PAD = 2.55         # legend + colorbar
LEGEND_H = 2.30          # vertical room the size legend takes, so the colorbar clears it
CBAR_H = 2.60
TOP_PAD = 1.45           # suptitle
BOTTOM_PAD = 0.95        # rotated archetype labels

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def layer_matrices(long, layer, rows, letters, keys):
    """{key: TF x archetype-letter matrix} for one subclass."""
    sub = long[(long['layer'] == layer) & (long['TF'].isin(rows))]
    return {k: sub.pivot_table(index='TF', columns='arch_primed', values=k, aggfunc='first')
                 .reindex(index=rows, columns=letters)
            for k in keys}


def cluster_rows(mats, rows, m41b):
    """Row order from hierarchical clustering of the trustworthy log2_enr profile."""
    log2 = mats['log2_enr'].values
    overlap = mats['overlap'].values
    tested = np.isfinite(log2)
    usable = tested & (overlap >= m41b.MASK_MIN_OVERLAP)
    X = np.where(usable, np.nan_to_num(log2), 0.0)
    if len(rows) < 3:
        return list(rows)
    Z = linkage(X, method=CLUSTER_METHOD, metric=CLUSTER_METRIC, optimal_ordering=True)
    return [rows[i] for i in leaves_list(Z)]


def draw_panel(ax, mats, rows, letters, label, m41b, m41f, cmap, norm):
    """One subclass's dot matrix; returns the colour mappable for the shared colorbar."""
    log2 = mats['log2_enr'].values
    overlap = mats['overlap'].values
    n_markers = mats['n_markers'].values
    tested = np.isfinite(log2)
    thin = tested & (overlap < m41b.MASK_MIN_OVERLAP)
    sig = (tested & (mats['fdr_strat'].values < m41f.STAR_FDR)
           & (log2 > m41f.STAR_LOG2ENR) & (overlap >= m41b.MASK_MIN_OVERLAP))

    frac = np.divide(overlap, n_markers, out=np.full_like(log2, np.nan), where=tested)
    assert np.nanmax(frac) <= FRAC_REF + 1e-9, \
        f'{label}: coverage {np.nanmax(frac):.3f} exceeds the size reference {FRAC_REF}'

    yy, xx = np.nonzero(tested)
    sizes = frac[yy, xx] / FRAC_REF * SIZE_REF
    mappable = None
    # masked and coloured dots are drawn separately so the gray fill is not run through cmap
    for keep, kw in [(thin[yy, xx], dict(color=m41b.MASK_COLOR)),
                     (~thin[yy, xx], dict(c=log2[yy, xx][~thin[yy, xx]], cmap=cmap, norm=norm))]:
        if not keep.any():
            continue
        edge = np.where(sig[yy, xx][keep], 'black', 'none')
        lw = np.where(sig[yy, xx][keep], BOX_LW, 0.0)
        sc = ax.scatter(xx[keep], yy[keep], s=sizes[keep], edgecolors=edge, linewidths=lw,
                        zorder=3, **kw)
        if 'c' in kw:
            mappable = sc

    ax.set_xticks(range(len(letters)))
    ax.set_xticklabels(letters, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_xlim(-0.6, len(letters) - 0.4)
    ax.set_ylim(len(rows) - 0.4, -0.6)          # first row on top
    ax.set_axisbelow(True)
    ax.grid(True, color='0.90', lw=0.5, zorder=0)
    ax.tick_params(length=0)
    ax.set_title(label, fontsize=10, pad=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('0.6')
    return mappable


def main():
    m41b = load_module(SCRIPT_41B, 'script41b')
    m41f = load_module(SCRIPT_41F, 'script41f')
    m41g = load_module(SCRIPT_41G, 'script41g')   # for the plotly -> matplotlib ramp

    primed = m41b.load_primed_labels()
    native = m41f.load_native(primed, m41b)
    native = native[native['regulation_direction'] == m41b.SIGN]

    assert os.path.exists(INPUT_SELECTION), \
        f'missing {INPUT_SELECTION}; run 41f.stratified_enriched_regulon_heatmap.py first'
    selected = list(pd.read_csv(INPUT_SELECTION, sep='\t')['TF'])
    print(f'  {len(selected)} regulons in 41f\'s selection; splitting the native panel by subclass')

    keys = ['log2_enr', 'fdr_strat', 'overlap', 'n_markers']
    panels, order_rows = [], []
    for layer, _token, label in m41b.LAYER_TOKEN:
        arch = sorted(primed[layer], key=lambda k: primed[layer][k])
        letters = [primed[layer][a] for a in arch]
        sub = native[(native['layer'] == layer) & (native['TF'].isin(selected))]

        present = [t for t in selected if t in set(sub['TF'])]
        if REQUIRE_UNMASKED_ROW:
            thick = set(sub.loc[sub['overlap'] >= m41b.MASK_MIN_OVERLAP, 'TF'])
            rows = [t for t in present if t in thick]
        else:
            rows = present
        assert rows, f'{label}: no regulon left after row filtering'

        mats = layer_matrices(native, layer, rows, letters, keys)
        rows = cluster_rows(mats, rows, m41b)
        mats = layer_matrices(native, layer, rows, letters, keys)
        print(f'  {label:5s}: {len(letters)} archetypes, {len(rows)} rows '
              f'({len(present) - len(rows)} dropped as fully masked, '
              f'{len(selected) - len(present)} absent from this subclass)')
        panels.append((label, letters, rows, mats))
        order_rows += [dict(subclass=label, position=i, TF=t) for i, t in enumerate(rows)]

    pd.DataFrame(order_rows).to_csv(OUT_ORDER, sep='\t', index=False)
    print(f'  wrote -> {OUT_ORDER}')

    cmap = m41g.plotly_to_mpl_cmap(m41f.build_colorscale())
    norm = Normalize(vmin=m41f.COLOR_MIN, vmax=m41f.COLOR_MAX)

    # explicit axes placement: every panel keeps the SAME cell pitch, so dot areas stay
    # comparable across subclasses even though the panels have different row and column counts
    widths = [len(p[1]) * CELL_W for p in panels]
    heights = [len(p[2]) * CELL_H for p in panels]
    fig_w = LEFT_PAD + sum(widths) + PANEL_GAP * (len(panels) - 1) + RIGHT_PAD
    fig_h = TOP_PAD + max(heights) + BOTTOM_PAD
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes, x_cursor = [], LEFT_PAD
    for (label, letters, rows, mats), w, h in zip(panels, widths, heights):
        ax = fig.add_axes([x_cursor / fig_w, (fig_h - TOP_PAD - h) / fig_h,
                           w / fig_w, h / fig_h])
        mappable = draw_panel(ax, mats, rows, letters, label, m41b, m41f, cmap, norm)
        axes.append(ax)
        x_cursor += w + PANEL_GAP

    # colorbar tucked directly under the size legend in the right margin, clamped so it stays
    # inside the figure when the tallest panel is short
    cbar_h = max(1.4, min(CBAR_H, max(heights) - LEGEND_H - 0.4))
    cbar_bottom = max(0.35, fig_h - TOP_PAD - LEGEND_H - 0.45 - cbar_h)
    cax = fig.add_axes([(fig_w - RIGHT_PAD + 1.75) / fig_w, cbar_bottom / fig_h,
                        0.16 / fig_w, cbar_h / fig_h])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label('log2 enrichment  (observed / expression-matched expectation)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.axhline(0.0, color='black', lw=0.8)     # the meaningful midpoint

    handles = [Line2D([], [], marker='o', linestyle='none', markerfacecolor='0.55',
                      markeredgecolor='none', markersize=np.sqrt(f / FRAC_REF * SIZE_REF),
                      label=f'{f:.2f}') for f in SIZE_LEGEND]
    handles += [Line2D([], [], marker='o', linestyle='none', markerfacecolor=m41b.MASK_COLOR,
                       markeredgecolor='none', markersize=7,
                       label=f'overlap<{m41b.MASK_MIN_OVERLAP}'),
                Line2D([], [], marker='o', linestyle='none', markerfacecolor='white',
                       markeredgecolor='black', markeredgewidth=BOX_LW, markersize=7,
                       label='significant')]
    axes[-1].legend(handles=handles, title='overlap / n_markers', loc='upper left',
                    bbox_to_anchor=(1.35, 1.0), frameon=False, fontsize=7, title_fontsize=8,
                    labelspacing=1.0, borderpad=0.6)

    fig.suptitle(
        f'Enriched regulons ({m41b.SIGN}) vs archetype markers — each subclass\'s own regulons, '
        f'split by subclass\n'
        f'rows clustered independently within each panel ({CLUSTER_METHOD} linkage on the '
        f'log2 enrichment profile), so rows do NOT align across panels\n'
        f'colour = log2 enrichment, area = fraction of the archetype marker set covered; '
        f'outlined = FDR<{m41f.STAR_FDR:g}, log2 enr>{m41f.STAR_LOG2ENR:g}, '
        f'overlap>={m41b.MASK_MIN_OVERLAP}; gray = overlap<{m41b.MASK_MIN_OVERLAP}; '
        f'blue = below the matched expectation',
        fontsize=9, y=1 - 0.18 / fig_h)

    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
