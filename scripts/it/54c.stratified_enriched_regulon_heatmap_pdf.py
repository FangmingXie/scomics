"""54b's two-panel heatmap as a vector PDF, for figure assembly.

54b draws the expression-stratified enrichment heatmap through plotly, which is interactive
but only exports HTML. This script renders the same two panels through matplotlib so the
result drops into a figure: vector quads and real text objects. The colorbar stays a
raster image, as matplotlib draws it by default -- it carries no text or geometry that
needs editing.

Nothing is recomputed and nothing is re-decided. Rows, columns, the statistic, the star
criterion, the mask rule and the colour ramp are all imported from 54b/41b, so this figure
cannot disagree with the HTML one:

  colour  = log2_enr          (54b.COLOR_MIN .. 54b.COLOR_MAX, 54b.build_colorscale)
  label   = overlap gene count
  gray    = overlap < MASK_MIN_OVERLAP     (too few shared genes to trust)
  outline = FDR < STAR_FDR AND log2_enr > STAR_LOG2ENR AND overlap >= MASK_MIN_OVERLAP
  blank   = the regulon does not exist in that subclass

Relation to 41d: 41d is the same idea one step earlier in the family -- a matplotlib PDF of
41c's *first* panel, drawn as a dot plot so area could carry marker-set coverage. This one
keeps the heatmap form, because the point of 54b is the colour scale rather than a second
channel, and it keeps both panels.

Reads:
  local_data/res/it/54.<layer>_stratified_enrichment.tsv    (panel 1, via 54b.load_native)
  local_data/res/it/54.l23set_stratified_enrichment.tsv     (panel 2)
  local_data/res/it/54b.enriched_regulon_selection.tsv       (row set + row order)
Outputs:
  local_data/fig/it/54c.stratified_enriched_regulon_heatmap.pdf
"""

import os
import re
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# fonttype 42 embeds TrueType so the PDF keeps real text objects; the default (3) writes
# Type 3 glyph procedures, which Illustrator/Inkscape open as uneditable outlines
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

SCRIPT_41B = os.path.join(SCRIPTS_DIR, 'it', '41b.selected_regulon_archetype_enrichment.py')
SCRIPT_54B = os.path.join(SCRIPTS_DIR, 'it', '54b.stratified_enriched_regulon_heatmap.py')
INPUT_SELECTION = os.path.join(RES_DIR, '54b.enriched_regulon_selection.tsv')
OUT_PDF = os.path.join(FIG_DIR, '54c.stratified_enriched_regulon_heatmap.pdf')

CELL_W, CELL_H = 0.50, 0.21      # inches per heatmap cell
GAP_LW = 0.6                     # white rule between cells (plotly's xgap/ygap)
BOX_LW = 1.6                     # outline width for significant cells
LABEL_SIZE = 7                   # in-cell overlap count

os.makedirs(FIG_DIR, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plotly_to_mpl_cmap(scale):
    """54b's plotly colorscale as a matplotlib colormap, break at zero preserved.

    54b's scale repeats a position (the hard step at log2_enr = 0). matplotlib requires
    strictly increasing stops, so the duplicate is nudged by one part in a million -- visually
    the same discontinuity, and it keeps the two figures on the same colours.
    """
    stops, eps = [], 1e-6
    for pos, css in scale:
        rgb = tuple(int(v) / 255 for v in re.findall(r'\d+', css)[:3])
        if stops and pos <= stops[-1][0]:
            pos = stops[-1][0] + eps
        stops.append((pos, rgb))
    lo, hi = stops[0][0], stops[-1][0]
    stops = [((p - lo) / (hi - lo), c) for p, c in stops]
    return LinearSegmentedColormap.from_list('log2enr', stops)


def draw_panel(ax, mats, rows, cols, primed, m41b, m54b, cmap, norm):
    """One heatmap panel; returns the mappable so the shared colorbar can use it."""
    log2 = mats['log2_enr'].values
    overlap = mats['overlap'].values
    tested = np.isfinite(log2)
    thin = tested & (overlap < m41b.MASK_MIN_OVERLAP)
    sig = (tested & (mats['fdr_strat'].values < m54b.STAR_FDR)
           & (log2 > m54b.STAR_LOG2ENR) & (overlap >= m41b.MASK_MIN_OVERLAP))

    x = np.arange(len(cols) + 1)
    y = np.arange(len(rows) + 1)
    # coloured cells; masked entries fall through to the axes face (white) or the gray layer
    mesh = ax.pcolormesh(x, y, np.ma.masked_where(~(tested & ~thin), log2),
                         cmap=cmap, norm=norm, edgecolors='white', linewidth=GAP_LW)
    if thin.any():
        gray = LinearSegmentedColormap.from_list('mask', [m41b.MASK_COLOR, m41b.MASK_COLOR])
        ax.pcolormesh(x, y, np.ma.masked_where(~thin, np.ones_like(log2)), cmap=gray,
                      vmin=0, vmax=1, edgecolors='white', linewidth=GAP_LW)

    for i, j in zip(*np.nonzero(tested)):
        ax.text(j + 0.5, i + 0.5, f'{int(overlap[i, j])}', ha='center', va='center',
                fontsize=LABEL_SIZE, color=m41b.TEXT_COLOR)
    for i, j in zip(*np.nonzero(sig)):
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=BOX_LW,
                               zorder=5))

    start = 0
    for layer, _token, _label in m41b.LAYER_TOKEN[:-1]:
        start += len(primed[layer])
        ax.axvline(start, color='black', lw=1.5, zorder=6)

    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)          # first row on top, as in the HTML
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('0.6')
    return mesh


def main():
    m41b = load_module(SCRIPT_41B, 'script41b')
    m54b = load_module(SCRIPT_54B, 'script54b')

    primed = m41b.load_primed_labels()
    cols = m41b.column_keys(primed)

    native = m54b.load_native(primed, m41b)
    assert os.path.exists(m54b.INPUT_L23SET), \
        f'missing {m54b.INPUT_L23SET}; run 54 first'
    l23set = m41b.to_col(pd.read_csv(m54b.INPUT_L23SET, sep='\t'), primed)
    native = native[native['regulation_direction'] == m41b.SIGN]
    l23set = l23set[l23set['regulation_direction'] == m41b.SIGN]

    assert os.path.exists(INPUT_SELECTION), \
        f'missing {INPUT_SELECTION}; run 54b.stratified_enriched_regulon_heatmap.py first'
    rows = list(pd.read_csv(INPUT_SELECTION, sep='\t')['TF'])
    print(f'  {len(rows)} regulons x {len(cols)} subclass-archetype columns, 2 panels')

    panels = [("each subclass's own regulons",
               m54b.to_matrices(native, rows, cols, m41b.SIGN)),
              ('L2/3 regulons applied to every subclass',
               m54b.to_matrices(l23set, rows, cols, m41b.SIGN))]

    cmap = plotly_to_mpl_cmap(m54b.build_colorscale())
    norm = Normalize(vmin=m54b.COLOR_MIN, vmax=m54b.COLOR_MAX)

    # constrained layout so the suptitle, the two panel titles and the shared colorbar are
    # packed against the axes instead of against fixed fractions of a tall figure
    fig, axes = plt.subplots(
        len(panels), 1, layout='constrained',
        figsize=(CELL_W * len(cols) + 3.4, len(panels) * (CELL_H * len(rows) + 1.1) + 1.2))
    for ax, (title, mats) in zip(np.atleast_1d(axes), panels):
        mesh = draw_panel(ax, mats, rows, cols, primed, m41b, m54b, cmap, norm)
        ax.set_title(title, fontsize=10, pad=8)
        tested = int(np.isfinite(mats['log2_enr'].values).sum())
        thin = int(((mats['overlap'].values < m41b.MASK_MIN_OVERLAP)
                    & np.isfinite(mats['log2_enr'].values)).sum())
        print(f'  {title}: {tested} of {len(rows) * len(cols)} cells populated, '
              f'{thin} masked (overlap<{m41b.MASK_MIN_OVERLAP})')

    cbar = fig.colorbar(mesh, ax=list(np.atleast_1d(axes)), fraction=0.022, pad=0.02,
                        aspect=40)
    cbar.set_label('log2 enrichment  (observed / expression-matched expectation)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.axhline(0.0, color='black', lw=0.8)     # the meaningful midpoint

    fig.suptitle(
        f'All enriched regulons ({m41b.SIGN}) — archetype marker enrichment across mouse IT '
        f'subclasses (expression-stratified)\n'
        f'rows = regulons starred in >=1 cell, grouped by peak column; cell label = overlap '
        f'gene count; outlined = FDR<{m54b.STAR_FDR:g}, log2 enr>{m54b.STAR_LOG2ENR:g}, '
        f'overlap>={m41b.MASK_MIN_OVERLAP}\n'
        f'gray = overlap<{m41b.MASK_MIN_OVERLAP}, too few shared genes to trust; '
        f'blue = below the matched expectation',
        fontsize=9)

    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
