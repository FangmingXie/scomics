"""54b's two panels as a dot matrix, sized by marker-set coverage (PDF).

54c renders 54b's heatmap as a PDF, in which colour (log2 enrichment) is the only encoded
quantity. This script draws the same two panels as dots and adds a second channel, exactly as
41d does one step earlier in the family:

  colour = log2_enr                    (as in 54b/54c, same [COLOR_MIN, COLOR_MAX] and ramp)
  area   = overlap / n_markers         (what fraction of the archetype's marker set the
                                        regulon's targets cover)

The two answer different questions. log2_enr says how much the overlap beats an
expression-matched background, which a small regulon can win on a handful of genes; the
coverage fraction says how much of the archetype's marker programme the regulon actually
accounts for. A dot that is dark *and* large is a regulon that both concentrates on the
archetype and explains a real share of it.

Difference from 41d, and the reason this script exists rather than a flag on 41d: 41d's colour
is 41's Fisher log2 odds ratio, which carries two systematic inflations (see 54). Here colour
is the expression-stratified log2 enrichment, so the ramp runs below zero -- a dot can now be
blue, meaning the regulon covers FEWER of the archetype's markers than an expression-matched
gene set would, which 41d's zero-floored scale could not express. 41d also drew only the
native panel; this keeps both, as 54b/54c do.

Rows, columns, statistics, ramp and the mask/significance rules are imported from 54b/41b so
this figure cannot disagree with them: gray fill = overlap < MASK_MIN_OVERLAP (too few shared
genes to trust), black outline = the star criterion, nothing drawn where the regulon does not
exist in that subclass.

Reads:
  local_data/res/it/54.<layer>_stratified_enrichment.tsv    (panel 1, via 54b.load_native)
  local_data/res/it/54.l23set_stratified_enrichment.tsv     (panel 2)
  local_data/res/it/54b.enriched_regulon_selection.tsv       (row set + row order)
Outputs:
  local_data/fig/it/54d.stratified_enriched_regulon_dotplot.pdf
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

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

SCRIPT_41B = os.path.join(SCRIPTS_DIR, 'it', '41b.selected_regulon_archetype_enrichment.py')
SCRIPT_54B = os.path.join(SCRIPTS_DIR, 'it', '54b.stratified_enriched_regulon_heatmap.py')
SCRIPT_54C = os.path.join(SCRIPTS_DIR, 'it', '54c.stratified_enriched_regulon_heatmap_pdf.py')
INPUT_SELECTION = os.path.join(RES_DIR, '54b.enriched_regulon_selection.tsv')
OUT_PDF = os.path.join(FIG_DIR, '54d.stratified_enriched_regulon_dotplot.pdf')

# Everything that depends on WHICH regulon catalogue is being drawn lives here, so the 55x
# entry point reuses this module instead of copying it. Thresholds and palette come from 54b
# and are shared by both catalogues -- see the note beside CFG_YOO25 there.
CFG_YOO25 = dict(
    tag='yoo25',
    label='mouse IT subclasses',
    native_tmpl=os.path.join(RES_DIR, '54.{layer}_stratified_enrichment.tsv'),
    l23set=os.path.join(RES_DIR, '54.l23set_stratified_enrichment.tsv'),
    selection=INPUT_SELECTION,
    out_pdf=OUT_PDF,
)

# area encoding: a dot at FRAC_REF covers SIZE_REF points^2, area scaling linearly with the
# fraction so twice the area reads as twice the coverage. FRAC_REF sits just above the
# observed maximum over both panels (0.382, the L2/3-set panel; the native panel reaches 0.353)
FRAC_REF = 0.40
SIZE_REF = 260.0
SIZE_LEGEND = [0.05, 0.15, 0.25, 0.35]
BOX_LW = 1.4             # outline width for significant cells
CELL_W, CELL_H = 0.50, 0.21      # inches per matrix cell, matching 54c

os.makedirs(FIG_DIR, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw_panel(ax, mats, rows, cols, primed, m41b, m54b, cmap, norm):
    """One dot panel; returns the colour mappable for the shared colorbar."""
    log2 = mats['log2_enr'].values
    overlap = mats['overlap'].values
    n_markers = mats['n_markers'].values
    tested = np.isfinite(log2)
    thin = tested & (overlap < m41b.MASK_MIN_OVERLAP)
    sig = (tested & (mats['fdr_strat'].values < m54b.STAR_FDR)
           & (log2 > m54b.STAR_LOG2ENR) & (overlap >= m41b.MASK_MIN_OVERLAP))

    frac = np.divide(overlap, n_markers, out=np.full_like(log2, np.nan), where=tested)
    assert np.nanmax(frac) <= FRAC_REF + 1e-9, \
        f'coverage {np.nanmax(frac):.3f} exceeds the size reference {FRAC_REF}'

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

    start = 0
    for layer, _token, _label in m41b.LAYER_TOKEN[:-1]:
        start += len(primed[layer])
        ax.axvline(start - 0.5, color='black', lw=1.2, zorder=2)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(len(rows) - 0.4, -0.6)          # first row on top
    ax.set_axisbelow(True)
    ax.grid(True, color='0.90', lw=0.5, zorder=0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('0.6')
    return mappable


def main(cfg=CFG_YOO25):
    m41b = load_module(SCRIPT_41B, 'script41b')
    m54b = load_module(SCRIPT_54B, 'script54b')
    m54c = load_module(SCRIPT_54C, 'script54c')   # for the plotly -> matplotlib ramp

    primed = m41b.load_primed_labels()
    cols = m41b.column_keys(primed)

    native = m54b.load_native(primed, m41b, cfg)
    assert os.path.exists(cfg['l23set']), \
        f'missing {cfg["l23set"]}; run the {cfg["tag"]} enrichment script first'
    l23set = m41b.to_col(pd.read_csv(cfg['l23set'], sep='\t'), primed)
    native = native[native['regulation_direction'] == m41b.SIGN]
    l23set = l23set[l23set['regulation_direction'] == m41b.SIGN]

    assert os.path.exists(cfg['selection']), \
        f'missing {cfg["selection"]}; run the {cfg["tag"]} heatmap script first'
    rows = list(pd.read_csv(cfg['selection'], sep='\t')['TF'])
    print(f'  {len(rows)} regulons x {len(cols)} subclass-archetype columns, 2 panels')

    panels = [("each subclass's own regulons",
               m54b.to_matrices(native, rows, cols, m41b.SIGN)),
              ('L2/3 regulons applied to every subclass',
               m54b.to_matrices(l23set, rows, cols, m41b.SIGN))]

    cmap = m54c.plotly_to_mpl_cmap(m54b.build_colorscale())
    norm = Normalize(vmin=m54b.COLOR_MIN, vmax=m54b.COLOR_MAX)

    fig, axes = plt.subplots(
        len(panels), 1, layout='constrained',
        figsize=(CELL_W * len(cols) + 4.6, len(panels) * (CELL_H * len(rows) + 1.1) + 1.2))
    axes = np.atleast_1d(axes)
    for ax, (title, mats) in zip(axes, panels):
        mappable = draw_panel(ax, mats, rows, cols, primed, m41b, m54b, cmap, norm)
        ax.set_title(title, fontsize=10, pad=8)
        tested = np.isfinite(mats['log2_enr'].values)
        thin = int((tested & (mats['overlap'].values < m41b.MASK_MIN_OVERLAP)).sum())
        frac = mats['overlap'].values / mats['n_markers'].values
        print(f'  {title}: {int(tested.sum())} of {len(rows) * len(cols)} cells populated, '
              f'{thin} masked; coverage max={np.nanmax(frac):.3f}, '
              f'unmasked median={np.nanmedian(frac[tested & ~(mats["overlap"].values < m41b.MASK_MIN_OVERLAP)]):.3f}')

    cbar = fig.colorbar(mappable, ax=list(axes), fraction=0.020, pad=0.02, aspect=40)
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
    axes[0].legend(handles=handles, title='overlap / n_markers', loc='upper left',
                   bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7, title_fontsize=8,
                   labelspacing=1.0, borderpad=0.6)

    fig.suptitle(
        f'All enriched regulons ({m41b.SIGN}) — archetype marker enrichment across '
        f'{cfg["label"]} (expression-stratified, {cfg["tag"]} regulons)\n'
        f'colour = log2 enrichment, area = fraction of the archetype marker set covered; '
        f'outlined = FDR<{m54b.STAR_FDR:g}, log2 enr>{m54b.STAR_LOG2ENR:g}, '
        f'overlap>={m41b.MASK_MIN_OVERLAP}\n'
        f'gray = overlap<{m41b.MASK_MIN_OVERLAP}, too few shared genes to trust; '
        f'blue = below the matched expectation',
        fontsize=9)

    fig.savefig(cfg['out_pdf'], bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {cfg['out_pdf']}")


if __name__ == '__main__':
    main()
