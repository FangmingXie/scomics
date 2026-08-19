"""Native-panel enrichment as a dot plot, sized by marker-set coverage (PDF).

41c draws two heatmaps in which color (log2 OR) is the only encoded quantity. This script
renders just its first panel -- each subclass's own regulons against its own archetype
markers -- as a dot matrix that adds a second channel:

  color = log2 OR                      (as in 41b/41c, same fixed [COLOR_MIN, COLOR_MAX])
  area  = overlap / n_markers          (what fraction of the archetype's marker set the
                                        regulon's targets cover)

The two answer different questions. log2 OR says how much the overlap beats chance, which
a small regulon can win on a handful of genes; the coverage fraction says how much of the
archetype's marker programme the regulon actually accounts for. A cell that is dark *and*
large is a regulon that both concentrates on the archetype and explains a real share of it.

Rows, columns, statistics, colormap and the mask/significance rules are all taken from
41b/41c so this figure cannot disagree with them: gray fill = overlap < MASK_MIN_OVERLAP
(too few shared genes to trust), black outline = the star criterion, nothing drawn where
the subclass has no regulon for that TF.

Output is PDF (vector, for figure assembly) rather than HTML -- this is the only script in
the 41 family that does not go through plotly.

Reads:
  local_data/res/it/41.<layer>_regulon_archetype_enrichment.tsv   (via 41b.load_native)
  local_data/res/it/41c.enriched_regulon_selection.tsv            (row set + row order)
Outputs:
  local_data/fig/it/41d.native_regulon_archetype_dotplot.pdf
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

SCRIPT_41B = os.path.join(SCRIPTS_DIR, 'it', '41b.selected_regulon_archetype_enrichment.py')
INPUT_SELECTION = os.path.join(RES_DIR, '41c.enriched_regulon_selection.tsv')
OUT_PDF = os.path.join(FIG_DIR, '41d.native_regulon_archetype_dotplot.pdf')

# area encoding: a dot at FRAC_REF covers SIZE_REF points^2, area scaling linearly with the
# fraction so twice the area reads as twice the coverage
FRAC_REF = 0.40          # just above the observed maximum coverage (0.353)
SIZE_REF = 300.0
SIZE_LEGEND = [0.05, 0.15, 0.25, 0.35]
BOX_LW = 1.6             # outline width for significant cells

os.makedirs(FIG_DIR, exist_ok=True)


def load_41b():
    spec = importlib.util.spec_from_file_location('script41b', SCRIPT_41B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def truncated_cmap(m41b):
    """41b's plotly ramp as a matplotlib colormap (both are ColorBrewer YlOrRd)."""
    base = plt.get_cmap('YlOrRd')
    return LinearSegmentedColormap.from_list('YlOrRd_trunc', base(np.linspace(0, 0.74, 256)))


def main():
    m41b = load_41b()
    primed = m41b.load_primed_labels()
    cols = m41b.column_keys(primed)

    native = m41b.load_native(primed)
    native = native[native['regulation_direction'] == m41b.SIGN]

    assert os.path.exists(INPUT_SELECTION), \
        f'missing {INPUT_SELECTION}; run 41c.enriched_regulon_archetype_enrichment.py first'
    rows = list(pd.read_csv(INPUT_SELECTION, sep='\t')['TF'])
    print(f'  {len(rows)} regulons x {len(cols)} subclass-archetype columns (native panel)')

    mats = m41b.to_matrices(native, rows, cols)
    log2or = mats['log2_or'].values
    overlap = mats['overlap'].values
    n_markers = native.pivot_table(index='TF', columns='col', values='n_markers',
                                   aggfunc='first').reindex(index=rows, columns=cols).values

    tested = np.isfinite(log2or)
    thin = tested & (overlap < m41b.MASK_MIN_OVERLAP)
    sig = (tested & (mats['fdr'].values < m41b.STAR_FDR)
           & (log2or > m41b.STAR_LOG2OR) & (overlap >= m41b.STAR_MIN_OVERLAP))
    frac = np.divide(overlap, n_markers, out=np.full_like(log2or, np.nan), where=tested)
    print(f'  {int(tested.sum())} cells populated, {int(thin.sum())} masked, '
          f'{int(sig.sum())} significant')
    print(f'  coverage overlap/n_markers: max={np.nanmax(frac):.3f} '
          f'(size reference {FRAC_REF}); unmasked median={np.nanmedian(frac[tested & ~thin]):.3f}')
    assert np.nanmax(frac) <= FRAC_REF + 1e-9, \
        f'coverage {np.nanmax(frac):.3f} exceeds the size reference {FRAC_REF}'

    cmap = truncated_cmap(m41b)
    norm = Normalize(vmin=m41b.COLOR_MIN, vmax=m41b.COLOR_MAX)

    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 5.2, 0.30 * len(rows) + 2.6))
    yy, xx = np.nonzero(tested)
    sizes = frac[yy, xx] / FRAC_REF * SIZE_REF

    # masked and colored dots are drawn separately so the gray fill is not run through cmap
    for keep, kw in [(thin[yy, xx], dict(color=m41b.MASK_COLOR)),
                     (~thin[yy, xx], dict(c=log2or[yy, xx][~thin[yy, xx]], cmap=cmap, norm=norm))]:
        if not keep.any():
            continue
        edge = np.where(sig[yy, xx][keep], 'black', 'none')
        lw = np.where(sig[yy, xx][keep], BOX_LW, 0.0)
        sc = ax.scatter(xx[keep], yy[keep], s=sizes[keep], edgecolors=edge, linewidths=lw,
                        zorder=3, **kw)
        if 'c' in kw:
            mappable = sc

    # subclass block separators
    start = 0
    for layer, _token, _label in m41b.LAYER_TOKEN[:-1]:
        start += len(primed[layer])
        ax.axvline(start - 0.5, color='black', lw=1.2, zorder=2)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(len(rows) - 0.4, -0.6)          # first row on top
    ax.set_axisbelow(True)
    ax.grid(True, color='0.90', lw=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('0.6')

    cbar = fig.colorbar(mappable, ax=ax, fraction=0.030, pad=0.02, aspect=28)
    cbar.set_label('log2 odds ratio', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    handles = [Line2D([], [], marker='o', linestyle='none', markerfacecolor='0.55',
                      markeredgecolor='none', markersize=np.sqrt(f / FRAC_REF * SIZE_REF),
                      label=f'{f:.2f}') for f in SIZE_LEGEND]
    handles += [Line2D([], [], marker='o', linestyle='none', markerfacecolor=m41b.MASK_COLOR,
                       markeredgecolor='none', markersize=8,
                       label=f'overlap<{m41b.MASK_MIN_OVERLAP}'),
                Line2D([], [], marker='o', linestyle='none', markerfacecolor='white',
                       markeredgecolor='black', markeredgewidth=BOX_LW, markersize=8,
                       label='significant')]
    ax.legend(handles=handles, title='overlap / n_markers', loc='upper left',
              bbox_to_anchor=(1.13, 1.0), frameon=False, fontsize=8, title_fontsize=9,
              labelspacing=1.1, borderpad=0.8)

    ax.set_title(f"Enriched regulons ({m41b.SIGN}) vs archetype markers — each subclass's own "
                 f'regulons\ncolor = log2 OR, area = fraction of the archetype marker set '
                 f'covered; outlined = FDR<{m41b.STAR_FDR:g}, '
                 f'log2 OR>{m41b.STAR_LOG2OR:g}, overlap>={m41b.STAR_MIN_OVERLAP}',
                 fontsize=10, pad=12)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
