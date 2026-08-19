"""Barplot of each regulon's mean target-gene loading on the human CCA1 axis — Jorstad23 L2/3 IT.

For EVERY activating (+/+) human Wang25 regulon (unfiltered — not the pre-selected 32 of script
42), this summarizes each regulon by the MEAN, across its target genes, of each gene's loading
on the human CCA1 axis. One bar per regulon, sorted descending, colored by the regulon's mouse
archetype enrichment (A/B/C vs other). Only regulons with >= MIN_HVG_TARGETS target genes among
the human HVGs (i.e. carrying a CCA1 loading) are plotted.

CCA1 = axis 1 of the cross-species (mouse Cheng22 vs human Jorstad23) canonical-correlation
matching of varimax (VX) loadings, from script 24. Script 24 fits CCA on 521 shared 1:1
orthologs and defines a per-gene human score as X @ x_weights (X = z-scored human VX loadings).
Here we EXTEND that score to all 2000 human HVGs by projecting each gene's VX loadings onto the
human CCA1 weight vector, standardizing with the same shared-ortholog z-score params the CCA was
fit on. No sign flip is applied (CCA axis sign is arbitrary; used as-is).

Bars are colored by the regulon's MOUSE archetype: the human TF is mapped to its mouse orthologue
(data/human_mouse_orthologs.tsv), and the mouse '<Sym>_+/+' regulon is assigned to the archetype
(A'/B'/C') with the largest log2 odds ratio among rows clearing overlap>=5 AND log2OR>2.0 AND
FDR<0.05 in the mouse L2/3 regulon-archetype enrichment (script 41; overlap = mouse regulon
targets ∩ mouse archetype markers). The overlap floor guards against small-N OR inflation. Human
regulons with no mouse orthologue, no mouse regulon, or no clearing archetype are 'other'.
Displayed as A/B/C/other.

Reads:
  local_data/res/l23_evo/05.varimax_loadings.tsv                (human gene x VX loadings)
  local_data/res/l23_evo/18.mouse_varimax_loadings.tsv          (mouse gene x VX; shared-set only)
  data/human_mouse_orthologs.tsv                                (1:1 ortholog pairs)
  local_data/res/l23_evo/24.orthoaxis_cca_weights_human.tsv     (human VX x CCA weights)
  local_data/res/l23_evo/27.human_wang25_regulon_targets.tsv    (human regulons, +/+ targets)
  local_data/res/it/41.L2_3_regulon_archetype_enrichment.tsv    (mouse regulon-archetype enrichment)
Outputs:
  local_data/fig/l23_evo/61.cca1_regulon_loadings_barplot_human.pdf      (all regulons)
  local_data/fig/l23_evo/61.cca1_regulon_loadings_barplot_human_AC.pdf   (A/C-colored only)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_HUMAN_VX    = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
IN_MOUSE_VX    = os.path.join(OUT_RES_DIR, '18.mouse_varimax_loadings.tsv')
IN_ORTHOLOGS   = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_CCA_WEIGHTS = os.path.join(OUT_RES_DIR, '24.orthoaxis_cca_weights_human.tsv')
IN_HUMAN_REG   = os.path.join(OUT_RES_DIR, '27.human_wang25_regulon_targets.tsv')
IN_MOUSE_ENRICH= os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '41.L2_3_regulon_archetype_enrichment.tsv')
OUT_PDF_BAR    = os.path.join(OUT_FIG_DIR, '61.cca1_regulon_loadings_barplot_human.pdf')
OUT_PDF_AC     = os.path.join(OUT_FIG_DIR, '61.cca1_regulon_loadings_barplot_human_AC.pdf')

# --- parameters ---
# The regulon set is UNFILTERED: every activating (+/+) regulon in the Wang25 human table
# (script 27) is built programmatically below (name/mouse/human keys), not hardcoded.
REG_DIRECTION  = '_+/+'   # activating regulons only
# CCA back-projection (must match script 24's HUMAN_VX_COLS / MOUSE_VX_COLS ortholog fit).
HUMAN_VX_COLS  = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
MOUSE_VX_COLS  = ['VX1', 'VX2', 'VX6', 'VX7', 'VX8', 'VX10']
CCA_AXIS       = 'CCA1'
# Mouse archetype enrichment significance (matches script 42/44 thresholds), plus a minimum
# overlap so small-N regulons aren't assigned on an inflated OR / knife-edge FDR. 'overlap' is
# the count of genes shared between the MOUSE regulon's targets and the MOUSE archetype's marker
# set (script 41, both restricted to the mouse expression universe). log2OR/FDR alone don't fix
# small-N inflation (e.g. NFIA: 4-gene overlap yet FDR 3e-4); the overlap count does.
LOG2OR_THRESH  = 2.0
FDR_THRESH     = 0.05
MIN_OVERLAP    = 5
# Only plot regulons with at least this many target genes among the human HVGs (i.e. carrying a
# CCA1 loading), so each bar's mean rests on enough genes to be meaningful.
MIN_HVG_TARGETS = 10
# Archetype coloring: mouse arch_letter (A'/B'/C') -> display letter -> color.
ARCH_COLORS    = {'A': 'C0', 'B': 'C1', 'C': 'C2', 'other': '#999999'}
ARCH_ORDER     = ['A', 'B', 'C', 'other']
DPI            = 300
YLABEL         = 'mean target CCA1 loading'
TITLE          = 'Wang25 human regulons — mean target-gene loading on human CCA1 (Jorstad23×Cheng22)'
# Criteria / annotation legend shown on both figures (values interpolate the thresholds above).
CRITERIA_LINES = [
    'Regulons: all activating (+/+) Wang25 human regulons',
    f'Plotted only if ≥{MIN_HVG_TARGETS} target genes are human HVGs (i.e. carry a CCA1 loading)',
    'Bar height = mean CCA1 loading across those HVG target genes',
    "Color = mouse archetype of the orthologous TF's mouse regulon, if it clears",
    f'    overlap≥{MIN_OVERLAP} AND log2OR>{LOG2OR_THRESH:g} AND FDR<{FDR_THRESH:g} (mouse enrichment, script 41);',
    "    otherwise 'other'.  overlap = mouse regulon targets ∩ mouse archetype markers",
    'xx/xx (bar tip) = HVG target genes with a loading / total target genes in the regulon',
]

os.makedirs(OUT_FIG_DIR, exist_ok=True)


def targets(df, key):
    return set(df.loc[df['regulon'] == key, 'Gene'])


# --- human CCA1 gene loadings: project ALL human HVGs onto the CCA1 direction ---
# X @ w as in script 24 (line 128), but standardizing every HVG with the shared-ortholog
# z-score params (mean/std over the 521 genes the CCA was fit on).
human_vx = pd.read_csv(IN_HUMAN_VX, sep='\t', index_col=0)
mouse_vx = pd.read_csv(IN_MOUSE_VX, sep='\t', index_col=0)
ortho = pd.read_csv(IN_ORTHOLOGS, sep='\t').drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
shared = ortho[ortho['human_symbol'].isin(human_vx.index) & ortho['mouse_symbol'].isin(mouse_vx.index)]
shared_sub = human_vx.loc[shared['human_symbol'].values, HUMAN_VX_COLS].values
mu = shared_sub.mean(axis=0)
sd = shared_sub.std(axis=0)                       # ddof=0, matching scipy.stats.zscore in script 24
w_cca1 = pd.read_csv(IN_CCA_WEIGHTS, sep='\t', index_col=0).loc[HUMAN_VX_COLS, CCA_AXIS].values
allz = (human_vx[HUMAN_VX_COLS].values - mu) / sd
cca1 = pd.Series(allz @ w_cca1, index=human_vx.index)   # per-gene human CCA1 loading (all 2000 HVG)
print(f'Human CCA1 gene loadings: {len(cca1)} HVGs (shared-ortholog fit n={len(shared)})')

# --- regulon target sets (+/+ only) ---
hreg = pd.read_csv(IN_HUMAN_REG, sep='\t')

# --- unfiltered regulon set: every activating (+/+) human regulon, mapped to its mouse orthologue ---
# human TF -> mouse symbol via the 1:1 ortholog table; mouse key '<Sym>_+/+' (None if no ortholog).
h2m = ortho.set_index('human_symbol')['mouse_symbol'].to_dict()
tf_by_reg = hreg.drop_duplicates('regulon').set_index('regulon')['TF']
REGULONS = []
for hk in sorted(r for r in hreg['regulon'].unique() if r.endswith(REG_DIRECTION)):
    msym = h2m.get(tf_by_reg[hk])
    REGULONS.append({'name': tf_by_reg[hk],
                     'mouse': f'{msym}{REG_DIRECTION}' if msym is not None else None,
                     'human': hk})
print(f'Unfiltered {REG_DIRECTION} human regulons: {len(REGULONS)}')

# --- mouse archetype assignment per regulon (top significant archetype, or 'other') ---
menr = pd.read_csv(IN_MOUSE_ENRICH, sep='\t')
msig = menr[(menr['overlap'] >= MIN_OVERLAP) &
            (menr['log2_or'] > LOG2OR_THRESH) & (menr['fdr'] < FDR_THRESH)]


def assign_archetype(mouse_key):
    """Return (display letter, mouse log2 OR) for the top clearing archetype, else ('other', nan)."""
    if mouse_key is None:
        return 'other', float('nan')
    rows = msig[msig['regulon'] == mouse_key]
    if rows.empty:
        return 'other', float('nan')
    top = rows.sort_values('log2_or', ascending=False).iloc[0]
    return top['arch_letter'].rstrip("'"), float(top['log2_or'])   # A' -> A, B' -> B, C' -> C


# --- per-regulon mean CCA1 loading over present targets + archetype color ---
rows = []
for r in REGULONS:
    tgs = targets(hreg, r['human'])
    present = [g for g in tgs if g in cca1.index]
    arch, arch_or = assign_archetype(r['mouse'])
    if len(present) < MIN_HVG_TARGETS:
        print(f"  {r['name']}: {len(present)}/{len(tgs)} HVG targets < {MIN_HVG_TARGETS} — skipped  (arch {arch})")
        continue
    mean_loading = cca1.loc[present].mean()
    print(f"  {r['name']}: {len(present)}/{len(tgs)} targets used, mean CCA1 = {mean_loading:.4f}  (arch {arch}, log2OR {arch_or:.2f})")
    rows.append({'name': r['name'], 'mean_loading': mean_loading,
                 'n_present': len(present), 'n_total': len(tgs), 'arch': arch, 'log2or': arch_or})

bar = pd.DataFrame(rows).sort_values('mean_loading', ascending=False).reset_index(drop=True)

# --- barplot (one bar per regulon, sorted descending, colored by mouse archetype) ---
plt.rcParams['pdf.fonttype'] = 42   # editable vector text


def draw_barplot(bar_df, out_pdf, label_fs, annot_fs, width_factor=0.16, show_log2or=False):
    """One bar per regulon (sorted descending), colored by mouse archetype -> out_pdf.

    If show_log2or, mark each bar at its base with the mouse log2 OR of its assigned archetype.
    """
    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(max(6.0, width_factor * len(bar_df)), 5.6))
        x = range(len(bar_df))
        colors = [ARCH_COLORS[a] for a in bar_df['arch'].values]
        ax.bar(x, bar_df['mean_loading'].values, color=colors, edgecolor='none', width=0.9)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(bar_df['name'].values, rotation=90, fontsize=label_fs)
        ax.set_ylabel(YLABEL)
        ax.set_title(TITLE)

        # annotate each bar tip with 'n genes with loading / n genes in regulon'
        ymax = bar_df['mean_loading'].abs().max()
        pad = 0.02 * ymax
        for xi, (val, npres, ntot) in enumerate(zip(bar_df['mean_loading'].values,
                                                    bar_df['n_present'].values, bar_df['n_total'].values)):
            va = 'bottom' if val >= 0 else 'top'
            ax.text(xi, val + (pad if val >= 0 else -pad), f'{npres}/{ntot}',
                    ha='center', va=va, fontsize=annot_fs, color='black', rotation=90)

        # mark each bar base with the mouse archetype log2 OR (A/C figure)
        if show_log2or:
            for xi, (val, lor) in enumerate(zip(bar_df['mean_loading'].values, bar_df['log2or'].values)):
                if not np.isfinite(lor):
                    continue
                va = 'bottom' if val >= 0 else 'top'
                ax.text(xi, (pad if val >= 0 else -pad), f'{lor:.1f}',
                        ha='center', va=va, fontsize=annot_fs + 1, color='black',
                        fontweight='bold', rotation=90)

        handles = [Patch(facecolor=ARCH_COLORS[a], label=a) for a in ARCH_ORDER
                   if (a in bar_df['arch'].values)]
        ax.legend(handles=handles, title='mouse archetype', frameon=False,
                  loc='upper right', fontsize=8, title_fontsize=8)

        # criteria / annotation legend (upper-right, below the color legend; that quadrant is empty
        # because bars are sorted so the right side holds the most-negative bars)
        lines = list(CRITERIA_LINES)
        if show_log2or:
            lines.append('bold label at bar base = mouse log2 OR of the assigned archetype')
        ax.text(0.995, 0.80, '\n'.join(lines), transform=ax.transAxes, ha='right', va='top',
                fontsize=6, family='monospace', linespacing=1.5,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))

        sns.despine(ax=ax)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)


# full set (all 174 regulons)
print(f'Writing {OUT_PDF_BAR}...')
draw_barplot(bar, OUT_PDF_BAR, label_fs=4, annot_fs=2.5)
print(f'Saved {OUT_PDF_BAR}')

# A/C-only subset (drop B and 'other'); mark each bar with mouse archetype log2 OR
bar_ac = bar[bar['arch'].isin(['A', 'C'])].reset_index(drop=True)
print(f'Writing {OUT_PDF_AC} ({len(bar_ac)} A/C regulons)...')
draw_barplot(bar_ac, OUT_PDF_AC, label_fs=6, annot_fs=4, width_factor=0.5, show_log2or=True)
print(f'Saved {OUT_PDF_AC}')
print('Done.')
