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
(A'/B'/C') with the largest log2 ENRICHMENT among the cells clearing 54b's star criterion in the
mouse L2/3 stratified enrichment (script 54; overlap = mouse regulon targets ∩ mouse archetype
markers). Human regulons with no mouse orthologue, no mouse regulon, or no clearing archetype are
'other'. Displayed as A/B/C/other.

That criterion and the statistic behind it are imported from 54b/41b rather than restated here,
so this figure's archetype calls cannot disagree with 54d's dot matrix: same expression-stratified
null, same FDR, same log2-enrichment and overlap floors. It replaces the earlier rule, which took
the largest Fisher log2 ODDS RATIO from script 41 at overlap>=5 / log2OR>2 / FDR<0.05. 54 exists
because that odds ratio carries two systematic inflations -- an unstratified null that ignores how
expression level drives both marker and target membership, and the Haldane-Anscombe correction on
thin cells -- so the switch is a tightening: on the yoo25 L2/3 catalogue 57 mouse regulons were
assigned under the old rule and 40 are under this one, with no regulon changing WHICH archetype it
is assigned to. The 17 that drop out become 'other', which now means "not enriched against an
expression-matched background" rather than "not enriched against a background of all genes". On
the human side that moves six regulons -- KLF12, KLF9, MBNL2, SOX5, THRB, ZEB1 -- out of A/B and
into 'other', leaving 9 A / 12 B / 6 C.

Reads:
  local_data/res/l23_evo/05.varimax_loadings.tsv                (human gene x VX loadings)
  local_data/res/l23_evo/18.mouse_varimax_loadings.tsv          (mouse gene x VX; shared-set only)
  data/human_mouse_orthologs.tsv                                (1:1 ortholog pairs)
  local_data/res/l23_evo/24.orthoaxis_cca_weights_human.tsv     (human VX x CCA weights)
  local_data/res/l23_evo/27.human_wang25_regulon_targets.tsv    (human regulons, +/+ targets)
  local_data/res/it/54.L2_3_stratified_enrichment.tsv            (mouse regulon-archetype
      enrichment, expression-stratified; thresholds imported from scripts/it/54b)
Outputs:
  Two representations, each as an all-regulon figure and an A/C-only 2-panel figure (CCA1 panel
  on top, cross-species target-overlap mirror below):
  local_data/fig/l23_evo/61.cca1_regulon_loadings_barplot_human.pdf       (bar: mean loading, all)
  local_data/fig/l23_evo/61.cca1_regulon_loadings_barplot_human_AC.pdf    (bar: mean loading, A/C)
  local_data/fig/l23_evo/61.cca1_regulon_loadings_boxplot_human.pdf       (box: distribution, all)
  local_data/fig/l23_evo/61.cca1_regulon_loadings_boxplot_human_AC.pdf    (box: distribution, A/C)
  local_data/fig/l23_evo/61.cca1_regulon_mean_loading_mouse_vs_human_AC.pdf  (scatter: per-regulon
      mean target CCA1 loading, human vs mouse — mouse loadings from the mouse CCA1 weights)
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

# --- file paths ---
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_HUMAN_VX    = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
IN_MOUSE_VX    = os.path.join(OUT_RES_DIR, '18.mouse_varimax_loadings.tsv')
IN_ORTHOLOGS   = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_CCA_WEIGHTS = os.path.join(OUT_RES_DIR, '24.orthoaxis_cca_weights_human.tsv')
IN_CCA_WEIGHTS_M = os.path.join(OUT_RES_DIR, '24.orthoaxis_cca_weights_mouse.tsv')
IN_HUMAN_REG   = os.path.join(OUT_RES_DIR, '27.human_wang25_regulon_targets.tsv')
IN_MOUSE_REG   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '40.yoo25_L2_3_regulon_targets.tsv')
IN_MOUSE_ENRICH= os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '54.L2_3_stratified_enrichment.tsv')
SCRIPT_54B     = os.path.join(SCRIPTS_DIR, 'it', '54b.stratified_enriched_regulon_heatmap.py')
# Both representations are produced: 'bar' (per-regulon mean loading) and 'box' (per-gene loading
# distribution). full = all regulons (single panel); AC = A/C only (2 panels + overlap).
OUT_PDF_BAR    = os.path.join(OUT_FIG_DIR, '61.cca1_regulon_loadings_barplot_human.pdf')
OUT_PDF_BOX    = os.path.join(OUT_FIG_DIR, '61.cca1_regulon_loadings_boxplot_human.pdf')
OUT_PDF_BAR_AC = os.path.join(OUT_FIG_DIR, '61.cca1_regulon_loadings_barplot_human_AC.pdf')
OUT_PDF_BOX_AC = os.path.join(OUT_FIG_DIR, '61.cca1_regulon_loadings_boxplot_human_AC.pdf')
OUT_PDF_SCATTER = os.path.join(OUT_FIG_DIR, '61.cca1_regulon_mean_loading_mouse_vs_human_AC.pdf')

# --- parameters ---
# The regulon set is UNFILTERED: every activating (+/+) regulon in the Wang25 human table
# (script 27) is built programmatically below (name/mouse/human keys), not hardcoded.
REG_DIRECTION  = '_+/+'   # activating regulons only
# CCA back-projection (must match script 24's HUMAN_VX_COLS / MOUSE_VX_COLS ortholog fit).
HUMAN_VX_COLS  = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
MOUSE_VX_COLS  = ['VX1', 'VX2', 'VX6', 'VX7', 'VX8', 'VX10']
CCA_AXIS       = 'CCA1'
# Mouse archetype enrichment significance: 54b's star criterion, imported rather than restated
# so this script cannot drift from 54d (fdr_strat < STAR_FDR AND log2_enr > STAR_LOG2ENR AND
# overlap >= MASK_MIN_OVERLAP). 'overlap' is the count of genes shared between the MOUSE
# regulon's targets and the MOUSE archetype's marker set (script 54, both restricted to the
# mouse expression universe); the overlap floor is what keeps a knife-edge FDR on a handful of
# shared genes from assigning an archetype (e.g. NFIA: 4-gene overlap yet FDR 3e-4).
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M54B = load_module(SCRIPT_54B, 'script54b')
# Only plot regulons with at least this many target genes among the human HVGs (i.e. carrying a
# CCA1 loading), so each bar's mean rests on enough genes to be meaningful.
MIN_HVG_TARGETS = 10
# For the human-vs-mouse mean-loading scatter: min mouse targets among mouse HVGs for a stable mean.
MIN_MOUSE_TARGETS = 5
# Archetype coloring: mouse arch_letter (A'/B'/C') -> display letter -> color.
ARCH_COLORS    = {'A': 'C0', 'B': 'C1', 'C': 'C2', 'other': '#999999'}
# Cross-species target-overlap panel (A/C figure): human targets up, mouse down, shared core.
HUMAN_COLOR    = '#b2182b'
MOUSE_COLOR    = '#2166ac'
SHARED_COLOR   = '#4d4d4d'
ARCH_ORDER     = ['A', 'B', 'C', 'other']
DPI            = 300
# Mode-dependent labels: 'bar' = per-regulon mean loading, 'box' = per-gene loading distribution.
YLABEL         = {'bar': 'mean target CCA1 loading', 'box': 'target CCA1 loading (per gene)'}
TITLE          = {
    'bar': 'Wang25 human regulons — mean target-gene loading on human CCA1 (Jorstad23×Cheng22)',
    'box': 'Wang25 human regulons — target-gene loading distribution on human CCA1 (Jorstad23×Cheng22)',
}


def criteria_lines(mode, show_log2enr):
    """Criteria/annotation legend text for the given representation (thresholds interpolated)."""
    stat = ('Bar height = mean CCA1 loading across those HVG target genes (sorted by mean)'
            if mode == 'bar' else
            'Box = distribution of CCA1 loadings across those HVG target genes (sorted by median)')
    tip = 'xx/xx (bar tip)' if mode == 'bar' else 'xx/xx (above box)'
    lines = [
        'Regulons: all activating (+/+) Wang25 human regulons',
        f'Plotted only if ≥{MIN_HVG_TARGETS} target genes are human HVGs (i.e. carry a CCA1 loading)',
        stat,
        "Color = mouse archetype of the orthologous TF's mouse regulon, if it clears",
        f'    overlap≥{M54B.MASK_MIN_OVERLAP} AND log2 enr>{M54B.STAR_LOG2ENR:g} AND '
        f'FDR<{M54B.STAR_FDR:g} (script 54, expression-stratified);',
        "    otherwise 'other'.  overlap = mouse regulon targets ∩ mouse archetype markers",
        f'{tip} = HVG target genes with a loading / total target genes in the regulon',
    ]
    if show_log2enr:
        lines.append('bold label at bar base = mouse log2 enrichment of the assigned archetype')
    return lines

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

# --- mouse CCA1 gene loadings: same back-projection with the mouse VX weights (aligned axis) ---
# The CCA jointly fits both species (canonical r>0), so mouse and human CCA1 share an orientation.
shared_sub_m = mouse_vx.loc[shared['mouse_symbol'].values, MOUSE_VX_COLS].values
mu_m = shared_sub_m.mean(axis=0)
sd_m = shared_sub_m.std(axis=0)
w_cca1_m = pd.read_csv(IN_CCA_WEIGHTS_M, sep='\t', index_col=0).loc[MOUSE_VX_COLS, CCA_AXIS].values
cca1_mouse = pd.Series(((mouse_vx[MOUSE_VX_COLS].values - mu_m) / sd_m) @ w_cca1_m, index=mouse_vx.index)

# --- regulon target sets (+/+ only), both species ---
hreg = pd.read_csv(IN_HUMAN_REG, sep='\t')
mreg = pd.read_csv(IN_MOUSE_REG, sep='\t')

# --- unfiltered regulon set: every activating (+/+) human regulon, mapped to its mouse orthologue ---
# human TF -> mouse symbol via the 1:1 ortholog table; mouse key '<Sym>_+/+' (None if no ortholog).
h2m = ortho.set_index('human_symbol')['mouse_symbol'].to_dict()
m2h = ortho.set_index('mouse_symbol')['human_symbol'].to_dict()
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
msig = menr[M54B.starred(menr)]
print(f'Mouse L2/3 stratified enrichment: {len(msig)} of {len(menr)} cells clear the star '
      f'criterion (overlap>={M54B.MASK_MIN_OVERLAP}, log2 enr>{M54B.STAR_LOG2ENR:g}, '
      f'FDR<{M54B.STAR_FDR:g}); {msig["regulon"].nunique()} regulons assignable')


def assign_archetype(mouse_key):
    """(display letter, mouse log2 enrichment) for the top clearing archetype, else ('other', nan).

    Ranked by log2_enr, the same quantity 54d colours its dots by -- so a regulon coloured A here
    is a regulon whose darkest starred dot in 54d's native L2/3 block sits in column A'.
    """
    if mouse_key is None:
        return 'other', float('nan')
    rows = msig[msig['regulon'] == mouse_key]
    if rows.empty:
        return 'other', float('nan')
    top = rows.sort_values('log2_enr', ascending=False).iloc[0]
    return top['arch_letter'].rstrip("'"), float(top['log2_enr'])   # A' -> A, B' -> B, C' -> C


# --- per-regulon CCA1 loading distribution over present targets + archetype color ---
rows = []
for r in REGULONS:
    tgs = targets(hreg, r['human'])
    present = [g for g in tgs if g in cca1.index]
    arch, arch_enr = assign_archetype(r['mouse'])
    if len(present) < MIN_HVG_TARGETS:
        print(f"  {r['name']}: {len(present)}/{len(tgs)} HVG targets < {MIN_HVG_TARGETS} — skipped  (arch {arch})")
        continue
    loadings = cca1.loc[present].values
    print(f"  {r['name']}: {len(present)}/{len(tgs)} targets used, median CCA1 = {np.median(loadings):.4f}  (arch {arch}, log2 enr {arch_enr:.2f})")
    rows.append({'name': r['name'], 'loadings': loadings, 'median_loading': float(np.median(loadings)),
                 'mean_loading': float(np.mean(loadings)),
                 'n_present': len(present), 'n_total': len(tgs), 'arch': arch, 'log2enr': arch_enr,
                 'human': r['human'], 'mouse': r['mouse']})

bar = pd.DataFrame(rows)

# --- boxplots (one box per regulon, sorted by median, colored by mouse archetype) ---
plt.rcParams['pdf.fonttype'] = 42   # editable vector text


def _draw_cca1_panel(ax, bar_df, mode, annot_fs, show_log2enr, criteria_fs, label_fs=None):
    """Draw the per-regulon CCA1 loading panel (colored by archetype) onto ax.

    mode='bar' -> one bar per regulon at its MEAN loading; mode='box' -> a boxplot of the per-gene
    loading distribution. label_fs=None hides x tick labels (shared-x top panel). If show_log2enr,
    mark each column base with the mouse archetype log2 enrichment.
    """
    x = np.arange(len(bar_df))
    sign = bar_df['mean_loading'].values if mode == 'bar' else bar_df['median_loading'].values

    if mode == 'bar':
        colors = [ARCH_COLORS[a] for a in bar_df['arch'].values]
        ax.bar(x, bar_df['mean_loading'].values, color=colors, edgecolor='none', width=0.9)
        pad = 0.02 * np.abs(bar_df['mean_loading'].values).max()
        ytips = bar_df['mean_loading'].values          # n/N annotation sits at the bar tip
    else:
        data = list(bar_df['loadings'].values)
        bp = ax.boxplot(data, positions=x, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=0.8))
        for patch, a in zip(bp['boxes'], bar_df['arch'].values):
            patch.set_facecolor(ARCH_COLORS[a])
            patch.set_edgecolor('0.3')
            patch.set_linewidth(0.4)
            patch.set_alpha(0.9)
        for part in ('whiskers', 'caps'):
            for line in bp[part]:
                line.set(color='0.4', linewidth=0.5)
        allvals = np.concatenate(data)
        pad = 0.015 * (allvals.max() - allvals.min())
        ytips = [bp['caps'][2 * xi + 1].get_ydata()[0] for xi in x]   # above each upper whisker

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlim(-0.6, len(bar_df) - 0.4)
    ax.set_xticks(x)
    if label_fs is None:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(bar_df['name'].values, rotation=90, fontsize=label_fs)
        ax.tick_params(labelbottom=True)   # override sharex auto-hide on the top panel
    ax.set_ylabel(YLABEL[mode])

    # annotate each column with 'n genes with loading / n genes in regulon'
    for xi, (yt, npres, ntot) in enumerate(zip(ytips, bar_df['n_present'].values, bar_df['n_total'].values)):
        va = 'bottom' if (mode == 'box' or sign[xi] >= 0) else 'top'
        ax.text(xi, yt + (pad if va == 'bottom' else -pad), f'{npres}/{ntot}',
                ha='center', va=va, fontsize=annot_fs, color='black', rotation=90)

    # mark each column base with the mouse archetype log2 enrichment (A/C figure)
    if show_log2enr:
        for xi, lor in enumerate(bar_df['log2enr'].values):
            if not np.isfinite(lor):
                continue
            va = 'bottom' if sign[xi] >= 0 else 'top'
            ax.text(xi, (pad if sign[xi] >= 0 else -pad), f'{lor:.1f}',
                    ha='center', va=va, fontsize=annot_fs + 1, color='black',
                    fontweight='bold', rotation=90)

    handles = [Patch(facecolor=ARCH_COLORS[a], label=a) for a in ARCH_ORDER
               if (a in bar_df['arch'].values)]
    ax.legend(handles=handles, title='mouse archetype', frameon=False,
              loc='upper right', fontsize=8, title_fontsize=8)

    # criteria / annotation legend (upper-right, below the color legend; that quadrant is empty
    # because columns are sorted so the right side holds the most-negative values)
    ax.text(0.995, 0.80, '\n'.join(criteria_lines(mode, show_log2enr)),
            transform=ax.transAxes, ha='right', va='top',
            fontsize=criteria_fs, family='monospace', linespacing=1.5,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))
    sns.despine(ax=ax)


def _draw_overlap_mirror(ax, bar_df, label_fs):
    """Mirrored stacked bars of cross-species regulon target counts onto ax.

    Human targets extend up, mouse down; the shared-ortholog core (overlap) sits symmetrically at
    the zero line. Each bar is labeled with the three counts (human up, mouse down, overlap core).
    """
    x = np.arange(len(bar_df))
    nh = bar_df['n_human'].values
    nm = bar_df['n_mouse'].values
    ov = bar_df['overlap'].values
    jac = bar_df['jaccard'].values
    ax.bar(x, ov, color=SHARED_COLOR, width=0.8)                     # shared core (up half)
    ax.bar(x, nh - ov, bottom=ov, color=HUMAN_COLOR, width=0.8)      # human-specific
    ax.bar(x, -ov, color=SHARED_COLOR, width=0.8)                    # shared core (down half)
    ax.bar(x, -(nm - ov), bottom=-ov, color=MOUSE_COLOR, width=0.8)  # mouse-specific
    ax.axhline(0, color='black', linewidth=0.8)

    for xi in x:
        ax.text(xi, nh[xi], f' {int(nh[xi])}', ha='center', va='bottom', fontsize=label_fs - 1,
                color=HUMAN_COLOR, rotation=90)
        ax.text(xi, -nm[xi], f'{int(nm[xi])} ', ha='center', va='top', fontsize=label_fs - 1,
                color=MOUSE_COLOR, rotation=90)
        # overlap count + Jaccard index, centered on the zero line with a white background box
        ax.text(xi, 0, f'{int(ov[xi])}\nJ={jac[xi]:.1%}', ha='center', va='center',
                fontsize=label_fs - 1.5, color='black', fontweight='bold', linespacing=1.2,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=SHARED_COLOR, linewidth=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels(bar_df['name'].values, rotation=90, fontsize=label_fs)
    ax.set_ylabel('regulon target genes\n(← mouse down / human up →)')
    handles = [Patch(facecolor=HUMAN_COLOR, label='human-specific'),
               Patch(facecolor=SHARED_COLOR, label='shared ortholog (overlap)'),
               Patch(facecolor=MOUSE_COLOR, label='mouse-specific')]
    ax.legend(handles=handles, title='box = overlap count & J (Jaccard, ortholog space)',
              frameon=False, loc='upper right', fontsize=7, title_fontsize=7)
    sns.despine(ax=ax)


def draw_full(bar_df, out_pdf, mode, label_fs, annot_fs, width_factor=0.16, criteria_fs=6):
    """Single-panel CCA1 figure (bar or box) over all regulons -> out_pdf."""
    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(max(6.0, width_factor * len(bar_df)), 5.6))
        ax.set_title(TITLE[mode])
        _draw_cca1_panel(ax, bar_df, mode, annot_fs, show_log2enr=False, criteria_fs=criteria_fs,
                         label_fs=label_fs)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)


def draw_ac_with_overlap(bar_df, out_pdf, mode, label_fs, annot_fs, width_factor=0.5, criteria_fs=6):
    """Two stacked panels sharing x: CCA1 panel (bar/box) on top + target-overlap mirror below."""
    with PdfPages(out_pdf) as pdf:
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True,
                                       figsize=(max(6.0, width_factor * len(bar_df)), 8.6),
                                       gridspec_kw={'height_ratios': [3, 2]})
        ax0.set_title(TITLE[mode])
        _draw_cca1_panel(ax0, bar_df, mode, annot_fs, show_log2enr=True, criteria_fs=criteria_fs,
                         label_fs=label_fs)
        _draw_overlap_mirror(ax1, bar_df, label_fs=label_fs)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)


def mouse_mean_loading(mouse_key):
    """(mean mouse CCA1 loading over the mouse regulon's HVG targets, n targets); (nan, 0) if none."""
    if mouse_key is None:
        return float('nan'), 0
    M = set(mreg.loc[mreg['regulon'] == mouse_key, 'Gene'])
    present = [g for g in M if g in cca1_mouse.index]
    return (float(cca1_mouse.loc[present].mean()) if present else float('nan')), len(present)


def draw_mouse_human_scatter(df, out_pdf):
    """Scatter of mean target CCA1 loading, human (x) vs mouse (y), one point per A/C regulon."""
    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        for a in ('A', 'C'):
            sub = df[df['arch'] == a]
            ax.scatter(sub['human_mean'], sub['mouse_mean'], c=ARCH_COLORS[a], s=48, label=a,
                       edgecolor='0.3', linewidth=0.4, zorder=3)
        for _, r in df.iterrows():
            ax.annotate(r['name'], (r['human_mean'], r['mouse_mean']), fontsize=6,
                        xytext=(3, 3), textcoords='offset points')
        ax.axhline(0, color='0.6', linewidth=0.6)
        ax.axvline(0, color='0.6', linewidth=0.6)

        # independent (non-equal-aspect) axis ranges: each species scaled to its own data range
        hmin, hmax = df['human_mean'].min(), df['human_mean'].max()
        mmin, mmax = df['mouse_mean'].min(), df['mouse_mean'].max()
        hpad, mpad = 0.10 * (hmax - hmin), 0.10 * (mmax - mmin)
        ax.set_xlim(hmin - hpad, hmax + hpad)
        ax.set_ylim(mmin - mpad, mmax + mpad)
        ax.set_aspect('auto')   # do NOT force equal aspect; mouse and human use independent scales

        r_p = np.corrcoef(df['human_mean'], df['mouse_mean'])[0, 1]
        r_s = df['human_mean'].corr(df['mouse_mean'], method='spearman')
        ax.set_xlabel('human mean target CCA1 loading')
        ax.set_ylabel('mouse mean target CCA1 loading')
        ax.set_title('A/C regulons — mean target CCA1 loading: human vs mouse')
        ax.legend(title='mouse archetype', frameon=False, loc='lower right', fontsize=8, title_fontsize=8)
        ax.text(0.03, 0.97,
                f'Pearson r = {r_p:.2f}\nSpearman ρ = {r_s:.2f}\n'
                f'n = {len(df)} regulons\n(human ≥{MIN_HVG_TARGETS}, mouse ≥{MIN_MOUSE_TARGETS} HVG targets)',
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))
        sns.despine(ax=ax)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)


def cross_species_overlap(human_key, mouse_key):
    """(n_human, n_mouse, overlap, jaccard) for a regulon's mouse vs human targets.

    overlap = ortholog pairs targeted in both species. jaccard is over the ortholog-comparable
    sets: |H_o ∩ M_h| / |H_o ∪ M_h|, where H_o = human targets with a 1:1 ortholog and M_h = mouse
    targets mapped to their human ortholog (so genes lacking an ortholog can't count as shared).
    """
    H = set(hreg.loc[hreg['regulon'] == human_key, 'Gene'])
    M = set(mreg.loc[mreg['regulon'] == mouse_key, 'Gene']) if mouse_key is not None else set()
    Ho = {g for g in H if g in h2m}
    Mh = {m2h[g] for g in M if g in m2h}
    overlap = len(Ho & Mh)
    union = len(Ho | Mh)
    jaccard = overlap / union if union else 0.0
    return len(H), len(M), overlap, jaccard


# Both representations, each as full (all regulons) + A/C (2-panel with overlap). 'bar' sorts by
# mean loading, 'box' by median.
SORT_KEY = {'bar': 'mean_loading', 'box': 'median_loading'}
for mode, out_full, out_ac in [('bar', OUT_PDF_BAR, OUT_PDF_BAR_AC),
                               ('box', OUT_PDF_BOX, OUT_PDF_BOX_AC)]:
    ordered = bar.sort_values(SORT_KEY[mode], ascending=False).reset_index(drop=True)

    print(f'Writing {out_full}...')
    draw_full(ordered, out_full, mode, label_fs=4, annot_fs=2.5, criteria_fs=11)
    print(f'Saved {out_full}')

    bar_ac = ordered[ordered['arch'].isin(['A', 'C'])].reset_index(drop=True)
    ov = bar_ac.apply(lambda r: cross_species_overlap(r['human'], r['mouse']), axis=1, result_type='expand')
    bar_ac[['n_human', 'n_mouse', 'overlap', 'jaccard']] = ov
    print(f'Writing {out_ac} ({len(bar_ac)} A/C regulons)...')
    draw_ac_with_overlap(bar_ac, out_ac, mode, label_fs=6, annot_fs=4)
    print(f'Saved {out_ac}')

# --- human vs mouse mean target CCA1 loading scatter (A/C regulons) ---
scat = bar[bar['arch'].isin(['A', 'C'])].copy()
scat['human_mean'] = scat['mean_loading']
mm = scat['mouse'].apply(lambda k: pd.Series(mouse_mean_loading(k), index=['mouse_mean', 'mouse_n']))
scat[['mouse_mean', 'mouse_n']] = mm
scat = scat[scat['mouse_n'] >= MIN_MOUSE_TARGETS].reset_index(drop=True)
for _, r in scat.iterrows():
    print(f"  {r['name']}: human_mean={r['human_mean']:.3f} mouse_mean={r['mouse_mean']:.3f} (mouse_n={int(r['mouse_n'])})")
print(f'Writing {OUT_PDF_SCATTER} ({len(scat)} regulons)...')
draw_mouse_human_scatter(scat, OUT_PDF_SCATTER)
print(f'Saved {OUT_PDF_SCATTER}')
print('Done.')
