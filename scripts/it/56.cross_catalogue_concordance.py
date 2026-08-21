"""plan/it/56 — do the yoo25 (54) and gao25 (55) catalogues agree, and is L2/3 really special?

Q1 (concordance) and Q2 (per-layer power) are one script because they are one question asked
twice: both are only answerable after conditioning on whether a comparison had the power to
come out either way, and both use the same two long-table families.

54 and 55 run the same expression-stratified enrichment against two independent SCENIC+
regulon catalogues over the same archetype marker sets. A cell is one
(layer, archetype, TF) triple at the activating sign, and a cell is *shared* when the TF's
activating regulon exists in both catalogues for that layer.

The whole point of this script is that the honest agreement statistic is conditional on
both catalogues having something to say. Two conditions have to be separated:

  thick in BOTH   overlap >= MASK_MIN_OVERLAP on each side -- both `log2_enr` values rest
                  on enough shared genes to mean anything. This is where a correlation is
                  legitimate.
  shared          the regulon merely exists in both. Most such cells sit on 0-4 genes in
                  one or both catalogues, which is exactly what 41b's MASK_MIN_OVERLAP was
                  introduced to gray out of the heatmaps; correlating over them measures
                  the noise floor, not the agreement.

Quoting the second number alone understates the agreement badly, so both are always
reported together, and the scatter draws the excluded cells in gray rather than dropping
them.

The same conditioning drives the replication cascade. A yoo25 hit whose TF has no gao25
regulon in that layer is *untestable*, not contradicted; only 53% of yoo25's significant
cells are even testable, so a raw "44% replicate" figure is mostly reporting catalogue
composition. The asymmetry that survives -- gao25's hits are close to a subset of yoo25's
-- is what a smaller, more conservative catalogue should look like.

--- Q2 -----------------------------------------------------------------------------------

L2/3 carries far more significant cells than the other layers, which invites the reading
that L2/3 archetypes are more transcription-factor-driven. They are not, and cell number is
not the cause either -- L4 has the most cells of any layer and a lower hit rate. The driver
is `exp`, the overlap expected under 54's stratified null (sum_b m_b t_b / n_b), which is a
pure power quantity: a cell with exp = 0.2 cannot reach overlap >= 5 and significance no
matter what the biology is.

Stratify the hit rate on `exp` and the L2/3 advantage disappears -- in the top quartile L6IT
beats it -- so the panel below reports rates within `exp` quartiles and never raw counts.
`exp` has two factors that must not be treated alike when interpreting the result:

  |T| regulon size    a property of the SCENIC+ run, not of the layer. Pure nuisance.
  |M| marker-set size partly biology -- how distinct that layer's archetypes are. Controlling
                      it away discards signal, so the table reports it rather than hiding it.

Nothing is recomputed. Every number here is read off the tables 54 and 55 wrote, and
significance uses 54b's three provisional cutoffs unchanged (imported, not restated, so
this file cannot drift from the figures).

Reads:
  local_data/res/it/54.<layer>_stratified_enrichment.tsv    (yoo25, native regulons)
  local_data/res/it/55.<layer>_stratified_enrichment.tsv    (gao25, native regulons)
Outputs:
  local_data/res/it/56.shared_cell_concordance.tsv     one row per shared cell, both sides
  local_data/res/it/56.concordance_stats.tsv           r / rho / median |delta|, overall + per layer
  local_data/res/it/56.replication_cascade.tsv         significant -> testable -> replicating
  local_data/res/it/56.shared_significant_tfs.tsv      per-archetype hit TFs, with an IEG flag
  local_data/res/it/56.validation_power_by_layer.tsv    Q2: hit rate per (catalogue, layer, exp quartile)
  local_data/fig/it/56.cross_catalogue_concordance.pdf  Q1: scatter + cascade
  local_data/fig/it/56.validation_power_by_layer.pdf    Q2: hit rate vs exp quartile, two panels
"""

import os
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
# fonttype 42 embeds TrueType so the PDF keeps real text objects; the default (3) writes
# Type 3 glyph procedures, which Illustrator/Inkscape open as uneditable outlines
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

SCRIPT_41B = os.path.join(SCRIPTS_DIR, 'it', '41b.selected_regulon_archetype_enrichment.py')
SCRIPT_54B = os.path.join(SCRIPTS_DIR, 'it', '54b.stratified_enriched_regulon_heatmap.py')
INPUT_YOO25_TMPL = os.path.join(RES_DIR, '54.{layer}_stratified_enrichment.tsv')
INPUT_GAO25_TMPL = os.path.join(RES_DIR, '55.{layer}_stratified_enrichment.tsv')

OUT_CELLS = os.path.join(RES_DIR, '56.shared_cell_concordance.tsv')
OUT_STATS = os.path.join(RES_DIR, '56.concordance_stats.tsv')
OUT_CASCADE = os.path.join(RES_DIR, '56.replication_cascade.tsv')
OUT_SHARED_TFS = os.path.join(RES_DIR, '56.shared_significant_tfs.tsv')
OUT_POWER = os.path.join(RES_DIR, '56.validation_power_by_layer.tsv')
OUT_PDF = os.path.join(FIG_DIR, '56.cross_catalogue_concordance.pdf')
OUT_POWER_PDF = os.path.join(FIG_DIR, '56.validation_power_by_layer.pdf')

# A cell is keyed by the subclass, the archetype within it, and the TF. The sign is fixed
# to 41b's activating direction, so it is a filter rather than part of the key.
KEY = ['layer', 'arch_letter', 'TF']
SIGN = '+/+'
# L5IT's shared subset is 16 cells; per-layer statistics are meaningless without their n,
# so every per-layer figure this script emits carries one (plan 56, Q1 caveat).
MIN_N_FOR_STAT = 3

# 41b's SELECTED_TFS is an IEG/*stress* set: eight classic immediate-early TFs (AP-1 and Egr)
# plus Atf6 (ER stress) and Smad3 (TGF-beta), which are neither immediate nor early. Any claim
# of the form "IEG regulons account for X% of the hits" has to say which set it means -- the
# two differ by 5 percentage points in yoo25 and 11 in gao25 -- so the strict core is used for
# the headline fraction and the broader set is reported alongside it.
IEG_STRESS_ONLY = ['Atf6', 'Smad3']

# Q2's `exp` quartiles are cut over ALL of one catalogue's activating cells, not within each
# layer. That is the whole point -- with common edges, comparing layers inside a quartile
# compares them at matched power, which per-layer edges would not do (L5IT's top quartile
# starts below L6IT's median). The edges do differ BETWEEN catalogues, so a quartile is
# comparable across layers but not across catalogues; only the overall rates are.
N_EXP_QUANTILES = 4

LAYER_COLOR = {'L2/3': '#4c72b0', 'L4': '#dd8452', 'L5IT': '#55a868', 'L6IT': '#c44e52'}
GRAY = '#c8c8c8'

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_catalogue(tmpl, m41b, tag):
    """The four native per-layer tables of one catalogue, activating regulons only."""
    frames = []
    for layer, _token, _label in m41b.LAYER_TOKEN:
        path = tmpl.format(layer=layer)
        assert os.path.exists(path), f'missing {path}; run the {tag} enrichment script first'
        frames.append(pd.read_csv(path, sep='\t'))
    long = pd.concat(frames, ignore_index=True)
    long = long[long['regulation_direction'] == SIGN].copy()
    label = {layer: lbl for layer, _t, lbl in m41b.LAYER_TOKEN}
    long['layer_label'] = long['layer'].map(label)
    assert not long.duplicated(KEY).any(), f'{tag}: duplicate (layer, archetype, TF) cells'
    return long


def merge_catalogues(yoo, gao, m54b):
    """One row per shared cell, plus the thickness/significance flags everything else uses."""
    cols = KEY + ['layer_label', 'archetype', 'overlap', 'n_markers', 'n_targets', 'exp',
                  'log2_enr', 'fdr_strat']
    m = yoo[cols].merge(gao[cols].drop(columns=['layer_label', 'archetype']),
                        on=KEY, suffixes=('_y', '_g'))
    for tag in ('y', 'g'):
        m[f'thick_{tag}'] = m[f'overlap_{tag}'] >= m54b.MASK_MIN_OVERLAP
        m[f'sig_{tag}'] = significant(m, tag, m54b)
    m['both_thick'] = m['thick_y'] & m['thick_g']
    m['delta_log2_enr'] = m['log2_enr_y'] - m['log2_enr_g']
    return m.sort_values(KEY).reset_index(drop=True)


def significant(df, tag, m54b):
    """54b's star criterion, on a suffixed side of the merged frame or on a raw table."""
    s = f'_{tag}' if tag else ''
    return ((df[f'fdr_strat{s}'] < m54b.STAR_FDR)
            & (df[f'log2_enr{s}'] > m54b.STAR_LOG2ENR)
            & (df[f'overlap{s}'] >= m54b.MASK_MIN_OVERLAP))


def concordance_stats(m):
    """r / rho / median |delta| per scope, for each of the three cell subsets."""
    subsets = {
        'both_thick': m['both_thick'],
        'either_thick': m['thick_y'] | m['thick_g'],
        'all_shared': pd.Series(True, index=m.index),
    }
    rows = []
    for scope, sel_scope in [('all layers', pd.Series(True, index=m.index))] + \
                            [(l, m['layer_label'] == l) for l in m['layer_label'].unique()]:
        for name, sel_sub in subsets.items():
            sub = m[sel_scope & sel_sub]
            row = dict(scope=scope, subset=name, n_cells=len(sub))
            if len(sub) >= MIN_N_FOR_STAT:
                row['pearson_r'] = pearsonr(sub['log2_enr_y'], sub['log2_enr_g'])[0]
                row['spearman_rho'] = spearmanr(sub['log2_enr_y'], sub['log2_enr_g'])[0]
                row['median_abs_delta'] = np.median(np.abs(sub['delta_log2_enr']))
            rows.append(row)
    return pd.DataFrame(rows)


def replication_cascade(yoo, gao, m54b):
    """significant -> regulon exists in the other catalogue -> significant there.

    The middle step is the one that matters: a hit whose TF is absent from the other
    catalogue is untestable, and lumping it in with genuine non-replication is what makes
    the naive replication fraction misleading.
    """
    rows = []
    for src_tag, src, other_tag, other in [('yoo25', yoo, 'gao25', gao),
                                           ('gao25', gao, 'yoo25', yoo)]:
        other_keys = set(map(tuple, other[KEY].values))
        other_sig_keys = set(map(tuple, other[significant(other, '', m54b)][KEY].values))
        for scope, sel in [('all layers', pd.Series(True, index=src.index))] + \
                          [(l, src['layer_label'] == l) for l in src['layer_label'].unique()]:
            sig = src[sel & significant(src, '', m54b)]
            keys = list(map(tuple, sig[KEY].values))
            n_testable = sum(k in other_keys for k in keys)
            n_repl = sum(k in other_sig_keys for k in keys)
            rows.append(dict(
                source=src_tag, target=other_tag, scope=scope,
                n_significant=len(keys),
                n_testable=n_testable,
                n_untestable=len(keys) - n_testable,
                n_replicating=n_repl,
                frac_testable=n_testable / len(keys) if keys else np.nan,
                frac_replicating_of_testable=n_repl / n_testable if n_testable else np.nan,
                frac_replicating_of_all=n_repl / len(keys) if keys else np.nan))
    return pd.DataFrame(rows)


def shared_significant_tfs(yoo, gao, m54b, ieg_tfs, ieg_stress):
    """Every (layer, archetype, TF) hit in either catalogue, labelled by what it is.

    status: `both` = significant in both, `<tag>_only` = significant in one and testable in
    the other, `<tag>_untestable` = significant in one and the regulon does not exist in the
    other. The third is the category the plan insists on keeping separate.
    """
    sig = {'yoo25': yoo[significant(yoo, '', m54b)], 'gao25': gao[significant(gao, '', m54b)]}
    exists = {tag: set(map(tuple, df[KEY].values)) for tag, df in [('yoo25', yoo), ('gao25', gao)]}
    sig_keys = {tag: set(map(tuple, df[KEY].values)) for tag, df in sig.items()}

    rows = []
    for key in sorted(sig_keys['yoo25'] | sig_keys['gao25']):
        layer, arch, tf = key
        in_y, in_g = key in sig_keys['yoo25'], key in sig_keys['gao25']
        if in_y and in_g:
            status = 'both'
        else:
            hit, miss = ('yoo25', 'gao25') if in_y else ('gao25', 'yoo25')
            status = f'{hit}_only' if key in exists[miss] else f'{hit}_untestable'
        row = dict(layer=layer, arch_letter=arch, TF=tf, status=status,
                   is_ieg=tf in ieg_tfs, is_ieg_or_stress=tf in ieg_stress)
        for tag, df in [('y', yoo), ('g', gao)]:
            cell = df[(df['layer'] == layer) & (df['arch_letter'] == arch) & (df['TF'] == tf)]
            for col in ['log2_enr', 'overlap', 'exp', 'fdr_strat']:
                row[f'{col}_{tag}'] = cell[col].iloc[0] if len(cell) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    label = dict(zip(yoo['layer'], yoo['layer_label']))
    order = {lbl: i for i, lbl in enumerate(dict.fromkeys(yoo['layer_label']))}
    out['layer_label'] = out['layer'].map(label)
    return (out.sort_values(['layer_label', 'arch_letter', 'TF'],
                            key=lambda s: s.map(order) if s.name == 'layer_label' else s)
               .drop(columns=['layer_label']).reset_index(drop=True))


def print_archetype_blocks(shared_tfs, yoo, gao):
    """The per-archetype yoo / gao / shared listing, in the plan's block form."""
    label = dict(zip(yoo['layer'], yoo['layer_label']))
    order = {lbl: i for i, lbl in enumerate(dict.fromkeys(yoo['layer_label']))}
    groups = sorted(shared_tfs.groupby(['layer', 'arch_letter']).groups,
                    key=lambda k: (order[label[k[0]]], k[1]))
    for layer, arch in groups:
        grp = shared_tfs[(shared_tfs['layer'] == layer) & (shared_tfs['arch_letter'] == arch)]
        y = sorted(grp[grp['status'].str.startswith(('yoo25', 'both'))]['TF'])
        g = sorted(grp[grp['status'].str.startswith(('gao25', 'both'))]['TF'])
        both = sorted(grp[grp['status'] == 'both']['TF'])
        n_ieg = int(grp[grp['status'] == 'both']['is_ieg'].sum())
        print(f'  {label[layer]:5s} {arch:3s} yoo: {" ".join(y) if y else "-"}')
        print(f'  {"":5s} {"":3s} gao: {" ".join(g) if g else "-"}')
        print(f'  {"":5s} {"":3s} shared: {" ".join(both) if both else "-"}'
              f'{f"   [{n_ieg} IEG]" if n_ieg else ""}')


# --------------------------------------------------------------------------- Q2: power

def power_by_layer(catalogues, m54b):
    """Hit rate per (catalogue, layer, `exp` quartile), plus the two factors behind `exp`.

    `exp` is cut into quartiles over the whole catalogue (see N_EXP_QUANTILES) so that a
    quartile means the same expected overlap in every layer. n_thick is carried alongside
    n_cells because the star criterion contains `overlap >= MASK_MIN_OVERLAP`: a cell that
    is not thick cannot be significant at any effect size, so `hit_rate_thick` says how
    often a cell that COULD have been called actually was.
    """
    rows = []
    for tag, df in catalogues.items():
        d = df.copy()
        d['sig'] = significant(d, '', m54b)
        d['thick'] = d['overlap'] >= m54b.MASK_MIN_OVERLAP
        edges = np.quantile(d['exp'], np.linspace(0, 1, N_EXP_QUANTILES + 1))
        d['exp_q'] = pd.qcut(d['exp'], N_EXP_QUANTILES, labels=range(1, N_EXP_QUANTILES + 1))
        groups = [('all', d.groupby('layer_label', sort=False))]
        groups += [(q, d[d['exp_q'] == q].groupby('layer_label', sort=False))
                   for q in range(1, N_EXP_QUANTILES + 1)]
        for q, grouped in groups:
            for layer, g in grouped:
                rows.append(dict(
                    catalogue=tag, layer=layer, exp_quartile=q,
                    exp_lo=np.nan if q == 'all' else edges[q - 1],
                    exp_hi=np.nan if q == 'all' else edges[q],
                    n_cells=len(g), n_thick=int(g['thick'].sum()),
                    n_significant=int(g['sig'].sum()),
                    hit_rate=g['sig'].mean(),
                    hit_rate_thick=g['sig'].sum() / g['thick'].sum() if g['thick'].any() else np.nan,
                    median_exp=g['exp'].median(),
                    median_n_markers=g['n_markers'].median(),
                    median_n_targets=g['n_targets'].median()))
    out = pd.DataFrame(rows)
    # How much of a layer sits in each quartile -- the direct evidence for a power deficit.
    # A layer with 44% of its cells in Q1 (yoo25 L5IT) is not being out-competed, it is
    # being asked a question it cannot answer.
    total = out[out['exp_quartile'] == 'all'].set_index(['catalogue', 'layer'])['n_cells']
    out['frac_of_layer_cells'] = (out['n_cells'].values
                                  / total.reindex(pd.MultiIndex.from_frame(
                                      out[['catalogue', 'layer']])).values)
    return out


def draw_power(power, m54b):
    """Hit rate vs `exp` quartile, one line per layer, one panel per catalogue.

    Two panels rather than one because the conclusion is the difference between them: the
    layer ordering inverts when the regulon catalogue is swapped (L6IT is yoo25's best layer
    and gao25's worst), which is the argument against reading per-layer hit counts as
    biology. Overall rates go in the legend, where they can be compared across panels; the
    quartile curves cannot be, since the edges are cut per catalogue.
    """
    tags = list(dict.fromkeys(power['catalogue']))
    fig, axes = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    qs = sorted(q for q in power['exp_quartile'].unique() if q != 'all')
    for ax, tag in zip(axes, tags):
        sub = power[power['catalogue'] == tag]
        for layer, color in LAYER_COLOR.items():
            byq = sub[(sub['layer'] == layer) & (sub['exp_quartile'] != 'all')] \
                    .set_index('exp_quartile').reindex(qs)
            overall = sub[(sub['layer'] == layer) & (sub['exp_quartile'] == 'all')]
            if overall.empty:
                continue
            o = overall.iloc[0]
            ax.plot(range(len(qs)), 100 * byq['hit_rate'], marker='o', ms=5, lw=1.6,
                    color=color,
                    label=f'{layer}   {100 * o["hit_rate"]:.1f}% overall  '
                          f'(median exp {o["median_exp"]:.2f}, |M| {o["median_n_markers"]:.0f}, '
                          f'|T| {o["median_n_targets"]:.0f})')
            for x, (_, r) in enumerate(byq.iterrows()):
                if r['n_significant']:
                    ax.annotate(f'{int(r["n_significant"])}/{int(r["n_cells"])}',
                                (x, 100 * r['hit_rate']), textcoords='offset points',
                                xytext=(0, 6), ha='center', fontsize=6, color=color)
        edges = sub[sub['exp_quartile'] != 'all'].drop_duplicates('exp_quartile') \
                   .set_index('exp_quartile').reindex(qs)
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels([f'Q{q}\n{lo:.2f}–{hi:.2f}' for q, lo, hi in
                            zip(qs, edges['exp_lo'], edges['exp_hi'])], fontsize=7.5)
        ax.set_xlabel('expected overlap `exp` under the stratified null\n'
                      '(quartiles cut over this catalogue)', fontsize=8.5)
        ax.set_title(tag, fontsize=10, loc='left')
        ax.legend(fontsize=6.5, frameon=False, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(f'% of cells significant\n(fdr < {m54b.STAR_FDR}, '
                       f'log2_enr > {m54b.STAR_LOG2ENR}, overlap $\\geq$ '
                       f'{m54b.MASK_MIN_OVERLAP})', fontsize=8.5)
    # headroom for the n/N annotations, which sit above the topmost marker
    axes[0].set_ylim(-1, 100 * power['hit_rate'].max() + 4)
    fig.suptitle('Power, not layer identity, sets the hit rate — and the layer ordering '
                 'inverts between catalogues',
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_POWER_PDF, bbox_inches='tight')
    plt.close(fig)


def draw_scatter(ax, m, stats, m54b):
    """log2_enr yoo25 vs gao25. Colour = jointly thick, gray = thick on one side only.

    Cells thick in neither are drawn faintest rather than dropped: the difference between
    the coloured cloud and the gray one IS the result, and hiding the gray would make the
    correlation look like it holds over the whole catalogue overlap.
    """
    lo = min(m['log2_enr_y'].min(), m['log2_enr_g'].min()) - 0.3
    hi = max(m['log2_enr_y'].max(), m['log2_enr_g'].max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color='0.4', lw=0.8, ls='--', zorder=1)
    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0)

    one_thick = (m['thick_y'] | m['thick_g']) & ~m['both_thick']
    neither = ~(m['thick_y'] | m['thick_g'])
    ax.scatter(m.loc[neither, 'log2_enr_y'], m.loc[neither, 'log2_enr_g'],
               s=8, c=GRAY, alpha=0.35, lw=0, zorder=2)
    ax.scatter(m.loc[one_thick, 'log2_enr_y'], m.loc[one_thick, 'log2_enr_g'],
               s=18, c=GRAY, alpha=0.85, lw=0.3, edgecolors='0.5', zorder=3)
    for layer, color in LAYER_COLOR.items():
        sel = m['both_thick'] & (m['layer_label'] == layer)
        if not sel.any():
            continue
        # area tracks the weaker of the two overlaps -- a cell is only as trustworthy as
        # the thinner side of the comparison
        size = 12 + 6 * np.minimum(m.loc[sel, 'overlap_y'], m.loc[sel, 'overlap_g'])
        ax.scatter(m.loc[sel, 'log2_enr_y'], m.loc[sel, 'log2_enr_g'], s=size, c=color,
                   alpha=0.85, lw=0.4, edgecolors='white', zorder=4, label=layer)

    top = stats[(stats['scope'] == 'all layers') & (stats['subset'] == 'both_thick')].iloc[0]
    allc = stats[(stats['scope'] == 'all layers') & (stats['subset'] == 'all_shared')].iloc[0]
    ax.text(0.03, 0.97,
            f'jointly thick (overlap $\\geq$ {m54b.MASK_MIN_OVERLAP} both sides), '
            f'n = {int(top["n_cells"])}\n'
            f'Pearson r = {top["pearson_r"]:.3f}   Spearman $\\rho$ = {top["spearman_rho"]:.3f}\n'
            f'median |$\\Delta$ log2 enr| = {top["median_abs_delta"]:.2f}\n'
            f'all {int(allc["n_cells"])} shared cells: $\\rho$ = {allc["spearman_rho"]:.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.7', lw=0.5))

    # Every per-layer rho carries its n AND how many of those cells are jointly thick. L5IT
    # has none, so its rho = -0.43 rests entirely on cells one catalogue could not measure --
    # noise, and it will be read as contradiction if the legend does not say so.
    handles = []
    for layer, color in LAYER_COLOR.items():
        row = stats[(stats['scope'] == layer) & (stats['subset'] == 'either_thick')]
        n = int(row['n_cells'].iloc[0]) if len(row) else 0
        n_both = int((m['both_thick'] & (m['layer_label'] == layer)).sum())
        rho = row['spearman_rho'].iloc[0] if len(row) else np.nan
        rho_txt = f'$\\rho$ = {rho:.2f}' if np.isfinite(rho) else 'n too small'
        handles.append(Line2D([], [], ls='', marker='o', ms=5, mfc=color, mec='white',
                              label=f'{layer}  {rho_txt} (n = {n}; {n_both} jointly thick)'))
    handles.append(Line2D([], [], ls='', marker='o', ms=4, mfc=GRAY, mec='0.5',
                          label='thick on one side only'))
    ax.legend(handles=handles, loc='lower right', fontsize=6.5, frameon=True,
              framealpha=0.9, title='per layer, cells thick in $\\geq$1 catalogue',
              title_fontsize=6.5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('yoo25 log2 enrichment (54)')
    ax.set_ylabel('gao25 log2 enrichment (55)')
    ax.set_title('a  Shared regulon $\\times$ archetype cells', loc='left', fontsize=10)


def draw_cascade(ax, cascade):
    """significant -> testable -> replicating, with untestable split out of the loss.

    Drawn as two stacked bars rather than a table because the point is proportional: the
    untestable block is about half of yoo25's hits, and that is the number a reader should
    take away instead of the replication fraction.
    """
    rows = cascade[cascade['scope'] == 'all layers'].reset_index(drop=True)
    y = np.arange(len(rows))[::-1]
    for i, r in rows.iterrows():
        base, yi = 0, y[i]
        for width, color, name in [
                (r['n_replicating'], '#4c72b0', 'significant in both'),
                (r['n_testable'] - r['n_replicating'], '#a9c0dc', 'testable, not significant'),
                (r['n_untestable'], GRAY, 'regulon absent from the other catalogue')]:
            ax.barh(yi, width, left=base, height=0.55, color=color, lw=0.4,
                    edgecolor='white', label=name if i == 0 else None)
            if width:
                ax.text(base + width / 2, yi, str(int(width)), ha='center', va='center',
                        fontsize=8, color='white' if color == '#4c72b0' else '0.15')
            base += width
        ax.text(base + 1.2, yi,
                f'{r["frac_testable"]:.0%} testable, '
                f'{r["frac_replicating_of_testable"]:.0%} of those replicate',
                va='center', fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{r["source"]} hits\n$\\rightarrow$ {r["target"]}'
                        for _, r in rows.iterrows()], fontsize=8)
    ax.set_xlim(0, rows['n_significant'].max() * 1.6)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    # two bars in a panel sized for a scatter would be drawn absurdly thick; shrink the drawn
    # box instead of the bar height, so the bars keep a normal aspect
    ax.set_box_aspect(0.42)
    ax.set_anchor('N')      # keep the shrunk box at the top, so both panel titles align
    ax.set_xlabel('significant regulon $\\times$ archetype cells')
    ax.legend(loc='lower right', fontsize=6.5, frameon=False)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_title('b  Replication cascade', loc='left', fontsize=10)


def main():
    m41b = load_module(SCRIPT_41B, 'script41b')
    m54b = load_module(SCRIPT_54B, 'script54b')
    ieg_stress = set(m41b.SELECTED_TFS)
    assert set(IEG_STRESS_ONLY) <= ieg_stress, f'{IEG_STRESS_ONLY} are not in 41b.SELECTED_TFS'
    ieg_tfs = ieg_stress - set(IEG_STRESS_ONLY)
    print(f'significance: fdr_strat < {m54b.STAR_FDR}, log2_enr > {m54b.STAR_LOG2ENR}, '
          f'overlap >= {m54b.MASK_MIN_OVERLAP}  (54b, provisional)')

    yoo = load_catalogue(INPUT_YOO25_TMPL, m41b, 'yoo25')
    gao = load_catalogue(INPUT_GAO25_TMPL, m41b, 'gao25')
    print(f'\nyoo25 {len(yoo)} activating cells, {yoo["TF"].nunique()} TFs; '
          f'gao25 {len(gao)} cells, {gao["TF"].nunique()} TFs')

    m = merge_catalogues(yoo, gao, m54b)
    assert len(m), 'no regulon is shared between the two catalogues'
    m.to_csv(OUT_CELLS, sep='\t', index=False)
    print(f'  wrote -> {OUT_CELLS}')

    stats = concordance_stats(m)
    stats.to_csv(OUT_STATS, sep='\t', index=False)
    print(f'  wrote -> {OUT_STATS}')
    print('\nconcordance of log2_enr:')
    for _, r in stats[stats['scope'] == 'all layers'].iterrows():
        print(f'  {r["subset"]:13s} n={int(r["n_cells"]):4d}  r={r["pearson_r"]:.3f}  '
              f'rho={r["spearman_rho"]:.3f}  median|delta|={r["median_abs_delta"]:.2f}')
    print('  per layer (cells thick in >=1 catalogue; n is not decoration, quote it):')
    for _, r in stats[(stats['subset'] == 'either_thick')
                      & (stats['scope'] != 'all layers')].iterrows():
        rho = f'{r["spearman_rho"]:.3f}' if np.isfinite(r['spearman_rho']) else '   n/a'
        print(f'    {r["scope"]:6s} n={int(r["n_cells"]):3d}  rho={rho}')

    cascade = replication_cascade(yoo, gao, m54b)
    cascade.to_csv(OUT_CASCADE, sep='\t', index=False)
    print(f'\n  wrote -> {OUT_CASCADE}')
    for _, r in cascade[cascade['scope'] == 'all layers'].iterrows():
        print(f'  {r["source"]} significant cells{"":18s}{int(r["n_significant"]):4d}')
        print(f'    ... regulon even EXISTS in {r["target"]}{"":8s}{int(r["n_testable"]):4d}'
              f'   ({r["frac_testable"]:.0%})   <- untestable, not contradicted')
        print(f'    ... and significant there{"":16s}{int(r["n_replicating"]):4d}'
              f'   ({r["frac_replicating_of_testable"]:.0%} of testable)')

    shared_tfs = shared_significant_tfs(yoo, gao, m54b, ieg_tfs, ieg_stress)
    shared_tfs.to_csv(OUT_SHARED_TFS, sep='\t', index=False)
    print(f'\n  wrote -> {OUT_SHARED_TFS}')
    print_archetype_blocks(shared_tfs, yoo, gao)
    for tag, df in [('yoo25', yoo), ('gao25', gao)]:
        sig = df[significant(df, '', m54b)]
        print(f'  IEG TFs are {sig["TF"].isin(ieg_tfs).mean():.0%} of {tag}\'s {len(sig)} '
              f'significant cells ({sig["TF"].isin(ieg_stress).mean():.0%} counting '
              f'{"/".join(IEG_STRESS_ONLY)} as IEG/stress)')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6),
                             gridspec_kw=dict(width_ratios=[1, 0.9]))
    draw_scatter(axes[0], m, stats, m54b)
    draw_cascade(axes[1], cascade)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  wrote -> {OUT_PDF}')

    # --- Q2 ------------------------------------------------------------------------------
    power = power_by_layer({'yoo25': yoo, 'gao25': gao}, m54b)
    power.to_csv(OUT_POWER, sep='\t', index=False)
    print(f'\n  wrote -> {OUT_POWER}')
    for tag in ['yoo25', 'gao25']:
        sub = power[power['catalogue'] == tag]
        wide = sub[sub['exp_quartile'] != 'all'].pivot(index='layer', columns='exp_quartile',
                                                       values='hit_rate')
        overall = sub[sub['exp_quartile'] == 'all'].set_index('layer')
        print(f'  {tag}: hit rate by exp quartile (%)')
        print(f'    {"layer":6s} ' + ' '.join(f'Q{q:<5d}' for q in wide.columns)
              + ' overall  med exp  med |M|  med |T|')
        for layer in overall.index:
            o = overall.loc[layer]
            print(f'    {layer:6s} ' + ' '.join(f'{100 * v:<6.1f}' for v in wide.loc[layer])
                  + f' {100 * o["hit_rate"]:<8.1f} {o["median_exp"]:<8.2f} '
                    f'{o["median_n_markers"]:<8.0f} {o["median_n_targets"]:<8.0f}')
        q1 = sub[sub['exp_quartile'] == 1].set_index('layer')['frac_of_layer_cells']
        print('    share of each layer in the bottom exp quartile: '
              + ', '.join(f'{l} {100 * v:.0f}%' for l, v in q1.items()))
        order = overall['hit_rate'].sort_values(ascending=False)
        print('    overall ordering: '
              + ' > '.join(f'{l} {100 * v:.1f}%' for l, v in order.items()))
    draw_power(power, m54b)
    print(f'  wrote -> {OUT_POWER_PDF}')
    print('  the two orderings above disagree on where L6IT sits; per-layer hit COUNTS are '
          'therefore not a biological readout, only exp-stratified rates are')


if __name__ == '__main__':
    main()
