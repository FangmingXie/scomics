"""Mouse-vs-human ortholog gene-loading joint plots on the conserved CCA axes (L2/3, boxplots).

Scatter of per-gene mouse-vs-human canonical projections, coloured by mouse archetype, with
one horizontal boxplot per group below the x-axis and one vertical boxplot per group along the
y-axis. The CCA1 and CCA2 joint plots are packed side-by-side into a single two-panel PDF.

Boxplots (rather than a density histogram + gaussian_kde per archetype) buy three things:
  1. No bandwidth to tune (a KDE_BW was doing real work at n=8, archetype B under
     `hvg_intersect`); a box has no smoothing parameter.
  2. The **"other" genes are drawn as a fourth, gray box**, so each archetype is read
     against the bulk of the transcriptome rather than against the other two archetypes.
     That is the comparison the "which pole does this archetype sit on" claim needs.
  3. Medians and IQRs are directly comparable across groups of very different n.

The trade-off is that multimodality is invisible in a box.

A Mann-Whitney U of each archetype against the "other" bulk is printed to stdout per axis and
per species, and annotated on the figure as significance stars beside each archetype's box
(ns / * .05 / ** .01 / *** .001, two-sided, vs the gray bulk). It is descriptive: the three
archetypes are not independent of each other and no multiplicity correction is applied. With
n_other in the thousands, even a small median shift reaches ***, so read the stars alongside
the rank-biserial effect sizes on stdout, not on their own.

Archetypes are keyed internally by A/B/C (= mouse archetype_1/2/3) for marker identity and
coloring, but *displayed* with the published primed labels via ARCH_RELABEL {A:C', B:B', C:A'}
and ordered A', B', C' (matching scripts/it/41,48,50). Below, "A/B/C" names the internal keys.

Reads (paths switch with UNIVERSE):
  local_data/res/it_evo/26.{human,mouse}_L23_varimax_loadings_full.tsv  (expanded)
  local_data/res/it_evo/02.human_L23_varimax_loadings.tsv               (HVG membership)
  local_data/res/it/19.cheng22_L23_varimax_loadings.tsv                 (HVG membership)
  local_data/res/it_evo/16.L23_axis_cca_weights_{human,mouse}<SUFFIX>.tsv
  local_data/res/it_evo/05.mouse_L23_archetype_markers.tsv
  data/human_mouse_orthologs.tsv
Outputs (CCA1 + CCA2 as the two panels of one figure):
  local_data/fig/it_evo/24b.human_mouse_L23_gene_loadings<SUFFIX>.pdf
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- config ---
UNIVERSE_SUFFIX = {'hvg_intersect': '', 'hvg_union': '_union'}
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--universe', choices=list(UNIVERSE_SUFFIX), default='hvg_union')
UNIVERSE     = _parser.parse_args().universe
SUFFIX       = UNIVERSE_SUFFIX[UNIVERSE]

HUMAN_VX     = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
MOUSE_VX     = ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9']
MOUSE_NOC    = 3
ALPHABET     = ['A', 'B', 'C', 'D', 'E', 'F']
# Internal keys A/B/C (= archetype_1/2/3) are display-relabeled for publication, matching
# scripts/it/41,48,50. Only labels change; keys drive all computation and coloring.
ARCH_RELABEL = {'A': "C'", 'B': "B'", 'C': "A'"}   # internal -> displayed
ARCH_ORDER   = ['C', 'B', 'A']                     # so labels read A', B', C'
# color follows the DISPLAYED label: A'->C0, B'->C1, C'->C2 (so internal C->C0, B->C1, A->C2)
ARCH_COLORS  = {'A': 'C2', 'B': 'C1', 'C': 'C0'}   # internal key -> color
BASE_COLOR   = '#bdbdbd'                            # "other" genes: neutral gray
POINT_SIZE   = {'hvg_intersect': 10, 'hvg_union': 4}[UNIVERSE]
BASE_ALPHA   = {'hvg_intersect': 1.0, 'hvg_union': 0.45}[UNIVERSE]
ARCH_BUMP    = {'hvg_intersect': 12, 'hvg_union': 6}[UNIVERSE]
TOP_N_LABEL  = 20                                   # label the top genes by |mouse.human| loading
BOX_WIDTH    = 0.62                                 # fraction of the unit slot each box fills
FLIER_SIZE   = 3.0                                  # outlier marker size (points)
SIG_LEVELS   = [(0.001, '***'), (0.01, '**'), (0.05, '*')]   # MWU vs "other"; else 'ns'
N_PERM       = 20000                                # gene-label permutations (mirror script 21)
PERM_SEED    = 0                                    # mirror script 21's SEED so p-values match exactly
R_MATCH_TOL  = 1e-6                                 # panel r vs canonical r (see draw_joint)

# --- file paths ---
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_H_HVG      = os.path.join(RES_DIR, '02.human_L23_varimax_loadings.tsv')
IN_M_HVG      = os.path.join(IT_RES_DIR, '19.cheng22_L23_varimax_loadings.tsv')
IN_H_FULL     = os.path.join(RES_DIR, '26.human_L23_varimax_loadings_full.tsv')
IN_M_FULL     = os.path.join(RES_DIR, '26.mouse_L23_varimax_loadings_full.tsv')
IN_W_HUMAN    = os.path.join(RES_DIR, f'16.L23_axis_cca_weights_human{SUFFIX}.tsv')
IN_W_MOUSE    = os.path.join(RES_DIR, f'16.L23_axis_cca_weights_mouse{SUFFIX}.tsv')
IN_MARKERS    = os.path.join(RES_DIR, '05.mouse_L23_archetype_markers.tsv')
IN_ORTHOLOGS  = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_PDF       = os.path.join(FIG_DIR, f'24b.human_mouse_L23_gene_loadings{SUFFIX}.pdf')

os.makedirs(FIG_DIR, exist_ok=True)

# --- shared orthologs, centered gene loading blocks (mirrors 16.load_loadings) ---
ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
hvg_h = pd.read_csv(IN_H_HVG, sep='\t', index_col=0)
hvg_m = pd.read_csv(IN_M_HVG, sep='\t', index_col=0)
if UNIVERSE == 'hvg_intersect':
    H, M = hvg_h, hvg_m
else:
    H = pd.read_csv(IN_H_FULL, sep='\t', index_col=0)
    M = pd.read_csv(IN_M_FULL, sep='\t', index_col=0)
shared = ortho[ortho['human_symbol'].isin(H.index) & ortho['mouse_symbol'].isin(M.index)]
if UNIVERSE == 'hvg_union':
    shared = shared[shared['human_symbol'].isin(hvg_h.index)
                    | shared['mouse_symbol'].isin(hvg_m.index)]
shared = shared.reset_index(drop=True)
symbols = shared['human_symbol'].values

X = H.loc[shared['human_symbol'].values, HUMAN_VX].values
Y = M.loc[shared['mouse_symbol'].values, MOUSE_VX].values
Xc = X - X.mean(0)
Yc = Y - Y.mean(0)

# --- CCA1/CCA2 gene loadings from 16's canonical weights ---
wdf_h = pd.read_csv(IN_W_HUMAN, sep='\t', index_col=0)
wdf_m = pd.read_csv(IN_W_MOUSE, sep='\t', index_col=0)


def cca_load(axis):
    # .loc[row, cols] keeps the row's object dtype (row mixes floats + the bool `stable`
    # column), so force float before the matmul.
    wm = wdf_m.loc[axis, MOUSE_VX].to_numpy(dtype=float)
    wh = wdf_h.loc[axis, HUMAN_VX].to_numpy(dtype=float)
    return Yc @ wm, Xc @ wh


# --- permutation significance of the cross-species canonical correlations (mirrors 21/24) ---
def _orthonormal(Mtx):
    return np.linalg.qr(Mtx - Mtx.mean(axis=0))[0]


def permutation_pvals():
    Qx, Qy = _orthonormal(X), _orthonormal(Y)
    n = X.shape[0]
    obs = np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), 0.0, 1.0)
    rng = np.random.default_rng(PERM_SEED)
    null = np.empty((N_PERM, Qy.shape[1]))
    for i in range(N_PERM):
        null[i] = np.clip(np.linalg.svd(Qx.T @ Qy[rng.permutation(n)], compute_uv=False), 0.0, 1.0)
    out = {}
    for comp, axis in [(0, 'CCA1'), (1, 'CCA2')]:
        n_ge = int((null[:, comp] >= obs[comp]).sum())
        out[axis] = {'r_cca': float(obs[comp]), 'n_ge': n_ge, 'p': (1 + n_ge) / (N_PERM + 1)}
    return out


# --- mouse archetype membership per gene (highest-log2FC archetype if multiple) ---
mk = pd.read_csv(IN_MARKERS, sep='\t')
mk['letter'] = mk['archetype'].map({f'archetype_{i+1}': ALPHABET[i] for i in range(MOUSE_NOC)})
mk = mk.sort_values('log2FC', ascending=False).drop_duplicates('gene')
gene2arch = dict(zip(mk['gene'], mk['letter']))
letters = np.array([gene2arch.get(g, '') for g in shared['mouse_symbol'].values])

# Group order runs "other" first so it sits nearest the scatter in both marginals and reads
# as the baseline the archetypes are compared against.
GROUPS = [('other', BASE_COLOR)] + [(L, ARCH_COLORS[L]) for L in ARCH_ORDER]


def draw_boxes(ax, values_by_group, orientation):
    """One box per group at positions 0..n-1, colored to match the scatter."""
    data = [values_by_group[g] for g, _ in GROUPS]
    bp = ax.boxplot(data, positions=range(len(GROUPS)), widths=BOX_WIDTH,
                    orientation=orientation, patch_artist=True, showfliers=True,
                    medianprops=dict(color='black', lw=1.2),
                    whiskerprops=dict(color='0.4', lw=0.8),
                    capprops=dict(color='0.4', lw=0.8),
                    flierprops=dict(marker='o', markersize=FLIER_SIZE, markerfacecolor='0.5',
                                    markeredgecolor='none', alpha=0.5))
    for patch, (_, color) in zip(bp['boxes'], GROUPS):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor('0.35')
        patch.set_linewidth(0.8)
    return bp


def group_values(vals):
    return {'other': vals[letters == ''], **{L: vals[letters == L] for L in ARCH_COLORS}}


def mwu_vs_other(vals):
    """Mann-Whitney U of each archetype against the 'other' bulk. letter -> (p, rank-biserial)."""
    g = group_values(vals)
    out = {}
    for L in ARCH_COLORS:
        if len(g[L]) < 1:
            continue
        u, p = mannwhitneyu(g[L], g['other'], alternative='two-sided')
        # rank-biserial effect size: +1 = archetype entirely above the bulk
        out[L] = (p, 2 * u / (len(g[L]) * len(g['other'])) - 1)
    return out


def stars(p):
    for thresh, mark in SIG_LEVELS:
        if p < thresh:
            return mark
    return 'ns'


def report_mwu(axis_label, ms, hs):
    """Print the same test the figure annotates, with effect sizes the stars cannot show."""
    for species, vals in (('mouse', ms), ('human', hs)):
        stats = mwu_vs_other(vals)
        parts = [f'{ARCH_RELABEL[L]}: med {np.median(group_values(vals)[L]):+.3f} rb {stats[L][1]:+.2f} '
                 f'p {stats[L][0]:.1e} {stars(stats[L][0])}' for L in ARCH_ORDER if L in stats]
        print(f'    {axis_label} {species:5s} '
              f'(bulk med {np.median(group_values(vals)["other"]):+.3f})  ' + ' | '.join(parts))


def annotate_sig(ax, vals, orientation):
    """Significance stars vs the 'other' bulk, at the outer edge of each archetype's slot."""
    trans = (ax.get_yaxis_transform() if orientation == 'horizontal'
             else ax.get_xaxis_transform())
    for L, (p, _) in mwu_vs_other(vals).items():
        pos = [g for g, _ in GROUPS].index(L)
        if orientation == 'horizontal':   # categories on y, values on x
            ax.text(0.995, pos, stars(p), transform=trans, ha='right', va='center',
                    fontsize=8, color='0.25')
        else:                             # categories on x, values on y
            ax.text(pos, 0.995, stars(p), transform=trans, ha='center', va='top',
                    fontsize=8, color='0.25')


def draw_joint(subfig, ms, hs, axis_label, sig):
    """Draw one axis' (mouse, human) joint plot into `subfig`; returns the panel Pearson r."""
    r = float(np.corrcoef(ms, hs)[0, 1])
    if not abs(abs(r) - sig['r_cca']) < R_MATCH_TOL:
        raise ValueError(
            f'{axis_label}: projected Pearson |r| = {abs(r):.6f} does not match the canonical '
            f'correlation {sig["r_cca"]:.6f} that the permutation p-value tests — 16\'s saved '
            f'weights are not the canonical vectors for this gene universe')
    p_str = (f'p < {1 / (N_PERM + 1):.0e}' if sig['n_ge'] == 0 else f'p = {sig["p"]:.1e}')
    top_idx = np.argsort(np.abs(ms * hs))[::-1][:TOP_N_LABEL]
    lim = np.array([min(ms.min(), hs.min()), max(ms.max(), hs.max())]) * 1.08

    gs = subfig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1],
                             wspace=0.04, hspace=0.04)
    ax_main = subfig.add_subplot(gs[0, 0])
    ax_boxx = subfig.add_subplot(gs[1, 0], sharex=ax_main)   # below x-axis
    ax_boxy = subfig.add_subplot(gs[0, 1], sharey=ax_main)   # along y-axis

    # --- main scatter (identical to 24) ---
    n_other = int((letters == '').sum())
    ax_main.scatter(ms[letters == ''], hs[letters == ''], s=POINT_SIZE, c=BASE_COLOR,
                    linewidths=0, alpha=BASE_ALPHA, rasterized=True,
                    label=f'other (n={n_other})')
    for L in ARCH_ORDER:
        color = ARCH_COLORS[L]
        m = letters == L
        ax_main.scatter(ms[m], hs[m], s=POINT_SIZE + ARCH_BUMP, c=color, linewidths=0, alpha=0.9,
                        label=f'mouse archetype {ARCH_RELABEL[L]} (n={int(m.sum())})', zorder=3)

    ax_main.axhline(0, color='0.75', lw=0.6, zorder=0)
    ax_main.axvline(0, color='0.75', lw=0.6, zorder=0)
    ax_main.plot(lim, lim, '--', color='0.6', lw=0.8, zorder=0)

    ax_main.scatter(ms[top_idx], hs[top_idx], s=POINT_SIZE + ARCH_BUMP + 6, facecolors='none',
                    edgecolors='black', linewidths=0.7, zorder=4)
    for i in top_idx:
        ax_main.annotate(symbols[i], (ms[i], hs[i]), textcoords='offset points',
                         xytext=(4, 3), fontsize=7, fontstyle='italic', zorder=5)

    ax_main.set_xlim(lim); ax_main.set_ylim(lim)
    ax_main.set_ylabel(f'Human Jorstad23 {axis_label} gene loading')
    ax_main.set_title(f'L2/3 {axis_label}: mouse vs human\n'
                      f'r = {r:.3f},  {p_str}  (gene-label permutation, {N_PERM} reps)\n'
                      f'universe: {UNIVERSE},  {len(shared)} orthologous genes',
                      fontsize=11)
    ax_main.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_main.tick_params(labelbottom=False)
    sns.despine(ax=ax_main)

    # --- marginal boxplots; "other" is the gray baseline box nearest the scatter ---
    labels = ['other' if g == 'other' else ARCH_RELABEL[g] for g, _ in GROUPS]
    draw_boxes(ax_boxx, group_values(ms), 'horizontal')
    draw_boxes(ax_boxy, group_values(hs), 'vertical')
    annotate_sig(ax_boxx, ms, 'horizontal')
    annotate_sig(ax_boxy, hs, 'vertical')

    ax_boxx.set_ylim(len(GROUPS) - 0.5, -0.5)     # group 0 ("other") nearest the scatter
    ax_boxx.set_yticks(range(len(GROUPS)))
    ax_boxx.set_yticklabels(labels, fontsize=8)
    ax_boxx.axvline(0, color='0.75', lw=0.6, zorder=0)
    ax_boxx.set_xlabel(f'Mouse Cheng22 {axis_label} gene loading')

    ax_boxy.set_xlim(-0.5, len(GROUPS) - 0.5)     # group 0 ("other") nearest the scatter
    ax_boxy.set_xticks(range(len(GROUPS)))
    ax_boxy.set_xticklabels(labels, fontsize=8)
    ax_boxy.axhline(0, color='0.75', lw=0.6, zorder=0)
    ax_boxy.tick_params(labelleft=False)          # shares y ticks with the main scatter
    ax_boxy.set_xlabel('Mann-Whitney U vs "other"\nns  * .05  ** .01  *** .001', fontsize=7)

    sns.despine(ax=ax_boxx)
    sns.despine(ax=ax_boxy)
    return r


print('--- L2/3 CCA gene-loading joint plots (boxplot marginals) ---')
print(f'  universe: {UNIVERSE}  ({len(shared)} shared orthologs)')
print(f'  archetype genes on scatter: '
      f'{ {ARCH_RELABEL[L]: int((letters == L).sum()) for L in ARCH_ORDER} } '
      f'of {len(gene2arch)} mouse markers')

print(f'  running {N_PERM} gene-label permutations for CCA1/CCA2 significance...')
sig = permutation_pvals()
for axis in ('CCA1', 'CCA2'):
    s = sig[axis]
    print(f'  {axis}: r_cca {s["r_cca"]:.3f}, {s["n_ge"]}/{N_PERM} >= obs, p = {s["p"]:.2e}')

ms_cca1, hs_cca1 = cca_load('CCA1')
ms_cca2, hs_cca2 = cca_load('CCA2')

print('  archetype vs "other" bulk (Mann-Whitney U, rank-biserial; descriptive, uncorrected):')
report_mwu('CCA1', ms_cca1, hs_cca1)
report_mwu('CCA2', ms_cca2, hs_cca2)

plt.rcParams['pdf.fonttype'] = 42
fig = plt.figure(figsize=(16, 8))
subfigs = fig.subfigures(1, 2, wspace=0.07)
r1 = draw_joint(subfigs[0], ms_cca1, hs_cca1, 'CCA1', sig['CCA1'])
r2 = draw_joint(subfigs[1], ms_cca2, hs_cca2, 'CCA2', sig['CCA2'])
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  panel r: CCA1 {r1:.3f}, CCA2 {r2:.3f}')
print(f'  Saved {OUT_PDF}')
print('\nDone.')
