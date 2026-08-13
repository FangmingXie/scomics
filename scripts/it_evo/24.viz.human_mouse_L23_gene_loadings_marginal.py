"""Polished CCA1/CCA2 mouse-vs-human ortholog gene-loading joint plots (L2/3 only, plots only).

A refinement of script 20's first two panels. Instead of one three-panel figure, this emits a
separate PDF per conserved canonical axis (CCA1, CCA2). Each figure is a joint plot:

  * center  — the mouse-vs-human gene-loading scatter (identical content to 20's panel: gray
              "other" genes, mouse archetype A/B/C genes colored A->C0/B->C1/C->C2, the y=x
              diagonal, and the top genes by |mouse.human| loading labelled with a black ring).
  * bottom  — a marginal histogram *below the x-axis* showing how the mouse loadings (x) of the
              A, B, C archetype genes are distributed, one colored curve per archetype.
  * right   — the matching marginal histogram *along the y-axis* for the human loadings (y) of
              the same A, B, C genes.

The marginals make explicit which pole of each conserved axis each archetype's program sits on:
a rightward-shifted A histogram means A genes carry positive loading on that axis in both species.

No CCA/loadings refit -- canonical weights read from 16's persisted TSVs, exactly as in 20.

Archetypes are keyed internally by A/B/C (= mouse archetype_1/2/3) for marker identity and
coloring, but *displayed* with the published primed labels via ARCH_RELABEL {A:C', B:B', C:A'}
and ordered A', B', C' (matching scripts/it/41,48,50). Below, "A/B/C" names the internal keys.

--universe selects the gene set (see 16's docstring): hvg_intersect (357) | hvg_union (3220,
default). The one-sided `human_hvg`/`mouse_hvg` selections that 16 also offers are not plotted
here. Under the historical `hvg_intersect` set only 51 of the 195 mouse archetype markers
survive the double-HVG intersection (A=20, B=8, C=23), which makes B's marginal KDE close to
meaningless; under `hvg_union` 177 do (A=82, B=35, C=60). Because the canonical correlation
depends on the gene universe, r is NOT comparable across universes -- the panel annotates the
universe and n for that reason.

Reads (paths switch with UNIVERSE):
  local_data/res/it_evo/26.{human,mouse}_L23_varimax_loadings_full.tsv  (expanded)
  local_data/res/it_evo/02.human_L23_varimax_loadings.tsv               (HVG membership)
  local_data/res/it/19.cheng22_L23_varimax_loadings.tsv                 (HVG membership)
  local_data/res/it_evo/16.L23_axis_cca_weights_{human,mouse}<SUFFIX>.tsv
  local_data/res/it_evo/05.mouse_L23_archetype_markers.tsv
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/fig/it_evo/24.human_mouse_L23_gene_loadings_CCA{1,2}<SUFFIX>.pdf
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- config ---
# Gene universe, mirroring 16's UNIVERSES/UNIVERSE_SUFFIX. 'hvg_union' is the primary
# expanded set; 'hvg_intersect' reproduces the original figures. Each writes its own
# suffixed PDFs, so the variants coexist rather than overwriting each other. 16 must have
# been run at the same universe first — this reads its canonical weights.
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
# The gray cloud grows 357 -> 3220 between universes; shrink and fade it so the archetype
# points stay legible on top of it.
POINT_SIZE   = {'hvg_intersect': 10, 'hvg_union': 4}[UNIVERSE]
BASE_ALPHA   = {'hvg_intersect': 1.0, 'hvg_union': 0.45}[UNIVERSE]
ARCH_BUMP    = {'hvg_intersect': 12, 'hvg_union': 6}[UNIVERSE]
TOP_N_LABEL  = 20                                   # label the top genes by |mouse.human| loading
N_BINS       = 24                                   # marginal-histogram bins (shared x/y range)
KDE_GRID     = 200                                  # points for the smoothed marginal density
KDE_BW       = 0.35                                 # gaussian_kde bandwidth factor (larger = smoother)
N_PERM       = 20000                                # gene-label permutations (mirror script 21)
PERM_SEED    = 0                                    # mirror script 21's SEED so p-values match exactly
R_MATCH_TOL  = 1e-6                                 # panel r vs canonical r (see make_joint_figure)

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
OUT_PDF_CCA1  = os.path.join(FIG_DIR, f'24.human_mouse_L23_gene_loadings_CCA1{SUFFIX}.pdf')
OUT_PDF_CCA2  = os.path.join(FIG_DIR, f'24.human_mouse_L23_gene_loadings_CCA2{SUFFIX}.pdf')

os.makedirs(FIG_DIR, exist_ok=True)

# --- shared orthologs, centered gene loading blocks ---
# Gene-set construction mirrors 16.load_loadings exactly; the 2000-HVG TSVs always define
# HVG membership, and only supply the loadings under 'hvg_intersect'.
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


# --- permutation significance of the cross-species canonical correlations (mirrors script 21) ---
# The r shown on each panel is the canonical correlation of that axis. Its null shuffles the mouse
# gene labels, destroying the pairing while each species' loading subspace is untouched; permuting
# rows of Y commutes with orthonormalization, so each replicate is one k*k SVD of Qx^T (P.Qy).
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


def make_joint_figure(ms, hs, axis_label, out_pdf, sig):
    """Scatter of (mouse, human) loadings with A/B/C marginal histograms below-x and along-y."""
    # The panel must quote ONE statistic: the Pearson r of the projected loadings and the
    # canonical correlation the p-value tests coincide only if 16's saved weights really are
    # the canonical vectors (16 normalizes and sign-fixes them, which preserves r). Assert
    # that rather than displaying an r from one estimator beside a p from another.
    r = float(np.corrcoef(ms, hs)[0, 1])
    if not abs(abs(r) - sig['r_cca']) < R_MATCH_TOL:
        raise ValueError(
            f'{axis_label}: projected Pearson |r| = {abs(r):.6f} does not match the canonical '
            f'correlation {sig["r_cca"]:.6f} that the permutation p-value tests — 16\'s saved '
            f'weights are not the canonical vectors for this gene universe')
    p_str = (f'p < {1 / (N_PERM + 1):.0e}' if sig['n_ge'] == 0 else f'p = {sig["p"]:.1e}')
    top_idx = np.argsort(np.abs(ms * hs))[::-1][:TOP_N_LABEL]
    lim = np.array([min(ms.min(), hs.min()), max(ms.max(), hs.max())]) * 1.08
    bins = np.linspace(lim[0], lim[1], N_BINS + 1)
    grid = np.linspace(lim[0], lim[1], KDE_GRID)

    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1],
                          wspace=0.04, hspace=0.04)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_histx = fig.add_subplot(gs[1, 0], sharex=ax_main)   # below x-axis
    ax_histy = fig.add_subplot(gs[0, 1], sharey=ax_main)   # along y-axis

    # --- main scatter (mirrors script 20's panel) ---
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
    # r depends on the gene universe and is not comparable across universes — always shown
    # with the universe name and the gene count.
    ax_main.set_title(f'L2/3 {axis_label}: mouse vs human\n'
                      f'r = {r:.3f},  {p_str}  (gene-label permutation, {N_PERM} reps)\n'
                      f'universe: {UNIVERSE},  {len(shared)} orthologous genes',
                      fontsize=11)
    ax_main.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_main.tick_params(labelbottom=False)   # x labels live on the marginal below
    sns.despine(ax=ax_main)

    # --- marginal distributions of A/B/C genes: density histogram (bars) + smoothed KDE curve.
    #     Both use unit-integral density so the faint bars and the KDE line share one y-scale. ---
    for L in ARCH_ORDER:
        color = ARCH_COLORS[L]
        m = letters == L
        ax_histx.hist(ms[m], bins=bins, density=True, color=color, alpha=0.18,
                      histtype='stepfilled', lw=0)
        ax_histy.hist(hs[m], bins=bins, density=True, color=color, alpha=0.18,
                      histtype='stepfilled', lw=0, orientation='horizontal')
        if m.sum() < 2:   # gaussian_kde needs >=2 points
            continue
        dx = gaussian_kde(ms[m], bw_method=KDE_BW)(grid)
        dy = gaussian_kde(hs[m], bw_method=KDE_BW)(grid)
        ax_histx.plot(grid, dx, color=color, lw=1.6)
        ax_histy.plot(dy, grid, color=color, lw=1.6)

    ax_histx.invert_yaxis()   # bars hang downward from the scatter (marginal "below" x-axis)
    ax_histx.axvline(0, color='0.75', lw=0.6, zorder=0)
    ax_histy.axhline(0, color='0.75', lw=0.6, zorder=0)
    ax_histx.set_xlabel(f'Mouse Cheng22 {axis_label} gene loading')
    ax_histx.set_ylabel('density')
    ax_histy.set_xlabel('density')
    ax_histy.tick_params(labelleft=False)   # shares y ticks with the main scatter
    sns.despine(ax=ax_histx)
    sns.despine(ax=ax_histy)

    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return r


print(f'--- L2/3 CCA gene-loading joint plots ---')
print(f'  universe: {UNIVERSE}  ({len(shared)} shared orthologs)')
print(f'  archetype genes on scatter: '
      f'{ {ARCH_RELABEL[L]: int((letters == L).sum()) for L in ARCH_ORDER} } '
      f'of {len(gene2arch)} mouse markers')

print(f'  running {N_PERM} gene-label permutations for CCA1/CCA2 significance...')
sig = permutation_pvals()
for axis in ('CCA1', 'CCA2'):
    s = sig[axis]
    print(f'  {axis}: r_cca {s["r_cca"]:.3f}, {s["n_ge"]}/{N_PERM} >= obs, '
          f'p = {s["p"]:.2e}')

ms_cca1, hs_cca1 = cca_load('CCA1')
ms_cca2, hs_cca2 = cca_load('CCA2')
r1 = make_joint_figure(ms_cca1, hs_cca1, 'CCA1', OUT_PDF_CCA1, sig['CCA1'])
r2 = make_joint_figure(ms_cca2, hs_cca2, 'CCA2', OUT_PDF_CCA2, sig['CCA2'])
print(f'  panel r: CCA1 {r1:.3f}, CCA2 {r2:.3f}')
print(f'  Saved {OUT_PDF_CCA1}')
print(f'  Saved {OUT_PDF_CCA2}')
print('\nDone.')
