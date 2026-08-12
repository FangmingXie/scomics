"""Mouse vs human ortholog gene loadings along the L2/3 conserved axes (L2/3 only, plots only).

Three panels, one per shared coordinate system. Each shared 1-to-1 ortholog is placed at
(mouse loading, human loading), where the loading is the gene's projection onto that species'
axis. Because these are gene canonical/PC variates, the Pearson correlation of each scatter is
the axis's cross-species agreement — for the CCA panels it equals the canonical correlation
(0.623, 0.488); the cloud's diagonal alignment *is* the conserved axis.

  1. CCA1  — the leading conserved canonical direction (script 16 weights).
  2. CCA2  — the second conserved direction.
  3. PC1   — each species' own leading PC of the Gate-A VX cell coordinates (plan 04's
             construction, sign-fixed so the largest |VX| coefficient is positive). Unlike
             CCA, PC1 is computed independently per species, so this panel shows whether the
             dominant *unsupervised* axes also align across species.

Overlays (all panels): the top genes by |mouse · human| loading are labelled (black ring);
mouse L2/3 archetype marker genes (05) that are shared orthologs are colored A→C0, B→C1,
C→C2, so one can read which pole of each axis an archetype's program sits on. The legend
reports how many genes fall in each category (A/B/C/other). No CCA/loadings refit — weights
read from 16's persisted TSVs.

Reads:
  local_data/res/it_evo/02.human_L23_varimax_{loadings,coords}.tsv
  local_data/res/it/19.cheng22_L23_varimax_{loadings,coords}.tsv
  local_data/res/it_evo/16.L23_axis_cca_weights_{human,mouse}.tsv
  local_data/res/it_evo/05.mouse_L23_archetype_markers.tsv
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/fig/it_evo/20.human_mouse_L23_gene_loadings.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_H_LOADINGS = os.path.join(RES_DIR, '02.human_L23_varimax_loadings.tsv')
IN_H_COORDS   = os.path.join(RES_DIR, '02.human_L23_varimax_coords.tsv')
IN_M_LOADINGS = os.path.join(IT_RES_DIR, '19.cheng22_L23_varimax_loadings.tsv')
IN_M_COORDS   = os.path.join(IT_RES_DIR, '19.cheng22_L23_varimax_coords.tsv')
IN_W_HUMAN    = os.path.join(RES_DIR, '16.L23_axis_cca_weights_human.tsv')
IN_W_MOUSE    = os.path.join(RES_DIR, '16.L23_axis_cca_weights_mouse.tsv')
IN_MARKERS    = os.path.join(RES_DIR, '05.mouse_L23_archetype_markers.tsv')
IN_ORTHOLOGS  = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_PDF       = os.path.join(FIG_DIR, '20.human_mouse_L23_gene_loadings.pdf')

# --- config ---
HUMAN_VX     = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
MOUSE_VX     = ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9']
MOUSE_NOC    = 3
ALPHABET     = ['A', 'B', 'C', 'D', 'E', 'F']
ARCH_COLORS  = {'A': 'C0', 'B': 'C1', 'C': 'C2'}   # A→C0, B→C1, C→C2
BASE_COLOR   = '#bdbdbd'                            # "other" genes: neutral gray
POINT_SIZE   = 10
TOP_N_LABEL  = 20                                   # label the top genes by |mouse·human| loading

os.makedirs(FIG_DIR, exist_ok=True)

# --- shared orthologs, centered gene loading blocks ---
ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
H = pd.read_csv(IN_H_LOADINGS, sep='\t', index_col=0)
M = pd.read_csv(IN_M_LOADINGS, sep='\t', index_col=0)
shared = ortho[ortho['human_symbol'].isin(H.index)
               & ortho['mouse_symbol'].isin(M.index)].reset_index(drop=True)
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


ms_cca1, hs_cca1 = cca_load('CCA1')
ms_cca2, hs_cca2 = cca_load('CCA2')

# --- PC1 gene loadings: PCA on the Gate-A VX cell coords per species (plan 04) ---
def species_pc1(coords_path, vx_cols):
    coords = pd.read_csv(coords_path, sep='\t', index_col=0)[vx_cols].values
    pc1 = PCA(n_components=len(vx_cols)).fit(coords).components_[0]
    return pc1 * np.sign(pc1[np.argmax(np.abs(pc1))])   # largest |VX| coef positive


ms_pc1 = Yc @ species_pc1(IN_M_COORDS, MOUSE_VX)
hs_pc1 = Xc @ species_pc1(IN_H_COORDS, HUMAN_VX)

# --- mouse archetype membership per gene (highest-log2FC archetype if multiple) ---
mk = pd.read_csv(IN_MARKERS, sep='\t')
mk['letter'] = mk['archetype'].map({f'archetype_{i+1}': ALPHABET[i] for i in range(MOUSE_NOC)})
mk = mk.sort_values('log2FC', ascending=False).drop_duplicates('gene')
gene2arch = dict(zip(mk['gene'], mk['letter']))
letters = np.array([gene2arch.get(g, '') for g in shared['mouse_symbol'].values])


def panel(ax, ms, hs, axis_label, show_legend):
    r = np.corrcoef(ms, hs)[0, 1]
    top_idx = np.argsort(np.abs(ms * hs))[::-1][:TOP_N_LABEL]

    n_other = int((letters == '').sum())
    ax.scatter(ms[letters == ''], hs[letters == ''], s=POINT_SIZE, c=BASE_COLOR,
               linewidths=0, rasterized=True, label=f'other (n={n_other})')
    for L, color in ARCH_COLORS.items():
        m = letters == L
        ax.scatter(ms[m], hs[m], s=POINT_SIZE + 12, c=color, linewidths=0, alpha=0.9,
                   label=f'mouse archetype {L} (n={int(m.sum())})', zorder=3)

    lim = np.array([min(ms.min(), hs.min()), max(ms.max(), hs.max())]) * 1.08
    ax.axhline(0, color='0.75', lw=0.6, zorder=0)
    ax.axvline(0, color='0.75', lw=0.6, zorder=0)
    ax.plot(lim, lim, '--', color='0.6', lw=0.8, zorder=0)

    # label the top genes by conserved loading magnitude (black ring + name)
    ax.scatter(ms[top_idx], hs[top_idx], s=POINT_SIZE + 18, facecolors='none',
               edgecolors='black', linewidths=0.7, zorder=4)
    for i in top_idx:
        ax.annotate(symbols[i], (ms[i], hs[i]), textcoords='offset points',
                    xytext=(4, 3), fontsize=7, fontstyle='italic', zorder=5)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'Mouse Cheng22 {axis_label} gene loading')
    ax.set_ylabel(f'Human Jorstad23 {axis_label} gene loading')
    ax.set_title(f'L2/3 {axis_label}: mouse vs human (r = {r:.3f})')
    if show_legend:
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    sns.despine(ax=ax)
    return r


print(f'--- L2/3 gene loadings, mouse vs human: {len(shared)} shared orthologs ---')
print(f'  archetype genes on scatter: '
      f'{ {L: int((letters == L).sum()) for L in ARCH_COLORS} }')

plt.rcParams['pdf.fonttype'] = 42
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
r1 = panel(axes[0], ms_cca1, hs_cca1, 'CCA1', show_legend=True)
r2 = panel(axes[1], ms_cca2, hs_cca2, 'CCA2', show_legend=False)
r3 = panel(axes[2], ms_pc1, hs_pc1, 'PC1', show_legend=False)
print(f'  panel r: CCA1 {r1:.3f}, CCA2 {r2:.3f}, PC1 {r3:.3f}')

fig.suptitle('L2/3 conserved axes — mouse vs human ortholog gene loadings')
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
