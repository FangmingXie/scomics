"""Mouse archetype scores on the L4 CCA1xCCA3 embedding (plots only).

Script 18c restricted to L4 and re-planed onto CCA1 x CCA3 instead of CCA1 x CCA2. Everything
else — the projection, the vertex carry-over, the primed relabelling, the per-panel colour
clipping — is identical to 18c; only CCA_AXES changes, and all axis labels/filenames derive
from it.

Why CCA3 for L4: the raw canonical correlations rank CCA1 > CCA2 > CCA3 (0.365 / 0.267 /
0.244), but the blocked-CV r from 16 — the generalization number, folds blocked by
co-expression module — ranks them CCA3 (0.264) > CCA1 (0.153) > CCA2 (0.124). L4's CCA2 is
the weakest of the three under cross-validation and its raw r sits only ~1.4x above CCA3's,
so the CCA1 x CCA2 plane 18c draws is not the best-supported 2D view of L4. CCA3 is also the
only L4 component whose held-out r exceeds its in-sample-fit ranking, i.e. the one that
degrades least out of fold. Note the two axes are NOT co-equal here: CCA1 has the larger raw
r, CCA3 the larger CV r, so read the plane as "best raw axis x best generalizing axis", not
as a variance-ordered embedding.

Both rows share one construction. A cell's CCA coordinate is `(VX coords - mean) . a_vx`, with
`a_vx` the Gate-A VX canonical weight vector from 16 (normalized + sign-fixed there); the species'
PCHA archetype vertices are carried into the same frame via `aa_vx = aa . inner_comps + inner_mean`
(scripts 04 human / 05 mouse) and the same projection, so cells and vertices share one coordinate
system per row. The human row uses the mouse-derived scores computed on human cells (06); the mouse
row uses the native mouse scores (05). Vertices carry the established primed figure labels from the
persisted depth arcs — human via `22.human_IT_joint_archetype_arc_order.tsv`, mouse via
`15.mouse_IT_joint_archetype_arc_order.tsv`. A' is the most superficial archetype; score panels are
ordered A', B', C' and each panel's colormap follows its displayed label (A'->C0, B'->C1, C'->C2).
Axes are shown in their raw orientation (no sign-flip). Colour limits are clipped per panel to the
5th-95th percentile of that panel's score.

Caveat: per script 21, L4's cross-species canonical correlations are weak in absolute terms
(CCA1 r=0.365, CCA3 r=0.244, both z>7 against the gene-label null but on n=479 orthologs), so
this embedding is exploratory. Only L2/3 shows clearly conserved axes; 18b remains the dedicated
L2/3 figure.

Reads:
  local_data/res/it_evo/06.mouse_L4_archetype_scores_on_human_cells.tsv
  local_data/res/it_evo/02.human_L4_varimax_coords.tsv
  local_data/res/it_evo/16.L4_axis_cca_weights_human.tsv
  local_data/res/it_evo/04.human_L4_pcha_{aa,inner_components,inner_mean}.tsv
  local_data/res/it_evo/05.mouse_L4_archetype_scores.tsv
  local_data/res/it/21.cheng22_L4_varimax_coords.tsv
  local_data/res/it_evo/16.L4_axis_cca_weights_mouse.tsv
  local_data/res/it_evo/05.mouse_L4_pcha_{aa,inner_components,inner_mean}.tsv
Outputs:
  local_data/fig/it_evo/18d.human_mouse_L4_scores_cca1_cca3.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- directories ---
RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
# established figure relabel/reorder (scripts 12-14): mouse via 15's depth arc, human via 22's
IN_M_ARC   = os.path.join(RES_DIR, '15.mouse_IT_joint_archetype_arc_order.tsv')
IN_H_ARC   = os.path.join(RES_DIR, '22.human_IT_joint_archetype_arc_order.tsv')

# --- subclass (Gate-A VX sets mirror script 21) ---
CFG = {'token': 'L4', 'label': 'L4',
       'human_vx': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'],
       'mouse_vx': ['VX1', 'VX4', 'VX5', 'VX6'],
       'mouse_coords': '21.cheng22_L4_varimax_coords.tsv'}

# --- config ---
CCA_AXES     = ['CCA1', 'CCA3']   # the plane; x then y. All labels/filenames derive from this.
ALPHABET     = ['A', 'B', 'C', 'D', 'E', 'F']
SCORE_PCTILE = (5, 95)   # clip each panel's color range to the 5th-95th pct of its score
POINT_SIZE   = 3
# archetype-tinted sequential colormaps keyed by the DISPLAYED (primed) label, so color follows
# the figure label: A'->C0, B'->C1, C'->C2, ... (independent of 05's arbitrary A/B/C order).
ARCH_BASE    = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
ARCH_CMAPS   = {f"{L}'": LinearSegmentedColormap.from_list(f'gray_{ARCH_BASE[i]}',
                                                          ['gainsboro', 'lightgray', ARCH_BASE[i]])
                for i, L in enumerate(ALPHABET)}

AXES_TAG = '_'.join(a.lower() for a in CCA_AXES)
OUT_PDF  = os.path.join(FIG_DIR, f'18d.human_mouse_{CFG["token"]}_scores_{AXES_TAG}.pdf')

# per-subclass, per-species old-letter -> new (primed) figure letter, from the persisted depth
# arcs (read, not hard-coded, so the labels match scripts 12-14 exactly).
M_ARC = pd.read_csv(IN_M_ARC, sep='\t')
H_ARC = pd.read_csv(IN_H_ARC, sep='\t')


def relabel_for(arc, token):
    """{old_letter: new_letter} for one subclass from a 15/22-style depth-arc table."""
    sub = arc[arc['token'] == token]
    return dict(zip(sub['old_letter'], sub['new_letter']))


os.makedirs(FIG_DIR, exist_ok=True)


def embed(coords_path, vx_cols, weights_path, aa_path, inner_cmp_path, inner_mean_path, relabel):
    """Project cells and PCHA archetype vertices into the shared CCA_AXES frame for one species.

    Vertices are labelled by `relabel` (05/04 old letter A/B/C.. -> the established primed figure
    letter from 15/22's depth arc). Returns (cell index, cell coords, ordered vertex coords,
    ordered vertex labels, [r of each CCA_AXES component]).
    """
    vx_df   = pd.read_csv(coords_path, sep='\t', index_col=0)
    weights = pd.read_csv(weights_path, sep='\t', index_col=0)
    aa_df   = pd.read_csv(aa_path, sep='\t', index_col=0)
    inner_cmp  = pd.read_csv(inner_cmp_path, sep='\t', index_col=0)
    inner_mean = pd.read_csv(inner_mean_path, sep='\t', index_col=0)[vx_cols].values.ravel()

    W = weights.loc[CCA_AXES, vx_cols].values.T          # (n_vx x 2 CCA), sign-fixed by 16
    r_cca = weights.loc[CCA_AXES, 'canonical_r'].values

    C = vx_df[vx_cols].values
    mean_vx = C.mean(axis=0)
    cca_cells = (C - mean_vx) @ W                        # cells x 2

    aa_vx = aa_df.values @ inner_cmp.loc[list(aa_df.columns), vx_cols].values + inner_mean
    cca_aa = (aa_vx - mean_vx) @ W                       # n_arch x 2

    # archetype_i -> established primed figure letter; order by angle for a clean (simple) polygon
    labels = np.array([relabel[ALPHABET[int(i.split('_')[1]) - 1]] for i in aa_df.index])
    ang = np.arctan2(cca_aa[:, 1] - cca_aa[:, 1].mean(), cca_aa[:, 0] - cca_aa[:, 0].mean())
    order = np.argsort(ang)
    return vx_df.index, cca_cells, cca_aa[order], labels[order], r_cca


def draw_panel(ax, coords, vals, cmap, aa, aa_labels, r_cca, title, fig):
    vmin, vmax = np.percentile(vals, SCORE_PCTILE)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=POINT_SIZE, linewidths=0, rasterized=True)
    ax.plot(list(aa[:, 0]) + [aa[0, 0]], list(aa[:, 1]) + [aa[0, 1]], '-', color='black', lw=1.0)
    ax.scatter(aa[:, 0], aa[:, 1], marker='D', color='black', s=30, zorder=3)
    for (ax_, ay_), label in zip(aa[:, :2], aa_labels):
        ax.annotate(label, (ax_, ay_), textcoords='offset points', xytext=(5, 5),
                    fontsize=8, fontweight='bold', color='black', zorder=4)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'{CCA_AXES[0]} (r={r_cca[0]:.3f})')
    ax.set_ylabel(f'{CCA_AXES[1]} (r={r_cca[1]:.3f})')
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label='archetype score [0-1]', shrink=0.8)
    sns.despine(ax=ax)


tok = CFG['token']
IN_H_SCORES     = os.path.join(RES_DIR, f'06.mouse_{tok}_archetype_scores_on_human_cells.tsv')
IN_H_COORDS     = os.path.join(RES_DIR, f'02.human_{tok}_varimax_coords.tsv')
IN_H_WEIGHTS    = os.path.join(RES_DIR, f'16.{tok}_axis_cca_weights_human.tsv')
IN_H_AA         = os.path.join(RES_DIR, f'04.human_{tok}_pcha_aa.tsv')
IN_H_INNER_CMP  = os.path.join(RES_DIR, f'04.human_{tok}_pcha_inner_components.tsv')
IN_H_INNER_MEAN = os.path.join(RES_DIR, f'04.human_{tok}_pcha_inner_mean.tsv')
IN_M_SCORES     = os.path.join(RES_DIR, f'05.mouse_{tok}_archetype_scores.tsv')
IN_M_COORDS     = os.path.join(IT_RES_DIR, CFG['mouse_coords'])
IN_M_WEIGHTS    = os.path.join(RES_DIR, f'16.{tok}_axis_cca_weights_mouse.tsv')
IN_M_AA         = os.path.join(RES_DIR, f'05.mouse_{tok}_pcha_aa.tsv')
IN_M_INNER_CMP  = os.path.join(RES_DIR, f'05.mouse_{tok}_pcha_inner_components.tsv')
IN_M_INNER_MEAN = os.path.join(RES_DIR, f'05.mouse_{tok}_pcha_inner_mean.tsv')

# native mouse scores define the archetype letters (score_A, score_B, ...)
m_scores = pd.read_csv(IN_M_SCORES, sep='\t', index_col=0)
h_scores = pd.read_csv(IN_H_SCORES, sep='\t', index_col=0)
if list(h_scores.columns) != list(m_scores.columns):
    raise ValueError(f'{tok}: score columns differ between {IN_H_SCORES} ({list(h_scores.columns)}) '
                     f'and {IN_M_SCORES} ({list(m_scores.columns)}).')
# native mouse score letters, reordered so panels read A', B', C' (depth order from 15)
m_relabel = relabel_for(M_ARC, tok)
h_relabel = relabel_for(H_ARC, tok)
letters = sorted((c.split('_')[1] for c in m_scores.columns), key=lambda L: m_relabel[L])

h_index, h_cells, h_aa, h_labels, h_r = embed(
    IN_H_COORDS, CFG['human_vx'], IN_H_WEIGHTS, IN_H_AA, IN_H_INNER_CMP, IN_H_INNER_MEAN,
    h_relabel)
if not h_scores.index.equals(h_index):
    raise ValueError(f'{tok}: human cell index mismatch: {IN_H_SCORES} vs {IN_H_COORDS} — '
                     f'scores and coords must cover the same cells in the same order.')

m_index, m_cells, m_aa, m_labels, m_r = embed(
    IN_M_COORDS, CFG['mouse_vx'], IN_M_WEIGHTS, IN_M_AA, IN_M_INNER_CMP, IN_M_INNER_MEAN,
    m_relabel)
if not m_scores.index.equals(m_index):
    raise ValueError(f'{tok}: mouse cell index mismatch: {IN_M_SCORES} vs {IN_M_COORDS} — '
                     f'scores and coords must cover the same cells in the same order.')

rows = [
    {'species': 'Human Jorstad23 (mouse scores)', 'cells': h_cells, 'aa': h_aa,
     'labels': h_labels, 'r': h_r, 'scores': h_scores},
    {'species': 'Mouse Cheng22 (native scores)', 'cells': m_cells, 'aa': m_aa,
     'labels': m_labels, 'r': m_r, 'scores': m_scores},
]

axes_str = ' x '.join(CCA_AXES)
print(f'--- {CFG["label"]}: mouse {[m_relabel[L] for L in letters]} scores on {axes_str} '
      f'(r = {h_r[0]:.3f}, {h_r[1]:.3f}) ---')
print(f'  human cells {len(h_cells)}, mouse cells {len(m_cells)}')

plt.rcParams['pdf.fonttype'] = 42
ncols = len(letters)
fig, axes = plt.subplots(len(rows), ncols, figsize=(4.2 * ncols, 4 * len(rows)), squeeze=False)
for i, row in enumerate(rows):
    for j, L in enumerate(letters):
        disp = m_relabel[L]   # displayed primed letter drives both title and color
        draw_panel(axes[i][j], row['cells'], row['scores'][f'score_{L}'].values,
                   ARCH_CMAPS[disp], row['aa'], row['labels'], row['r'],
                   f'{row["species"]} — Score {disp}', fig)

fig.suptitle(f'{CFG["label"]} mouse Cheng22 archetype scores on the {axes_str} embedding '
             f'— human vs mouse cells')
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
