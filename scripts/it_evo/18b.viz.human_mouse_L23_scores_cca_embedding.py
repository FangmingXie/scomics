"""Mouse L2/3 archetype scores on the conserved CCA1xCCA2 embedding, human AND mouse (plots only).

Script 18 rendered the mouse-derived L2/3 scores (score_B and the A-C contrast) on the *human*
CCA1xCCA2 embedding. This script replots those two panels and adds the matching pair for *mouse*
cells positioned on the *mouse* CCA1xCCA2 embedding, colored by the native mouse archetype scores.
The result is a 2x2 grid: rows are species (top = human Jorstad23, bottom = mouse Cheng22),
columns are the two score readouts, shown with the published primed labels (Score B', Score
C'-A'; internal score_B and score_A-score_C, via ARCH_RELABEL {A:C', B:B', C:A'}).

Both rows share one construction. A cell's CCA coordinate is `(VX coords - mean) . a_vx`, with
`a_vx` the Gate-A VX canonical weight vector from 16 (normalized + sign-fixed there); the species'
PCHA archetype vertices are carried into the same frame via `aa_vx = aa . inner_comps + inner_mean`
(scripts 04 human / 05 mouse) and the same projection, so cells and vertices share one coordinate
system per row. The two species' CCA axes are the matched cross-species canonical directions from
16, so both rows carry the same canonical correlations (r = 0.623, 0.488).

The human row uses the same mouse-derived scores as script 18 (computed on human cells by 06); the
human PCHA vertices are the four human archetypes, labelled with primed letters (A'..D') to mark
that they are *not* the mouse archetypes that define the scores. The mouse row uses the native
mouse archetype scores (05) and the three mouse archetype vertices. Both rows are shown reflected
along CCA1 (axis labelled CCA1') so the two embeddings share one orientation, and the mouse
vertices are relabelled A->C', B->B', C->A' to line up with the human layout — cosmetic only, the
coordinates/scores/canonical correlation are unchanged. Color limits are clipped per panel to the
5th-95th percentile of that panel's score (95th of |A-C| for the symmetric contrast), so the same
colormap means the same score range within a panel, not across species (the two rows are different
measurements: imputed-on-human vs native-on-mouse).

Reads:
  local_data/res/it_evo/06.mouse_L23_archetype_scores_on_human_cells.tsv
  local_data/res/it_evo/02.human_L23_varimax_coords.tsv
  local_data/res/it_evo/16.L23_axis_cca_weights_human.tsv
  local_data/res/it_evo/04.human_L23_pcha_{aa,inner_components,inner_mean}.tsv
  local_data/res/it_evo/05.mouse_L23_archetype_scores.tsv
  local_data/res/it/19.cheng22_L23_varimax_coords.tsv
  local_data/res/it_evo/16.L23_axis_cca_weights_mouse.tsv
  local_data/res/it_evo/05.mouse_L23_pcha_{aa,inner_components,inner_mean}.tsv
Outputs:
  local_data/fig/it_evo/18b.human_mouse_L23_scores_cca.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
# human row (mouse-derived scores on human cells, human embedding) — script 18's inputs
IN_H_SCORES   = os.path.join(RES_DIR, '06.mouse_L23_archetype_scores_on_human_cells.tsv')
IN_H_COORDS   = os.path.join(RES_DIR, '02.human_L23_varimax_coords.tsv')
IN_H_WEIGHTS  = os.path.join(RES_DIR, '16.L23_axis_cca_weights_human.tsv')
IN_H_AA       = os.path.join(RES_DIR, '04.human_L23_pcha_aa.tsv')
IN_H_INNER_CMP  = os.path.join(RES_DIR, '04.human_L23_pcha_inner_components.tsv')
IN_H_INNER_MEAN = os.path.join(RES_DIR, '04.human_L23_pcha_inner_mean.tsv')
# mouse row (native mouse scores on mouse cells, mouse embedding)
IN_M_SCORES   = os.path.join(RES_DIR, '05.mouse_L23_archetype_scores.tsv')
IN_M_COORDS   = os.path.join(IT_RES_DIR, '19.cheng22_L23_varimax_coords.tsv')
IN_M_WEIGHTS  = os.path.join(RES_DIR, '16.L23_axis_cca_weights_mouse.tsv')
IN_M_AA       = os.path.join(RES_DIR, '05.mouse_L23_pcha_aa.tsv')
IN_M_INNER_CMP  = os.path.join(RES_DIR, '05.mouse_L23_pcha_inner_components.tsv')
IN_M_INNER_MEAN = os.path.join(RES_DIR, '05.mouse_L23_pcha_inner_mean.tsv')
OUT_PDF       = os.path.join(FIG_DIR, '18b.human_mouse_L23_scores_cca.pdf')

# --- config ---
CCA_AXES  = ['CCA1', 'CCA2']
HUMAN_VX  = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']   # Gate-A human L2/3 (04/16)
MOUSE_VX  = ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9']    # Gate-A mouse L2/3 (05/16)
# internal mouse archetype keys A/B/C index score_* columns; displayed with the published
# primed labels (matching scripts/it/41,48,50, script 18, and the primed vertices below).
ARCH_RELABEL = {'A': "C'", 'B': "B'", 'C': "A'"}          # internal -> displayed
PANELS    = [{'name': ARCH_RELABEL['B'], 'pos': 'B'},     # columns: sequential score, then contrast
             {'name': f"{ARCH_RELABEL['A']} - {ARCH_RELABEL['C']}", 'pos': 'A', 'neg': 'C'}]
# human PCHA vertices are the four human archetypes -> primed letters (script 07/18); mouse PCHA
# vertices are the three mouse archetypes that define the scores. The mouse row is shown reflected
# (CCA1 -> -CCA1, axis labelled CCA1') and its vertices relabelled A->C', B->B', C->A' purely so
# the mouse embedding lines up visually with the human one; the scores/coordinates are unchanged.
AA_RENAME_H = {'archetype_1': "D'", 'archetype_2': "C'", 'archetype_3': "B'", 'archetype_4': "A'"}
AA_RENAME_M = {'archetype_1': "C'", 'archetype_2': "B'", 'archetype_3': "A'"}

# --- parameters ---
SCORE_PCTILE = (5, 95)   # clip sequential-score color range to 5th-95th pct
DIFF_PCTILE  = 95        # clip A-C contrast color range to the 95th pct of |A-C|
POINT_SIZE   = 3
CMAP_B       = LinearSegmentedColormap.from_list('gray_C1', ['gainsboro', 'lightgray', 'C1'])
CMAP_DIFF    = LinearSegmentedColormap.from_list('C0_gray_C2', ['C0', 'lightgray', 'C2'])

os.makedirs(FIG_DIR, exist_ok=True)


def embed(coords_path, vx_cols, weights_path, aa_path, inner_cmp_path, inner_mean_path,
          aa_rename, flip_cca1=False):
    """Project cells and PCHA archetype vertices into the shared CCA1xCCA2 frame for one species.

    flip_cca1 negates the CCA1 coordinate (x) of both cells and vertices — a visual reflection
    only (the canonical correlation and every score are unchanged).
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

    if flip_cca1:                                        # visual reflection along CCA1 only
        cca_cells[:, 0] *= -1
        cca_aa[:, 0] *= -1

    # order vertices by angle for a clean (non-self-intersecting) polygon, labels attached
    labels = np.array([aa_rename[i] for i in aa_df.index])
    ang = np.arctan2(cca_aa[:, 1] - cca_aa[:, 1].mean(), cca_aa[:, 0] - cca_aa[:, 0].mean())
    order = np.argsort(ang)
    return vx_df.index, cca_cells, cca_aa[order], labels[order], r_cca


def panel_specs(scores_df):
    """(vals, cmap, vlim, cbar) per column, computed on this species' own scores (07/18 rules)."""
    specs = []
    for p in PANELS:
        vals = scores_df[f'score_{p["pos"]}'].values
        if 'neg' in p:
            vals = vals - scores_df[f'score_{p["neg"]}'].values
            lim = np.percentile(np.abs(vals), DIFF_PCTILE)
            specs.append((vals, CMAP_DIFF, (-lim, lim), 'score difference'))
        else:
            specs.append((vals, CMAP_B, tuple(np.percentile(vals, SCORE_PCTILE)),
                          'archetype score [0-1]'))
    return specs


def draw_panel(ax, coords, vals, cmap, vlim, cbar, aa, aa_labels, r_cca, title, fig, cca1_label):
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                    s=POINT_SIZE, linewidths=0, rasterized=True)
    ax.plot(list(aa[:, 0]) + [aa[0, 0]], list(aa[:, 1]) + [aa[0, 1]], '-', color='black', lw=1.0)
    ax.scatter(aa[:, 0], aa[:, 1], marker='D', color='black', s=30, zorder=3)
    for (ax_, ay_), label in zip(aa[:, :2], aa_labels):
        ax.annotate(label, (ax_, ay_), textcoords='offset points', xytext=(5, 5),
                    fontsize=8, fontweight='bold', color='black', zorder=4)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'{cca1_label} (r={r_cca[0]:.3f})')
    ax.set_ylabel(f'CCA2 (r={r_cca[1]:.3f})')
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=cbar, shrink=0.8)
    sns.despine(ax=ax)


# --- human row: mouse-derived scores on human cells, human CCA embedding (script 18) ---
h_scores = pd.read_csv(IN_H_SCORES, sep='\t', index_col=0)
h_index, h_cells, h_aa, h_labels, h_r = embed(
    IN_H_COORDS, HUMAN_VX, IN_H_WEIGHTS, IN_H_AA, IN_H_INNER_CMP, IN_H_INNER_MEAN, AA_RENAME_H,
    flip_cca1=True)   # reflect along CCA1 too, so both rows share the CCA1' orientation
if not h_scores.index.equals(h_index):
    raise ValueError(f'Human cell index mismatch: {IN_H_SCORES} vs {IN_H_COORDS} — mouse scores '
                     f'and human coords must cover the same cells in the same order.')

# --- mouse row: native mouse scores on mouse cells, mouse CCA embedding ---
m_scores = pd.read_csv(IN_M_SCORES, sep='\t', index_col=0)
m_index, m_cells, m_aa, m_labels, m_r = embed(
    IN_M_COORDS, MOUSE_VX, IN_M_WEIGHTS, IN_M_AA, IN_M_INNER_CMP, IN_M_INNER_MEAN, AA_RENAME_M,
    flip_cca1=True)   # reflect the mouse embedding along CCA1 for visual alignment (CCA1')
if not m_scores.index.equals(m_index):
    raise ValueError(f'Mouse cell index mismatch: {IN_M_SCORES} vs {IN_M_COORDS} — mouse scores '
                     f'and mouse coords must cover the same cells in the same order.')

rows = [
    {'species': 'Human Jorstad23 (mouse scores)', 'cells': h_cells, 'aa': h_aa,
     'labels': h_labels, 'r': h_r, 'specs': panel_specs(h_scores), 'cca1_label': "CCA1'"},
    {'species': 'Mouse Cheng22 (native scores)', 'cells': m_cells, 'aa': m_aa,
     'labels': m_labels, 'r': m_r, 'specs': panel_specs(m_scores), 'cca1_label': "CCA1'"},
]

print(f'--- L2/3 mouse {ARCH_RELABEL["B"]} / {ARCH_RELABEL["A"]}-{ARCH_RELABEL["C"]} scores '
      f'on conserved CCA1xCCA2 (r = {h_r[0]:.3f}, {h_r[1]:.3f}) ---')
print(f'  human cells {len(h_cells)}, mouse cells {len(m_cells)}')

plt.rcParams['pdf.fonttype'] = 42
fig, axes = plt.subplots(len(rows), len(PANELS), figsize=(4.2 * len(PANELS), 4 * len(rows)),
                         squeeze=False)
for i, row in enumerate(rows):
    for j, (p, (vals, cmap, vlim, cbar)) in enumerate(zip(PANELS, row['specs'])):
        draw_panel(axes[i][j], row['cells'], vals, cmap, vlim, cbar, row['aa'], row['labels'],
                   row['r'], f'{row["species"]} — Score {p["name"]}', fig, row['cca1_label'])
        if 'neg' in p:   # the contrast panel
            print(f'  {row["species"]}: {p["name"]} range [{vals.min():.3f}, {vals.max():.3f}], '
                  f'color limit +/-{vlim[1]:.3f}')

fig.suptitle('L2/3 mouse Cheng22 archetype scores on the conserved CCA1xCCA2 embedding — '
             'human vs mouse cells')
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
