"""Mouse L2/3 archetype scores rendered on the human CCA1×CCA2 embedding (L2/3 only, plots only).

Script 07's figure, re-based from the human PCHA PC1/PC2 embedding onto the cross-species
conserved axes: the human CCA1 and CCA2 canonical directions from script 16. L2/3 is the only
subclass where those directions are conserved and estimable (16's report), so this is the one
place the re-basing is meaningful.

The same mouse-derived scores as script 07 (score_B and the A−C contrast, computed on human
cells by script 06) color the human cells, now positioned by their projection onto the human
canonical weights. Mouse score panels carry the published primed labels (internal score_B → B',
score_A−score_C → C'−A', via ARCH_RELABEL), matching the primed human vertices and scripts/it/41,48,50. A cell's CCA coordinate is `(VX coords − mean) · a_vx`, where `a_vx` is the
Gate-A VX canonical weight vector from 16; the human PCHA archetype vertices are carried into
the same frame via `aa_vx = aa · inner_comps + inner_mean` (script 04) followed by the same
projection, so cells and vertices share one coordinate system. Styling matches 07 exactly
(SCORE_PCTILE=(2,98), symmetric DIFF_PCTILE=98, CMAP_B, CMAP_DIFF). No recomputation of scores
or weights — a pure re-embedding.

Reads:
  local_data/res/it_evo/06.mouse_L23_archetype_scores_on_human_cells.tsv
  local_data/res/it_evo/02.human_L23_varimax_coords.tsv
  local_data/res/it_evo/16.L23_axis_cca_weights_human.tsv
  local_data/res/it_evo/04.human_L23_pcha_{aa,inner_components,inner_mean}.tsv
Outputs:
  local_data/fig/it_evo/18.human_L23_mouse_b_ca_scores_cca.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_score_scatter_pdf

# --- file paths ---
RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_SCORES    = os.path.join(RES_DIR, '06.mouse_L23_archetype_scores_on_human_cells.tsv')
IN_VX_COORDS = os.path.join(RES_DIR, '02.human_L23_varimax_coords.tsv')
IN_WEIGHTS   = os.path.join(RES_DIR, '16.L23_axis_cca_weights_human.tsv')
IN_AA        = os.path.join(RES_DIR, '04.human_L23_pcha_aa.tsv')
IN_INNER_CMP = os.path.join(RES_DIR, '04.human_L23_pcha_inner_components.tsv')
IN_INNER_MEAN = os.path.join(RES_DIR, '04.human_L23_pcha_inner_mean.tsv')
OUT_PDF      = os.path.join(FIG_DIR, '18.human_L23_mouse_b_ca_scores_cca.pdf')

# --- config (mirrors 07's L23 entry) ---
HUMAN_SUBCLASS = 'L2/3 IT'
MOUSE_SUBCLASS = 'L2/3'
VX_COLS        = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']   # Gate-A human L2/3 (04/16)
CCA_AXES       = ['CCA1', 'CCA2']
# internal mouse archetype keys A/B/C index score_* columns; displayed with the published
# primed labels (matching scripts/it/41,48,50 and the primed human vertices below).
ARCH_RELABEL   = {'A': "C'", 'B': "B'", 'C': "A'"}   # internal -> displayed
PANELS         = [{'name': ARCH_RELABEL['B'], 'pos': 'B'},
                  {'name': f"{ARCH_RELABEL['A']} − {ARCH_RELABEL['C']}", 'pos': 'A', 'neg': 'C'}]
# human PCHA archetype vertices are archetype_1..4; 07's L23 rename to primed letters
AA_RENAME      = {'archetype_1': "D'", 'archetype_2': "C'", 'archetype_3': "B'", 'archetype_4': "A'"}

# --- parameters (07's styling, verbatim) ---
SCORE_PCTILE = (2, 98)
DIFF_PCTILE  = 98
CMAP_B       = LinearSegmentedColormap.from_list('gray_C1', ['gainsboro', 'lightgray', 'C1'])
CMAP_DIFF    = LinearSegmentedColormap.from_list('C0_gray_C2', ['C0', 'lightgray', 'C2'])

os.makedirs(FIG_DIR, exist_ok=True)

# --- load ---
scores_df = pd.read_csv(IN_SCORES, sep='\t', index_col=0)
vx_df     = pd.read_csv(IN_VX_COORDS, sep='\t', index_col=0)
weights   = pd.read_csv(IN_WEIGHTS, sep='\t', index_col=0)
aa_df     = pd.read_csv(IN_AA, sep='\t', index_col=0)
inner_cmp = pd.read_csv(IN_INNER_CMP, sep='\t', index_col=0)
inner_mean = pd.read_csv(IN_INNER_MEAN, sep='\t', index_col=0)[VX_COLS].values.ravel()

if not scores_df.index.equals(vx_df.index):
    raise ValueError(
        f'Cell index mismatch: {IN_SCORES} ({len(scores_df)} cells) vs {IN_VX_COORDS} '
        f'({len(vx_df)} cells) — mouse scores and human coords must cover the same cells '
        f'in the same order.')

# --- human CCA1/CCA2 weight vectors (Gate-A VX basis; normalized + sign-fixed by 16) ---
W = weights.loc[CCA_AXES, VX_COLS].values.T          # (6 VX × 2 CCA)
r_cca = weights.loc[CCA_AXES, 'canonical_r'].values

# --- project cells and PCHA archetype vertices into the shared CCA frame ---
C = vx_df[VX_COLS].values                             # cells × 6
mean_vx = C.mean(axis=0)
cca_cells = (C - mean_vx) @ W                         # cells × 2

# aa (PC space) → VX space (04's aa_vx) → CCA, same centering as the cells
aa_vx = aa_df.values @ inner_cmp.loc[list(aa_df.columns), VX_COLS].values + inner_mean
cca_aa = (aa_vx - mean_vx) @ W                        # 4 × 2

# order the 4 vertices by angle for a clean (non-self-intersecting) polygon, labels attached
aa_labels = np.array([AA_RENAME[i] for i in aa_df.index])
ang = np.arctan2(cca_aa[:, 1] - cca_aa[:, 1].mean(), cca_aa[:, 0] - cca_aa[:, 0].mean())
order = np.argsort(ang)
cca_aa, aa_labels = cca_aa[order], aa_labels[order]

# --- build panels (same scores/colormaps as 07) ---
panel_vals, panel_names, panel_cmaps, panel_cbar, panel_vlims = [], [], [], [], []
for p in PANELS:
    vals = scores_df[f'score_{p["pos"]}'].values
    if 'neg' in p:
        vals = vals - scores_df[f'score_{p["neg"]}'].values
        lim = np.percentile(np.abs(vals), DIFF_PCTILE)
        panel_cmaps.append(CMAP_DIFF)
        panel_cbar.append('score difference')
        panel_vlims.append((-lim, lim))
        print(f'  {p["name"]} range: [{vals.min():.3f}, {vals.max():.3f}], color limit ±{lim:.3f}')
    else:
        panel_cmaps.append(CMAP_B)
        panel_cbar.append('archetype score [0–1]')
        panel_vlims.append(None)
    panel_vals.append(vals)
    panel_names.append(p['name'])

print(f'--- L2/3: mouse {MOUSE_SUBCLASS} {ARCH_RELABEL["B"]} / '
      f'{ARCH_RELABEL["A"]}−{ARCH_RELABEL["C"]} scores on human CCA1×CCA2 '
      f'(r = {r_cca[0]:.3f}, {r_cca[1]:.3f}) ---')

save_score_scatter_pdf(
    cca_cells, np.column_stack(panel_vals), panel_names, cca_aa,
    title=f'Jorstad23 human {HUMAN_SUBCLASS} — mouse Cheng22 {MOUSE_SUBCLASS} archetype '
          f'scores on the conserved CCA1×CCA2 embedding',
    out_path=OUT_PDF,
    cmap=panel_cmaps, pctile=SCORE_PCTILE,
    aa_labels=list(aa_labels),
    colorbar_title=panel_cbar,
    vlims=panel_vlims,
    axis_labels=(f'CCA1 (r={r_cca[0]:.3f})', f'CCA2 (r={r_cca[1]:.3f})'),
)

print('\nDone.')
