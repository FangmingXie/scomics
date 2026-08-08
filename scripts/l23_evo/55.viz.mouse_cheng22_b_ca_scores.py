"""Mouse B and C−A archetype scores on the mouse PCHA embedding (plots only).

Mouse-on-mouse counterpart of 54.viz, and a two-panel restyling of 53.viz — identical
data, identical embedding and display conventions, visualization changes only:
  * only Score B and the C-minus-A contrast are shown;
  * Score B uses a sequential lightgray -> C1 colormap;
  * Score C − A uses a diverging C0 -> lightgray -> C2 colormap, with symmetric color
    limits so lightgray falls exactly at zero.
No recomputation here.

Reads:
  local_data/res/l23_evo/21.mouse_archetype_scores.tsv
  local_data/res/l23_evo/21.mouse_pcha_xp.tsv
  local_data/res/l23_evo/21.mouse_pcha_aa.tsv
Outputs:
  local_data/fig/l23_evo/55.mouse_b_ca_scores.pdf
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
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_SCORES    = os.path.join(OUT_RES_DIR, '21.mouse_archetype_scores.tsv')
IN_PCHA_XP   = os.path.join(OUT_RES_DIR, '21.mouse_pcha_xp.tsv')
IN_PCHA_AA   = os.path.join(OUT_RES_DIR, '21.mouse_pcha_aa.tsv')
OUT_PDF      = os.path.join(OUT_FIG_DIR, '55.mouse_b_ca_scores.pdf')

# --- parameters ---
ARCHETYPE_NAMES = ['A', 'B', 'C']
SCORE_PCTILE    = (2, 98)
DIFF_NAME       = 'C − A'
DIFF_PCTILE     = 98        # |C − A| percentile setting the symmetric color limit
CMAP_B          = LinearSegmentedColormap.from_list('gray_C1', ['gainsboro', 'lightgray', 'C1'])
CMAP_DIFF       = LinearSegmentedColormap.from_list('C0_gray_C2', ['C0', 'lightgray', 'C2'])

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load cached results ---
scores_df = pd.read_csv(IN_SCORES, sep='\t', index_col=0)
xp_df     = pd.read_csv(IN_PCHA_XP, sep='\t', index_col=0)
aa_df     = pd.read_csv(IN_PCHA_AA, sep='\t', index_col=0)

if not scores_df.index.equals(xp_df.index):
    raise ValueError(
        f'Cell index mismatch: {IN_SCORES} ({len(scores_df)} cells) vs '
        f'{IN_PCHA_XP} ({len(xp_df)} cells) — scores and embedding must cover the same '
        f'cells in the same order.'
    )

# --- two panels: Score B, and the C − A contrast ---
diff     = scores_df['score_C'].values - scores_df['score_A'].values
diff_lim = np.percentile(np.abs(diff), DIFF_PCTILE)
scores   = np.column_stack([scores_df['score_B'].values, diff])
PANEL_NAMES = ['B', DIFF_NAME]
PANEL_CMAPS = [CMAP_B, CMAP_DIFF]
PANEL_CBAR  = ['archetype score [0–1]', 'score difference']
PANEL_VLIMS = [None, (-diff_lim, diff_lim)]   # symmetric -> lightgray sits at zero
print(f'C − A range: [{diff.min():.3f}, {diff.max():.3f}], color limit ±{diff_lim:.3f}')

# ===========================================================================
# VISUALIZATION-ONLY AXIS FLIP — DO NOT PROPAGATE TO ANALYSIS
# PC2 = -PC2 ONLY (PC1 UNCHANGED), MATCHING 21.viz AND 53.viz, SO THE ARCHETYPE TRIANGLE
# IS ORIENTED IDENTICALLY IN ALL THREE FIGURES. THE SAME SIGN FLIP IS APPLIED TO BOTH
# CELL COORDS (xp) AND ARCHETYPE COORDS (aa) SO THEY STAY CONSISTENT.
# THIS DOES NOT TOUCH THE CACHED TSVs OR ANY COMPUTED RESULT.
# ===========================================================================
FLIP = np.array([1.0, -1.0])             # PC1 KEPT, PC2 FLIPPED FOR DISPLAY ONLY
xp = xp_df[['PC1', 'PC2']].values * FLIP
aa = aa_df[['PC1', 'PC2']].values * FLIP
# ===========================================================================

# --- score scatter PDF (PC1 vs PC2, two panels) ---
save_score_scatter_pdf(
    xp, scores, PANEL_NAMES, aa,
    title='Cheng22 mouse L2/3 IT — archetype scores on mouse cells '
          f'(NOC={len(ARCHETYPE_NAMES)})',
    out_path=OUT_PDF,
    cmap=PANEL_CMAPS, pctile=SCORE_PCTILE,
    aa_labels=ARCHETYPE_NAMES,   # aa rows are archetype_1..3 = A, B, C
    colorbar_title=PANEL_CBAR,
    vlims=PANEL_VLIMS,
)
print('Done.')
