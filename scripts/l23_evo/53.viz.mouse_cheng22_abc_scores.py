"""Mouse Cheng22 ABC archetype scores on the mouse PCHA embedding (plots only).

Mouse-on-mouse counterpart of 52.viz (mouse scores on human cells): colors Cheng22 mouse
L2/3 IT cells by their own A/B/C archetype scores from script 21, on the script-21 mouse
PCHA embedding, with the same four-panel layout — A, B, C, and the C-minus-A contrast
(diverging colormap centered at zero) — and the archetype polygon annotated A/B/C.
Same display flip as 21.viz. No recomputation here.

Reads:
  local_data/res/l23_evo/21.mouse_archetype_scores.tsv
  local_data/res/l23_evo/21.mouse_pcha_xp.tsv
  local_data/res/l23_evo/21.mouse_pcha_aa.tsv
Outputs:
  local_data/fig/l23_evo/53.mouse_abc_scores.pdf
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_score_scatter_pdf

# --- file paths ---
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_SCORES    = os.path.join(OUT_RES_DIR, '21.mouse_archetype_scores.tsv')
IN_PCHA_XP   = os.path.join(OUT_RES_DIR, '21.mouse_pcha_xp.tsv')
IN_PCHA_AA   = os.path.join(OUT_RES_DIR, '21.mouse_pcha_aa.tsv')
OUT_PDF      = os.path.join(OUT_FIG_DIR, '53.mouse_abc_scores.pdf')

# --- parameters ---
ARCHETYPE_NAMES = ['A', 'B', 'C']
SCORE_PCTILE    = (5, 95)
DIFF_NAME       = 'C − A'
DIFF_PCTILE     = 95        # |C − A| percentile setting the symmetric color limit

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

scores = scores_df[[f'score_{n}' for n in ARCHETYPE_NAMES]].values

# --- C − A contrast panel: symmetric color limits so white sits exactly at zero ---
diff     = scores_df['score_C'].values - scores_df['score_A'].values
diff_lim = np.percentile(np.abs(diff), DIFF_PCTILE)
scores   = np.column_stack([scores, diff])
PANEL_NAMES = ARCHETYPE_NAMES + [DIFF_NAME]
PANEL_CBAR  = ['archetype score [0–1]'] * len(ARCHETYPE_NAMES) + ['score difference']
PANEL_VLIMS = [None] * len(ARCHETYPE_NAMES) + [(-diff_lim, diff_lim)]
print(f'C − A range: [{diff.min():.3f}, {diff.max():.3f}], color limit ±{diff_lim:.3f}')

# ===========================================================================
# VISUALIZATION-ONLY AXIS FLIP — DO NOT PROPAGATE TO ANALYSIS
# PC2 = -PC2 ONLY (PC1 UNCHANGED), MATCHING 21.viz, SO THE ARCHETYPE TRIANGLE IS
# ORIENTED IDENTICALLY IN BOTH FIGURES. THE SAME SIGN FLIP IS APPLIED TO BOTH CELL
# COORDS (xp) AND ARCHETYPE COORDS (aa) SO THEY STAY CONSISTENT.
# THIS DOES NOT TOUCH THE CACHED TSVs OR ANY COMPUTED RESULT.
# ===========================================================================
FLIP = np.array([1.0, -1.0])             # PC1 KEPT, PC2 FLIPPED FOR DISPLAY ONLY
xp = xp_df[['PC1', 'PC2']].values * FLIP
aa = aa_df[['PC1', 'PC2']].values * FLIP
# ===========================================================================

# --- score scatter PDF (PC1 vs PC2, one panel per score + contrast) ---
save_score_scatter_pdf(
    xp, scores, PANEL_NAMES, aa,
    title='Cheng22 mouse L2/3 IT — archetype scores on mouse cells '
          f'(NOC={len(ARCHETYPE_NAMES)})',
    out_path=OUT_PDF,
    cmap='RdBu_r', pctile=SCORE_PCTILE,
    aa_labels=ARCHETYPE_NAMES,   # aa rows are archetype_1..3 = A, B, C
    colorbar_title=PANEL_CBAR,
    vlims=PANEL_VLIMS,
)
print('Done.')
