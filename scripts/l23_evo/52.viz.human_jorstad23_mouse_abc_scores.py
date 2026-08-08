"""Mouse Cheng22 ABC archetype scores on the human Jorstad23 PCHA embedding (plots only).

Colors human L2/3 IT cells by the mouse-derived archetype scores from script 22 (mouse
A/B/C markers mapped to human orthologs, then scored on human expression), rendered on
the script-25 human PCHA embedding — the same embedding and display conventions as
25.viz, so the two figures are directly comparable. The human archetype polygon is
annotated with the primed labels A'-D' used in 25.viz. A fourth panel shows the
C-minus-A contrast (diverging colormap centered at zero). No recomputation here.

Reads:
  local_data/res/l23_evo/22.human_archetype_scores.tsv
  local_data/res/l23_evo/25.human_pcha_xp.tsv
  local_data/res/l23_evo/25.human_pcha_aa.tsv
Outputs:
  local_data/fig/l23_evo/52.human_mouse_abc_scores.pdf
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
IN_SCORES    = os.path.join(OUT_RES_DIR, '22.human_archetype_scores.tsv')
IN_PCHA_XP   = os.path.join(OUT_RES_DIR, '25.human_pcha_xp.tsv')
IN_PCHA_AA   = os.path.join(OUT_RES_DIR, '25.human_pcha_aa.tsv')
OUT_PDF      = os.path.join(OUT_FIG_DIR, '52.human_mouse_abc_scores.pdf')

# --- parameters ---
MOUSE_ARCHETYPE_NAMES = ['A', 'B', 'C']
SCORE_PCTILE          = (5, 95)
DIFF_NAME             = 'C − A'
DIFF_PCTILE           = 95        # |C − A| percentile setting the symmetric color limit

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load cached results ---
scores_df = pd.read_csv(IN_SCORES, sep='\t', index_col=0)
xp_df     = pd.read_csv(IN_PCHA_XP, sep='\t', index_col=0)
aa_df     = pd.read_csv(IN_PCHA_AA, sep='\t', index_col=0)

if not scores_df.index.equals(xp_df.index):
    raise ValueError(
        f'Cell index mismatch: {IN_SCORES} ({len(scores_df)} cells) vs '
        f'{IN_PCHA_XP} ({len(xp_df)} cells) — mouse scores and human embedding must '
        f'cover the same cells in the same order.'
    )

scores = scores_df[[f'score_{n}' for n in MOUSE_ARCHETYPE_NAMES]].values

# --- C − A contrast panel: symmetric color limits so white sits exactly at zero ---
diff      = scores_df['score_C'].values - scores_df['score_A'].values
diff_lim  = np.percentile(np.abs(diff), DIFF_PCTILE)
scores    = np.column_stack([scores, diff])
PANEL_NAMES     = MOUSE_ARCHETYPE_NAMES + [DIFF_NAME]
PANEL_CBAR      = ['archetype score [0–1]'] * len(MOUSE_ARCHETYPE_NAMES) + ['score difference']
PANEL_VLIMS     = [None] * len(MOUSE_ARCHETYPE_NAMES) + [(-diff_lim, diff_lim)]
print(f'C − A range: [{diff.min():.3f}, {diff.max():.3f}], color limit ±{diff_lim:.3f}')

# ===========================================================================
# VISUALIZATION-ONLY ARCHETYPE RENAME — DO NOT PROPAGATE TO ANALYSIS
# THE HUMAN ARCHETYPES (ROWS archetype_1..4 OF 25.human_pcha_aa.tsv = A, B, C, D)
# ARE DISPLAYED UNDER THE PRIMED LABELS USED BY 25.viz: (A, B, C, D) -> (D', C', B', A').
# ONLY THE POLYGON VERTEX LABELS ARE AFFECTED; THE MOUSE A/B/C SCORES KEEP THEIR OWN
# LABELS, AND NO CACHED TSV IS TOUCHED.
# ===========================================================================
RENAME    = {'A': 'D', 'B': 'C', 'C': 'B', 'D': 'A'}   # old human label -> new letter
AA_LABELS = [f"{RENAME[old]}'" for old in ['A', 'B', 'C', 'D']]   # aa row order
# ===========================================================================

# ===========================================================================
# VISUALIZATION-ONLY AXIS FLIP — DO NOT PROPAGATE TO ANALYSIS
# PC1 = -PC1 ONLY (PC2 UNCHANGED), MATCHING 25.viz, SO THE ARCHETYPE POLYGON IS
# ORIENTED IDENTICALLY IN BOTH FIGURES. THE SAME SIGN FLIP IS APPLIED TO BOTH CELL
# COORDS (xp) AND ARCHETYPE COORDS (aa) SO THEY STAY CONSISTENT.
# ===========================================================================
FLIP = np.array([-1.0, 1.0])             # PC1 FLIPPED, PC2 KEPT (DISPLAY ONLY)
xp = xp_df[['PC1', 'PC2']].values * FLIP
aa = aa_df[['PC1', 'PC2']].values * FLIP
# ===========================================================================

# --- score scatter PDF (PC1 vs PC2, one panel per mouse score) ---
save_score_scatter_pdf(
    xp, scores, PANEL_NAMES, aa,
    title='Jorstad23 human L2/3 IT — mouse Cheng22 ABC archetype scores '
          "(human archetypes labeled A'–D')",
    out_path=OUT_PDF,
    cmap='RdBu_r', pctile=SCORE_PCTILE,
    aa_labels=AA_LABELS,
    colorbar_title=PANEL_CBAR,
    vlims=PANEL_VLIMS,
)
print('Done.')
