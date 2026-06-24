"""Archetype-score scatter PDF — two-dataset L4 IT, n=3 (plots only).

Reads the cached results from 33.follow.two_L4_archetype_scores.py and renders a static,
publication-ready figure: per-cell A/B/C archetype scores on PC1 vs PC2, one panel each,
points rasterized but saved as a vectorized PDF (axes/text/archetype overlay stay vector).
No recomputation here.

Reads:
  local_data/res/it/33.follow.two_L4_archetype_scores.tsv
  local_data/res/it/33.follow.two_L4_pcha_xp.tsv
  local_data/res/it/33.follow.two_L4_pcha_aa.tsv
Outputs:
  local_data/fig/it/33.follow.two_L4_archetype_scores.pdf
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_score_scatter_pdf

# --- file paths ---
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
IN_SCORES    = os.path.join(OUT_RES_DIR, '33.follow.two_L4_archetype_scores.tsv')
IN_PCHA_XP   = os.path.join(OUT_RES_DIR, '33.follow.two_L4_pcha_xp.tsv')
IN_PCHA_AA   = os.path.join(OUT_RES_DIR, '33.follow.two_L4_pcha_aa.tsv')
OUT_PDF      = os.path.join(OUT_FIG_DIR, '33.follow.two_L4_archetype_scores.pdf')

# --- parameters ---
ARCHETYPE_NAMES = ['A', 'B', 'C']
NOC             = len(ARCHETYPE_NAMES)
SCORE_PCTILE    = (5, 95)

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load cached results (index-aligned: same cells, same order) ---
scores_df = pd.read_csv(IN_SCORES, sep='\t', index_col=0)
xp_df     = pd.read_csv(IN_PCHA_XP, sep='\t', index_col=0)
aa_df     = pd.read_csv(IN_PCHA_AA, sep='\t', index_col=0)

scores_df = scores_df.reindex(xp_df.index)   # defensive alignment to PC coords
scores = scores_df[[f'score_{n}' for n in ARCHETYPE_NAMES]].values

# ===========================================================================
# VISUALIZATION-ONLY AXIS FLIP — DO NOT PROPAGATE TO ANALYSIS
# Display knob to re-orient the archetype triangle. Defaults to NO flip; set
# either entry to -1.0 after a first look if the triangle reads better mirrored.
# The same flip is applied to both cell coords (xp) and archetype coords (aa) so
# they stay consistent. THIS DOES NOT TOUCH THE CACHED TSVs OR ANY COMPUTED RESULT.
# ===========================================================================
FLIP = np.array([1.0, 1.0])              # [PC1, PC2] display sign; 1.0 = unchanged
xp = xp_df[['PC1', 'PC2']].values * FLIP
aa = aa_df[['PC1', 'PC2']].values * FLIP
# ===========================================================================

# --- score scatter PDF (PC1 vs PC2, one panel per score) ---
save_score_scatter_pdf(
    xp, scores, ARCHETYPE_NAMES, aa,
    title=f'Two-dataset (cheng22+yoo25) L4 IT — archetype scores (NOC={NOC})',
    out_path=OUT_PDF,
    cmap='RdBu_r', pctile=SCORE_PCTILE,
)
print('Done.')
