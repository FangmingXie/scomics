"""Archetype-score scatter PDF — Jorstad23 human L2/3 IT (plots only).

Reads the cached results from 25.human_jorstad23_archetype_scores.py and renders a static,
publication-ready figure: per-cell A/B/C/D archetype scores on PC1 vs PC2, one panel each,
points rasterized (rasterized=True) but saved as a vectorized PDF (axes/text/archetype
overlay stay vector). No recomputation here. Human analog of 21.viz (4 archetypes).

Reads:
  local_data/res/l23_evo/25.human_archetype_scores.tsv
  local_data/res/l23_evo/25.human_pcha_xp.tsv
  local_data/res/l23_evo/25.human_pcha_aa.tsv
Outputs:
  local_data/fig/l23_evo/25.human_archetype_scores.pdf
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
IN_SCORES    = os.path.join(OUT_RES_DIR, '25.human_archetype_scores.tsv')
IN_PCHA_XP   = os.path.join(OUT_RES_DIR, '25.human_pcha_xp.tsv')
IN_PCHA_AA   = os.path.join(OUT_RES_DIR, '25.human_pcha_aa.tsv')
OUT_PDF      = os.path.join(OUT_FIG_DIR, '25.human_archetype_scores.pdf')

# --- parameters ---
ARCHETYPE_NAMES = ['A', 'B', 'C', 'D']
NOC             = len(ARCHETYPE_NAMES)
SCORE_PCTILE    = (5, 95)

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load cached results (index-aligned: same cells, same order) ---
scores_df = pd.read_csv(IN_SCORES, sep='\t', index_col=0)
xp_df     = pd.read_csv(IN_PCHA_XP, sep='\t', index_col=0)
aa_df     = pd.read_csv(IN_PCHA_AA, sep='\t', index_col=0)

scores_df = scores_df.reindex(xp_df.index)   # defensive alignment to PC coords

# ===========================================================================
# VISUALIZATION-ONLY ARCHETYPE RENAME — DO NOT PROPAGATE TO ANALYSIS
# RELABEL (A, B, C, D) -> (D', C', B', A') FOR DISPLAY (PRIME SUFFIX MARKS THE
# RENAMED LABELS), THEN SHOW PANELS IN THE NEW ALPHABETICAL ORDER A', B', C', D'.
# THE CACHED score_* COLUMNS ARE UNCHANGED; WE ONLY REMAP WHICH ORIGINAL SCORE
# IS SHOWN UNDER EACH NEW LABEL:
#   new A' = old D  (score_D)
#   new B' = old C  (score_C)
#   new C' = old B  (score_B)
#   new D' = old A  (score_A)
# ===========================================================================
RENAME        = {'A': 'D', 'B': 'C', 'C': 'B', 'D': 'A'}   # old label -> new letter
DISPLAY_NAMES = ["A'", "B'", "C'", "D'"]                    # new primed labels, panel order
OLD_FOR_NEW   = {new: old for old, new in RENAME.items()}  # new letter -> old score column
scores = scores_df[[f'score_{OLD_FOR_NEW[n]}' for n in ['A', 'B', 'C', 'D']]].values
# ===========================================================================

# ===========================================================================
# VISUALIZATION-ONLY AXIS FLIP — DO NOT PROPAGATE TO ANALYSIS
# PC1 = -PC1 ONLY (PC2 UNCHANGED) PURELY TO ORIENT THE ARCHETYPE POLYGON FOR
# DISPLAY. THE SAME SIGN FLIP IS APPLIED TO BOTH CELL COORDS (xp) AND
# ARCHETYPE COORDS (aa) SO THEY STAY CONSISTENT.
# THIS DOES NOT TOUCH THE CACHED TSVs OR ANY COMPUTED RESULT.
# ===========================================================================
FLIP = np.array([-1.0, 1.0])             # PC1 FLIPPED, PC2 KEPT (DISPLAY ONLY)
xp = xp_df[['PC1', 'PC2']].values * FLIP
aa = aa_df[['PC1', 'PC2']].values * FLIP
# ===========================================================================

# --- score scatter PDF (PC1 vs PC2, one panel per score; new A/B/C/D order) ---
save_score_scatter_pdf(
    xp, scores, DISPLAY_NAMES, aa,
    title=f'Jorstad23 human L2/3 IT — archetype scores (NOC={NOC})',
    out_path=OUT_PDF,
    cmap='RdBu_r', pctile=SCORE_PCTILE,
)
print('Done.')
