"""The cross-subclass survey figure: every mouse IT archetype score on every human IT embedding.

Rows are the four human subclass PCHA embeddings, columns are the 11 mouse Cheng22
archetypes, and the color scale is shared down each column, so a column reads as "where in
human IT does this mouse archetype live". Script 10 renders the same kind of panels but
only for one subclass pair at a time — i.e. only the four diagonal cells of this grid — so
it cannot show whether a mouse archetype score is subclass-specific or generic to IT.

Thin caller: all of the computation is script 11, which scores every mouse archetype on
every human cell against one pooled per-gene scale (the shared column colorbar needs it).
The four diagonal cells — the pairs script 09 quantifies — are outlined.

Reads (per TOKEN in L23 / L4 / L5IT / L6IT):
  local_data/res/it_evo/11.mouse_all_archetype_scores_on_human_<TOKEN>.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_xp.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_aa.tsv
Outputs:
  local_data/fig/it_evo/12.human_mouse_all_scores_grid.pdf
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_score_grid_pdf

# --- file paths ---
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
OUT_PDF     = os.path.join(OUT_FIG_DIR, '12.human_mouse_all_scores_grid.pdf')

# ===========================================================================
# VISUALIZATION-ONLY SETTINGS — DO NOT PROPAGATE TO ANALYSIS
# `rename` / `flip` are carried for the L23 row only, identical to 07 and 10, so this
# figure sits in the same orientation as local_data/fig/l23_evo/57.human_mouse_b_ca_scores.pdf.
# The flip is applied to both cell coords (xp) and archetype coords (aa).
# ===========================================================================
SUBCLASSES = [
    {'token': 'L23', 'human_subclass': 'L2/3 IT', 'mouse_subclass': 'L2/3',
     'rename': {'archetype_1': "D'", 'archetype_2': "C'",
                'archetype_3': "B'", 'archetype_4': "A'"},
     'flip': [-1.0, 1.0]},
    {'token': 'L4',   'human_subclass': 'L4 IT', 'mouse_subclass': 'L4'},
    {'token': 'L5IT', 'human_subclass': 'L5 IT', 'mouse_subclass': 'L5IT'},
    {'token': 'L6IT', 'human_subclass': 'L6 IT', 'mouse_subclass': 'L6IT'},
]
# ===========================================================================

# --- parameters ---
# matches 06/10: gene_expr_scatter_html(colorscale='RdBu_r', pctile_low=5, pctile_high=95)
SCORE_CMAP   = 'RdBu_r'
SCORE_PCTILE = (5, 95)
ALPHABET     = ['A', 'B', 'C', 'D', 'E', 'F']

os.makedirs(OUT_FIG_DIR, exist_ok=True)

rows, scores, col_keys = [], [], None

for cfg in SUBCLASSES:
    token = cfg['token']

    in_scores  = os.path.join(
        OUT_RES_DIR, f'11.mouse_all_archetype_scores_on_human_{token}.tsv')
    in_pcha_xp = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_xp.tsv')
    in_pcha_aa = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_aa.tsv')

    scores_df = pd.read_csv(in_scores, sep='\t', index_col=0)
    xp_df     = pd.read_csv(in_pcha_xp, sep='\t', index_col=0)
    aa_df     = pd.read_csv(in_pcha_aa, sep='\t', index_col=0)

    if not scores_df.index.equals(xp_df.index):
        raise ValueError(
            f'Cell index mismatch: {in_scores} ({len(scores_df)} cells) vs '
            f'{in_pcha_xp} ({len(xp_df)} cells) — mouse scores and human embedding must '
            f'cover the same cells in the same order.')

    if col_keys is None:
        col_keys = list(scores_df.columns)
    elif list(scores_df.columns) != col_keys:
        raise ValueError(f'{in_scores} has columns {list(scores_df.columns)}, expected '
                         f'{col_keys} — every row of the grid must carry the same 11 '
                         f'mouse archetypes in the same order')

    aa_labels = ([cfg['rename'][i] for i in aa_df.index]
                 if 'rename' in cfg else ALPHABET[:len(aa_df)])
    flip = np.array(cfg.get('flip', [1.0, 1.0]))

    rows.append({'label': f'human {cfg["human_subclass"]}\n({len(xp_df)} cells)',
                 'xp': xp_df[['PC1', 'PC2']].values * flip,
                 'aa': aa_df[['PC1', 'PC2']].values * flip,
                 'aa_labels': aa_labels})
    scores.append([scores_df[k].values for k in col_keys])
    print(f'{token}: {len(xp_df)} cells x {len(col_keys)} mouse archetype scores')

# column keys are `<mouse token>_<letter>`, in script 11's L23 -> L6IT order
display = {cfg['token']: cfg['mouse_subclass'] for cfg in SUBCLASSES}
col_names = [f'mouse {display[k.rsplit("_", 1)[0]]} {k.rsplit("_", 1)[1]}' for k in col_keys]
diagonal  = {(i, j) for i, cfg in enumerate(SUBCLASSES)
             for j, k in enumerate(col_keys) if k.rsplit('_', 1)[0] == cfg['token']}
print(f'Columns: {", ".join(col_names)}')
print(f'Diagonal (script-09) cells outlined: {sorted(diagonal)}')

save_score_grid_pdf(
    rows, col_names, scores,
    title='Mouse Cheng22 IT archetype scores across all human Jorstad23 IT subclasses',
    out_path=OUT_PDF,
    cmap=SCORE_CMAP, pctile=SCORE_PCTILE,
    highlight=diagonal,
)

print('\nDone.')
