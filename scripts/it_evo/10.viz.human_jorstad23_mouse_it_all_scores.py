"""All mouse archetype scores on the human embedding, one panel each (plots only).

The static counterpart to `06.mouse_<TOKEN>_archetype_scores_on_human_cells.html`: that
figure shows one mouse archetype score at a time behind a dropdown, so the archetypes
cannot be compared side by side or dropped into a document. This renders the same data as
a matplotlib PDF with every score (A, B, C, ...) laid out together.

Complements `07.viz`, which renders only the two hand-picked panels inherited from the
l23_evo L2/3 figure (one score, plus the C - A contrast). Nothing is hand-picked here: the
score columns are read off the script-06 TSV, so the panel count follows the mouse NOC
(3 for L23 / L4 / L6IT, 2 for L5IT) with no per-token configuration.

Styling matches `06` rather than `07`: script 06 calls gene_expr_scatter_html with the
default RdBu_r colorscale and 5/95 percentile clipping, which are also
save_score_scatter_pdf's defaults, so the PDF reproduces the HTML's appearance.

Reads (per TOKEN in L23 / L4 / L5IT / L6IT):
  local_data/res/it_evo/06.mouse_<TOKEN>_archetype_scores_on_human_cells.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_xp.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_aa.tsv
Outputs:
  local_data/fig/it_evo/10.human_<TOKEN>_mouse_all_scores.pdf
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_score_scatter_pdf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokens', nargs='*', default=None,
                    help='subset of tokens to render (default: all four)')
args = parser.parse_args()

# --- file paths ---
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')

# ===========================================================================
# VISUALIZATION-ONLY SETTINGS — DO NOT PROPAGATE TO ANALYSIS
# `rename` / `flip` are carried for the L23 control only, identical to 07, so this
# figure sits in the same orientation as 07.human_L23_mouse_b_ca_scores.pdf and
# local_data/fig/l23_evo/57.human_mouse_b_ca_scores.pdf. The flip is applied to both
# cell coords (xp) and archetype coords (aa) so they stay consistent.
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
if args.tokens:
    SUBCLASSES = [c for c in SUBCLASSES if c['token'] in args.tokens]

# --- parameters ---
# 06 renders with gene_expr_scatter_html(colorscale='RdBu_r', pctile_low=5, pctile_high=95)
SCORE_CMAP   = 'RdBu_r'
SCORE_PCTILE = (5, 95)
ALPHABET     = ['A', 'B', 'C', 'D', 'E', 'F']

os.makedirs(OUT_FIG_DIR, exist_ok=True)

for cfg in SUBCLASSES:
    token          = cfg['token']
    human_subclass = cfg['human_subclass']
    mouse_subclass = cfg['mouse_subclass']

    in_scores  = os.path.join(
        OUT_RES_DIR, f'06.mouse_{token}_archetype_scores_on_human_cells.tsv')
    in_pcha_xp = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_xp.tsv')
    in_pcha_aa = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_aa.tsv')
    out_pdf    = os.path.join(OUT_FIG_DIR, f'10.human_{token}_mouse_all_scores.pdf')

    scores_df = pd.read_csv(in_scores, sep='\t', index_col=0)
    xp_df     = pd.read_csv(in_pcha_xp, sep='\t', index_col=0)
    aa_df     = pd.read_csv(in_pcha_aa, sep='\t', index_col=0)

    if not scores_df.index.equals(xp_df.index):
        raise ValueError(
            f'Cell index mismatch: {in_scores} ({len(scores_df)} cells) vs '
            f'{in_pcha_xp} ({len(xp_df)} cells) — mouse scores and human embedding must '
            f'cover the same cells in the same order.'
        )

    # every score column, in file order — no hand-picked subset
    score_cols  = [c for c in scores_df.columns if c.startswith('score_')]
    panel_names = [c[len('score_'):] for c in score_cols]
    print(f'\n--- {token}: mouse {mouse_subclass} scores ({", ".join(panel_names)}) '
          f'on human {human_subclass} ---')

    aa_labels = ([cfg['rename'][i] for i in aa_df.index]
                 if 'rename' in cfg else ALPHABET[:len(aa_df)])
    flip = np.array(cfg.get('flip', [1.0, 1.0]))
    xp = xp_df[['PC1', 'PC2']].values * flip
    aa = aa_df[['PC1', 'PC2']].values * flip

    save_score_scatter_pdf(
        xp, scores_df[score_cols].values, panel_names, aa,
        title=f'Jorstad23 human {human_subclass} — all mouse Cheng22 {mouse_subclass} '
              f'archetype scores',
        out_path=out_pdf,
        cmap=SCORE_CMAP, pctile=SCORE_PCTILE,
        aa_labels=aa_labels,
        colorbar_title='archetype score [0–1]',
    )

print('\nDone.')
