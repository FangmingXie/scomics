"""Mouse archetype scores rendered on the human PCHA embedding, per IT subclass (plots only).

The l23_evo/57 figure for the four subclass pairs: mouse-derived scores from script 06
(mouse markers of script 05 mapped to human orthologs) on the human embedding from
script 04. Styling is 57's exactly — SCORE_PCTILE=(2,98), a symmetric DIFF_PCTILE=98 limit
so `lightgray` sits at zero, CMAP_B = gainsboro→lightgray→C1, CMAP_DIFF = C0→lightgray→C2.
No recomputation here.

Panels per subclass follow the mouse NOC: NOC=3 gets two panels (one score, plus a
two-pole contrast), NOC=2 gets the single contrast.

Reads:
  local_data/res/it_evo/06.mouse_<TOKEN>_archetype_scores_on_human_cells.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_{xp,aa}.tsv
Outputs:
  local_data/fig/it_evo/07.human_<TOKEN>_mouse_b_ca_scores.pdf
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
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')

# ===========================================================================
# VISUALIZATION-ONLY SETTINGS — DO NOT PROPAGATE TO ANALYSIS
# `panels`  which mouse score(s) to draw, in this record's OWN archetype labels;
#           {'name', 'pos'} for a plain score, {'name', 'pos', 'neg'} for a contrast.
# `rename`  human archetype display labels (aa row order archetype_1..N -> letter).
# `flip`    per-axis sign applied to BOTH cell coords (xp) and archetype coords (aa),
#           so they stay consistent. Orientation only; no cached TSV is touched.
# Only the L23 control carries `rename`/`flip`, so that its figure is directly
# comparable to local_data/fig/l23_evo/57.human_mouse_b_ca_scores.pdf.
# ===========================================================================
SUBCLASSES = [
    # L23 control. The mouse vertices of script 05 are a permutation of l23_evo/58's:
    # cross-Jaccard of the marker sets is diagonal under 05:1<->58:3, 05:2<->58:2,
    # 05:3<->58:1. So 57's 'B' panel is this record's B, and 57's 'C - A' contrast is
    # this record's 'A - C' — same quantity, same colormap orientation, honest label.
    {'token': 'L23', 'human_subclass': 'L2/3 IT', 'mouse_subclass': 'L2/3',
     'panels': [{'name': 'B', 'pos': 'B'},
                {'name': 'A − C', 'pos': 'A', 'neg': 'C'}],
     'rename': {'archetype_1': "D'", 'archetype_2': "C'",
                'archetype_3': "B'", 'archetype_4': "A'"},
     'flip': [-1.0, 1.0]},
    {'token': 'L4', 'human_subclass': 'L4 IT', 'mouse_subclass': 'L4',
     'panels': [{'name': 'B', 'pos': 'B'},
                {'name': 'C − A', 'pos': 'C', 'neg': 'A'}]},
    {'token': 'L5IT', 'human_subclass': 'L5 IT', 'mouse_subclass': 'L5IT',
     'panels': [{'name': 'B − A', 'pos': 'B', 'neg': 'A'}]},
    {'token': 'L6IT', 'human_subclass': 'L6 IT', 'mouse_subclass': 'L6IT',
     'panels': [{'name': 'B', 'pos': 'B'},
                {'name': 'C − A', 'pos': 'C', 'neg': 'A'}]},
]
# ===========================================================================

# --- parameters ---
SCORE_PCTILE = (2, 98)
DIFF_PCTILE  = 98        # |difference| percentile setting the symmetric color limit
CMAP_B       = LinearSegmentedColormap.from_list('gray_C1', ['gainsboro', 'lightgray', 'C1'])
CMAP_DIFF    = LinearSegmentedColormap.from_list('C0_gray_C2', ['C0', 'lightgray', 'C2'])
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
    out_pdf    = os.path.join(OUT_FIG_DIR, f'07.human_{token}_mouse_b_ca_scores.pdf')

    print(f'\n--- {token}: mouse {mouse_subclass} scores on human {human_subclass} ---')

    scores_df = pd.read_csv(in_scores, sep='\t', index_col=0)
    xp_df     = pd.read_csv(in_pcha_xp, sep='\t', index_col=0)
    aa_df     = pd.read_csv(in_pcha_aa, sep='\t', index_col=0)

    if not scores_df.index.equals(xp_df.index):
        raise ValueError(
            f'Cell index mismatch: {in_scores} ({len(scores_df)} cells) vs '
            f'{in_pcha_xp} ({len(xp_df)} cells) — mouse scores and human embedding must '
            f'cover the same cells in the same order.'
        )

    panel_vals, panel_names, panel_cmaps, panel_cbar, panel_vlims = [], [], [], [], []
    for p in cfg['panels']:
        vals = scores_df[f'score_{p["pos"]}'].values
        if 'neg' in p:
            vals = vals - scores_df[f'score_{p["neg"]}'].values
            lim = np.percentile(np.abs(vals), DIFF_PCTILE)
            panel_cmaps.append(CMAP_DIFF)
            panel_cbar.append('score difference')
            panel_vlims.append((-lim, lim))   # symmetric -> lightgray sits at zero
            print(f'  {p["name"]} range: [{vals.min():.3f}, {vals.max():.3f}], '
                  f'color limit ±{lim:.3f}')
        else:
            panel_cmaps.append(CMAP_B)
            panel_cbar.append('archetype score [0–1]')
            panel_vlims.append(None)
        panel_vals.append(vals)
        panel_names.append(p['name'])

    # aa rows are archetype_1..N; label them A, B, ... unless a rename is configured
    aa_labels = ([cfg['rename'][i] for i in aa_df.index]
                 if 'rename' in cfg else ALPHABET[:len(aa_df)])
    flip = np.array(cfg.get('flip', [1.0, 1.0]))
    xp = xp_df[['PC1', 'PC2']].values * flip
    aa = aa_df[['PC1', 'PC2']].values * flip

    save_score_scatter_pdf(
        xp, np.column_stack(panel_vals), panel_names, aa,
        title=f'Jorstad23 human {human_subclass} — mouse Cheng22 {mouse_subclass} '
              f'archetype scores',
        out_path=out_pdf,
        cmap=panel_cmaps, pctile=SCORE_PCTILE,
        aa_labels=aa_labels,
        colorbar_title=panel_cbar,
        vlims=panel_vlims,
    )

print('\nDone.')
