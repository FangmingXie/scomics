"""Mouse archetype scores rendered on the mouse PCHA embedding, per IT subclass (plots only).

The l23_evo/59 companion to script 07: the same panels and styling, but scores, cell
coords and archetype coords all come from script 05, so each figure is self-consistent
with its own seeded re-fit. No recomputation here.

Panel selection mirrors script 07 exactly, so the mouse and human renderings of a
subclass show the same contrast side by side.

Reads:
  local_data/res/it_evo/05.mouse_<TOKEN>_archetype_scores.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_pcha_{xp,aa}.tsv
Outputs:
  local_data/fig/it_evo/08.mouse_<TOKEN>_b_ca_scores.pdf
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
# `panels` are the same contrasts script 07 draws, in this record's own archetype labels.
# `flip` applies a per-axis sign to BOTH cell coords (xp) and archetype coords (aa) so
# they stay consistent; orientation only, no cached TSV is touched.
# ===========================================================================
SUBCLASSES = [
    # L23 control. This record's vertices are a permutation of l23_evo/58's
    # (05:1<->58:3, 05:2<->58:2, 05:3<->58:1), so 59's 'C - A' contrast is this
    # record's 'A - C' — same quantity, same colormap orientation.
    # This record's aa correlates +0.994 with l23_evo/58's under [-1,1] (once the vertex
    # permutation is undone), and 59 displays 58 under [1,-1] — so [-1,-1] puts this
    # triangle in 59's orientation. Note the vertex LABELS still differ from 59's: this
    # record's A sits where 58's C sat, which is the same permutation the panel choice
    # above accounts for.
    {'token': 'L23', 'mouse_subclass': 'L2/3', 'noc': 3,
     'panels': [{'name': 'B', 'pos': 'B'},
                {'name': 'A − C', 'pos': 'A', 'neg': 'C'}],
     'flip': [-1.0, -1.0]},
    {'token': 'L4', 'mouse_subclass': 'L4', 'noc': 3,
     'panels': [{'name': 'B', 'pos': 'B'},
                {'name': 'C − A', 'pos': 'C', 'neg': 'A'}]},
    {'token': 'L5IT', 'mouse_subclass': 'L5IT', 'noc': 2,
     'panels': [{'name': 'B − A', 'pos': 'B', 'neg': 'A'}]},
    {'token': 'L6IT', 'mouse_subclass': 'L6IT', 'noc': 3,
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
    token    = cfg['token']
    subclass = cfg['mouse_subclass']
    noc      = cfg['noc']

    in_scores  = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_archetype_scores.tsv')
    in_pcha_xp = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_xp.tsv')
    in_pcha_aa = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_aa.tsv')
    out_pdf    = os.path.join(OUT_FIG_DIR, f'08.mouse_{token}_b_ca_scores.pdf')

    print(f'\n--- {token}: mouse {subclass} scores on mouse embedding (NOC={noc}) ---')

    scores_df = pd.read_csv(in_scores, sep='\t', index_col=0)
    xp_df     = pd.read_csv(in_pcha_xp, sep='\t', index_col=0)
    aa_df     = pd.read_csv(in_pcha_aa, sep='\t', index_col=0)

    if not scores_df.index.equals(xp_df.index):
        raise ValueError(
            f'Cell index mismatch: {in_scores} ({len(scores_df)} cells) vs '
            f'{in_pcha_xp} ({len(xp_df)} cells) — scores and embedding must cover the '
            f'same cells in the same order.'
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

    flip = np.array(cfg.get('flip', [1.0, 1.0]))
    xp = xp_df[['PC1', 'PC2']].values * flip
    aa = aa_df[['PC1', 'PC2']].values * flip

    save_score_scatter_pdf(
        xp, np.column_stack(panel_vals), panel_names, aa,
        title=f'Cheng22 mouse {subclass} — archetype scores on mouse cells (NOC={noc})',
        out_path=out_pdf,
        cmap=panel_cmaps, pctile=SCORE_PCTILE,
        aa_labels=ALPHABET[:noc],   # aa rows are archetype_1..N = A, B, ...
        colorbar_title=panel_cbar,
        vlims=panel_vlims,
    )

print('\nDone.')
