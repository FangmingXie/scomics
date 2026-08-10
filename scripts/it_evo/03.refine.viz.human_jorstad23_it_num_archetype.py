"""Archetype number selection — Jorstad23 human IT subclasses (plots only).

Renders the metrics and per-donor-overlay HTMLs from the results computed by
03.refine.human_jorstad23_it_num_archetype.py. No computation here — rerun the compute
script if inputs are missing.

Reads (per TOKEN in L23 / L4 / L5IT / L6IT):
  local_data/res/it_evo/03.refine.human_<TOKEN>_num_archetype_metrics.tsv
  local_data/res/it_evo/03.refine.human_<TOKEN>_num_archetype_plotdata.pkl
Outputs:
  local_data/fig/it_evo/03.refine.human_<TOKEN>_num_archetype_metrics.html
  local_data/fig/it_evo/03.refine.human_<TOKEN>_num_archetype_interactive.html
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_metrics_err_plot_html, scatter_per_group_html

# --- file paths ---
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')

SUBCLASSES = [
    {'token': 'L23',  'human_subclass': 'L2/3 IT'},
    {'token': 'L4',   'human_subclass': 'L4 IT'},
    {'token': 'L5IT', 'human_subclass': 'L5 IT'},
    {'token': 'L6IT', 'human_subclass': 'L6 IT'},
]

os.makedirs(OUT_FIG_DIR, exist_ok=True)

for cfg in SUBCLASSES:
    token    = cfg['token']
    subclass = cfg['human_subclass']

    in_metrics_tsv   = os.path.join(
        OUT_RES_DIR, f'03.refine.human_{token}_num_archetype_metrics.tsv')
    in_plotdata      = os.path.join(
        OUT_RES_DIR, f'03.refine.human_{token}_num_archetype_plotdata.pkl')
    out_metrics_html = os.path.join(
        OUT_FIG_DIR, f'03.refine.human_{token}_num_archetype_metrics.html')
    out_interactive  = os.path.join(
        OUT_FIG_DIR, f'03.refine.human_{token}_num_archetype_interactive.html')

    print(f'\n--- {token} (human {subclass}) ---')

    metrics_df     = pd.read_csv(in_metrics_tsv, sep='\t')
    noc_grid       = metrics_df['NOC'].values
    ev_grid        = metrics_df['EV'].values
    arv_mean       = metrics_df['ARV_mean'].values
    arv_std        = metrics_df['ARV_std'].values
    av_rep_grid    = metrics_df['ARV_rep'].values
    effev_mean     = metrics_df['effEV_mean'].values
    effev_std      = metrics_df['effEV_std'].values
    effev_rep_grid = metrics_df['effEV_rep'].values

    with open(in_plotdata, 'rb') as f:
        plotdata = pickle.load(f)
    ndim         = plotdata['ndim']
    groups       = plotdata['groups']
    aa_reps_grid = plotdata['aa_reps_grid']
    # xp is NOC-independent; scatter_per_group_html indexes xp_grid[i] per NOC.
    xp_grid      = [plotdata['xp']] * len(noc_grid)

    save_metrics_err_plot_html(
        noc_grid, ev_grid, arv_mean, arv_std, av_rep_grid,
        effev_mean, effev_std, effev_rep_grid, ndim,
        f'Archetype selection — Jorstad23 human {subclass} VX subspace (NDIM={ndim})',
        out_metrics_html)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    group_to_color = {g: cycle[i % len(cycle)] for i, g in enumerate(np.unique(groups))}

    scatter_per_group_html(
        noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
        groups, group_to_color,
        f'Per-donor archetype overlay — Jorstad23 human {subclass} VX subspace (NDIM={ndim})',
        out_interactive)

print('\nDone.')
