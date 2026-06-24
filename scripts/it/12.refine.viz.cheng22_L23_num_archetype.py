"""Archetype number selection on PCA subspace — cheng22 mouse L2/3 IT (plots, refined).

Reads the computed results from 12.refine.cheng22_L23_num_archetype.py
(top-5-PC baseline) and renders the interactive HTML figures. The metrics plot shows
bootstrap ARV and effective-EV as mean ± std error bars. No computation here — rerun
the compute script if inputs are missing.

Reads:
  local_data/res/it/12.refine.cheng22_L23_num_archetype_metrics.tsv
  local_data/res/it/12.refine.cheng22_L23_num_archetype_plotdata.pkl
Outputs:
  local_data/fig/it/12.refine.cheng22_L23_num_archetype_metrics.html
  local_data/fig/it/12.refine.cheng22_L23_num_archetype_interactive.html
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_metrics_err_plot_html, scatter_per_group_html

# --- file paths ---
OUT_RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
OUT_FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
IN_METRICS_TSV   = os.path.join(OUT_RES_DIR, '12.refine.cheng22_L23_num_archetype_metrics.tsv')
IN_PLOTDATA      = os.path.join(OUT_RES_DIR, '12.refine.cheng22_L23_num_archetype_plotdata.pkl')
OUT_METRICS_HTML = os.path.join(OUT_FIG_DIR, '12.refine.cheng22_L23_num_archetype_metrics.html')
OUT_INTERACTIVE  = os.path.join(OUT_FIG_DIR, '12.refine.cheng22_L23_num_archetype_interactive.html')

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load computed results ---
metrics_df = pd.read_csv(IN_METRICS_TSV, sep='\t')
noc_grid       = metrics_df['NOC'].values
ev_grid        = metrics_df['EV'].values
arv_mean       = metrics_df['ARV_mean'].values
arv_std        = metrics_df['ARV_std'].values
av_rep_grid    = metrics_df['ARV_rep'].values
effev_mean     = metrics_df['effEV_mean'].values
effev_std      = metrics_df['effEV_std'].values
effev_rep_grid = metrics_df['effEV_rep'].values

with open(IN_PLOTDATA, 'rb') as f:
    plotdata = pickle.load(f)
NDIM         = plotdata['ndim']
samples      = plotdata['samples']
aa_reps_grid = plotdata['aa_reps_grid']
# xp is NOC-independent; scatter_per_group_html indexes xp_grid[i] per NOC.
xp_grid      = [plotdata['xp']] * len(noc_grid)

# --- metrics plot with bootstrap error bars ---
save_metrics_err_plot_html(noc_grid, ev_grid, arv_mean, arv_std, av_rep_grid,
                           effev_mean, effev_std, effev_rep_grid, NDIM,
                           f'Archetype selection — cheng22 mouse L2/3 PCA subspace (NDIM={NDIM})',
                           OUT_METRICS_HTML)

# --- per-sample archetype overlay ---
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sample_to_color = {s: cycle[i % len(cycle)] for i, s in enumerate(np.unique(samples))}

scatter_per_group_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                       samples, sample_to_color,
                       f'Per-sample archetype overlay — cheng22 mouse L2/3 PCA subspace (NDIM={NDIM})',
                       OUT_INTERACTIVE)

print('Done.')
