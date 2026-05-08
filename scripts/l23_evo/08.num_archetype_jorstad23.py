"""Archetype number selection on varimax subspace — Jorstad23 L2/3 IT.

Runs PCHA NOC sweep on the VX2/VX6–VX10 subspace (6 varimax components).
The VX coords are passed directly as the feature matrix; SCA's internal PCA
reduces them to NDIM=5 (fitting 6 PCs on 6D data, dropping the last).

Reads:  local_data/res/l23_evo/05.varimax_coords.tsv
Outputs:
  local_data/fig/l23_evo/08.num_archetype_metrics.png
  local_data/fig/l23_evo/08.num_archetype_interactive.html
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from common import run_noc_sweep
from viz import save_metrics_plot, scatter_per_group_html

from scomics.main import SCA

import pandas as pd

# --- file paths ---
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_VARIMAX   = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
OUT_METRICS  = os.path.join(OUT_FIG_DIR, '08.num_archetype_metrics.png')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '08.num_archetype_interactive.html')

VX_COLS   = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
NDIM      = 5        # n_fit = NDIM+1 = 6 = len(VX_COLS); drops last PC
NOC_MIN   = 2
NOC_MAX   = 10
NREPEATS  = 10
CLUSTER_COL = 'WithinArea_cluster'
DONOR_COL   = 'donor_id'

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load varimax coords ---
vx_df   = pd.read_csv(IN_VARIMAX, sep='\t', index_col=0)
xn      = vx_df[VX_COLS].values                   # (n_cells, 6) — feature matrix
types   = vx_df[CLUSTER_COL].values
donors  = vx_df[DONOR_COL].values

# --- archetype sweep ---
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
print(f'Running NOC sweep {NOC_MIN}–{NOC_MAX}, NDIM={NDIM}, NREPEATS={NREPEATS}...')

ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, donors)

# --- plots ---
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
donor_to_color = {d: cycle[i % len(cycle)] for i, d in enumerate(np.unique(donors))}

save_metrics_plot(noc_grid, ev_grid, av_grid, av_rep_grid,
                  NDIM, f'Archetype selection — Jorstad23 VX subspace (NDIM={NDIM})',
                  OUT_METRICS)

scatter_per_group_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                       donors, donor_to_color,
                       f'Per-donor archetype overlay — Jorstad23 VX subspace (NDIM={NDIM})',
                       OUT_HTML)

print('Done.')
