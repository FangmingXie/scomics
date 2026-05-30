"""Archetype number selection on varimax subspace — P56 gao25 gray-matter astrocytes.

Runs PCHA NOC sweep on the analyst-selected VX column subspace from script 33.2
(P56, no mt genes, Arch4 excluded).

Reads:  local_data/res/astro/33.2.varimax_coords.tsv
Outputs:
  local_data/fig/astro/34.2.num_archetype_metrics.png
  local_data/fig/astro/34.2.num_archetype_interactive.html
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import run_noc_sweep
from viz import save_metrics_plot, scatter_per_group_html
from scomics.main import SCA

# --- file paths ---
OUT_RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
IN_VARIMAX_COORDS = os.path.join(OUT_RES_DIR, '33.2.varimax_coords.tsv')
OUT_METRICS_PNG   = os.path.join(OUT_FIG_DIR, '34.2.num_archetype_metrics.png')
OUT_SCATTER_HTML  = os.path.join(OUT_FIG_DIR, '34.2.num_archetype_interactive.html')

# analyst-set after inspecting 33.2.variance_partition.html
VX_COLS  = ['VX3', 'VX4', 'VX6', 'VX7', 'VX9']
NDIM     = 4
NOC_MIN  = 2
NOC_MAX  = 8
NREPEATS = 10

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load varimax coords ---
vx_df  = pd.read_csv(IN_VARIMAX_COORDS, sep='\t', index_col=0)
xn     = vx_df[VX_COLS].values
types  = vx_df['archetype'].values
donors = vx_df['donor_name'].values

# --- archetype sweep ---
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
print(f'Running NOC sweep {NOC_MIN}–{NOC_MAX}, NDIM={NDIM}, NREPEATS={NREPEATS}...')
print(f'VX_COLS: {VX_COLS}')

ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, donors)

# --- plots ---
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
donor_to_color = {d: cycle[i % len(cycle)] for i, d in enumerate(np.unique(donors))}

save_metrics_plot(noc_grid, ev_grid, av_grid, av_rep_grid,
                  NDIM, f'Archetype selection — P56 gao25 GM VX subspace (NDIM={NDIM})',
                  OUT_METRICS_PNG)

scatter_per_group_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                       donors, donor_to_color,
                       f'Per-donor archetype overlay — P56 gao25 GM VX subspace (NDIM={NDIM})',
                       OUT_SCATTER_HTML)

print('Done.')
