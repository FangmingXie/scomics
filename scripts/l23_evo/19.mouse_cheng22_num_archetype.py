"""Archetype number selection on varimax subspace — Cheng22 mouse L2/3 IT.

Runs PCHA NOC sweep on a subset of varimax components (high Type R²).
VX_COLS should be updated after inspecting 18.mouse_vx_variance_partition.tsv:
select components where cell_type R² dominates over sample/library_size R².

Reads:  local_data/res/l23_evo/18.mouse_varimax_coords.tsv
Outputs:
  local_data/res/l23_evo/19.mouse_num_archetype_metrics.tsv
  local_data/fig/l23_evo/19.mouse_num_archetype_metrics.html
  local_data/fig/l23_evo/19.mouse_num_archetype_interactive.html
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import run_noc_sweep
from viz import save_metrics_plot_html, scatter_per_group_html
from scomics.main import SCA

# --- file paths ---
OUT_RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_VX_COORDS     = os.path.join(OUT_RES_DIR, '18.mouse_varimax_coords.tsv')
OUT_METRICS_TSV  = os.path.join(OUT_RES_DIR, '19.mouse_num_archetype_metrics.tsv')
OUT_METRICS_HTML = os.path.join(OUT_FIG_DIR, '19.mouse_num_archetype_metrics.html')
OUT_INTERACTIVE  = os.path.join(OUT_FIG_DIR, '19.mouse_num_archetype_interactive.html')

# --- parameters ---
CLUSTER_COL = 'Type'
SAMPLE_COL  = 'sample'
# Update VX_COLS after inspecting 18.mouse_vx_variance_partition.tsv:
# select components where cell_type R² dominates (analogous to VX2,VX6-VX10 for human)
VX_COLS     = ['VX1', 'VX2', 'VX6', 'VX7', 'VX8', 'VX10']
NDIM        = len(VX_COLS) - 1   # 5
NOC_MIN     = 2
NOC_MAX     = 6
NREPEATS    = 10

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load varimax coords ---
vx_df   = pd.read_csv(IN_VX_COORDS, sep='\t', index_col=0)
xn      = vx_df[VX_COLS].values
types   = vx_df[CLUSTER_COL].values
samples = vx_df[SAMPLE_COL].values

# --- archetype sweep ---
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
print(f'Running NOC sweep {NOC_MIN}–{NOC_MAX}, NDIM={NDIM}, NREPEATS={NREPEATS}...')

ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, samples)

# --- persist metric grids (cheap future replots) ---
metrics_df = pd.DataFrame({
    'NOC':       noc_grid,
    'EV':        ev_grid,
    'ARV':       av_grid,
    'ARV_rep':   av_rep_grid,
    'effEV':     ev_grid * (1 - av_grid),
    'effEV_rep': ev_grid * (1 - av_rep_grid),
})
metrics_df.to_csv(OUT_METRICS_TSV, sep='\t', index=False)
print(f'  Saved {OUT_METRICS_TSV}')

# --- plots ---
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sample_to_color = {s: cycle[i % len(cycle)] for i, s in enumerate(np.unique(samples))}

save_metrics_plot_html(noc_grid, ev_grid, av_grid, av_rep_grid,
                       NDIM, f'Archetype selection — Cheng22 mouse VX subspace (NDIM={NDIM})',
                       OUT_METRICS_HTML)

scatter_per_group_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                       samples, sample_to_color,
                       f'Per-sample archetype overlay — Cheng22 mouse VX subspace (NDIM={NDIM})',
                       OUT_INTERACTIVE)

print('Done.')
