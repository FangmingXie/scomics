"""Archetype number selection on varimax subspace — Yoo25 mouse L4 IT (compute).

Runs PCHA NOC sweep on the varimax components where cell-type R² dominates over
sample / library_size (selected from 01.yoo25_L4_vx_variance_partition.tsv):
VX1, VX2, VX6, VX7, VX9.

This script only computes and persists results; plotting lives in
02.viz.yoo25_L4_num_archetype.py (reads the outputs below).

Reads:  local_data/res/it/01.yoo25_L4_varimax_coords.tsv
Outputs:
  local_data/res/it/02.yoo25_L4_num_archetype_metrics.tsv   (metric grids)
  local_data/res/it/02.yoo25_L4_num_archetype_plotdata.pkl  (proj + per-group archetypes)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import run_noc_sweep
from scomics.main import SCA

# --- file paths ---
OUT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IN_VX_COORDS    = os.path.join(OUT_RES_DIR, '01.yoo25_L4_varimax_coords.tsv')
OUT_METRICS_TSV = os.path.join(OUT_RES_DIR, '02.yoo25_L4_num_archetype_metrics.tsv')
OUT_PLOTDATA    = os.path.join(OUT_RES_DIR, '02.yoo25_L4_num_archetype_plotdata.pkl')

# --- parameters ---
CLUSTER_COL = 'Type'
SAMPLE_COL  = 'Sample'
# VX components where cell_type R² dominates (from 01.yoo25_L4_vx_variance_partition.tsv)
VX_COLS     = ['VX1', 'VX2', 'VX6', 'VX7', 'VX9']
NDIM        = len(VX_COLS) - 1   # 4
NOC_MIN     = 2
NOC_MAX     = 6
NREPEATS    = 10

os.makedirs(OUT_RES_DIR, exist_ok=True)

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

# --- persist metric grids (human-readable; drives the metrics plot) ---
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

# --- persist projection + per-group archetypes (drives the per-sample scatter) ---
# xp is the ndim projection (NOC-independent), so store one array, not the full grid.
plotdata = {
    'noc_grid':     noc_grid,
    'ndim':         NDIM,
    'samples':      samples,
    'xp':           xp_grid[0],
    'aa_reps_grid': aa_reps_grid,
}
with open(OUT_PLOTDATA, 'wb') as f:
    pickle.dump(plotdata, f)
print(f'  Saved {OUT_PLOTDATA}')

print('Done.')
