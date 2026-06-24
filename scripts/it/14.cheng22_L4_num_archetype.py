"""Archetype number selection on PCA subspace — cheng22 mouse L4 IT (compute).

Skips varimax / VX selection: runs the PCHA NOC sweep directly on the top 5 PCs
(PC1–PC5). Mirrors the yoo25 direct-PCA set for cheng22, L4 subclass.

This script only computes and persists results; plotting lives in
14.viz.cheng22_L4_num_archetype.py (reads the outputs below).

Reads:  local_data/res/it/13.cheng22_L4_pca_coords.tsv
Outputs:
  local_data/res/it/14.cheng22_L4_num_archetype_metrics.tsv   (metric grids)
  local_data/res/it/14.cheng22_L4_num_archetype_plotdata.pkl  (proj + per-group archetypes)
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
IN_PC_COORDS    = os.path.join(OUT_RES_DIR, '13.cheng22_L4_pca_coords.tsv')
OUT_METRICS_TSV = os.path.join(OUT_RES_DIR, '14.cheng22_L4_num_archetype_metrics.tsv')
OUT_PLOTDATA    = os.path.join(OUT_RES_DIR, '14.cheng22_L4_num_archetype_plotdata.pkl')

# --- parameters ---
CLUSTER_COL = 'Type'
SAMPLE_COL  = 'Sample'
# Top 5 PCs directly (no varimax / VX selection)
PC_COLS     = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
NDIM        = len(PC_COLS) - 1   # 4
NOC_MIN     = 2
NOC_MAX     = 6
NREPEATS    = 10

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- load PCA coords ---
pc_df   = pd.read_csv(IN_PC_COORDS, sep='\t', index_col=0)
xn      = pc_df[PC_COLS].values
types   = pc_df[CLUSTER_COL].values
samples = pc_df[SAMPLE_COL].values

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
