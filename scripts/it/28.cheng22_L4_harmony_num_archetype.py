"""Archetype number selection on Harmony PCs — cheng22 mouse L4 IT (compute).

New procedure: PCHA NOC sweep run directly on the top 5 Harmony-corrected PCs
(H1–H5, NDIM=5, no PC dropped — drop_pcs=[]). No varimax, no VX selection.
Front-end (library-size regression + PCA + Harmony) is 27.cheng22_L4_harmony_pca.py.

This script only computes and persists results; plotting lives in
28.viz.cheng22_L4_harmony_num_archetype.py (reads the outputs below).

Reads:  local_data/res/it/27.cheng22_L4_harmony_coords.tsv
Outputs:
  local_data/res/it/28.cheng22_L4_harmony_num_archetype_metrics.tsv   (metric grids)
  local_data/res/it/28.cheng22_L4_harmony_num_archetype_plotdata.pkl  (proj + per-group archetypes)
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
IN_COORDS       = os.path.join(OUT_RES_DIR, '27.cheng22_L4_harmony_coords.tsv')
OUT_METRICS_TSV = os.path.join(OUT_RES_DIR, '28.cheng22_L4_harmony_num_archetype_metrics.tsv')
OUT_PLOTDATA    = os.path.join(OUT_RES_DIR, '28.cheng22_L4_harmony_num_archetype_plotdata.pkl')

# --- parameters ---
CLUSTER_COL = 'Type'
SAMPLE_COL  = 'Sample'
# Top 5 Harmony-corrected PCs, used directly as the PCHA space (no drop)
PC_COLS     = ['H1', 'H2', 'H3', 'H4', 'H5']
NDIM        = len(PC_COLS)   # 5
DROP_PCS    = []             # keep all NDIM dims (no PC dropped)
NOC_MIN     = 2
NOC_MAX     = 6
NREPEATS    = 10

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- load Harmony coords ---
df      = pd.read_csv(IN_COORDS, sep='\t', index_col=0)
xn      = df[PC_COLS].values
types   = df[CLUSTER_COL].values
samples = df[SAMPLE_COL].values

# --- archetype sweep ---
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
print(f'Running NOC sweep {NOC_MIN}–{NOC_MAX}, NDIM={NDIM} (drop_pcs={DROP_PCS}), NREPEATS={NREPEATS}...')

ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, samples, drop_pcs=DROP_PCS)

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
