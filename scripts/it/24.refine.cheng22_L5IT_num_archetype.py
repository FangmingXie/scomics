"""Archetype number selection on varimax subspace — cheng22 mouse L5IT IT (compute, refined).

Refinement of 24.cheng22_L5IT_num_archetype.py: the bootstrap-ARV procedure
is repeated N_OUTER times so the metrics plot can show mean ± std error bars. EV and the
per-group (rep) ARV stay single-valued. VX selection (single-Type subclass: cell-type selection is undefined (cell_type R²=0 on every component), so this uses the top varimax axes by variance after dropping the two technical-dominated axes VX3 (sample) and VX4 (library_size)): ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'].

Reads:  local_data/res/it/23.cheng22_L5IT_varimax_coords.tsv
Outputs:
  local_data/res/it/24.refine.cheng22_L5IT_num_archetype_metrics.tsv   (metric grids w/ ARV mean/std)
  local_data/res/it/24.refine.cheng22_L5IT_num_archetype_plotdata.pkl  (proj + per-group archetypes)
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
from scomics.utils import get_relative_variation

# --- file paths ---
OUT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IN_VX_COORDS    = os.path.join(OUT_RES_DIR, '23.cheng22_L5IT_varimax_coords.tsv')
OUT_METRICS_TSV = os.path.join(OUT_RES_DIR, '24.refine.cheng22_L5IT_num_archetype_metrics.tsv')
OUT_PLOTDATA    = os.path.join(OUT_RES_DIR, '24.refine.cheng22_L5IT_num_archetype_plotdata.pkl')

# --- parameters ---
CLUSTER_COL = 'Type'
SAMPLE_COL  = 'Sample'
# VX components selected from 23.cheng22_L5IT_vx_variance_partition.tsv
# (single-Type subclass: cell-type selection is undefined (cell_type R²=0 on every component), so this uses the top varimax axes by variance after dropping the two technical-dominated axes VX3 (sample) and VX4 (library_size))
VX_COLS     = ['VX1', 'VX2', 'VX5', 'VX6', 'VX7']
NDIM        = len(VX_COLS) - 1
NOC_MIN     = 2
NOC_MAX     = 6
NREPEATS    = 10   # bootstrap resamples per ARV estimate
N_OUTER     = 20   # repeated ARV estimates → mean ± std

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- load varimax coords ---
vx_df   = pd.read_csv(IN_VX_COORDS, sep='\t', index_col=0)
xn      = vx_df[VX_COLS].values
types   = vx_df[CLUSTER_COL].values
samples = vx_df[SAMPLE_COL].values

# --- archetype sweep (single pass: EV, per-group ARV, projection, per-group archetypes) ---
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
print(f'Running NOC sweep {NOC_MIN}–{NOC_MAX}, NDIM={NDIM}, NREPEATS={NREPEATS}...')

ev_grid, _, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, samples)

# --- repeated bootstrap ARV → mean ± std ---
print(f'Repeating bootstrap ARV N_OUTER={N_OUTER} times per NOC...')
arv_mean, arv_std, effev_mean, effev_std = [], [], [], []
for i, noc in enumerate(noc_grid):
    arv_reps = np.array([
        get_relative_variation(sca.bootstrap_proj_pcha(NDIM, noc, nrepeats=NREPEATS))
        for _ in range(N_OUTER)
    ])
    effev_reps = ev_grid[i] * (1 - arv_reps)
    arv_mean.append(arv_reps.mean())
    arv_std.append(arv_reps.std())
    effev_mean.append(effev_reps.mean())
    effev_std.append(effev_reps.std())
    print(f"  NOC={noc}  ARV={arv_reps.mean():.4f}±{arv_reps.std():.4f}"
          f"  effEV={effev_reps.mean():.4f}±{effev_reps.std():.4f}")

arv_mean, arv_std = np.array(arv_mean), np.array(arv_std)
effev_mean, effev_std = np.array(effev_mean), np.array(effev_std)
effev_rep_grid = ev_grid * (1 - av_rep_grid)

# --- persist metric grids ---
metrics_df = pd.DataFrame({
    'NOC':        noc_grid,
    'EV':         ev_grid,
    'ARV_mean':   arv_mean,
    'ARV_std':    arv_std,
    'ARV_rep':    av_rep_grid,
    'effEV_mean': effev_mean,
    'effEV_std':  effev_std,
    'effEV_rep':  effev_rep_grid,
})
metrics_df.to_csv(OUT_METRICS_TSV, sep='\t', index=False)
print(f'  Saved {OUT_METRICS_TSV}')

# --- persist projection + per-group archetypes (drives the per-sample scatter) ---
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
