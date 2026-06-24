"""Archetype number selection — Jorstad23 human L2/3 IT (compute, refined w/ bootstrap error bars).

Refinement of 08.num_archetype_jorstad23.py: instead of a single bootstrap ARV per
NOC, the bootstrap-ARV procedure is repeated N_OUTER times so the metrics plot can
show mean ± std bands. EV and the per-group (rep) ARV stay single-valued.

Each outer repeat draws NREPEATS downsamples (DOWNSAMP_P fraction of cells, without
replacement) and reduces them to one ARV via get_relative_variation; mean/std are taken
across the N_OUTER repeats.

Reads:  local_data/res/l23_evo/05.varimax_coords.tsv
Outputs:
  local_data/res/l23_evo/08.refine.num_archetype_metrics.tsv   (metric grids w/ ARV mean/std)
  local_data/res/l23_evo/08.refine.num_archetype_plotdata.pkl  (proj + per-group archetypes)
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
OUT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
IN_VARIMAX      = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
OUT_METRICS_TSV = os.path.join(OUT_RES_DIR, '08.refine.num_archetype_metrics.tsv')
OUT_PLOTDATA    = os.path.join(OUT_RES_DIR, '08.refine.num_archetype_plotdata.pkl')

# --- parameters ---
CLUSTER_COL = 'WithinArea_cluster'
GROUP_COL   = 'donor_id'
VX_COLS     = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
NDIM        = 5     # n_fit = NDIM+1 = 6 = len(VX_COLS); drops last PC
NOC_MIN     = 2
NOC_MAX     = 6
NREPEATS    = 10    # downsample draws per ARV estimate
N_OUTER     = 20    # repeated ARV estimates → mean ± std
DOWNSAMP_P  = 0.2   # fraction of cells per downsample draw (without replacement)

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- load varimax coords ---
vx_df   = pd.read_csv(IN_VARIMAX, sep='\t', index_col=0)
xn      = vx_df[VX_COLS].values
types   = vx_df[CLUSTER_COL].values
groups  = vx_df[GROUP_COL].values

# --- archetype sweep (single pass: EV, per-group ARV, projection, per-group archetypes) ---
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
print(f'Running NOC sweep {NOC_MIN}–{NOC_MAX}, NDIM={NDIM}, NREPEATS={NREPEATS}...')

ev_grid, _, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, groups)

# --- repeated bootstrap ARV → mean ± std ---
print(f'Repeating bootstrap ARV N_OUTER={N_OUTER} times per NOC...')
arv_mean, arv_std, effev_mean, effev_std = [], [], [], []
for i, noc in enumerate(noc_grid):
    arv_reps = np.array([
        get_relative_variation(sca.bootstrap_proj_pcha(
            NDIM, noc, nrepeats=NREPEATS, is_bootstrap=False, downsamp_p=DOWNSAMP_P))
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

# --- persist projection + per-group archetypes (drives the per-donor scatter) ---
plotdata = {
    'noc_grid':     noc_grid,
    'ndim':         NDIM,
    'groups':       groups,
    'xp':           xp_grid[0],
    'aa_reps_grid': aa_reps_grid,
}
with open(OUT_PLOTDATA, 'wb') as f:
    pickle.dump(plotdata, f)
print(f'  Saved {OUT_PLOTDATA}')

print('Done.')
