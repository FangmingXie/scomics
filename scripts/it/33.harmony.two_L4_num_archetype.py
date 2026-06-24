"""Archetype number selection — two-dataset L4 IT (NOC sweep on Harmony PCs).

Reads the Harmony-corrected embedding produced by 33.harmony.two_L4_embed.py
(cheng22 + yoo25, gao25 dropped) and sweeps the number-of-archetypes (NOC) on the
top 5 Harmony-corrected PCs (H1-H5, NDIM=5), with N_OUTER bootstrap repeats for
mean +/- std error bars. Per-group (dataset:sample) archetypes are also fit for the
scatter overlay. No embedding here — rerun 33.harmony.two_L4_embed.py if the coords
are missing.

Reads:
  local_data/res/it/33.harmony.two_L4_coords.tsv   (H1..H10 + Type + Sample + Dataset)
Outputs:
  local_data/res/it/33.harmony.two_L4_num_archetype_metrics.tsv   (metric grids w/ ARV mean/std)
  local_data/res/it/33.harmony.two_L4_num_archetype_plotdata.pkl  (proj + per-group archetypes)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import run_noc_sweep
from scomics.main import SCA
from scomics.utils import get_relative_variation

# --- file paths ---
OUT_RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
OUT_LOG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'log', 'it')
IN_HARMONY_COORDS  = os.path.join(OUT_RES_DIR, '33.harmony.two_L4_coords.tsv')
OUT_METRICS_TSV    = os.path.join(OUT_RES_DIR, '33.harmony.two_L4_num_archetype_metrics.tsv')
OUT_PLOTDATA       = os.path.join(OUT_RES_DIR, '33.harmony.two_L4_num_archetype_plotdata.pkl')
OUT_LOG            = os.path.join(OUT_LOG_DIR, '33.harmony.two_L4_num_archetype.log')

# --- parameters ---
CLUSTER_COL    = 'Type'
SAMPLE_COL     = 'Sample'
N_PCS          = 10
# Top 5 Harmony-corrected PCs used directly as the PCHA space (no drop)
N_ARCH_PCS     = 5
NDIM           = N_ARCH_PCS   # 5
DROP_PCS       = []           # keep all NDIM dims (no PC dropped)
NOC_MIN        = 2
NOC_MAX        = 6
NREPEATS       = 10   # bootstrap resamples per ARV estimate
N_OUTER        = 3    # repeated ARV estimates -> mean +/- std

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_LOG_DIR, exist_ok=True)


# --- unbuffered tee: mirror stdout+stderr to console and a log file, flush every write ---
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_fh = open(OUT_LOG, 'w', buffering=1)   # line-buffered file handle
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)  # tqdm writes here
print(f'Logging to {OUT_LOG}')

# ===================== load Harmony embedding =====================

print(f'Loading Harmony coords from {IN_HARMONY_COORDS}...')
coords = pd.read_csv(IN_HARMONY_COORDS, sep='\t', index_col=0)
h_cols = [f'H{i+1}' for i in range(N_PCS)]
harmony_scores = coords[h_cols].values
types   = coords[CLUSTER_COL].astype(str).values
samples = coords[SAMPLE_COL].astype(str).values
print(f'  {harmony_scores.shape[0]} cells, {len(np.unique(samples))} dataset:sample groups')

# ===================== archetype sweep =====================

# Feature matrix = top 5 Harmony-corrected PCs (used directly, no drop)
xf = harmony_scores[:, :N_ARCH_PCS]
sca = SCA(xf, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
print(f'Running NOC sweep {NOC_MIN}-{NOC_MAX}, NDIM={NDIM} (drop_pcs={DROP_PCS}), NREPEATS={NREPEATS}...')

ev_grid, _, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, samples, drop_pcs=DROP_PCS)

# --- repeated bootstrap ARV -> mean +/- std ---
print(f'Repeating bootstrap ARV N_OUTER={N_OUTER} times per NOC...')
arv_mean, arv_std, effev_mean, effev_std = [], [], [], []
for i, noc in enumerate(tqdm(noc_grid, desc='Bootstrap ARV (per NOC)')):
    arv_reps = np.array([
        get_relative_variation(sca.bootstrap_proj_pcha(NDIM, noc, nrepeats=NREPEATS, drop_pcs=DROP_PCS))
        for _ in tqdm(range(N_OUTER), desc=f'  NOC={noc} repeats', leave=False)
    ])
    effev_reps = ev_grid[i] * (1 - arv_reps)
    arv_mean.append(arv_reps.mean())
    arv_std.append(arv_reps.std())
    effev_mean.append(effev_reps.mean())
    effev_std.append(effev_reps.std())
    tqdm.write(f"  NOC={noc}  ARV={arv_reps.mean():.4f}+/-{arv_reps.std():.4f}"
               f"  effEV={effev_reps.mean():.4f}+/-{effev_reps.std():.4f}")

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
