"""Categorical scatter of WithinArea_cluster in reprojected VX subspace — Jorstad23 L2/3 IT.

Same coordinate space as script 10 (PC1/PC2/PC3 of VX subspace, NOC=4 archetypes overlaid),
colored by WithinArea_cluster with downsampling for a compact HTML.

Reads:
  local_data/res/l23_evo/05.varimax_coords.tsv
Output:
  local_data/fig/l23_evo/11.archetype_clusters_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from viz import scatter_categorical_html

from scomics.main import SCA

# --- file paths ---
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_VARIMAX   = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '11.archetype_clusters_scatter.html')

# --- parameters ---
VX_COLS      = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
NOC          = 4
NDIM         = 5
N_DOWNSAMPLE = 5000
CLUSTER_COL  = 'WithinArea_cluster'

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- fit PCHA at NOC=4 ---
print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
vx_df = pd.read_csv(IN_VARIMAX, sep='\t', index_col=0)
xn    = vx_df[VX_COLS].values
types = vx_df[CLUSTER_COL].values

sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)

# --- downsample ---
n_cells = sca.xp.shape[0]
if N_DOWNSAMPLE is not None and N_DOWNSAMPLE < n_cells:
    rng  = np.random.default_rng(0)
    sidx = rng.choice(n_cells, size=N_DOWNSAMPLE, replace=False)
    print(f'Downsampled {n_cells} → {N_DOWNSAMPLE} cells for plotting.')
else:
    sidx = np.arange(n_cells)
    print(f'Using all {n_cells} cells.')

xp_plot = sca.xp[sidx]
cell_metadata = {CLUSTER_COL: types[sidx]}

# --- plot ---
scatter_categorical_html(
    xp_grid=[xp_plot],
    cell_metadata=cell_metadata,
    title=f'Jorstad23 L2/3 IT — WithinArea_cluster (NOC={NOC}, reprojected VX subspace)',
    out_path=OUT_HTML,
    panels=[(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')],
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    aa_grid=[sca.aa],
)

print(f'Saved {OUT_HTML}')
