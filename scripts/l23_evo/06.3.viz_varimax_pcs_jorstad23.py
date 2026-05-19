"""Scatter plots of PC1/PC2/PC3 in the reprojected VX subspace — Jorstad23 L2/3 IT.

Same coordinate space as script 11 (PC1/PC2/PC3 of VX subspace after PCHA projection),
but colored by cluster and covariates like script 06.2.

Reads: local_data/res/l23_evo/05.varimax_coords.tsv
Output: local_data/fig/l23_evo/06.3.varimax_pcs_scatter.html
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from viz import scatter_categorical_html

from scomics.main import SCA

# --- file paths ---
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_VARIMAX  = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
OUT_HTML    = os.path.join(OUT_FIG_DIR, '06.3.varimax_pcs_scatter.html')

# --- parameters (match script 11) ---
VX_COLS        = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
NOC            = 4
NDIM           = 5
CLUSTER_COL    = 'WithinArea_cluster'
COVARIATE_COLS = ['donor_id', 'Source', 'development_stage']

os.makedirs(OUT_FIG_DIR, exist_ok=True)

vx_df = pd.read_csv(IN_VARIMAX, sep='\t', index_col=0)
xn    = vx_df[VX_COLS].values
types = vx_df[CLUSTER_COL].values

print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)

cell_metadata = {col: vx_df[col].values for col in [CLUSTER_COL] + COVARIATE_COLS}

scatter_categorical_html(
    xp_grid=[sca.xp],
    cell_metadata=cell_metadata,
    title='Jorstad23 L2/3 IT — VX subspace PC1 vs PC2 vs PC3',
    out_path=OUT_HTML,
    panels=[(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')],
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    arch_vis=sca.aa,
)

print(f'Saved {OUT_HTML}')
