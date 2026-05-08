"""Scatter plots of varimax-rotated components — Jorstad23 L2/3 IT.

Reads: local_data/res/l23_evo/05.varimax_coords.tsv
Output: local_data/fig/l23_evo/06.varimax_scatter.html
"""

import os
import sys
import pandas as pd

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from viz import scatter_categorical_html

OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_VARIMAX  = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
OUT_HTML    = os.path.join(OUT_FIG_DIR, '06.varimax_scatter.html')

CLUSTER_COL    = 'WithinArea_cluster'
COVARIATE_COLS = ['donor_id', 'Source', 'development_stage']

os.makedirs(OUT_FIG_DIR, exist_ok=True)

vx_df = pd.read_csv(IN_VARIMAX, sep='\t', index_col=0)

vx_coords = vx_df[['VX2', 'VX6', 'VX7']].values
cell_metadata = {col: vx_df[col].values for col in [CLUSTER_COL] + COVARIATE_COLS}

scatter_categorical_html(
    xp_grid=[vx_coords],
    cell_metadata=cell_metadata,
    title='Jorstad23 L2/3 IT — Varimax components VX2 vs VX6 vs VX7',
    out_path=OUT_HTML,
    panels=[(0, 1, 'VX2', 'VX6'), (0, 2, 'VX2', 'VX7'), (1, 2, 'VX6', 'VX7')],
    panel_3d=(0, 1, 2, 'VX2', 'VX6', 'VX7'),
)

print(f'Saved {OUT_HTML}')
