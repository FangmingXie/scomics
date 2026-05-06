"""Scatter plots of cell-type-robust PC axes (PC1, PC3, PC5) — Jorstad23 L2/3 IT.

Reads: local_data/res/l23_evo/01.pca_coords.tsv
Output: local_data/fig/l23_evo/04.pcs_scatter.html
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
IN_PCA      = os.path.join(OUT_RES_DIR, '01.pca_coords.tsv')
OUT_HTML    = os.path.join(OUT_FIG_DIR, '04.pcs_scatter.html')

CLUSTER_COL    = 'WithinArea_cluster'
COVARIATE_COLS = ['donor_id', 'Source', 'development_stage']

os.makedirs(OUT_FIG_DIR, exist_ok=True)

pca_df = pd.read_csv(IN_PCA, sep='\t', index_col=0)

# PC indices (0-based): PC1=0, PC3=2, PC5=4
pc_coords = pca_df[['PC1', 'PC3', 'PC5']].values
cell_metadata = {col: pca_df[col].values for col in [CLUSTER_COL] + COVARIATE_COLS}

scatter_categorical_html(
    xp_grid=[pc_coords],
    cell_metadata=cell_metadata,
    title='Jorstad23 L2/3 IT — PC1 vs PC3 vs PC5 (cell-type axes)',
    out_path=OUT_HTML,
    panels=[(0, 1, 'PC1', 'PC3'), (0, 2, 'PC1', 'PC5'), (1, 2, 'PC3', 'PC5')],
    panel_3d=(0, 1, 2, 'PC1', 'PC3', 'PC5'),
)

print(f'Saved {OUT_HTML}')
