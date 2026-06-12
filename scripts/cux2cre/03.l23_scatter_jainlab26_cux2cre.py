"""Scatter plot visualization of jainlab26_cux2cre L2/3 IT PCA and varimax coords.

Reads coords TSVs from script 02 and writes two HTMLs, each with three subpanels:
  PC1 vs PC2 / PC1 vs PC3 / PC1 vs PC4  (colored by library size)
  VX1 vs VX2 / VX1 vs VX3 / VX1 vs VX4  (colored by library size)

Reads:
  local_data/res/cux2cre/02.l23_pc_coords.tsv
  local_data/res/cux2cre/02.l23_varimax_coords.tsv
Outputs:
  local_data/fig/cux2cre/03.l23_pc_scatter.html
  local_data/fig/cux2cre/03.l23_vx_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html

OUT_FIG_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre')
INPUT_PC_COORDS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '02.l23_pc_coords.tsv')
INPUT_VX_COORDS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '02.l23_varimax_coords.tsv')
OUT_PC_HTML     = os.path.join(OUT_FIG_DIR, '03.l23_pc_scatter.html')
OUT_VX_HTML     = os.path.join(OUT_FIG_DIR, '03.l23_vx_scatter.html')

os.makedirs(OUT_FIG_DIR, exist_ok=True)

LIBSIZE_COL  = 'total_counts'
SUBCLASS_COL = 'Subclass_transferred'


def plot_scatter(coords_df, comp_prefix, panels, out_path, title):
    comp_cols = [c for c in coords_df.columns if c.startswith(comp_prefix)]
    xp = coords_df[comp_cols].values  # (n_cells, n_components)

    scatter_categorical_html(
        xp_grid=[xp],
        cell_metadata={LIBSIZE_COL: coords_df[LIBSIZE_COL].values},
        title=title,
        out_path=out_path,
        panels=panels,
    )
    print(f'Saved {out_path}')


print('Loading PC coords...')
pc_df = pd.read_csv(INPUT_PC_COORDS, sep='\t', index_col=0)
print(f'  {len(pc_df)} cells')

print('Loading VX coords...')
vx_df = pd.read_csv(INPUT_VX_COORDS, sep='\t', index_col=0)

PC_PANELS = [
    (0, 1, 'PC1', 'PC2'),
    (0, 2, 'PC1', 'PC3'),
    (0, 3, 'PC1', 'PC4'),
]
VX_PANELS = [
    (0, 1, 'VX1', 'VX2'),
    (0, 2, 'VX1', 'VX3'),
    (0, 3, 'VX1', 'VX4'),
    (1, 3, 'VX2', 'VX4'),
]

plot_scatter(pc_df, 'PC', PC_PANELS, OUT_PC_HTML, 'jainlab26 L2/3 IT — PCA scatter (colored by library size)')
plot_scatter(vx_df, 'VX', VX_PANELS, OUT_VX_HTML, 'jainlab26 L2/3 IT — Varimax scatter (colored by library size)')

print('Done.')
