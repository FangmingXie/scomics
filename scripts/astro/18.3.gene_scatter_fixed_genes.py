# Gene expression scatter for P56 astrocytes using a fixed gene list.
# Subset of script 18.2 — only the gene expression visualization.

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import gene_expr_scatter_html

from scomics.utils import norm

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE     = os.path.join(PROJECT_ROOT, 'links', 'astro', 'gao25_scrna_astro.h5ad')
RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
PARQUET_ALL_IN      = os.path.join(RES_DIR, '17.labels_all_ages.parquet')
ARCHETYPE_VERTICES_IN = os.path.join(RES_DIR, '17.archetype_vertices_knn.parquet')
HTML_GENE_SCATTER = os.path.join(FIG_DIR, '18.3.gene_scatter.html')

# (col_x, col_y, xlabel, ylabel) — retained cols: 0=PC1, 1=PC3, 2=PC4 (PC2 dropped)
PANELS    = [(0, 1, 'PC1', 'PC3'), (0, 2, 'PC1', 'PC4'), (1, 2, 'PC3', 'PC4')]
PANEL_3D  = (0, 1, 2, 'PC1', 'PC3', 'PC4')
FIXED_GENES = [
    'Trpm3', 'Nckap5', 'Cdh13', 'Cst3', 'Gfap', 'Ddit4l', 'B3galt2', 'Nrxn1',
    'Grin2c', 'Thbs4', 'Kcnj6', 'Fkbp5',
    'Aldoc', 'Slc1a3', 'Chrdl1', 'Scel', 'Il33', 'Id3', 'Spry1', 'Eogt',
    'Adipor2', 'Grm3', 'Kirrel2', 'Igfbp2', 'Irak2', 'Chd9', 'Inka2',
    'Slc25a34', 'Axin2', 'Tnfrsf19', 'Apoe', 'Mfge8', 'Itm2c', 'Id1',
    'Phgdh', 'Efhd2', 'Lrp1b', 'Dhcr24', 'Paqr6', 'Ddhd1', 'Nr1d1',
    'Sox21', 'Stat2', 'Klf3', 'Rfx4', 'Mertk', 'Sdc4',
    'Unc13c', 'Agt', 'Gria2', 'Slc7a10', 'Gabrg1',
]

os.makedirs(FIG_DIR, exist_ok=True)

df_all = pd.read_parquet(PARQUET_ALL_IN)
pc_cols = [c for c in df_all.columns if c.startswith('PC')]

df_vertices = pd.read_parquet(ARCHETYPE_VERTICES_IN)
aa_vis = df_vertices.values.T[:3, :]  # (3, NOC)

adata = ad.read_h5ad(INPUT_FILE)
adata_p56 = adata[adata.obs['Age'] == 'P56']
x_p56  = adata_p56.X.toarray() if sp.issparse(adata_p56.X) else np.array(adata_p56.X)
depths = x_p56.sum(axis=1)
xn_p56 = norm(x_p56, depths)

var_names = np.array(adata_p56.var_names)
df_p56 = df_all[df_all['age'] == 'P56'].reset_index(drop=True)
xp_p56 = df_p56[pc_cols].values

gene_vals = {}
for gene in FIXED_GENES:
    matches = np.where(var_names == gene)[0]
    if len(matches) == 0:
        print(f'  WARNING: gene {gene!r} not found in var_names, skipping')
        continue
    gene_vals[gene] = xn_p56[:, matches[0]]
    print(f'  Found gene {gene!r} at index {matches[0]}')

gene_expr_scatter_html(
    x=None, y=None,
    xp=xp_p56,
    gene_vals=gene_vals,
    aa=aa_vis,
    title='P56 (no PC2, joint PCA top 5 PCs) NOC=4 — fixed gene expression',
    out_path=HTML_GENE_SCATTER,
    panels=PANELS,
    panel_3d=PANEL_3D,
    marker_size=5,
    bg_color='white',
)
print(f'Saved {HTML_GENE_SCATTER}')
