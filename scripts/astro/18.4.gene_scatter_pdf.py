# P56 astrocyte gene-expression scatter as a vectorized PDF (plots only).
#
# Static, publication-ready counterpart of the interactive gene scatter produced by
# scripts 16 and 18.3 (which both also do non-plotting work and handle more than P56).
# This script is plotting-only and P56-only: it loads the cached P56 labels/coords and
# vertices, normalizes the P56 expression matrix, and renders one PDF page per fixed gene
# (PC1-PC3, PC1-PC4, PC3-PC4 panels). Scatter points are rasterized; axes/text/archetype
# overlay stay vector. No recomputation of labels, PCA, or archetypes.

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import save_gene_scatter_pdf

from scomics.utils import norm

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE            = os.path.join(PROJECT_ROOT, 'links', 'astro', 'gao25_scrna_astro.h5ad')
RES_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
PARQUET_ALL_IN        = os.path.join(RES_DIR, '17.labels_all_ages.parquet')
ARCHETYPE_VERTICES_IN = os.path.join(RES_DIR, '17.archetype_vertices_knn.parquet')
PDF_GENE_SCATTER      = os.path.join(FIG_DIR, '18.4.gene_scatter.pdf')

# (col_x, col_y, xlabel, ylabel) — retained cols: 0=PC1, 1=PC3, 2=PC4 (PC2 dropped)
PANELS = [(0, 1, 'PC1', 'PC3'), (0, 2, 'PC1', 'PC4'), (1, 2, 'PC3', 'PC4')]
FIXED_GENES = [
    'Trpm3', 'Cst3', 'Gfap', 'Nrxn1', 'Grin2c', 'Aldoc', 'Slc1a3', 'Chrdl1',
    'Il33', 'Id3', 'Id1', 'Spry1', 'Eogt', 'Grm3', 'Igfbp2', 'Inka2',
    'Slc25a34', 'Apoe', 'Mfge8', 'Efhd2', 'Ddhd1', 'Rfx4', 'Mertk', 'Sdc4',
    'Gria2', 'Slc7a10', 'Gabrg1',
]

os.makedirs(FIG_DIR, exist_ok=True)

# --- load cached coords + archetype vertices (P56 joint PCA, PC2 dropped) ---
df_all  = pd.read_parquet(PARQUET_ALL_IN)
pc_cols = [c for c in df_all.columns if c.startswith('PC')]

df_vertices = pd.read_parquet(ARCHETYPE_VERTICES_IN)
aa_vis = df_vertices.values.T[:3, :]  # (3, NOC) — rows PC1, PC3, PC4

df_p56 = df_all[df_all['age'] == 'P56'].reset_index(drop=True)
xp_p56 = df_p56[pc_cols].values

# --- normalize P56 expression (CP10k -> log2 -> z-score per gene) ---
adata = ad.read_h5ad(INPUT_FILE)
adata_p56 = adata[adata.obs['Age'] == 'P56']
x_p56  = adata_p56.X.toarray() if sp.issparse(adata_p56.X) else np.array(adata_p56.X)
depths = x_p56.sum(axis=1)
xn_p56 = norm(x_p56, depths)
var_names = np.array(adata_p56.var_names)

assert xn_p56.shape[0] == xp_p56.shape[0], (
    f'P56 cell count mismatch: expression {xn_p56.shape[0]} vs coords {xp_p56.shape[0]}'
)

# --- collect per-gene z-scores (fail loud on missing genes) ---
gene_vals = {}
for gene in FIXED_GENES:
    matches = np.where(var_names == gene)[0]
    if len(matches) == 0:
        print(f'  WARNING: gene {gene!r} not found in var_names, skipping')
        continue
    gene_vals[gene] = xn_p56[:, matches[0]]
print(f'Plotting {len(gene_vals)} / {len(FIXED_GENES)} genes for {xp_p56.shape[0]} P56 cells')

save_gene_scatter_pdf(
    xp=xp_p56,
    gene_vals=gene_vals,
    panels=PANELS,
    aa=aa_vis,
    title='P56 astrocytes (joint PCA, no PC2) NOC=4 — fixed gene expression',
    out_path=PDF_GENE_SCATTER,
)
print('Done.')
