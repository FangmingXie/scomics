# Visualization of sequential kNN archetype label transfer (from script 17).
# Same as script 18 but uses a fixed gene list instead of distance-based gene scoring.
# Loads combined parquet; renders per-age archetype scatter + all-ages scatter + abundance barplot
# + gene expression scatter (P56 cells) for a fixed set of genes.

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pandas as pd
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import scatter_categorical_html, stacked_bar_html, gene_expr_scatter_html

from scomics.utils import norm

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE     = os.path.join(PROJECT_ROOT, 'links', 'astro', 'gao25_scrna_astro.h5ad')
RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
PARQUET_ALL_IN      = os.path.join(RES_DIR, '17.labels_all_ages.parquet')
ARCHETYPE_VERTICES_IN = os.path.join(RES_DIR, '17.archetype_vertices_knn.parquet')
HTML_PER_AGE   = os.path.join(FIG_DIR, '18.2.scatter_{age}.html')
HTML_ALL_AGES  = os.path.join(FIG_DIR, '18.2.scatter_all_ages.html')
HTML_BARPLOT   = os.path.join(FIG_DIR, '18.2.archetype_abundance.html')
HTML_GENE_SCATTER = os.path.join(FIG_DIR, '18.2.gene_scatter.html')

SCATTER_AGES      = ['P0', 'P7', 'P14', 'P21', 'P28', 'P56']
# (col_x, col_y, xlabel, ylabel) — retained cols: 0=PC1, 1=PC3, 2=PC4 (PC2 dropped)
PANELS            = [(0, 1, 'PC1', 'PC3'), (0, 2, 'PC1', 'PC4'), (1, 2, 'PC3', 'PC4')]
PANEL_3D          = (0, 1, 2, 'PC1', 'PC3', 'PC4')
FIXED_GENES       = ['Trpm3', 'Nckap5', 'Cdh13', 'Cst3', 'Gfap', 'Ddit4l', 'B3galt2', 'Nrxn1']

os.makedirs(FIG_DIR, exist_ok=True)

df_all = pd.read_parquet(PARQUET_ALL_IN)
print(f'Loaded {len(df_all)} cells from {PARQUET_ALL_IN}')

pc_cols = [c for c in df_all.columns if c.startswith('PC')]

# Load PCHA archetype vertices projected into kNN visualization space (from script 17)
df_vertices = pd.read_parquet(ARCHETYPE_VERTICES_IN)  # (NOC, n_knn_dims)
aa_vis = df_vertices.values.T[:3, :]                  # (3, NOC) — first 3 dims used by PANELS
print(f'Loaded PCHA archetype vertices (kNN space):\n{df_vertices}')

# Per-age scatter colored by archetype (PC2 already dropped by script 17)
for age_val in SCATTER_AGES:
    df_age = df_all[df_all['age'] == age_val].reset_index(drop=True)
    xp_age = df_age[pc_cols].values

    scatter_categorical_html(
        xp_grid=[xp_age],
        cell_metadata={
            'archetype':  df_age['archetype'].values,
            'donor_name': df_age['donor_name'].values,
        },
        title=f'{age_val} — kNN-transferred archetype labels (P56 joint PCA top 5 PCs, no PC2)',
        out_path=HTML_PER_AGE.format(age=age_val),
        panels=PANELS,
        panel_3d=PANEL_3D,
        arch_vis=aa_vis,
    )
    print(f'  Saved {HTML_PER_AGE.format(age=age_val)}')

# All-ages scatter colored by archetype or age
xp_all = df_all[pc_cols].values

scatter_categorical_html(
    xp_grid=[xp_all],
    cell_metadata={
        'archetype':  df_all['archetype'].values,
        'age':        df_all['age'].values,
        'donor_name': df_all['donor_name'].values,
    },
    title='All postnatal ages — kNN-transferred archetype labels (P56 joint PCA top 5 PCs, no PC2)',
    out_path=HTML_ALL_AGES,
    ordered_labels=('age',),
    panels=PANELS,
    panel_3d=PANEL_3D,
    arch_vis=aa_vis,
)
print(f'Saved {HTML_ALL_AGES}')

# Archetype abundance barplot across all ages
age_order = natsorted(df_all['age'].unique())
archetype_order = sorted(df_all['archetype'].unique())

counts = df_all.groupby(['age', 'archetype']).size().unstack(fill_value=0)
counts = counts.reindex(columns=archetype_order, fill_value=0)
frac = counts.div(counts.sum(axis=1), axis=0)

stacked_bar_html(
    panel_data=[('Archetype fraction by age', age_order, frac)],
    celltypes=archetype_order,
    title='Archetype abundance across postnatal ages (sequential kNN transfer, top 5 PCs)',
    out_path=HTML_BARPLOT,
    panel_width=1000,
)
print(f'Saved {HTML_BARPLOT}')

# --- Fixed gene list expression scatter (P56 cells) ---
adata = ad.read_h5ad(INPUT_FILE)
adata_p56 = adata[adata.obs['Age'] == 'P56']
x_p56  = adata_p56.X.toarray() if sp.issparse(adata_p56.X) else np.array(adata_p56.X)
depths = x_p56.sum(axis=1)

var_names = np.array(adata_p56.var_names)
xn_p56 = norm(x_p56, depths)

df_p56  = df_all[df_all['age'] == 'P56'].reset_index(drop=True)
xp_p56   = df_p56[pc_cols].values

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
)
print(f'Saved {HTML_GENE_SCATTER}')
