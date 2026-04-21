# Visualization of sequential kNN archetype label transfer (from script 15).
# Loads combined parquet; renders per-age archetype scatter + all-ages scatter + abundance barplot
# + archetype-specific gene expression scatter (P56 cells).

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pandas as pd
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg
from viz import scatter_categorical_html, stacked_bar_html, gene_expr_scatter_html

from scomics.utils import norm

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE     = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')
RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
PARQUET_ALL_IN = os.path.join(RES_DIR, '15.labels_all_ages.parquet')
HTML_PER_AGE   = os.path.join(FIG_DIR, '16.scatter_{age}.html')
HTML_ALL_AGES  = os.path.join(FIG_DIR, '16.scatter_all_ages.html')
HTML_BARPLOT   = os.path.join(FIG_DIR, '16.archetype_abundance.html')
HTML_GENE_SCATTER = os.path.join(FIG_DIR, '16.gene_scatter.html')

SCATTER_AGES      = ['P0', 'P7', 'P14', 'P21', 'P28', 'P56']
# (col_x, col_y, xlabel, ylabel) — retained cols: 0=PC1, 1=PC3, 2=PC4 (PC2 dropped)
PANELS            = [(0, 1, 'PC1', 'PC3'), (0, 2, 'PC1', 'PC4'), (1, 2, 'PC3', 'PC4')]
PANEL_3D          = (0, 1, 2, 'PC1', 'PC3', 'PC4')
N_TOP_GENES       = 2000
N_ARCHETYPE_CELLS = 300
N_TOP_ARCHETYPE   = 5

os.makedirs(FIG_DIR, exist_ok=True)

df_all = pd.read_parquet(PARQUET_ALL_IN)
print(f'Loaded {len(df_all)} cells from {PARQUET_ALL_IN}')

pc_cols = [c for c in df_all.columns if c.startswith('PC')]

# Archetype centroids in visualization space (P56 cells, first 3 PC dims)
df_p56_all = df_all[df_all['age'] == 'P56']
arch_order  = sorted(df_p56_all['archetype'].unique())
aa_vis = np.stack([
    df_p56_all[df_p56_all['archetype'] == a][pc_cols[:3]].values.mean(axis=0)
    for a in arch_order
], axis=1)  # shape (3, noc)
print(f'Archetype centroids (P56 visualization space): {aa_vis}')

# Per-age scatter colored by archetype (PC2 already dropped by script 15)
for age_val in SCATTER_AGES:
    df_age = df_all[df_all['age'] == age_val].reset_index(drop=True)
    xp_age = df_age[pc_cols].values

    scatter_categorical_html(
        xp_grid=[xp_age],
        cell_metadata={
            'archetype':  df_age['archetype'].values,
            'donor_name': df_age['donor_name'].values,
        },
        title=f'{age_val} — kNN-transferred archetype labels (P56 joint PCA, no PC2)',
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
    title='All postnatal ages — kNN-transferred archetype labels (P56 joint PCA, no PC2)',
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
    title='Archetype abundance across postnatal ages (sequential kNN transfer)',
    out_path=HTML_BARPLOT,
    panel_width=1000,
)
print(f'Saved {HTML_BARPLOT}')

# --- Archetype-specific gene expression scatter (P56 cells) ---
adata = ad.read_h5ad(INPUT_FILE)
adata_p56 = adata[adata.obs['Age'] == 'P56']
x_p56  = adata_p56.X.toarray() if sp.issparse(adata_p56.X) else np.array(adata_p56.X)
depths = x_p56.sum(axis=1)

hvg_mask   = select_hvg(x_p56, depths, N_TOP_GENES)
gene_names = np.array(adata_p56.var_names)[hvg_mask]
xn_p56     = norm(x_p56[:, hvg_mask], depths)

df_p56  = df_all[df_all['age'] == 'P56'].reset_index(drop=True)
arch_p56 = df_p56['archetype'].values
xp_p56   = df_p56[pc_cols].values

# Distance-based gene scoring: N_ARCHETYPE_CELLS closest to each archetype centroid vs. rest
noc       = len(np.unique(arch_p56))
centroids = np.stack([xp_p56[arch_p56 == f'Arch{k+1}'].mean(axis=0) for k in range(noc)])
dists_all = np.stack([np.linalg.norm(xp_p56 - centroids[k], axis=1) for k in range(noc)], axis=1)

arch_specific_idx = []
for k in range(noc):
    closest   = np.argsort(dists_all[:, k])[:N_ARCHETYPE_CELLS]
    rest      = np.argsort(dists_all[:, k])[N_ARCHETYPE_CELLS:]
    mean_diff = xn_p56[closest].mean(axis=0) - xn_p56[rest].mean(axis=0)
    top_idx   = np.argsort(mean_diff)[::-1][:N_TOP_ARCHETYPE]
    arch_specific_idx.append(top_idx)
    print(f'  Arch{k+1} top genes: {list(gene_names[top_idx])}')

seen, ordered_idx = set(), []
for top_idx in arch_specific_idx:
    for idx in top_idx:
        if idx not in seen:
            seen.add(idx)
            ordered_idx.append(idx)

gene_vals = {gene_names[i]: xn_p56[:, i] for i in ordered_idx}

gene_expr_scatter_html(
    x=None, y=None,
    xp=xp_p56,
    gene_vals=gene_vals,
    aa=aa_vis,
    title='P56 (no PC2, joint PCA) NOC=4 — archetype-specific gene expression',
    out_path=HTML_GENE_SCATTER,
    panels=PANELS,
    panel_3d=PANEL_3D,
)
print(f'Saved {HTML_GENE_SCATTER}')
