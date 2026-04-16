# Visualization of all ages projected onto P56 PCA space (from script 13).
# Reproduces all script-12 plots for P56 archetype analysis, plus the all-ages scatter.
# Ages colored with turbo colormap (natsorted order), matching script 02.

import os
import sys
import glob
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg
from viz import (save_metrics_plot, scatter_per_group_html,
                 scatter_categorical_html, gene_expr_scatter_html)

from scomics.utils import norm

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
H5AD_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')

PARQUET_AGE_GLOB         = os.path.join(RES_DIR, '13.pca_projected_*.parquet')
PARQUET_AGE_TEMPLATE     = os.path.join(RES_DIR, '13.pca_projected_{age}.parquet')
PARQUET_P56_ARCHETYPES   = os.path.join(RES_DIR, '13.p56_archetypes.parquet')
PARQUET_METRICS          = os.path.join(RES_DIR, '13.metrics.parquet')
PARQUET_P56_ARCH_PCA     = os.path.join(RES_DIR, '13.p56_archetype_pca.parquet')
PARQUET_P56_ARCH_VERTS   = os.path.join(RES_DIR, '13.p56_archetype_vertices.parquet')
PARQUET_P56_DONOR_REPS   = os.path.join(RES_DIR, '13.p56_archetype_donor_reps.parquet')

FIG_METRICS                   = os.path.join(FIG_DIR, '14.p56_archetype_metrics.png')
FIG_ARCH_REP                  = os.path.join(FIG_DIR, '14.p56_archetype_rep.html')
FIG_ARCH_PCA_CAT              = os.path.join(FIG_DIR, '14.p56_archetype_pca_metadata.html')
FIG_GENE_SCATTER              = os.path.join(FIG_DIR, '14.p56_archetype_gene_scatter.html')
FIG_ALL_AGES_SCATTER          = os.path.join(FIG_DIR, '14.all_ages_pca_scatter.html')
FIG_GENE_SCATTER_AGE_TEMPLATE = os.path.join(FIG_DIR, '14.gene_scatter_{age}.html')

P56_AGE_VAL        = 'P56'
NDIM               = 5
NOC                = 4
N_TOP_GENES        = 2000
N_ARCHETYPE_CELLS  = 300
N_TOP_ARCHETYPE    = 5
AGE_MIN_CELLS      = 50
GENE_SCATTER_AGES  = ['P1', 'P7', 'P14', 'P28', 'P56']

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load archetype-space data
# ---------------------------------------------------------------------------
metrics       = pd.read_parquet(PARQUET_METRICS)
df_p56_arch   = pd.read_parquet(PARQUET_P56_ARCH_PCA)
df_aa         = pd.read_parquet(PARQUET_P56_ARCH_VERTS)
df_donor_reps = pd.read_parquet(PARQUET_P56_DONOR_REPS)

adim_cols = [f'APC{i+1}' for i in range(NDIM)]
noc_grid  = metrics['noc'].values
ev_grid   = metrics['ev'].values
av_grid   = metrics['av'].values
av_rep_grid = metrics['av_rep'].values

xp = df_p56_arch[adim_cols].values          # (n_p56, NDIM)
aa = df_aa[adim_cols].values.T               # (NDIM, NOC)
types_p56  = df_p56_arch['cluster_label'].values
donors_p56 = df_p56_arch['donor_name'].values
arch_labels_p56 = df_p56_arch['archetype'].values

# reconstruct aa_reps_grid (list of lists of (donor, aa_d))
aa_reps = []
for donor in df_donor_reps['donor'].unique():
    sub  = df_donor_reps[df_donor_reps['donor'] == donor].sort_values('archetype_id')
    aa_d = sub[adim_cols].values.T  # (NDIM, NOC)
    aa_reps.append((donor, aa_d))
aa_reps_grid = [aa_reps]

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
donor_to_color = {d: cycle[i % len(cycle)] for i, d in enumerate(np.unique(donors_p56))}

# ---------------------------------------------------------------------------
# Plot 1: Metrics PNG
# ---------------------------------------------------------------------------
save_metrics_plot(noc_grid, ev_grid, av_grid, av_rep_grid,
                  NDIM, f'Archetype selection P56 (no PC2) NOC={NOC} (NDIM={NDIM})',
                  FIG_METRICS)

# ---------------------------------------------------------------------------
# Plot 2: Per-donor archetype overlay
# ---------------------------------------------------------------------------
scatter_per_group_html(
    noc_grid, ev_grid, av_rep_grid, [xp], aa_reps_grid,
    donors_p56, donor_to_color,
    f'Per-donor archetype overlay P56 (no PC2) NOC={NOC} (NDIM={NDIM})',
    FIG_ARCH_REP,
)

# ---------------------------------------------------------------------------
# Plot 3: P56 PCA scatter colored by metadata (archetype space)
# ---------------------------------------------------------------------------
scatter_categorical_html(
    xp_grid=[xp],
    cell_metadata={'cluster_label': types_p56, 'donor_name': donors_p56, 'archetype': arch_labels_p56},
    title=f'P56 astrocytes (no PC2) NOC={NOC} — PCA scatter colored by metadata (NDIM={NDIM})',
    out_path=FIG_ARCH_PCA_CAT,
    noc_grid=noc_grid, ev_grid=ev_grid, av_grid=av_grid, aa_grid=[aa],
)

# ---------------------------------------------------------------------------
# Plot 4: Archetype-specific gene scatter (P56, archetype space)
# ---------------------------------------------------------------------------
adata = ad.read_h5ad(H5AD_FILE)
adata_p56 = adata[adata.obs['Age'] == P56_AGE_VAL]
x_raw  = adata_p56.X.toarray() if sp.issparse(adata_p56.X) else np.array(adata_p56.X)
depths = x_raw.sum(axis=1)
hvg_mask   = select_hvg(x_raw, depths, N_TOP_GENES)
gene_names = np.array(adata_p56.var_names)[hvg_mask]
xn = norm(x_raw[:, hvg_mask], depths)

dists_all = np.stack([np.linalg.norm(xp - aa[:, k], axis=1) for k in range(NOC)], axis=1)

arch_specific_idx = []
for k in range(NOC):
    closest   = np.argsort(dists_all[:, k])[:N_ARCHETYPE_CELLS]
    rest      = np.argsort(dists_all[:, k])[N_ARCHETYPE_CELLS:]
    mean_diff = xn[closest].mean(axis=0) - xn[rest].mean(axis=0)
    top_idx   = np.argsort(mean_diff)[::-1][:N_TOP_ARCHETYPE]
    arch_specific_idx.append(top_idx)
    print(f'  Archetype {k+1} top genes: {list(gene_names[top_idx])}')

seen, ordered_idx = set(), []
for top_idx in arch_specific_idx:
    for idx in top_idx:
        if idx not in seen:
            seen.add(idx)
            ordered_idx.append(idx)

gene_vals = {gene_names[i]: xn[:, i] for i in ordered_idx}

gene_expr_scatter_html(
    x=xp[:, 0], y=xp[:, 1], z=xp[:, 2],
    gene_vals=gene_vals,
    aa=aa,
    title=f'P56 (no PC2) NOC={NOC} — archetype-specific gene expression',
    out_path=FIG_GENE_SCATTER,
    xlabel='PC1', ylabel='PC3', zlabel='PC4',
)

# ---------------------------------------------------------------------------
# Plot 5: All-ages scatter in pca_save space (turbo age coloring)
# ---------------------------------------------------------------------------
parquet_files = sorted(glob.glob(PARQUET_AGE_GLOB))
if not parquet_files:
    raise FileNotFoundError(f'No parquet files found matching {PARQUET_AGE_GLOB}')

df_all = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
age_counts = df_all['age'].value_counts()
qualifying_ages = {a for a, c in age_counts.items() if a.startswith('P') and c > AGE_MIN_CELLS}
df_all = df_all[df_all['age'].isin(qualifying_ages)].reset_index(drop=True)
print(f'Total cells (postnatal, >{AGE_MIN_CELLS} cells/age): {len(df_all)}')

pc_cols = [c for c in df_all.columns if c.startswith('PC')]
xp_all  = df_all[pc_cols].values
xp_all  = np.delete(xp_all, 1, axis=1)  # drop PC2 (index 1)

scatter_categorical_html(
    xp_grid=[xp_all],
    cell_metadata={
        'age':           df_all['age'].values,
        'cluster_label': df_all['cluster_label'].values,
        'archetype':     df_all['archetype'].fillna('other').values,
    },
    title='All ages projected onto P56 PCA — NOC=4 (no PC2)',
    out_path=FIG_ALL_AGES_SCATTER,
    ordered_labels=('age',),
    xlabel='PC1', ylabel='PC3', zlabel='PC4',
)

# ---------------------------------------------------------------------------
# Plot 6: Per-age gene scatter (pca_save space, PC2 dropped)
# ---------------------------------------------------------------------------
# Archetype centroids in pca_save space (drop PC2, take first 3 dims)
df_arch_save = pd.read_parquet(PARQUET_P56_ARCHETYPES)
pc_cols_save = [c for c in df_arch_save.columns if c.startswith('PC')]
aa_save = np.delete(df_arch_save[pc_cols_save].values, 1, axis=1)[:, :3].T  # (3, NOC)

# Filter adata to qualifying ages (mirrors script 13's keep mask for cell-order consistency)
ages_adata = adata.obs['Age'].values
age_counts_adata = pd.Series(ages_adata).value_counts()
qualifying_ages_set = {a for a, c in age_counts_adata.items()
                       if a.startswith('P') and c > AGE_MIN_CELLS}
adata_qual = adata[np.array([a in qualifying_ages_set for a in ages_adata])]

for age_val in GENE_SCATTER_AGES:
    df_age   = pd.read_parquet(PARQUET_AGE_TEMPLATE.format(age=age_val))
    xp_age   = np.delete(df_age[pc_cols_save].values, 1, axis=1)  # drop PC2

    adata_age  = adata_qual[adata_qual.obs['Age'] == age_val]
    x_age      = adata_age.X.toarray() if sp.issparse(adata_age.X) else np.array(adata_age.X)
    depths_age = x_age.sum(axis=1)
    xn_age     = norm(x_age[:, hvg_mask], depths_age)

    gene_vals_age = {gene_names[i]: xn_age[:, i] for i in ordered_idx}

    gene_expr_scatter_html(
        x=xp_age[:, 0], y=xp_age[:, 1], z=xp_age[:, 2],
        gene_vals=gene_vals_age,
        aa=aa_save,
        title=f'{age_val} — archetype-specific gene expression (P56 PCA, no PC2)',
        out_path=FIG_GENE_SCATTER_AGE_TEMPLATE.format(age=age_val),
        xlabel='PC1', ylabel='PC3', zlabel='PC4',
    )
