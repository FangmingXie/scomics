"""Map all postnatal ages into P56 varimax archetype space — gao25 astrocytes.

Applies the fitted P56 PCA → varimax → inner-PCA pipeline algebraically to all
postnatal ages simultaneously, then visualises the developmental trajectory.

Reads:
  links/astro/gao25_scrna_astro.h5ad
  local_data/res/astro/33.varimax_loadings.tsv   (HVG gene list)
  local_data/res/astro/33.pca_components.tsv
  local_data/res/astro/33.pca_mean.tsv
  local_data/res/astro/33.varimax_R.tsv
  local_data/res/astro/35.inner_pca_components.tsv
  local_data/res/astro/35.inner_pca_mean.tsv
  local_data/res/astro/35.pcha_xp.tsv
  local_data/res/astro/35.pcha_aa.tsv
  local_data/res/astro/17.labels_all_ages.parquet
Outputs:
  local_data/res/astro/36.all_ages_xp.tsv
  local_data/fig/astro/36.all_ages_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg
from viz import scatter_categorical_html
from scomics.utils import norm

# --- file paths ---
INPUT_H5AD        = os.path.join(PROJECT_ROOT, 'links', 'astro', 'gao25_scrna_astro.h5ad')
OUT_RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
IN_VX_LOADINGS    = os.path.join(OUT_RES_DIR, '33.varimax_loadings.tsv')
IN_PCA_COMPS      = os.path.join(OUT_RES_DIR, '33.pca_components.tsv')
IN_PCA_MEAN       = os.path.join(OUT_RES_DIR, '33.pca_mean.tsv')
IN_VARIMAX_R      = os.path.join(OUT_RES_DIR, '33.varimax_R.tsv')
IN_INNER_COMPS    = os.path.join(OUT_RES_DIR, '35.inner_pca_components.tsv')
IN_INNER_MEAN     = os.path.join(OUT_RES_DIR, '35.inner_pca_mean.tsv')
IN_XP_P56         = os.path.join(OUT_RES_DIR, '35.pcha_xp.tsv')
IN_AA             = os.path.join(OUT_RES_DIR, '35.pcha_aa.tsv')
IN_ARCH_LABELS    = os.path.join(OUT_RES_DIR, '17.labels_all_ages.parquet')
OUT_ALL_AGES_TSV  = os.path.join(OUT_RES_DIR, '36.all_ages_xp.tsv')
OUT_ALL_AGES_HTML = os.path.join(OUT_FIG_DIR, '36.all_ages_scatter.html')

# must match scripts 33, 34, 35
VX_COLS   = ['VX1', 'VX2', 'VX3', 'VX5', 'VX6']
MIN_CELLS = 50

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load h5ad; apply same postnatal filter as script 33 ---
print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
print(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

ages = adata.obs['Age'].values
postnatal_mask = np.array([a.startswith('P') for a in ages])
adata = adata[postnatal_mask].copy()
ages  = adata.obs['Age'].values

age_counts = pd.Series(ages).value_counts()
valid_ages  = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
ages   = adata.obs['Age'].values
donors = adata.obs['donor_name'].values
print(f'  Postnatal cells after filter: {adata.shape[0]}')

x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
depths = x.sum(axis=1)

# --- HVG gene list from script 33 loadings ---
load_df    = pd.read_csv(IN_VX_LOADINGS, sep='\t', index_col=0)
hvg_genes  = load_df.index.values
gene_names = adata.var_names.values
hvg_idx    = np.where(np.isin(gene_names, hvg_genes))[0]
assert len(hvg_idx) == len(hvg_genes), (
    f'HVG gene count mismatch: expected {len(hvg_genes)}, found {len(hvg_idx)} in h5ad'
)

# reorder hvg_idx to match the order in hvg_genes (loadings index order)
gene_to_idx = {g: i for i, g in enumerate(gene_names)}
hvg_idx_ordered = np.array([gene_to_idx[g] for g in hvg_genes])

# --- normalize all postnatal cells jointly (identical to script 33) ---
xn = norm(x[:, hvg_idx_ordered], depths)
print(f'  xn shape: {xn.shape}')

# --- load PCA/varimax parameters ---
pca_comps  = pd.read_csv(IN_PCA_COMPS, sep='\t', index_col=0).values.astype(np.float64)  # (N_PCS, n_hvg)
pca_mean   = pd.read_csv(IN_PCA_MEAN,  sep='\t', index_col=0).values.astype(np.float64).squeeze()  # (n_hvg,)
varimax_R  = pd.read_csv(IN_VARIMAX_R, sep='\t', index_col=0).values.astype(np.float64)  # (N_PCS, N_PCS)
inner_comps = pd.read_csv(IN_INNER_COMPS, sep='\t', index_col=0).values.astype(np.float64)  # (NDIM, n_vx_cols)
inner_mean  = pd.read_csv(IN_INNER_MEAN,  sep='\t', index_col=0).values.astype(np.float64).squeeze()  # (n_vx_cols,)

vx_col_idx = [int(c[2:]) - 1 for c in VX_COLS]   # VX2→1, VX3→2, etc.
NDIM = inner_comps.shape[0]

# --- 4-step algebraic projection ---
print('Projecting all postnatal cells into P56 archetype space...')
pca_scores = (xn - pca_mean) @ pca_comps.T        # (n, N_PCS)
vx_all     = pca_scores @ varimax_R                # (n, N_PCS) — already reordered
vx_sub     = vx_all[:, vx_col_idx]                # (n, n_vx_cols)
xp_all     = (vx_sub - inner_mean) @ inner_comps.T # (n, NDIM)
print(f'  xp_all shape: {xp_all.shape}')

# --- join Arch1-4 labels from parquet (positional per age group) ---
labels_df  = pd.read_parquet(IN_ARCH_LABELS)
arch_col   = np.empty(len(adata), dtype=object)

for age_val in np.unique(ages):
    age_mask_adata  = ages == age_val
    labels_age      = labels_df[labels_df['age'] == age_val].reset_index(drop=True)
    assert len(labels_age) == age_mask_adata.sum(), (
        f'Label count mismatch for {age_val}: parquet {len(labels_age)} vs adata {age_mask_adata.sum()}'
    )
    arch_col[age_mask_adata] = labels_age['archetype'].values

# --- save TSV ---
pc_cols  = [f'PC{i+1}' for i in range(NDIM)]
out_df   = pd.DataFrame(xp_all, index=adata.obs_names, columns=pc_cols)
out_df['age']        = ages
out_df['archetype']  = arch_col
out_df['donor_name'] = donors
out_df.to_csv(OUT_ALL_AGES_TSV, sep='\t')
print(f'Saved {OUT_ALL_AGES_TSV}')

# --- visualization ---
aa = pd.read_csv(IN_AA, sep='\t', index_col=0).values.T.astype(np.float64)  # (NDIM, NOC)
NOC = aa.shape[1]

scatter_categorical_html(
    xp_grid=[xp_all],
    cell_metadata={'age': ages, 'archetype': arch_col},
    title='gao25 astrocytes — all postnatal ages in P56 archetype space',
    out_path=OUT_ALL_AGES_HTML,
    noc_grid=np.array([NOC]),
    aa_grid=[aa],
)
print(f'Saved {OUT_ALL_AGES_HTML}')
print('Done.')
