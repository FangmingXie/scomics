# Sequential kNN archetype label transfer: P56 → all younger postnatal ages.
# Fits joint PCA (N_DIMS_KNN=20) on P56, projects all postnatal ages.
# P56 archetype labels assigned via PCHA; relayed to younger ages via kNN majority vote.

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg, run_noc_sweep

from scomics.main import SCA
from scomics.utils import norm

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE            = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')
RES_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
PARQUET_AGE_TEMPLATE  = os.path.join(RES_DIR, '15.labels_{age}.parquet')
PARQUET_ALL           = os.path.join(RES_DIR, '15.labels_all_ages.parquet')

NDIM        = 5
NOC         = 4
NREPEATS    = 10
N_TOP_GENES = 2000
DROP_PCS    = [1]
N_DIMS_KNN  = 20
K_NEIGHBORS = 15

P56_AGE_VAL = 'P56'

os.makedirs(RES_DIR, exist_ok=True)

# Load all cells
adata = ad.read_h5ad(INPUT_FILE)
print(f'Total cells: {adata.shape[0]}')

x      = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
depths = x.sum(axis=1)
donors = adata.obs['donor_name'].values
ages   = adata.obs['Age'].values

# Keep only postnatal ages (no min-cell filter)
keep = np.array([a.startswith('P') for a in ages])
x, depths, donors, ages = x[keep], depths[keep], donors[keep], ages[keep]
print(f'Postnatal cells: {keep.sum()}')

# HVG selection from P56 only
p56_mask = ages == P56_AGE_VAL
print(f'P56 cells: {p56_mask.sum()}')
hvg_mask = select_hvg(x[p56_mask], depths[p56_mask], N_TOP_GENES)

# Joint normalization on HVG columns for all cells
xn = norm(x[:, hvg_mask], depths)
xn_p56 = xn[p56_mask]

# Fit joint PCA on P56 cells; project all postnatal cells
pca_joint = PCA(n_components=N_DIMS_KNN)
pca_joint.fit(xn_p56)
xp_all = pca_joint.transform(xn)
print(f'Joint PCA fitted on P56; projected all postnatal cells → shape {xp_all.shape}')

pc_cols = [f'PC{i+1}' for i in range(N_DIMS_KNN)]

# P56 archetype assignment via PCHA
sca = SCA(xn_p56, donors[p56_mask])
sca.setup_feature_matrix(method='data')
noc_grid = np.array([NOC])

_, _, _, xp_grid, aa_grid, _ = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, donors[p56_mask], drop_pcs=DROP_PCS)

xp_pcha = xp_grid[0]   # (n_p56, NDIM)
aa      = aa_grid[0]   # (NDIM, NOC)
dists_p56 = np.stack([np.linalg.norm(xp_pcha - aa[:, k], axis=1) for k in range(NOC)], axis=1)
arch_assign_p56 = np.argmin(dists_p56, axis=1)
arch_labels_p56 = np.array([f'Arch{k+1}' for k in arch_assign_p56])
print(f'P56 archetype counts: {pd.Series(arch_labels_p56).value_counts().to_dict()}')

# Sort postnatal ages oldest → youngest by numeric value after 'P'
unique_ages = np.unique(ages)
age_order = sorted(unique_ages, key=lambda a: int(a[1:]), reverse=True)
print(f'Age order (oldest→youngest): {age_order}')

# Initialise label store: P56 cells already labeled
age_labels = {}
age_labels[P56_AGE_VAL] = arch_labels_p56

# Sequential kNN relay
for i in range(len(age_order) - 1):
    older_age   = age_order[i]
    younger_age = age_order[i + 1]

    older_mask   = ages == older_age
    younger_mask = ages == younger_age

    xp_older   = xp_all[older_mask]
    xp_younger = xp_all[younger_mask]
    labels_older = age_labels[older_age]

    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    knn.fit(xp_older, labels_older)
    labels_younger = knn.predict(xp_younger)

    age_labels[younger_age] = labels_younger
    print(f'  {older_age} → {younger_age}: {younger_mask.sum()} cells transferred; '
          f'counts={pd.Series(labels_younger).value_counts().to_dict()}')

# Collect per-age DataFrames and save
dfs = []
for age_val in age_order:
    age_mask = ages == age_val
    df = pd.DataFrame(xp_all[age_mask], columns=pc_cols)
    df['donor_name'] = donors[age_mask]
    df['age']        = age_val
    df['archetype']  = age_labels[age_val]

    out_path = PARQUET_AGE_TEMPLATE.format(age=age_val)
    df.to_parquet(out_path, index=False)
    print(f'  Saved {out_path}  ({age_mask.sum()} cells)')
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all.to_parquet(PARQUET_ALL, index=False)
print(f'Saved {PARQUET_ALL}  ({len(df_all)} cells total)')
print(f'Unique archetypes: {sorted(df_all["archetype"].unique())}')
