# Archetype analysis — P56 only (no PC2, NOC=4); project all ages onto P56 PCA.
# Saves top-10 PCA coordinates + metadata as parquet files per age.
# HVGs selected from P56; all cells normalized jointly.
# Groups/replicates: donor_name

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg, run_noc_sweep

from scomics.main import SCA
from scomics.utils import norm

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')
RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')

PARQUET_AGE_TEMPLATE      = os.path.join(RES_DIR, '13.pca_projected_{age}.parquet')
PARQUET_P56_ARCHETYPES    = os.path.join(RES_DIR, '13.p56_archetypes.parquet')
PARQUET_METRICS           = os.path.join(RES_DIR, '13.metrics.parquet')
PARQUET_P56_ARCH_PCA      = os.path.join(RES_DIR, '13.p56_archetype_pca.parquet')
PARQUET_P56_ARCH_VERTS    = os.path.join(RES_DIR, '13.p56_archetype_vertices.parquet')
PARQUET_P56_DONOR_REPS    = os.path.join(RES_DIR, '13.p56_archetype_donor_reps.parquet')

P56_AGE_VAL       = 'P56'
NDIM              = 5
NOC               = 4
NREPEATS          = 10
N_TOP_GENES       = 2000
DROP_PCS          = [1]
N_ARCHETYPE_CELLS = 300
N_PCS_SAVE        = 10
AGE_MIN_CELLS     = 50

# Load all cells
adata = ad.read_h5ad(INPUT_FILE)
print(f'Total cells: {adata.shape[0]}')

x      = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
depths = x.sum(axis=1)
types  = adata.obs['cluster_label'].values
donors = adata.obs['donor_name'].values
ages   = adata.obs['Age'].values

# Filter to postnatal ages with at least AGE_MIN_CELLS cells
age_counts = pd.Series(ages).value_counts()
qualifying_ages = {a for a, c in age_counts.items() if a.startswith('P') and c > AGE_MIN_CELLS}
keep = np.array([a in qualifying_ages for a in ages])
x, depths, types, donors, ages = x[keep], depths[keep], types[keep], donors[keep], ages[keep]
print(f'Cells after filtering (postnatal, >{AGE_MIN_CELLS} cells/age): {keep.sum()}')

# Select HVGs from P56 only (same gene set as script 12)
p56_mask = ages == P56_AGE_VAL
print(f'P56 cells: {p56_mask.sum()}')
hvg_mask = select_hvg(x[p56_mask], depths[p56_mask], N_TOP_GENES)

# Normalize all cells jointly using P56-derived HVG columns
xn = norm(x[:, hvg_mask], depths)
xn_p56 = xn[p56_mask]

# Archetype analysis on P56 only
sca = SCA(xn_p56, types[p56_mask])
sca.setup_feature_matrix(method='data')

os.makedirs(RES_DIR, exist_ok=True)
noc_grid = np.array([NOC])

ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, donors[p56_mask], drop_pcs=DROP_PCS)

# Archetype assignment on P56 (in the 5D archetype PCA space)
xp = xp_grid[0]   # (n_p56, NDIM)
aa = aa_grid[0]   # (NDIM, NOC)
dists_all = np.stack([np.linalg.norm(xp - aa[:, k], axis=1) for k in range(NOC)], axis=1)
arch_assign = np.argmin(dists_all, axis=1)
arch_labels_p56 = np.array([f'Arch{k+1}' for k in arch_assign])

# Fit a separate PCA on P56 for saving (N_PCS_SAVE raw components, no drop)
pca_save = PCA(n_components=N_PCS_SAVE)
pca_save.fit(xn_p56)

pc_cols = [f'PC{i+1}' for i in range(N_PCS_SAVE)]

# Project and save each age
for age_val in np.unique(ages):
    age_mask = ages == age_val
    xp_age = pca_save.transform(xn[age_mask])

    df = pd.DataFrame(xp_age, columns=pc_cols)
    df['cluster_label'] = types[age_mask]
    df['donor_name']    = donors[age_mask]
    df['age']           = age_val
    if age_val == P56_AGE_VAL:
        df['archetype'] = arch_labels_p56
    else:
        df['archetype'] = pd.NA

    out_path = PARQUET_AGE_TEMPLATE.format(age=age_val)
    df.to_parquet(out_path, index=False)
    print(f'  Saved {out_path}  ({age_mask.sum()} cells)')

# Save archetype centroids in pca_save space
arch_rows = []
for k in range(NOC):
    closest  = np.argsort(dists_all[:, k])[:N_ARCHETYPE_CELLS]
    centroid = pca_save.transform(xn_p56[closest]).mean(axis=0)
    row = {pc_cols[i]: centroid[i] for i in range(N_PCS_SAVE)}
    row['archetype'] = f'Arch{k+1}'
    arch_rows.append(row)

pd.DataFrame(arch_rows).to_parquet(PARQUET_P56_ARCHETYPES, index=False)
print(f'  Saved {PARQUET_P56_ARCHETYPES}')

# Save archetype-space data for script 14
adim_cols = [f'APC{i+1}' for i in range(NDIM)]

pd.DataFrame({'noc': noc_grid, 'ev': ev_grid, 'av': av_grid, 'av_rep': av_rep_grid}).to_parquet(
    PARQUET_METRICS, index=False)
print(f'  Saved {PARQUET_METRICS}')

df_p56_arch = pd.DataFrame(xp, columns=adim_cols)
df_p56_arch['cluster_label'] = types[p56_mask]
df_p56_arch['donor_name']    = donors[p56_mask]
df_p56_arch['archetype']     = arch_labels_p56
df_p56_arch.to_parquet(PARQUET_P56_ARCH_PCA, index=False)
print(f'  Saved {PARQUET_P56_ARCH_PCA}')

df_aa = pd.DataFrame(aa.T, columns=adim_cols)
df_aa['archetype'] = [f'Arch{k+1}' for k in range(NOC)]
df_aa.to_parquet(PARQUET_P56_ARCH_VERTS, index=False)
print(f'  Saved {PARQUET_P56_ARCH_VERTS}')

rows = []
for donor, aa_d in aa_reps_grid[0]:
    for k in range(NOC):
        row = {adim_cols[i]: aa_d[i, k] for i in range(NDIM)}
        row['donor'] = donor
        row['archetype_id'] = k
        rows.append(row)
pd.DataFrame(rows).to_parquet(PARQUET_P56_DONOR_REPS, index=False)
print(f'  Saved {PARQUET_P56_DONOR_REPS}')
