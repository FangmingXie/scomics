import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap

warnings.filterwarnings('ignore')

# --- file paths ---
SCRIPTS_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../scripts/
PROJECT_ROOT  = os.path.dirname(SCRIPTS_DIR)                                   # .../SingleCellArchetype/
INPUT_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')
OUT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_STANDARD  = os.path.join(OUT_RES_DIR, '01.standard_results.parquet')
OUT_P56_PROJ  = os.path.join(OUT_RES_DIR, '01.p56_projection_results.parquet')

# --- config ---
MIN_CELLS_PER_AGE = 100
N_HVG             = 2000
N_PCS             = 50
N_NEIGHBORS       = 15
N_CLUSTERS        = 10
P56_AGE_VAL       = 'P56'
METADATA_COLS     = ['Age', 'timepoint', 'age_bin', 'cluster_label', 'subcluster_label', 'roi', 'sex']

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- load ---
print(f'Loading {INPUT_FILE}')
adata = ad.read_h5ad(INPUT_FILE)
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes')

# --- filter to ages with >MIN_CELLS_PER_AGE cells ---
age_counts = adata.obs['Age'].value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS_PER_AGE].index
adata = adata[adata.obs['Age'].isin(valid_ages)].copy()
print(f'  After filtering (>{MIN_CELLS_PER_AGE} cells/age): {adata.shape[0]} cells, {len(valid_ages)} ages')
print(f'  Ages retained: {sorted(valid_ages.tolist())}')

# --- normalize: CP10k + log1p (shared) ---
X = adata.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)
depths = X.sum(axis=1, keepdims=True)
depths = np.where(depths == 0, 1, depths)
X_norm = np.log1p(X / depths * 1e4)
print(f'  Normalized: CP10k + log1p')

# =============================================================================
# Part A: Standard analysis (all filtered cells)
# =============================================================================
print('\n--- Part A: Standard analysis ---')

# HVG selection by variance
gene_var_a = X_norm.var(axis=0)
hvg_idx_a  = np.argsort(gene_var_a)[::-1][:N_HVG]
X_hvg_a    = X_norm[:, hvg_idx_a]
print(f'  HVGs selected: {N_HVG}')

# scale
scaler_a   = StandardScaler()
X_scaled_a = scaler_a.fit_transform(X_hvg_a)

# PCA
print(f'  Running PCA (n_components={N_PCS})...')
pca_a   = PCA(n_components=N_PCS, random_state=0)
X_pca_a = pca_a.fit_transform(X_scaled_a)

# UMAP
print(f'  Running UMAP (n_neighbors={N_NEIGHBORS})...')
reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, n_components=2, random_state=0)
X_umap  = reducer.fit_transform(X_pca_a)

# KMeans clustering
print(f'  Running KMeans (k={N_CLUSTERS})...')
km             = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init='auto')
kmeans_labels  = km.fit_predict(X_pca_a)

# save
pc_cols   = [f'PC{i+1}' for i in range(N_PCS)]
out_a     = pd.DataFrame(X_pca_a, index=adata.obs_names, columns=pc_cols)
out_a.index.name = 'cell_id'
out_a.insert(0, 'UMAP2', X_umap[:, 1])
out_a.insert(0, 'UMAP1', X_umap[:, 0])
out_a['kmeans_cluster'] = kmeans_labels
for col in METADATA_COLS:
    out_a[col] = adata.obs[col].values
out_a.to_parquet(OUT_STANDARD)
print(f'  Saved -> {OUT_STANDARD}')

# =============================================================================
# Part B: P56-anchored projection (all filtered cells projected to P56 PCA space)
# =============================================================================
print('\n--- Part B: P56 projection ---')

p56_mask = (adata.obs['Age'] == P56_AGE_VAL).values
print(f'  P56 reference cells: {p56_mask.sum()}')

# HVG selection from P56 cells only
gene_var_b = X_norm[p56_mask, :].var(axis=0)
hvg_idx_b  = np.argsort(gene_var_b)[::-1][:N_HVG]
print(f'  HVGs selected from P56: {N_HVG}')

# fit scaler and PCA on P56 cells
scaler_b     = StandardScaler()
X_p56_scaled = scaler_b.fit_transform(X_norm[p56_mask, :][:, hvg_idx_b])

print(f'  Fitting PCA on P56 cells (n_components={N_PCS})...')
pca_b     = PCA(n_components=N_PCS, random_state=0)
pca_b.fit(X_p56_scaled)

# project all filtered cells
X_all_scaled = scaler_b.transform(X_norm[:, hvg_idx_b])
X_all_pca_b  = pca_b.transform(X_all_scaled)

# save (top 5 PCs only)
out_b = pd.DataFrame(X_all_pca_b[:, :5], index=adata.obs_names, columns=pc_cols[:5])
out_b.index.name = 'cell_id'
for col in METADATA_COLS:
    out_b[col] = adata.obs[col].values
out_b.to_parquet(OUT_P56_PROJ)
print(f'  Saved -> {OUT_P56_PROJ}')

print('\nDone.')
