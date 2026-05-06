# Cross-dataset astrocyte integration: gao25 P56 (reference) → cheng22 P28/P38 + yoo25 P21.
# Finds common genes across all three datasets, selects HVG from gao25 P56 cells,
# fits PCA on P56 (10 components), projects all datasets, applies harmonypy batch correction,
# transfers archetype labels via kNN in harmony space, computes joint UMAP,
# and saves combined parquet + archetype centroid vertices.

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import harmonypy as hm
import umap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg

from scomics.utils import norm

SCRIPTS_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT  = os.path.dirname(SCRIPTS_DIR)
INPUT_GAO25   = os.path.join(PROJECT_ROOT, 'links', 'astro', 'gao25_scrna_astro.h5ad')
INPUT_CHENG22 = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
INPUT_YOO25   = os.path.join(PROJECT_ROOT, 'links', 'astro', 'yoo25_astro.h5ad')
LABELS_P56_IN = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '17.labels_P56.parquet')
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
PARQUET_COMBINED = os.path.join(RES_DIR, '22.combined_labels.parquet')
PARQUET_VERTICES = os.path.join(RES_DIR, '22.archetype_vertices.parquet')

N_TOP_GENES  = 2000
NDIM_PCA     = 10
K_NEIGHBORS  = 15
# cheng22: P28/P38 NR (no suffix) and DR (_dr suffix); P28_dl excluded
CHENG22_AGES = ['P28', 'P28_dr', 'P38', 'P38_dr']
# yoo25: P21 NR and P21DR (no underscore before DR)
YOO25_AGES   = ['P21', 'P21DR']

os.makedirs(RES_DIR, exist_ok=True)

pc_cols = [f'PC{i+1}' for i in range(NDIM_PCA)]

# --- Load gao25 P56 reference ---
print('Loading gao25 P56 cells...')
adata_gao25 = ad.read_h5ad(INPUT_GAO25)
adata_p56   = adata_gao25[adata_gao25.obs['Age'] == 'P56'].copy()
print(f'  gao25 P56 cells: {adata_p56.shape[0]}')

df_labels = pd.read_parquet(LABELS_P56_IN)
if len(df_labels) != adata_p56.shape[0]:
    raise ValueError(
        f'P56 label count {len(df_labels)} does not match adata_p56 cell count {adata_p56.shape[0]}'
    )
arch_labels_p56 = df_labels['archetype'].values
print(f'  Archetype counts: {pd.Series(arch_labels_p56).value_counts().to_dict()}')

# --- Load cheng22 ---
print('Loading cheng22...')
adata_cheng22_full = ad.read_h5ad(INPUT_CHENG22)
missing = [a for a in CHENG22_AGES if a not in adata_cheng22_full.obs['Age'].values]
if missing:
    raise ValueError(f'Ages not found in cheng22: {missing}')
adata_cheng22 = adata_cheng22_full[adata_cheng22_full.obs['Age'].isin(CHENG22_AGES)].copy()
print(f'  cheng22 cells: {adata_cheng22.shape[0]}  ages: {sorted(adata_cheng22.obs["Age"].unique())}')

# --- Load yoo25 ---
print('Loading yoo25...')
adata_yoo25_full = ad.read_h5ad(INPUT_YOO25)
missing = [a for a in YOO25_AGES if a not in adata_yoo25_full.obs['Age'].values]
if missing:
    raise ValueError(f'Ages not found in yoo25: {missing}')
adata_yoo25 = adata_yoo25_full[adata_yoo25_full.obs['Age'].isin(YOO25_AGES)].copy()
print(f'  yoo25 cells: {adata_yoo25.shape[0]}  ages: {sorted(adata_yoo25.obs["Age"].unique())}')

# --- Common genes ---
common_genes = sorted(
    set(adata_gao25.var_names) & set(adata_cheng22.var_names) & set(adata_yoo25.var_names)
)
print(f'Common genes: {len(common_genes)}')

adata_p56     = adata_p56[:, common_genes]
adata_cheng22 = adata_cheng22[:, common_genes]
adata_yoo25   = adata_yoo25[:, common_genes]

# --- Raw count matrices ---
def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)

x_p56     = _to_array(adata_p56.X)
x_cheng22 = _to_array(adata_cheng22.X)
x_yoo25   = _to_array(adata_yoo25.X)

depths_p56     = x_p56.sum(axis=1)
depths_cheng22 = x_cheng22.sum(axis=1)
depths_yoo25   = x_yoo25.sum(axis=1)

# --- HVG selection on gao25 P56 (common genes) ---
hvg_mask = select_hvg(x_p56, depths_p56, N_TOP_GENES)
print(f'HVG selected: {hvg_mask.sum()} genes')

# --- Normalize (each dataset independently) ---
xn_p56     = norm(x_p56[:, hvg_mask],     depths_p56)
xn_cheng22 = norm(x_cheng22[:, hvg_mask], depths_cheng22)
xn_yoo25   = norm(x_yoo25[:, hvg_mask],   depths_yoo25)

# --- PCA: fit on P56, project all ---
pca = PCA(n_components=NDIM_PCA)
pca.fit(xn_p56)
xp_p56_raw     = pca.transform(xn_p56)
xp_cheng22_raw = pca.transform(xn_cheng22)
xp_yoo25_raw   = pca.transform(xn_yoo25)
print(f'PCA done — P56: {xp_p56_raw.shape}, cheng22: {xp_cheng22_raw.shape}, yoo25: {xp_yoo25_raw.shape}')

# --- Harmony batch correction ---
n_p56, n_cheng22, n_yoo25 = len(xp_p56_raw), len(xp_cheng22_raw), len(xp_yoo25_raw)
xp_all_raw   = np.vstack([xp_p56_raw, xp_cheng22_raw, xp_yoo25_raw])
batch_labels = ['gao25'] * n_p56 + ['cheng22'] * n_cheng22 + ['yoo25'] * n_yoo25
meta_df      = pd.DataFrame({'batch': batch_labels})
ho           = hm.run_harmony(xp_all_raw, meta_df, 'batch', random_state=42)
xp_all_h     = ho.Z_corr      # harmonypy 2.x returns (ncells, ndim) directly
xp_p56       = xp_all_h[:n_p56]
xp_cheng22   = xp_all_h[n_p56 : n_p56 + n_cheng22]
xp_yoo25     = xp_all_h[n_p56 + n_cheng22:]
print(f'Harmony done — P56: {xp_p56.shape}, cheng22: {xp_cheng22.shape}, yoo25: {xp_yoo25.shape}')

# --- kNN label transfer (in harmony-corrected space) ---
knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
knn.fit(xp_p56, arch_labels_p56)
labels_cheng22 = knn.predict(xp_cheng22)
labels_yoo25   = knn.predict(xp_yoo25)
print(f'cheng22 transferred: {pd.Series(labels_cheng22).value_counts().to_dict()}')
print(f'yoo25   transferred: {pd.Series(labels_yoo25).value_counts().to_dict()}')

# --- Joint UMAP (on harmony-corrected embeddings) ---
print(f'Running UMAP on {xp_all_h.shape[0]} cells...')
reducer = umap.UMAP(n_components=2, random_state=42)
umap_coords = reducer.fit_transform(xp_all_h)
umap_p56     = umap_coords[:n_p56]
umap_cheng22 = umap_coords[n_p56:n_p56 + n_cheng22]
umap_yoo25   = umap_coords[n_p56 + n_cheng22:]
print('UMAP done.')

# --- Archetype centroid vertices (in PCA space from P56 cells) ---
arch_names = sorted(np.unique(arch_labels_p56))
centroids  = np.stack([xp_p56[arch_labels_p56 == a].mean(axis=0) for a in arch_names])
df_vertices = pd.DataFrame(centroids, index=arch_names, columns=pc_cols)
df_vertices.index.name = 'archetype'
df_vertices.to_parquet(PARQUET_VERTICES)
print(f'Saved archetype vertices → {PARQUET_VERTICES}')

# --- Build combined DataFrame ---
def _make_df(xp, umap_xy, labels, ages, donors, dataset, cell_types=None):
    df = pd.DataFrame(xp, columns=pc_cols)
    df['UMAP1']      = umap_xy[:, 0]
    df['UMAP2']      = umap_xy[:, 1]
    df['dataset']    = dataset
    df['age']        = ages
    df['donor_name'] = donors
    df['cell_type']  = cell_types if cell_types is not None else ''
    df['archetype']  = labels
    return df

df_p56 = _make_df(
    xp_p56, umap_p56, arch_labels_p56,
    ages      = adata_p56.obs['Age'].values,
    donors    = adata_p56.obs['donor_name'].values,
    dataset   = 'gao25',
)
df_cheng22 = _make_df(
    xp_cheng22, umap_cheng22, labels_cheng22,
    ages       = adata_cheng22.obs['Age'].values,
    donors     = adata_cheng22.obs['Sample'].values,
    dataset    = 'cheng22',
    cell_types = adata_cheng22.obs['Type'].values,
)
df_yoo25 = _make_df(
    xp_yoo25, umap_yoo25, labels_yoo25,
    ages       = adata_yoo25.obs['Age'].values,
    donors     = adata_yoo25.obs['Sample'].values,
    dataset    = 'yoo25',
    cell_types = adata_yoo25.obs['Type'].values,
)

df_combined = pd.concat([df_p56, df_cheng22, df_yoo25], ignore_index=True)
df_combined.to_parquet(PARQUET_COMBINED, index=False)
print(f'Saved combined parquet → {PARQUET_COMBINED}')
print(f'  Shape: {df_combined.shape}')
print(f'  Columns: {df_combined.columns.tolist()}')
print(f'  Dataset counts: {df_combined["dataset"].value_counts().to_dict()}')
