# First-pass analysis of jainlab26_cux2cre (21k cells, 10x h5).
# Transfers Subclass labels from yoo25_P21 reference via Harmony + kNN,
# saves labeled h5ad, and writes UMAP HTML figures colored by source and subclass.

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import h5py
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import harmonypy as hm
import umap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg
from viz import scatter_2d_categorical_html

from scomics.utils import norm

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

INPUT_TARGET_H5  = os.path.join(PROJECT_ROOT, 'links', 'cux2cre', 'filtered_feature_bc_matrix_jainlab26_cux2cre.h5')
INPUT_REF_H5AD   = os.path.join(PROJECT_ROOT, 'links', 'cux2cre', 'yoo25_P21.h5ad')
OUT_H5AD         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00.jainlab26_cux2cre_labeled.h5ad')
OUT_FIG_SOURCE   = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '00.umap_datasource.html')
OUT_FIG_SUBCLASS = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '00.umap_subclass.html')

N_HVG       = 3000
N_PCS       = 30
N_NEIGHBORS = 15
MIN_GENES   = 200
MIN_COUNTS  = 500

os.makedirs(os.path.dirname(OUT_H5AD), exist_ok=True)
os.makedirs(os.path.dirname(OUT_FIG_SOURCE), exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


def _read_10x_h5(path):
    """Read a 10x Genomics HDF5 file into AnnData."""
    with h5py.File(path, 'r') as f:
        m = f['matrix']
        data    = m['data'][:]
        indices = m['indices'][:]
        indptr  = m['indptr'][:]
        shape   = m['shape'][:]  # (n_genes, n_cells)
        barcodes = m['barcodes'][:].astype(str)
        gene_names = m['features']['name'][:].astype(str)
    X = sp.csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))
    obs = pd.DataFrame(index=barcodes)
    obs.index.name = None
    var = pd.DataFrame(index=gene_names)
    var.index.name = None
    return ad.AnnData(X=X, obs=obs, var=var)


# --- Load target ---
print('Loading target (jainlab26_cux2cre)...')
target = _read_10x_h5(INPUT_TARGET_H5)
target.var_names_make_unique()
x_qc = _to_array(target.X)
target.obs['n_genes_by_counts'] = (x_qc > 0).sum(axis=1)
target.obs['total_counts']      = x_qc.sum(axis=1)
n_before = target.n_obs
target = target[(target.obs['n_genes_by_counts'] >= MIN_GENES) &
                (target.obs['total_counts'] >= MIN_COUNTS)].copy()
print(f'  Cells after QC: {target.n_obs} (removed {n_before - target.n_obs})')
target.obs['source'] = 'jainlab26'

# --- Load reference ---
print('Loading reference (yoo25_P21)...')
ref = ad.read_h5ad(INPUT_REF_H5AD)
ref.obs['source'] = 'yoo25'
print(f'  Reference: {ref.n_obs} cells, {ref.n_vars} genes')
print(f'  Subclass counts:\n{ref.obs["Subclass"].value_counts()}')

# --- Shared genes and HVG selection ---
shared_genes = np.intersect1d(target.var_names, ref.var_names)
print(f'Shared genes: {len(shared_genes)}')
ref_sub    = ref[:, shared_genes]
target_sub = target[:, shared_genes]

x_ref    = _to_array(ref_sub.X)
x_target = _to_array(target_sub.X)
depths_ref    = x_ref.sum(axis=1)
depths_target = x_target.sum(axis=1)

hvg_mask = select_hvg(x_ref, depths_ref, N_HVG)
print(f'HVGs selected: {hvg_mask.sum()}')

# --- Normalize (each dataset independently) ---
# norm() z-scores per gene; zero-variance genes produce NaN → set to 0
xn_ref    = norm(x_ref[:, hvg_mask],    depths_ref)
xn_target = norm(x_target[:, hvg_mask], depths_target)
xn_ref    = np.nan_to_num(xn_ref,    nan=0.0)
xn_target = np.nan_to_num(xn_target, nan=0.0)

# --- PCA: fit on reference, project both ---
print('Running PCA...')
pca = PCA(n_components=N_PCS, random_state=0)
pca.fit(xn_ref)
xp_ref_raw    = pca.transform(xn_ref)
xp_target_raw = pca.transform(xn_target)
print(f'  ref: {xp_ref_raw.shape}, target: {xp_target_raw.shape}')

# --- Harmony batch correction ---
print('Running Harmony...')
n_ref_cells, n_target_cells = len(xp_ref_raw), len(xp_target_raw)
xp_all_raw   = np.vstack([xp_ref_raw, xp_target_raw])
batch_labels = ['yoo25'] * n_ref_cells + ['jainlab26'] * n_target_cells
meta_df      = pd.DataFrame({'source': batch_labels})
ho           = hm.run_harmony(xp_all_raw, meta_df, 'source', random_state=42)
xp_all_h     = ho.Z_corr  # harmonypy 2.x: (ncells, ndim)
xp_ref_h     = xp_all_h[:n_ref_cells]
xp_target_h  = xp_all_h[n_ref_cells:]
print(f'  Harmony done — ref: {xp_ref_h.shape}, target: {xp_target_h.shape}')

# --- kNN label transfer in Harmony space ---
print('Running kNN label transfer...')
subclass_labels = ref.obs['Subclass'].values
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric='euclidean')
knn.fit(xp_ref_h, subclass_labels)
pred_subclass = knn.predict(xp_target_h)
pred_proba    = knn.predict_proba(xp_target_h)
pred_max_prob = pred_proba.max(axis=1)
print(f'  Transferred subclass distribution:\n{pd.Series(pred_subclass).value_counts()}')

# --- UMAP on all cells (Harmony space) ---
print(f'Running UMAP on {n_ref_cells + n_target_cells} cells...')
reducer     = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
umap_coords = reducer.fit_transform(xp_all_h)
umap_ref    = umap_coords[:n_ref_cells]
umap_target = umap_coords[n_ref_cells:]
print('  UMAP done.')

# --- Save labeled target h5ad ---
target.obs['Subclass_transferred'] = pred_subclass
target.obs['Subclass_max_prob']    = pred_max_prob
target.obsm['X_harmony'] = xp_target_h
target.obsm['X_umap']    = umap_target
target.write_h5ad(OUT_H5AD)
print(f'Saved labeled h5ad → {OUT_H5AD}')

# --- Visualize: UMAP colored by data source ---
umap_all  = np.vstack([umap_ref, umap_target])
source_all = np.array(['yoo25'] * n_ref_cells + ['jainlab26'] * n_target_cells)
subclass_all = np.concatenate([
    ref.obs['Subclass'].values,
    pred_subclass,
])

scatter_2d_categorical_html(
    xp_grid=[umap_all],
    cell_metadata={'source': source_all},
    title='UMAP — data source',
    out_path=OUT_FIG_SOURCE,
    xlabel='UMAP1', ylabel='UMAP2',
)
print(f'Saved source UMAP → {OUT_FIG_SOURCE}')

scatter_2d_categorical_html(
    xp_grid=[umap_all],
    cell_metadata={'Subclass': subclass_all},
    title='UMAP — Subclass (yoo25 ground truth + jainlab26 transferred)',
    out_path=OUT_FIG_SUBCLASS,
    xlabel='UMAP1', ylabel='UMAP2',
)
print(f'Saved subclass UMAP → {OUT_FIG_SUBCLASS}')
print('Done.')
