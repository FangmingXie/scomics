# First-pass analysis of jainlab26_cux2cre + jainlab26_wt (v2).
# Integrates both targets with yoo25_P21 reference via three-batch Harmony + kNN,
# transfers Subclass labels, saves labeled h5ads, and writes UMAP HTML figures.

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

INPUT_CUX2CRE_H5  = os.path.join(PROJECT_ROOT, 'links', 'cux2cre', 'filtered_feature_bc_matrix_jainlab26_cux2cre.h5')
INPUT_WT_H5       = os.path.join(PROJECT_ROOT, 'links', 'cux2cre', 'filtered_feature_bc_matrix_jainlab26_wt.h5')
INPUT_REF_H5AD    = os.path.join(PROJECT_ROOT, 'links', 'cux2cre', 'yoo25_P21.h5ad')
OUT_CUX2CRE_H5AD  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_cux2cre_labeled.h5ad')
OUT_WT_H5AD       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_wt_labeled.h5ad')
OUT_FIG_SOURCE    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '00_v2.umap_datasource.html')
OUT_FIG_SUBCLASS  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '00_v2.umap_subclass.html')

N_HVG       = 3000
N_PCS       = 30
N_NEIGHBORS = 15
MIN_GENES   = 200
MIN_COUNTS  = 500

os.makedirs(os.path.dirname(OUT_CUX2CRE_H5AD), exist_ok=True)
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


def _qc_filter(adata, source_label):
    x = _to_array(adata.X)
    adata.obs['n_genes_by_counts'] = (x > 0).sum(axis=1)
    adata.obs['total_counts']      = x.sum(axis=1)
    n_before = adata.n_obs
    adata = adata[(adata.obs['n_genes_by_counts'] >= MIN_GENES) &
                  (adata.obs['total_counts'] >= MIN_COUNTS)].copy()
    print(f'  Cells after QC: {adata.n_obs} (removed {n_before - adata.n_obs})')
    adata.obs['source'] = source_label
    return adata


# --- Load and QC targets ---
print('Loading target (jainlab26_cux2cre)...')
cux2cre = _read_10x_h5(INPUT_CUX2CRE_H5)
cux2cre.var_names_make_unique()
cux2cre = _qc_filter(cux2cre, 'jainlab26_cux2cre')

print('Loading target (jainlab26_wt)...')
wt = _read_10x_h5(INPUT_WT_H5)
wt.var_names_make_unique()
wt = _qc_filter(wt, 'jainlab26_wt')

# --- Load reference ---
print('Loading reference (yoo25_P21)...')
ref = ad.read_h5ad(INPUT_REF_H5AD)
ref.obs['source'] = 'yoo25'
print(f'  Reference: {ref.n_obs} cells, {ref.n_vars} genes')
print(f'  Subclass counts:\n{ref.obs["Subclass"].value_counts()}')

# --- Shared genes and HVG selection ---
shared_genes = np.intersect1d(np.intersect1d(ref.var_names, cux2cre.var_names), wt.var_names)
print(f'Shared genes: {len(shared_genes)}')
ref_sub     = ref[:, shared_genes]
cux2cre_sub = cux2cre[:, shared_genes]
wt_sub      = wt[:, shared_genes]

x_ref     = _to_array(ref_sub.X)
x_cux2cre = _to_array(cux2cre_sub.X)
x_wt      = _to_array(wt_sub.X)
depths_ref     = x_ref.sum(axis=1)
depths_cux2cre = x_cux2cre.sum(axis=1)
depths_wt      = x_wt.sum(axis=1)

hvg_mask = select_hvg(x_ref, depths_ref, N_HVG)
print(f'HVGs selected: {hvg_mask.sum()}')

# --- Normalize (each dataset independently) ---
xn_ref     = norm(x_ref[:, hvg_mask],     depths_ref)
xn_cux2cre = norm(x_cux2cre[:, hvg_mask], depths_cux2cre)
xn_wt      = norm(x_wt[:, hvg_mask],      depths_wt)
xn_ref     = np.nan_to_num(xn_ref,     nan=0.0)
xn_cux2cre = np.nan_to_num(xn_cux2cre, nan=0.0)
xn_wt      = np.nan_to_num(xn_wt,      nan=0.0)

# --- PCA: fit on reference, project all ---
print('Running PCA...')
pca = PCA(n_components=N_PCS, random_state=0)
pca.fit(xn_ref)
xp_ref_raw     = pca.transform(xn_ref)
xp_cux2cre_raw = pca.transform(xn_cux2cre)
xp_wt_raw      = pca.transform(xn_wt)
print(f'  ref: {xp_ref_raw.shape}, cux2cre: {xp_cux2cre_raw.shape}, wt: {xp_wt_raw.shape}')

# --- Harmony batch correction (three batches) ---
print('Running Harmony...')
n_ref, n_cux2cre, n_wt = len(xp_ref_raw), len(xp_cux2cre_raw), len(xp_wt_raw)
xp_all_raw = np.vstack([xp_ref_raw, xp_cux2cre_raw, xp_wt_raw])
meta_df    = pd.DataFrame({'source': ['yoo25'] * n_ref + ['jainlab26_cux2cre'] * n_cux2cre + ['jainlab26_wt'] * n_wt})
ho         = hm.run_harmony(xp_all_raw, meta_df, 'source', random_state=42)
xp_all_h   = ho.Z_corr
xp_ref_h     = xp_all_h[:n_ref]
xp_cux2cre_h = xp_all_h[n_ref:n_ref + n_cux2cre]
xp_wt_h      = xp_all_h[n_ref + n_cux2cre:]
print(f'  Harmony done — ref: {xp_ref_h.shape}, cux2cre: {xp_cux2cre_h.shape}, wt: {xp_wt_h.shape}')

# --- kNN label transfer in Harmony space ---
print('Running kNN label transfer...')
subclass_labels = ref.obs['Subclass'].values
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric='euclidean')
knn.fit(xp_ref_h, subclass_labels)

pred_cux2cre      = knn.predict(xp_cux2cre_h)
proba_cux2cre     = knn.predict_proba(xp_cux2cre_h)
maxprob_cux2cre   = proba_cux2cre.max(axis=1)

pred_wt           = knn.predict(xp_wt_h)
proba_wt          = knn.predict_proba(xp_wt_h)
maxprob_wt        = proba_wt.max(axis=1)

print(f'  cux2cre Subclass:\n{pd.Series(pred_cux2cre).value_counts()}')
print(f'  wt Subclass:\n{pd.Series(pred_wt).value_counts()}')

# --- UMAP on all cells (Harmony space) ---
print(f'Running UMAP on {n_ref + n_cux2cre + n_wt} cells...')
reducer     = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
umap_coords = reducer.fit_transform(xp_all_h)
umap_ref     = umap_coords[:n_ref]
umap_cux2cre = umap_coords[n_ref:n_ref + n_cux2cre]
umap_wt      = umap_coords[n_ref + n_cux2cre:]
print('  UMAP done.')

# --- Save labeled h5ads ---
cux2cre.obs['Subclass_transferred'] = pred_cux2cre
cux2cre.obs['Subclass_max_prob']    = maxprob_cux2cre
cux2cre.obsm['X_harmony'] = xp_cux2cre_h
cux2cre.obsm['X_umap']    = umap_cux2cre
cux2cre.write_h5ad(OUT_CUX2CRE_H5AD)
print(f'Saved cux2cre h5ad → {OUT_CUX2CRE_H5AD}')

wt.obs['Subclass_transferred'] = pred_wt
wt.obs['Subclass_max_prob']    = maxprob_wt
wt.obsm['X_harmony'] = xp_wt_h
wt.obsm['X_umap']    = umap_wt
wt.write_h5ad(OUT_WT_H5AD)
print(f'Saved wt h5ad → {OUT_WT_H5AD}')

# --- Visualize: UMAP colored by data source ---
umap_all     = np.vstack([umap_ref, umap_cux2cre, umap_wt])
source_all   = np.array(['yoo25'] * n_ref + ['jainlab26_cux2cre'] * n_cux2cre + ['jainlab26_wt'] * n_wt)
subclass_all = np.concatenate([ref.obs['Subclass'].values, pred_cux2cre, pred_wt])

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
