# L2/3-only Harmony integration and label transfer for jainlab26_cux2cre + jainlab26_wt (v2).
# Repeats script 00_v2 pipeline restricted to L2/3 neurons from both targets + reference.
# Transfers fine-grained Type labels (L2/3_A/B/C) from yoo25_P21 reference.

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
from viz import scatter_2d_categorical_html

from scomics.utils import norm

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

INPUT_CUX2CRE_H5AD  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_H5AD        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_wt_labeled.h5ad')
INPUT_REF_H5AD       = os.path.join(PROJECT_ROOT, 'links', 'cux2cre', 'yoo25_P21.h5ad')
OUT_CUX2CRE_H5AD     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_jainlab26_cux2cre_labeled.h5ad')
OUT_WT_H5AD          = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_jainlab26_wt_labeled.h5ad')
OUT_HARMONY_COORDS   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_harmony_coords.tsv')
OUT_FIG_SOURCE       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '04_v2.l23_umap_datasource.html')
OUT_FIG_TYPE         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '04_v2.l23_umap_type.html')

N_HVG       = 3000
N_PCS       = 30
N_NEIGHBORS = 15
SUBCLASS    = 'L2/3'

os.makedirs(os.path.dirname(OUT_CUX2CRE_H5AD), exist_ok=True)
os.makedirs(os.path.dirname(OUT_FIG_SOURCE), exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


# --- Load and filter targets to L2/3 ---
print('Loading target (jainlab26_cux2cre L2/3)...')
cux2cre = ad.read_h5ad(INPUT_CUX2CRE_H5AD)
cux2cre = cux2cre[cux2cre.obs['Subclass_transferred'] == SUBCLASS].copy()
cux2cre.obs['source'] = 'jainlab26_cux2cre'
print(f'  {cux2cre.n_obs} L2/3 cells')

print('Loading target (jainlab26_wt L2/3)...')
wt = ad.read_h5ad(INPUT_WT_H5AD)
wt = wt[wt.obs['Subclass_transferred'] == SUBCLASS].copy()
wt.obs['source'] = 'jainlab26_wt'
print(f'  {wt.n_obs} L2/3 cells')

# --- Load and filter reference to L2/3 ---
print('Loading reference (yoo25_P21 L2/3)...')
ref = ad.read_h5ad(INPUT_REF_H5AD)
ref = ref[ref.obs['Subclass'] == SUBCLASS].copy()
ref.obs['source'] = 'yoo25'
print(f'  {ref.n_obs} L2/3 cells')
print(f'  Type counts:\n{ref.obs["Type"].value_counts()}')

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

# --- kNN label transfer (Type) in Harmony space ---
print('Running kNN label transfer...')
type_labels = ref.obs['Type'].values
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric='euclidean')
knn.fit(xp_ref_h, type_labels)

pred_cux2cre    = knn.predict(xp_cux2cre_h)
proba_cux2cre   = knn.predict_proba(xp_cux2cre_h)
maxprob_cux2cre = proba_cux2cre.max(axis=1)

pred_wt         = knn.predict(xp_wt_h)
proba_wt        = knn.predict_proba(xp_wt_h)
maxprob_wt      = proba_wt.max(axis=1)

print(f'  cux2cre Type:\n{pd.Series(pred_cux2cre).value_counts()}')
print(f'  wt Type:\n{pd.Series(pred_wt).value_counts()}')

# --- UMAP on all cells (Harmony space) ---
print(f'Running UMAP on {n_ref + n_cux2cre + n_wt} cells...')
reducer     = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
umap_coords = reducer.fit_transform(xp_all_h)
umap_ref     = umap_coords[:n_ref]
umap_cux2cre = umap_coords[n_ref:n_ref + n_cux2cre]
umap_wt      = umap_coords[n_ref + n_cux2cre:]
print('  UMAP done.')

# --- Save combined harmony coords (ref + both targets) ---
pc_cols      = [f'PC{i+1}' for i in range(N_PCS)]
all_barcodes = np.concatenate([ref.obs_names, cux2cre.obs_names, wt.obs_names])
harmony_df   = pd.DataFrame(xp_all_h, index=all_barcodes, columns=pc_cols)
harmony_df['source'] = np.array(['yoo25'] * n_ref + ['jainlab26_cux2cre'] * n_cux2cre + ['jainlab26_wt'] * n_wt)
harmony_df['Type']   = np.concatenate([ref.obs['Type'].values, pred_cux2cre, pred_wt])
harmony_df.index.name = 'cell'
harmony_df.to_csv(OUT_HARMONY_COORDS, sep='\t')
print(f'Saved harmony coords → {OUT_HARMONY_COORDS}')

# --- Save labeled target h5ads ---
cux2cre.obs['Type_transferred'] = pred_cux2cre
cux2cre.obs['Type_max_prob']    = maxprob_cux2cre
cux2cre.obsm['X_harmony'] = xp_cux2cre_h
cux2cre.obsm['X_umap']    = umap_cux2cre
cux2cre.write_h5ad(OUT_CUX2CRE_H5AD)
print(f'Saved cux2cre h5ad → {OUT_CUX2CRE_H5AD}')

wt.obs['Type_transferred'] = pred_wt
wt.obs['Type_max_prob']    = maxprob_wt
wt.obsm['X_harmony'] = xp_wt_h
wt.obsm['X_umap']    = umap_wt
wt.write_h5ad(OUT_WT_H5AD)
print(f'Saved wt h5ad → {OUT_WT_H5AD}')

# --- Visualize UMAPs ---
umap_all   = np.vstack([umap_ref, umap_cux2cre, umap_wt])
source_all = np.array(['yoo25'] * n_ref + ['jainlab26_cux2cre'] * n_cux2cre + ['jainlab26_wt'] * n_wt)
type_all   = np.concatenate([ref.obs['Type'].values, pred_cux2cre, pred_wt])

scatter_2d_categorical_html(
    xp_grid=[umap_all],
    cell_metadata={'source': source_all},
    title='UMAP — data source (L2/3 only)',
    out_path=OUT_FIG_SOURCE,
    xlabel='UMAP1', ylabel='UMAP2',
)
print(f'Saved source UMAP → {OUT_FIG_SOURCE}')

scatter_2d_categorical_html(
    xp_grid=[umap_all],
    cell_metadata={'Type': type_all},
    title='UMAP — Type (yoo25 ground truth + jainlab26 transferred, L2/3 only)',
    out_path=OUT_FIG_TYPE,
    xlabel='UMAP1', ylabel='UMAP2',
)
print(f'Saved type UMAP → {OUT_FIG_TYPE}')
print('Done.')
