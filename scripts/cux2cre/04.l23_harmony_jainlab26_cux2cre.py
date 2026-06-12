# L2/3-only Harmony integration and label transfer for jainlab26_cux2cre.
# Repeats script 00 pipeline restricted to L2/3 neurons from both datasets.
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

INPUT_TARGET_H5AD = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00.jainlab26_cux2cre_labeled.h5ad')
INPUT_REF_H5AD    = os.path.join(PROJECT_ROOT, 'links', 'cux2cre', 'yoo25_P21.h5ad')
OUT_H5AD          = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04.l23_jainlab26_labeled.h5ad')
OUT_HARMONY_COORDS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04.l23_harmony_coords.tsv')
OUT_FIG_SOURCE    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '04.l23_umap_datasource.html')
OUT_FIG_TYPE      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '04.l23_umap_type.html')

N_HVG       = 3000
N_PCS       = 30
N_NEIGHBORS = 15
SUBCLASS    = 'L2/3'

os.makedirs(os.path.dirname(OUT_H5AD), exist_ok=True)
os.makedirs(os.path.dirname(OUT_FIG_SOURCE), exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


# --- Load and filter target to L2/3 ---
print('Loading target (jainlab26 L2/3)...')
target = ad.read_h5ad(INPUT_TARGET_H5AD)
target = target[target.obs['Subclass_transferred'] == SUBCLASS].copy()
target.obs['source'] = 'jainlab26'
print(f'  {target.n_obs} L2/3 cells')

# --- Load and filter reference to L2/3 ---
print('Loading reference (yoo25_P21 L2/3)...')
ref = ad.read_h5ad(INPUT_REF_H5AD)
ref = ref[ref.obs['Subclass'] == SUBCLASS].copy()
ref.obs['source'] = 'yoo25'
print(f'  {ref.n_obs} L2/3 cells')
print(f'  Type counts:\n{ref.obs["Type"].value_counts()}')

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
meta_df      = pd.DataFrame({'source': ['yoo25'] * n_ref_cells + ['jainlab26'] * n_target_cells})
ho           = hm.run_harmony(xp_all_raw, meta_df, 'source', random_state=42)
xp_all_h     = ho.Z_corr  # harmonypy 2.x: (ncells, ndim)
xp_ref_h     = xp_all_h[:n_ref_cells]
xp_target_h  = xp_all_h[n_ref_cells:]
print(f'  Harmony done — ref: {xp_ref_h.shape}, target: {xp_target_h.shape}')

# --- kNN label transfer (Type) in Harmony space ---
print('Running kNN label transfer...')
type_labels = ref.obs['Type'].values
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric='euclidean')
knn.fit(xp_ref_h, type_labels)
pred_type     = knn.predict(xp_target_h)
pred_proba    = knn.predict_proba(xp_target_h)
pred_max_prob = pred_proba.max(axis=1)
print(f'  Transferred Type distribution:\n{pd.Series(pred_type).value_counts()}')

# --- UMAP on all cells (Harmony space) ---
print(f'Running UMAP on {n_ref_cells + n_target_cells} cells...')
reducer     = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
umap_coords = reducer.fit_transform(xp_all_h)
umap_ref    = umap_coords[:n_ref_cells]
umap_target = umap_coords[n_ref_cells:]
print('  UMAP done.')

# --- Save combined harmony coords (ref + target) ---
pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
all_barcodes = np.concatenate([ref.obs_names, target.obs_names])
harmony_df = pd.DataFrame(xp_all_h, index=all_barcodes, columns=pc_cols)
harmony_df['source'] = np.array(['yoo25'] * n_ref_cells + ['jainlab26'] * n_target_cells)
harmony_df['Type']   = np.concatenate([ref.obs['Type'].values, pred_type])
harmony_df.index.name = 'cell'
harmony_df.to_csv(OUT_HARMONY_COORDS, sep='\t')
print(f'Saved harmony coords → {OUT_HARMONY_COORDS}')

# --- Save labeled target h5ad ---
target.obs['Type_transferred'] = pred_type
target.obs['Type_max_prob']    = pred_max_prob
target.obsm['X_harmony'] = xp_target_h
target.obsm['X_umap']    = umap_target
target.write_h5ad(OUT_H5AD)
print(f'Saved labeled h5ad → {OUT_H5AD}')

# --- Visualize UMAPs ---
umap_all     = np.vstack([umap_ref, umap_target])
source_all   = np.array(['yoo25'] * n_ref_cells + ['jainlab26'] * n_target_cells)
type_all     = np.concatenate([ref.obs['Type'].values, pred_type])

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
