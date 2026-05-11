"""Direct projection of mouse L2/3 IT (Yoo25) into human VX archetype space.

Instead of the cosine-similarity neighbor search (script 12) — which collapses
the mouse manifold onto ~73 hub human cells — this script projects mouse cells
directly into the human VX/PCHA coordinate space using the human PCA gene
loadings restricted to shared ortholog genes.

Pipeline:
  1. Recover combined rotation + reorder matrix R_total from saved PCA and VX
     loadings (lstsq: L_pca @ R_total = L_vx).
  2. Find all HVGs with 1-to-1 mouse orthologs present in the mouse dataset.
  3. For those shared genes:
       - Compute human mean/std from h5ad (= StandardScaler stats used in script 01)
       - Scale mouse log-normalized expression using those stats
       - Partial PCA projection: X_mouse_scaled @ L_sub → approximate PC coords
       - Apply R_total → VX coords; select VX_COLS → 6D VX subspace
  4. Refit human SCA on VX_COLS (same as script 09) to get the PCHA PCA transform;
     apply proj_transform to mouse VX coords → mouse cells in PCHA space.
  5. Visualize human (downsampled) + mouse overlaid with archetype simplex.
  6. Hub diagnostic: compare manifold spread vs script 12.

Reads:
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
  links/l23_evo/yoo25_mouse_IT_P21.h5ad
  local_data/res/l23_evo/01.pca_loadings.tsv
  local_data/res/l23_evo/05.varimax_loadings.tsv
  local_data/res/l23_evo/05.varimax_coords.tsv
  data/human_mouse_orthologs.tsv
  local_data/res/l23_evo/09.pcha_aa.tsv
Output:
  local_data/res/l23_evo/13.mouse_projected_coords.tsv
  local_data/fig/l23_evo/13.mouse_projected_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from viz import scatter_categorical_html

from scomics.main import SCA
from scomics.utils import proj_transform

# --- file paths ---
OUT_RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
INPUT_HUMAN      = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
INPUT_MOUSE      = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'yoo25_mouse_IT_P21.h5ad')
IN_PCA_LOADINGS  = os.path.join(OUT_RES_DIR, '01.pca_loadings.tsv')
IN_VX_LOADINGS   = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
IN_VX_COORDS     = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
IN_ORTHOLOGS     = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_AA            = os.path.join(OUT_RES_DIR, '09.pcha_aa.tsv')
OUT_MOUSE_COORDS = os.path.join(OUT_RES_DIR, '13.mouse_projected_coords.tsv')
OUT_HTML         = os.path.join(OUT_FIG_DIR, '13.mouse_projected_scatter.html')

# --- parameters ---
VX_COLS        = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
NOC            = 4
NDIM           = 5
MOUSE_SUBCLASS = 'L2/3'
N_DOWNSAMPLE   = 5000
CLUSTER_COL    = 'WithinArea_cluster'

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- 1. Recover combined rotation+reorder matrix R_total ---
# L_pca (genes × PCs) and L_vx (genes × VX, reordered) are both saved.
# L_pca @ R_total = L_vx  →  R_total = lstsq(L_pca, L_vx)
print('Recovering varimax rotation matrix...')
L_pca_df = pd.read_csv(IN_PCA_LOADINGS, sep='\t', index_col=0)   # (2000, 10)
L_vx_df  = pd.read_csv(IN_VX_LOADINGS,  sep='\t', index_col=0)   # (2000, 10)
hvg_genes = L_pca_df.index.values   # ordered list of 2000 HVGs

L_pca = L_pca_df.values.astype(np.float64)
L_vx  = L_vx_df.values.astype(np.float64)
R_total, residuals, rank, _ = np.linalg.lstsq(L_pca, L_vx, rcond=None)   # (10, 10)
reconstruction_err = np.max(np.abs(L_pca @ R_total - L_vx))
print(f'  R_total shape: {R_total.shape},  max reconstruction error: {reconstruction_err:.2e}')

# column indices for VX_COLS (VX2 → col 1, VX6 → col 5, etc.)
vx_all_cols = L_vx_df.columns.tolist()   # ['VX1', ..., 'VX10']
VX_COLS_IDX = [vx_all_cols.index(c) for c in VX_COLS]

# --- 2. Find shared HVGs with 1-to-1 mouse orthologs ---
print('Finding shared HVGs with mouse orthologs...')
ortho  = pd.read_csv(IN_ORTHOLOGS, sep='\t')
ortho  = ortho.drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
h2m    = ortho.set_index('human_symbol')['mouse_symbol'].to_dict()

m_adata      = ad.read_h5ad(INPUT_MOUSE)
m_adata      = m_adata[m_adata.obs['Subclass'] == MOUSE_SUBCLASS]
m_gene_names = m_adata.var_names.values
m_gene_set   = set(m_gene_names)

# use ALL shared HVGs (not just top-50-per-VX) for a better projection
shared = [(g, h2m[g], i) for i, g in enumerate(hvg_genes)
          if g in h2m and h2m[g] in m_gene_set]
human_shared  = [t[0] for t in shared]
mouse_shared  = [t[1] for t in shared]
hvg_shared_idx = [t[2] for t in shared]   # positions in the 2000-HVG list
print(f'  {len(shared)} shared HVGs ({len(hvg_genes)} total HVGs)')

L_sub = L_pca[hvg_shared_idx, :]   # (n_shared, 10) — PCA loadings for shared genes

# --- 3. Compute human gene mean/std for scaling (= StandardScaler stats from script 01) ---
print('Loading human expression for scaling stats...')
h_adata      = ad.read_h5ad(INPUT_HUMAN)
h_gene_names = (h_adata.var['feature_name'].values
                if 'feature_name' in h_adata.var.columns
                else h_adata.var_names.values)
h_gene_map   = {g: i for i, g in enumerate(h_gene_names)}
h_shared_idx = np.array([h_gene_map[g] for g in human_shared])

print(f'  Extracting {len(human_shared)} shared gene columns from human h5ad...')
X_human_shared = h_adata.X[:, h_shared_idx].toarray().astype(np.float64)
human_mean = X_human_shared.mean(axis=0)   # (n_shared,)
human_std  = X_human_shared.std(axis=0)    # (n_shared,), ddof=0 — matches StandardScaler
human_std[human_std == 0] = 1
types = h_adata.obs[CLUSTER_COL].values
print(f'  X_human_shared shape: {X_human_shared.shape}')
del X_human_shared   # free memory

# --- 4. Normalize and scale mouse expression, project into VX space ---
print('Projecting mouse cells into VX space...')
m_shared_idx = np.array([np.where(m_gene_names == g)[0][0] for g in mouse_shared])
X_mouse_raw  = m_adata.X[:, m_shared_idx].toarray().astype(np.float64)
depths       = X_mouse_raw.sum(axis=1, keepdims=True); depths[depths == 0] = 1
X_mouse_log  = np.log2(X_mouse_raw / depths * 1e4 + 1)
X_mouse_scaled = (X_mouse_log - human_mean) / human_std   # center + scale like StandardScaler
del X_mouse_raw, X_mouse_log

# partial PCA projection (using n_shared / 2000 genes)
mouse_pc  = X_mouse_scaled @ L_sub        # (n_mouse, 10) — approximate PC coords
mouse_vx  = mouse_pc @ R_total            # (n_mouse, 10) — VX coords, correct ordering
mouse_vx6 = mouse_vx[:, VX_COLS_IDX]     # (n_mouse, 6)  — select VX_COLS subspace
print(f'  mouse_vx6 shape: {mouse_vx6.shape}')

# --- 5. Refit human SCA on VX_COLS to get PCHA transform ---
print('Fitting human SCA (refitting PCHA transform)...')
vx_df = pd.read_csv(IN_VX_COORDS, sep='\t', index_col=0)
xn    = vx_df[VX_COLS].values.astype(np.float64)

sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)
print(f'  Human xp shape: {sca.xp.shape}, aa shape: {sca.aa.shape}')

# project mouse VX coords using the same transform (no re-fitting)
mouse_xp = proj_transform(mouse_vx6, sca.pca_)   # (n_mouse, NDIM)
print(f'  mouse_xp shape: {mouse_xp.shape}')

# --- 6. Hub diagnostic: spread of mouse cells in PCHA space ---
print('\n--- Hub diagnostic ---')
print(f'Human PCHA space range (PC1-PC{NDIM}):')
for i in range(NDIM):
    print(f'  PC{i+1}: human [{sca.xp[:,i].min():.2f}, {sca.xp[:,i].max():.2f}]  '
          f'mouse [{mouse_xp[:,i].min():.2f}, {mouse_xp[:,i].max():.2f}]')
# pairwise spread: std of coordinates
print(f'Mouse coord std per dim: {mouse_xp.std(axis=0).round(3)}')
print(f'Human coord std per dim: {sca.xp.std(axis=0).round(3)}')
print(f'Mouse/human std ratio:   {(mouse_xp.std(axis=0)/sca.xp.std(axis=0)).round(3)}')

# --- 7. Save mouse coords ---
ndim = sca.xp.shape[1]
mouse_coords_df = pd.DataFrame(mouse_xp, index=m_adata.obs_names,
                                columns=[f'PC{i+1}' for i in range(ndim)])
mouse_coords_df['Subclass'] = m_adata.obs['Subclass'].values
if 'Type_leiden' in m_adata.obs.columns:
    mouse_coords_df['Type_leiden'] = m_adata.obs['Type_leiden'].values
mouse_coords_df.to_csv(OUT_MOUSE_COORDS, sep='\t')
print(f'\nSaved {OUT_MOUSE_COORDS}  ({len(mouse_coords_df)} rows)')

# --- 8. Visualize ---
print('Generating visualization...')
rng  = np.random.default_rng(0)
hidx = rng.choice(sca.xp.shape[0], min(N_DOWNSAMPLE, sca.xp.shape[0]), replace=False)

xp_combined = np.vstack([sca.xp[hidx], mouse_xp])
species_col = np.array(['human'] * len(hidx) + ['mouse'] * mouse_xp.shape[0])
cell_metadata = {'species': species_col}
if 'Type_leiden' in m_adata.obs.columns:
    cell_metadata['Type_leiden'] = np.concatenate([
        types[hidx], m_adata.obs['Type_leiden'].values
    ])

aa = pd.read_csv(IN_AA, sep='\t', index_col=0).values.T.astype(np.float32)  # (NDIM, NOC)

scatter_categorical_html(
    xp_grid=[xp_combined],
    cell_metadata=cell_metadata,
    title='Jorstad23 (human) + Yoo25 L2/3 (mouse) — direct VX projection',
    out_path=OUT_HTML,
    panels=[(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')],
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    aa_grid=[aa],
)
print(f'Saved {OUT_HTML}')
print('Done.')
