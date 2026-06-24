"""Harmony archetype embedding — cheng22 mouse L4 IT (compute).

New archetype-inference front-end (prototype on cheng22 L4):
  1) z-score normalization (CP10k → log2(1+x) → z-score per gene) on top-2000 HVGs
  2) regress out library size (log depth) in expression space
  3) PCA (10 comps)
  4) Harmony batch correction across samples (Sample)
Persists the Harmony-corrected PC coords; the top 5 are fed directly to PCHA
(no varimax, no VX selection) in 28.cheng22_L4_harmony_num_archetype.py.

Mirrors the normalization + regression + Harmony steps of
scripts/astro/41.embed_cheng22_nr_harmony_arch12.py.

Reads:
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
Outputs:
  local_data/res/it/27.cheng22_L4_harmony_coords.tsv     (H1..H10 + Type + Sample)
  local_data/res/it/27.cheng22_L4_harmony_pca_coords.tsv (raw PCA, pre-Harmony)
  local_data/res/it/27.cheng22_L4_harmony_loadings.tsv   (PCA gene loadings)
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import harmonypy as hm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg
from scomics.utils import norm

# --- file paths ---
OUT_RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
INPUT_MOUSE       = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
OUT_HARMONY_COORDS = os.path.join(OUT_RES_DIR, '27.cheng22_L4_harmony_coords.tsv')
OUT_PCA_COORDS     = os.path.join(OUT_RES_DIR, '27.cheng22_L4_harmony_pca_coords.tsv')
OUT_PCA_LOADINGS   = os.path.join(OUT_RES_DIR, '27.cheng22_L4_harmony_loadings.tsv')

# --- parameters ---
MOUSE_SUBCLASS = 'L4'
CLUSTER_COL    = 'Type'
SAMPLE_COL     = 'Sample'
N_HVG          = 2000
N_PCS          = 10

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- 1. Load and filter to L4 ---
print('Loading mouse cheng22 data...')
adata = ad.read_h5ad(INPUT_MOUSE)
# remove mitochondrial genes (mirror astro/41)
mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
print(f'  Removed {mt_mask.sum()} mito genes; {(~mt_mask).sum()} remaining')
adata = adata[adata.obs['Subclass'] == MOUSE_SUBCLASS].copy()
print(f'  {adata.n_obs} L4 cells, {adata.n_vars} genes')

types   = adata.obs[CLUSTER_COL].values
samples = adata.obs[SAMPLE_COL].values

# raw counts from .raw (cheng22 .X is log1p; .raw holds integer counts)
x = adata.raw[:, adata.var_names].X
x = x.toarray() if sp.issparse(x) else np.asarray(x, dtype=np.float64)
depths = x.sum(axis=1)

# --- 2. HVG selection ---
print(f'Selecting top {N_HVG} HVGs...')
hvg_mask = select_hvg(x, depths, N_HVG)
gene_names = adata.var_names.values[hvg_mask]

# --- 3. normalize (CP10k → log2(1+x) → z-score per gene) ---
xn = norm(x[:, hvg_mask], depths)

# --- 4. regress out library size (log depth) ---
print('Regressing out library size...')
log_depth = np.log(depths).reshape(-1, 1)
reg = LinearRegression().fit(log_depth, xn)
xn = xn - reg.predict(log_depth)

# --- 5. PCA ---
print(f'Fitting PCA (N_PCS={N_PCS})...')
pca = PCA(N_PCS, random_state=0)
pca_scores = pca.fit_transform(xn)   # (n, N_PCS)
L = pca.components_.T                 # (n_hvg, N_PCS)

# --- 6. Harmony batch correction across samples ---
print('Running Harmony batch correction (by Sample)...')
meta_df = pd.DataFrame({SAMPLE_COL: samples})
ho = hm.run_harmony(pca_scores, meta_df, SAMPLE_COL, random_state=0)
Z = np.asarray(ho.Z_corr)
# harmonypy returns Z_corr as (d, N); orient to (n_cells, N_PCS)
harmony_scores = Z.T if Z.shape[0] == N_PCS else Z
assert harmony_scores.shape == (adata.n_obs, N_PCS), \
    f'unexpected Harmony shape {harmony_scores.shape}, expected {(adata.n_obs, N_PCS)}'
print(f'  Harmony corrected embedding: {harmony_scores.shape}')

# --- 7. Save coords ---
h_cols = [f'H{i+1}' for i in range(N_PCS)]
h_df = pd.DataFrame(harmony_scores, index=adata.obs_names, columns=h_cols)
h_df[CLUSTER_COL] = types
h_df[SAMPLE_COL]  = samples
h_df.to_csv(OUT_HARMONY_COORDS, sep='\t')
print(f'Saved {OUT_HARMONY_COORDS}')

pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
pc_df = pd.DataFrame(pca_scores, index=adata.obs_names, columns=pc_cols)
pc_df[CLUSTER_COL] = types
pc_df[SAMPLE_COL]  = samples
pc_df.to_csv(OUT_PCA_COORDS, sep='\t')
print(f'Saved {OUT_PCA_COORDS}')

pd.DataFrame(L, index=gene_names, columns=pc_cols).to_csv(OUT_PCA_LOADINGS, sep='\t')
print(f'Saved {OUT_PCA_LOADINGS}')

print('Done.')
