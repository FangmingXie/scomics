"""Mouse L4 IT (cheng22, P28NR) PCA coords — no varimax (compute).

Direct-PCA front-end (matches the yoo25 set, scripts 03/05/07/09): load → normalize
→ HVG → PCA(10), stopping at PCA and persisting the raw PC scores. Feeds the
"skip VX selection — use top PCs directly" archetype sweep in
14.cheng22_L4_num_archetype.py.

Normalization: cheng22 .X is log1p(CP10k), so we re-normalize the integer raw
counts in .raw with log2(CP10k + 1) to stay faithful to the Cheng22 procedure.

Reads:
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
Outputs:
  local_data/res/it/13.cheng22_L4_pca_coords.tsv
  local_data/res/it/13.cheng22_L4_pca_loadings.tsv
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

# --- file paths ---
OUT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
INPUT_MOUSE     = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
OUT_PC_COORDS   = os.path.join(OUT_RES_DIR, '13.cheng22_L4_pca_coords.tsv')
OUT_PC_LOADINGS = os.path.join(OUT_RES_DIR, '13.cheng22_L4_pca_loadings.tsv')

# --- parameters ---
MOUSE_SUBCLASS = 'L4'
CLUSTER_COL    = 'Type'
SAMPLE_COL     = 'Sample'
N_HVG          = 2000
N_PCS          = 10

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- 1. Load and filter to L4 ---
print('Loading mouse cheng22 data...')
m_adata = ad.read_h5ad(INPUT_MOUSE)
m_adata = m_adata[m_adata.obs['Subclass'] == MOUSE_SUBCLASS]
print(f'  {m_adata.n_obs} cells, {m_adata.n_vars} genes')

types   = m_adata.obs[CLUSTER_COL].values
samples = m_adata.obs[SAMPLE_COL].values

# --- 2. Normalize: raw counts → log2(CP10k + 1) ---
print('Normalizing...')
X_raw = m_adata.raw[:, m_adata.var_names].X.toarray().astype(np.float32)
depths = X_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_norm = np.log2(X_raw / depths * 1e4 + 1)

# --- 3. HVG selection ---
print(f'Selecting top {N_HVG} HVGs...')
gene_var = X_norm.var(axis=0)
hvg_idx  = np.argsort(gene_var)[::-1][:N_HVG]
gene_names = m_adata.var_names.values[hvg_idx]
X_hvg = X_norm[:, hvg_idx]

# --- 4. Scale + PCA ---
print('Scaling and PCA...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hvg)

pca = PCA(n_components=N_PCS, random_state=0)
scores = pca.fit_transform(X_scaled)   # (n_cells, N_PCS)
L      = pca.components_.T             # (N_HVG, N_PCS) gene loadings

pc_cols = [f'PC{i+1}' for i in range(N_PCS)]

# --- 5. Save PC coords ---
pc_df = pd.DataFrame(scores, index=m_adata.obs_names, columns=pc_cols)
pc_df[CLUSTER_COL] = types
pc_df[SAMPLE_COL]  = samples
pc_df.to_csv(OUT_PC_COORDS, sep='\t')
print(f'Saved {OUT_PC_COORDS}')

# --- 6. Save PC loadings ---
pc_load_df = pd.DataFrame(L, index=gene_names, columns=pc_cols)
pc_load_df.to_csv(OUT_PC_LOADINGS, sep='\t')
print(f'Saved {OUT_PC_LOADINGS}')

print('Done.')
