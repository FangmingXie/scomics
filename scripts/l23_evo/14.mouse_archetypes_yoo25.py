"""Mouse L2/3 IT (Yoo25, P21) varimax rotation + PCHA — within-mouse subtype variation.

Mirrors human pipeline (scripts 01 + 05 + 09) applied to mouse.
Produces mouse subtype-informative genes and archetype structure.

Reads:
  links/l23_evo/yoo25_mouse_IT_P21.h5ad
Outputs:
  local_data/res/l23_evo/14.mouse_varimax_coords.tsv
  local_data/res/l23_evo/14.mouse_varimax_loadings.tsv
  local_data/res/l23_evo/14.mouse_pcha_xp.tsv
  local_data/res/l23_evo/14.mouse_pcha_aa.tsv
  local_data/fig/l23_evo/14.mouse_varimax_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html
from scomics.main import SCA

# --- file paths ---
OUT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
INPUT_MOUSE     = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'yoo25_mouse_IT_P21.h5ad')
OUT_VX_COORDS   = os.path.join(OUT_RES_DIR, '14.mouse_varimax_coords.tsv')
OUT_VX_LOADINGS = os.path.join(OUT_RES_DIR, '14.mouse_varimax_loadings.tsv')
OUT_PCHA_XP     = os.path.join(OUT_RES_DIR, '14.mouse_pcha_xp.tsv')
OUT_PCHA_AA     = os.path.join(OUT_RES_DIR, '14.mouse_pcha_aa.tsv')
OUT_HTML        = os.path.join(OUT_FIG_DIR, '14.mouse_varimax_scatter.html')

# --- parameters ---
MOUSE_SUBCLASS = 'L2/3'
CLUSTER_COL    = 'Type_leiden'
N_HVG          = 2000
N_PCS          = 10
NOC            = 3
NDIM           = 4   # NOC+1 PCs fitted; last dropped → 4D PCHA space

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def varimax(L, gamma=1.0, max_iter=1000, tol=1e-6):
    """Kaiser varimax rotation of loading matrix L (n_vars × n_factors)."""
    n, p = L.shape
    R = np.eye(p)
    for _ in range(max_iter):
        R_old = R.copy()
        for i in range(p - 1):
            for j in range(i + 1, p):
                Lr = L @ R
                u = Lr[:, i] ** 2 - Lr[:, j] ** 2
                v = 2 * Lr[:, i] * Lr[:, j]
                A = u.sum()
                B = v.sum()
                C = (u ** 2 - v ** 2).sum()
                D = 2 * (u * v).sum()
                theta = 0.25 * np.arctan2(
                    D - gamma * 2 * A * B / n,
                    C - gamma * (A ** 2 - B ** 2) / n,
                )
                c, s = np.cos(theta), np.sin(theta)
                Rij = np.eye(p)
                Rij[i, i] = Rij[j, j] = c
                Rij[i, j] = -s
                Rij[j, i] = s
                R = R @ Rij
        if np.max(np.abs(R - R_old)) < tol:
            break
    return R


def r2(X, y):
    reg = LinearRegression().fit(X, y)
    ss_res = np.sum((y - reg.predict(X)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


# --- 1. Load and filter to L2/3 ---
print('Loading mouse data...')
m_adata = ad.read_h5ad(INPUT_MOUSE)
m_adata = m_adata[m_adata.obs['Subclass'] == MOUSE_SUBCLASS]
print(f'  {m_adata.n_obs} cells, {m_adata.n_vars} genes')

types = m_adata.obs[CLUSTER_COL].values

# --- 2. Normalize: raw counts → log2(CP10k + 1) ---
print('Normalizing...')
X_raw = m_adata.X.toarray().astype(np.float32)
depths = X_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_norm = np.log2(X_raw / depths * 1e4 + 1)

# --- 3. HVG selection ---
print(f'Selecting top {N_HVG} HVGs...')
gene_var = X_norm.var(axis=0)
hvg_idx = np.argsort(gene_var)[::-1][:N_HVG]
gene_names = m_adata.var_names.values[hvg_idx]
X_hvg = X_norm[:, hvg_idx]

# --- 4. Scale + PCA ---
print('Scaling and PCA...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hvg)

pca = PCA(n_components=N_PCS, random_state=0)
scores = pca.fit_transform(X_scaled)   # (n_cells, N_PCS)
L = pca.components_.T                   # (N_HVG, N_PCS) gene loadings

# --- 5. Varimax rotation ---
print('Running varimax...')
R = varimax(L)
vx_scores   = scores @ R
vx_loadings = L @ R

# --- 6. Reorder by descending variance ---
vx_var_order = np.argsort(vx_scores.var(axis=0))[::-1]
vx_scores    = vx_scores[:, vx_var_order]
vx_loadings  = vx_loadings[:, vx_var_order]
vx_cols      = [f'VX{i+1}' for i in range(N_PCS)]

# --- 7. Variance partitioning by Type_leiden ---
print('\nVariance partitioning by Type_leiden:')
X_type = pd.get_dummies(types).values.astype(float)
print(f'  {"VX":<6}  {"R2_type":>9}  {"R2_resid":>10}')
for i, col in enumerate(vx_cols):
    y = vx_scores[:, i]
    r2_type = r2(X_type, y)
    r2_resid = max(1.0 - r2_type, 0.0)
    print(f'  {col:<6}  {r2_type:>9.3f}  {r2_resid:>10.3f}')

# --- 8. Save varimax coords + loadings ---
vx_df = pd.DataFrame(vx_scores, index=m_adata.obs_names, columns=vx_cols)
vx_df[CLUSTER_COL] = types
vx_df.to_csv(OUT_VX_COORDS, sep='\t')
print(f'\nSaved {OUT_VX_COORDS}')

vx_load_df = pd.DataFrame(vx_loadings, index=gene_names, columns=vx_cols)
vx_load_df.to_csv(OUT_VX_LOADINGS, sep='\t')
print(f'Saved {OUT_VX_LOADINGS}')

# --- 9. PCHA ---
print(f'\nFitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(vx_scores, types)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)

pd.DataFrame(sca.xp, index=m_adata.obs_names,
             columns=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_PCHA_XP, sep='\t')
pd.DataFrame(sca.aa.T, columns=[f'PC{i+1}' for i in range(NDIM)],
             index=[f'archetype_{k+1}' for k in range(NOC)]).to_csv(OUT_PCHA_AA, sep='\t')
print(f'Saved {OUT_PCHA_XP} and {OUT_PCHA_AA}')

# --- 10. Visualize ---
print('\nGenerating visualization...')
scatter_categorical_html(
    xp_grid=[sca.xp],
    cell_metadata={CLUSTER_COL: types},
    title='Yoo25 mouse L2/3 IT — varimax PCHA space (NOC=3)',
    out_path=OUT_HTML,
    panels=[(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')],
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    aa_grid=[sca.aa],
)
print(f'Saved {OUT_HTML}')
print('Done.')
