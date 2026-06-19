"""PCA + Harmony donor correction + varimax + UMAP + Leiden for cheng22 Arch1+Arch2 astrocytes.

Same as script 41 but includes all labeled P28/P38 samples (NR, DR, DL).

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/26.combined_labels.parquet
Outputs:
  local_data/res/astro/41.v2.cheng22_arch12_harmony.h5ad
  local_data/fig/astro/41.v2.umap.html
  local_data/fig/astro/41.v2.pc12.html
  local_data/fig/astro/41.v2.vx12.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg
import umap
import harmonypy as hm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg
from scomics.utils import norm
from viz import scatter_2d_categorical_html, gene_expr_scatter_html

# --- file paths ---
INPUT_H5AD         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_COMBINED_LABELS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
OUT_RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_FIG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_H5AD           = os.path.join(OUT_RES_DIR, '41.v2.cheng22_arch12_harmony.h5ad')
OUT_UMAP           = os.path.join(OUT_FIG_DIR, '41.v2.umap.html')
OUT_PC12           = os.path.join(OUT_FIG_DIR, '41.v2.pc12.html')
OUT_VX12           = os.path.join(OUT_FIG_DIR, '41.v2.vx12.html')

LABELED_AGES      = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
ARCH_KEEP         = ['Arch1', 'Arch2']
N_HVG             = 2000
N_PCS             = 10
MIN_CELLS         = 50
N_NEIGHBORS       = 30
LEIDEN_RESOLUTION = 0.5
GENES             = ['Gfap', 'Apoe', 'Mfge8', 'Id3', 'Lama3', 'Trpm3', 'Il33']

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


# --- load data ---
print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
print(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

# remove mitochondrial genes
mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
print(f'  Removed {mt_mask.sum()} mitochondrial genes; {(~mt_mask).sum()} remaining')

# filter to ages with > MIN_CELLS cells
ages = adata.obs['Age'].values
age_counts = pd.Series(ages).value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
ages = adata.obs['Age'].values
print(f'  Total cells after age filter: {adata.shape[0]}')

x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float64)
depths = x.sum(axis=1)
donors = adata.obs['Sample'].values

# --- load arch labels ---
print(f'Loading arch labels from {IN_COMBINED_LABELS}')
df_combined = pd.read_parquet(IN_COMBINED_LABELS)
labels_c22 = df_combined[df_combined['dataset'] == 'cheng22'].reset_index(drop=True)

labeled_mask = np.isin(ages, LABELED_AGES)
assert len(labels_c22) == labeled_mask.sum(), (
    f'Labeled cell count mismatch: parquet {len(labels_c22)} vs adata {labeled_mask.sum()}'
)

# select all labeled cells (P28 NR/DL/DR + P38 NR/DR), restrict to Arch1 and Arch2
arch_labels_all = labels_c22['archetype'].values
labeled_indices = np.where(labeled_mask)[0]

arch12_mask = np.isin(arch_labels_all, ARCH_KEEP)
sel_indices  = labeled_indices[arch12_mask]
arch_labels  = arch_labels_all[arch12_mask]
print(f'  Total labeled cells: {labeled_mask.sum()}')
print(f'  Arch1+Arch2 cells: {len(sel_indices)}')
print(f'  Archetype distribution: {pd.Series(arch_labels).value_counts().to_dict()}')
print(f'  Age distribution: {pd.Series(ages[sel_indices]).value_counts().to_dict()}')

# --- HVG selection ---
hvg_mask = select_hvg(x[sel_indices], depths[sel_indices], N_HVG)
gene_names = adata.var_names.values[hvg_mask]
print(f'  HVGs selected: {hvg_mask.sum()}')

# --- normalize (CP10k → log2(1+x) → z-score per gene) ---
xn = norm(x[:, hvg_mask], depths)
xn_sel = xn[sel_indices]

# --- regress out library size ---
print('Regressing out library size...')
log_depth = np.log(depths[sel_indices]).reshape(-1, 1)
reg = LinearRegression().fit(log_depth, xn_sel)
xn_sel = xn_sel - reg.predict(log_depth)
print(f'  Done; xn_sel shape: {xn_sel.shape}')

# --- PCA ---
print(f'Fitting PCA (N_PCS={N_PCS})...')
pca = PCA(N_PCS, random_state=0)
pca.fit(xn_sel)
pca_scores = pca.transform(xn_sel)
L = pca.components_.T

# --- Harmony donor correction ---
print('Running Harmony batch correction (by Sample)...')
meta_df = pd.DataFrame({'Sample': donors[sel_indices]})
ho = hm.run_harmony(pca_scores, meta_df, 'Sample', random_state=0)
pca_scores_h = ho.Z_corr
print(f'  Harmony converged; corrected embedding shape: {pca_scores_h.shape}')

# --- varimax rotation ---
print('Running varimax rotation...')
R = varimax(L)
vx_var_order = np.argsort((pca_scores_h @ R).var(axis=0))[::-1]
R = R[:, vx_var_order]
vx_scores   = pca_scores_h @ R
vx_loadings = L @ R

# --- UMAP ---
print(f'Running UMAP (n_neighbors={N_NEIGHBORS})...')
reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, n_components=2, random_state=0)
umap_coords = reducer.fit_transform(pca_scores_h)

# --- Leiden clustering ---
print(f'Running Leiden (resolution={LEIDEN_RESOLUTION}, n_neighbors={N_NEIGHBORS})...')
nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric='euclidean')
nn.fit(pca_scores_h)
_, knn_indices = nn.kneighbors(pca_scores_h)
knn_indices = knn_indices[:, 1:]

n = len(pca_scores_h)
edges_set = set()
for i in range(n):
    for j in knn_indices[i]:
        edges_set.add((min(i, int(j)), max(i, int(j))))

graph = ig.Graph(n=n, edges=list(edges_set), directed=False)
partition = leidenalg.find_partition(
    graph, leidenalg.RBConfigurationVertexPartition,
    resolution_parameter=LEIDEN_RESOLUTION, seed=0,
)
leiden_labels = [f'l{c}' for c in partition.membership]
print(f'  {len(set(leiden_labels))} Leiden clusters')

# --- assemble and save h5ad ---
print('Assembling h5ad...')
adata_out = adata[sel_indices, hvg_mask].copy()
adata_out.X = xn_sel.astype(np.float32)

adata_out.obs['archetype'] = arch_labels
adata_out.obs['leiden']    = leiden_labels
adata_out.obs['depth']     = depths[sel_indices]

adata_out.obsm['X_pca']     = pca_scores.astype(np.float32)
adata_out.obsm['X_harmony'] = pca_scores_h.astype(np.float32)
adata_out.obsm['X_vx']      = vx_scores.astype(np.float32)
adata_out.obsm['X_umap']    = umap_coords.astype(np.float32)

adata_out.varm['PCA_loadings'] = pca.components_.T.astype(np.float32)
adata_out.varm['VX_loadings']  = vx_loadings.astype(np.float32)

adata_out.write_h5ad(OUT_H5AD)
print(f'Saved {OUT_H5AD}')

# --- gene values: log2(CP10k) from raw counts ---
print('Extracting gene expression for visualization...')
missing = [g for g in GENES if g not in adata.var_names]
if missing:
    raise ValueError(f'Genes not found: {missing}')
gene_idx = {g: np.where(adata.var_names == g)[0][0] for g in GENES}
x_genes = x[sel_indices][:, [gene_idx[g] for g in GENES]]
x_lognorm = np.log2(1 + x_genes / depths[sel_indices].reshape(-1, 1) * 1e4)
gene_vals = {g: x_lognorm[:, i] for i, g in enumerate(GENES)}
gene_vals['library_size'] = depths[sel_indices]

cell_metadata = {
    'Type':      adata_out.obs['Type'].values,
    'Age':       adata_out.obs['Age'].values,
    'archetype': arch_labels,
    'leiden':    leiden_labels,
    'Sample':    donors[sel_indices],
}


def _make_html(coords, xlabel, ylabel, title_cat, title_gene):
    html_cat = scatter_2d_categorical_html(
        xp_grid=[coords], cell_metadata=cell_metadata,
        title=title_cat, out_path=None,
        xlabel=xlabel, ylabel=ylabel, return_html=True,
    )
    html_gene = gene_expr_scatter_html(
        x=coords[:, 0], y=coords[:, 1], gene_vals=gene_vals,
        title=title_gene, out_path=None,
        xlabel=xlabel, ylabel=ylabel, return_html=True,
    )
    return f'<html><body>{html_cat}{html_gene}</body></html>'


print('Building UMAP HTML...')
with open(OUT_UMAP, 'w') as f:
    f.write(_make_html(umap_coords, 'UMAP1', 'UMAP2',
                       'cheng22 all P28+P38 Arch1+Arch2 — UMAP (Harmony corrected)',
                       'cheng22 all P28+P38 Arch1+Arch2 — gene expression on UMAP'))
print(f'Saved {OUT_UMAP}')

print('Building PC1-2 HTML...')
with open(OUT_PC12, 'w') as f:
    f.write(_make_html(pca_scores[:, :2], 'PC1', 'PC2',
                       'cheng22 all P28+P38 Arch1+Arch2 — PC1 vs PC2 (raw PCA)',
                       'cheng22 all P28+P38 Arch1+Arch2 — gene expression on PC1-2'))
print(f'Saved {OUT_PC12}')

print('Building VX1-2 HTML...')
with open(OUT_VX12, 'w') as f:
    f.write(_make_html(vx_scores[:, :2], 'VX1', 'VX2',
                       'cheng22 all P28+P38 Arch1+Arch2 — VX1 vs VX2 (varimax on Harmony)',
                       'cheng22 all P28+P38 Arch1+Arch2 — gene expression on VX1-2'))
print(f'Saved {OUT_VX12}')
print('Done.')
