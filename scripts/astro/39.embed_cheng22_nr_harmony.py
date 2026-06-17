"""PCA + Harmony donor correction + varimax + UMAP + Leiden for cheng22 P28NR + P38NR astrocytes.

Extends script 37 with two correction steps:
  1. Library size regressed out from normalized expression (before PCA).
  2. Harmony donor batch correction applied to PCA scores (before varimax/UMAP/Leiden).

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/26.combined_labels.parquet
Outputs:
  local_data/res/astro/39.cheng22_nr_harmony.h5ad
  local_data/fig/astro/39.umap.html
  local_data/fig/astro/39.pc12.html
  local_data/fig/astro/39.vx12.html
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
OUT_H5AD           = os.path.join(OUT_RES_DIR, '39.cheng22_nr_harmony.h5ad')
OUT_UMAP           = os.path.join(OUT_FIG_DIR, '39.umap.html')
OUT_PC12           = os.path.join(OUT_FIG_DIR, '39.pc12.html')
OUT_VX12           = os.path.join(OUT_FIG_DIR, '39.vx12.html')

LABELED_AGES      = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES           = ['P28', 'P38']
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

# select NR cells (P28 + P38), all archetypes
arch_labels_all = labels_c22['archetype'].values
nr_mask_in_labeled = np.isin(ages[labeled_mask], NR_AGES)
nr_indices = np.where(labeled_mask)[0][nr_mask_in_labeled]
arch_labels = arch_labels_all[nr_mask_in_labeled]
print(f'  NR cells (P28 + P38): {len(nr_indices)}')
print(f'  Archetype distribution: {pd.Series(arch_labels).value_counts().to_dict()}')

# --- HVG selection on all NR cells ---
hvg_mask = select_hvg(x[nr_indices], depths[nr_indices], N_HVG)
gene_names = adata.var_names.values[hvg_mask]
print(f'  HVGs selected: {hvg_mask.sum()}')

# --- normalize (CP10k → log2(1+x) → z-score per gene) ---
xn = norm(x[:, hvg_mask], depths)
xn_nr = xn[nr_indices]

# --- regress out library size from expression space ---
print('Regressing out library size...')
log_depth = np.log(depths[nr_indices]).reshape(-1, 1)
reg = LinearRegression().fit(log_depth, xn_nr)
xn_nr = xn_nr - reg.predict(log_depth)
print(f'  Done; xn_nr shape: {xn_nr.shape}')

# --- PCA on all NR cells ---
print(f'Fitting PCA (N_PCS={N_PCS}) on all NR cells...')
pca = PCA(N_PCS, random_state=0)
pca.fit(xn_nr)
pca_scores = pca.transform(xn_nr)   # (n_nr, N_PCS)
L = pca.components_.T               # (n_hvg, N_PCS)

# --- Harmony donor correction ---
print('Running Harmony batch correction (by Sample)...')
meta_df = pd.DataFrame({'Sample': donors[nr_indices]})
ho = hm.run_harmony(pca_scores, meta_df, 'Sample', random_state=0)
pca_scores_h = ho.Z_corr            # (n_nr, N_PCS), donor-corrected
print(f'  Harmony converged; corrected embedding shape: {pca_scores_h.shape}')

# --- varimax rotation on harmony-corrected scores ---
print('Running varimax rotation...')
R = varimax(L)
vx_var_order = np.argsort((pca_scores_h @ R).var(axis=0))[::-1]
R = R[:, vx_var_order]
vx_scores   = pca_scores_h @ R     # (n_nr, N_PCS)
vx_loadings = L @ R                 # (n_hvg, N_PCS)

# --- UMAP on harmony-corrected scores ---
print(f'Running UMAP (n_neighbors={N_NEIGHBORS})...')
reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, n_components=2, random_state=0)
umap_coords = reducer.fit_transform(pca_scores_h)

# --- Leiden clustering via leidenalg + igraph ---
# (scanpy unavailable in this env: matplotlib metaclass conflict in scanpy.plotting)
# Use n_neighbors+1 because kneighbors(X) returns each point as its own 0th neighbor.
print(f'Running Leiden (resolution={LEIDEN_RESOLUTION}, n_neighbors={N_NEIGHBORS})...')
nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric='euclidean')
nn.fit(pca_scores_h)
_, knn_indices = nn.kneighbors(pca_scores_h)
knn_indices = knn_indices[:, 1:]   # drop self

n = len(pca_scores_h)
# Deduplicate: keep only (min, max) pairs for an undirected graph
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
adata_out = adata[nr_indices, hvg_mask].copy()
adata_out.X = xn_nr.astype(np.float32)

adata_out.obs['archetype'] = arch_labels
adata_out.obs['leiden']    = leiden_labels
adata_out.obs['depth']     = depths[nr_indices]

adata_out.obsm['X_pca']     = pca_scores.astype(np.float32)
adata_out.obsm['X_harmony'] = pca_scores_h.astype(np.float32)
adata_out.obsm['X_vx']      = vx_scores.astype(np.float32)
adata_out.obsm['X_umap']    = umap_coords.astype(np.float32)

adata_out.varm['PCA_loadings'] = pca.components_.T.astype(np.float32)
adata_out.varm['VX_loadings']  = vx_loadings.astype(np.float32)

adata_out.write_h5ad(OUT_H5AD)
print(f'Saved {OUT_H5AD}')

# --- gene values from z-scored adata_out.X ---
print('Extracting gene expression for visualization...')
present = [g for g in GENES if g in adata_out.var_names]
missing = [g for g in GENES if g not in adata_out.var_names]
if missing:
    print(f'  Warning: genes not in HVG set, skipping: {missing}')
X_dense = adata_out.X.toarray() if hasattr(adata_out.X, 'toarray') else np.array(adata_out.X)
gene_vals = {g: X_dense[:, adata_out.var_names.get_loc(g)] for g in present}
gene_vals['library_size'] = adata_out.obs['depth'].values

cell_metadata = {
    'Type':      adata_out.obs['Type'].values,
    'archetype': arch_labels,
    'leiden':    leiden_labels,
    'Sample':    donors[nr_indices],
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
                       'cheng22 NR astrocytes — UMAP (Harmony corrected)',
                       'cheng22 NR astrocytes — gene expression on UMAP'))
print(f'Saved {OUT_UMAP}')

print('Building PC1-2 HTML...')
with open(OUT_PC12, 'w') as f:
    f.write(_make_html(pca_scores[:, :2], 'PC1', 'PC2',
                       'cheng22 NR astrocytes — PC1 vs PC2 (raw PCA)',
                       'cheng22 NR astrocytes — gene expression on PC1-2'))
print(f'Saved {OUT_PC12}')

print('Building VX1-2 HTML...')
with open(OUT_VX12, 'w') as f:
    f.write(_make_html(vx_scores[:, :2], 'VX1', 'VX2',
                       'cheng22 NR astrocytes — VX1 vs VX2 (varimax on Harmony)',
                       'cheng22 NR astrocytes — gene expression on VX1-2'))
print(f'Saved {OUT_VX12}')
print('Done.')
