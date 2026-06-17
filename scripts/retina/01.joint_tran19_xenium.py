# Joint analysis of tran19 snRNA-seq reference and Xenium spatial trial data.
# Uses Harmony for batch correction, kNN for label transfer of tran19 subtypes to Xenium.
# Normalization: log1p CP10k on shared genes.

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
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg
from viz import scatter_2d_categorical_html

# --- File paths ---
IN_TRAN19 = os.path.join(PROJECT_ROOT, 'links', 'retina', 'tran19_retina.h5ad')
IN_XENIUM = os.path.join(PROJECT_ROOT, 'links', 'retina', 'trial', 'retina_xenium.h5ad')

OUT_FIG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'retina')
OUT_XENIUM_H5AD    = IN_XENIUM
OUT_FIG_UMAP_JOINT  = os.path.join(OUT_FIG_DIR, '01.umap_joint.html')
OUT_FIG_UMAP_GENES  = os.path.join(OUT_FIG_DIR, '01.umap_genes.html')
OUT_FIG_LEIDEN_BAR  = os.path.join(OUT_FIG_DIR, '01.leiden_barplot.html')

# --- Parameters ---
N_HVG          = 2000
N_PCS          = 30
N_NEIGHBORS    = 15
UMAP_SEED      = 42
SUBSAMPLE_FRAC = 0.20
SUBSAMPLE_SEED = 0

SELECTED_GENES = ['Slc17a6', 'Slc17a7', 'Rbpms', 'Spp1']

os.makedirs(OUT_FIG_DIR, exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


def _normalize(x, depths):
    return np.log1p(x / depths[:, None] * 1e4)


# --- Load data ---
print('Loading tran19...')
tran19 = ad.read_h5ad(IN_TRAN19)
print(f'  {tran19.n_obs} cells × {tran19.n_vars} genes')
print(f'  Subtypes: {tran19.obs["subtype"].nunique()} unique')

print('Loading xenium...')
xenium = ad.read_h5ad(IN_XENIUM)
print(f'  {xenium.n_obs} cells × {xenium.n_vars} genes')

# --- Shared genes ---
shared_genes = sorted(set(tran19.var_names) & set(xenium.var_names))
print(f'Shared genes: {len(shared_genes)}')

# --- Count matrices on shared genes ---
x_tran19 = _to_array(tran19[:, shared_genes].X)
x_xenium  = _to_array(xenium[:, shared_genes].X)

depths_tran19 = x_tran19.sum(axis=1)
depths_xenium  = x_xenium.sum(axis=1)

# --- HVG selection (on tran19 reference) ---
hvg_mask = select_hvg(x_tran19, depths_tran19, N_HVG)
print(f'HVGs selected: {hvg_mask.sum()}')

# --- Normalize on HVG subset ---
xn_tran19 = _normalize(x_tran19[:, hvg_mask], depths_tran19)
xn_xenium  = _normalize(x_xenium[:, hvg_mask],  depths_xenium)
xn_tran19  = np.nan_to_num(xn_tran19, nan=0.0)
xn_xenium  = np.nan_to_num(xn_xenium,  nan=0.0)

# --- PCA: fit on tran19, project both ---
print('Running PCA...')
pca = PCA(n_components=N_PCS, random_state=0)
xp_tran19 = pca.fit_transform(xn_tran19)
xp_xenium  = pca.transform(xn_xenium)
print(f'  Cumulative variance ({N_PCS} PCs): {np.cumsum(pca.explained_variance_ratio_)[-1]:.3f}')

# --- Harmony batch correction ---
print('Running Harmony...')
n_tran19  = tran19.n_obs
n_xenium  = xenium.n_obs
xp_all    = np.vstack([xp_tran19, xp_xenium])
meta_df   = pd.DataFrame({'batch': ['tran19'] * n_tran19 + ['xenium'] * n_xenium})
ho        = hm.run_harmony(xp_all, meta_df, 'batch', random_state=42)
xp_all_h  = ho.Z_corr  # harmonypy 2.x: (ncells, ndim)
print('  Harmony done.')

xp_tran19_h = xp_all_h[:n_tran19]
xp_xenium_h  = xp_all_h[n_tran19:]

# --- kNN label transfer: tran19 subtypes → xenium ---
print('Running kNN label transfer...')
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
knn.fit(xp_tran19_h, tran19.obs['subtype'].values)
xenium.obs['tran19_subtype'] = knn.predict(xp_xenium_h)
print(f'  Transferred {xenium.obs["tran19_subtype"].nunique()} subtypes.')

# --- Joint UMAP ---
print(f'Running joint UMAP ({n_tran19 + n_xenium} cells)...')
reducer     = umap.UMAP(n_components=2, random_state=UMAP_SEED)
umap_coords = reducer.fit_transform(xp_all_h)
umap_tran19 = umap_coords[:n_tran19]
umap_xenium  = umap_coords[n_tran19:]
print('  UMAP done.')

# --- Subsample for visualization ---
n_total = n_tran19 + n_xenium
n_sub   = int(n_total * SUBSAMPLE_FRAC)
rng     = np.random.default_rng(SUBSAMPLE_SEED)
sub_idx = np.sort(rng.choice(n_total, size=n_sub, replace=False))
print(f'  Subsampled {n_sub} / {n_total} cells for visualization.')

# --- Gene expression (log1p CP10k on shared genes) ---
shared_idx = {g: i for i, g in enumerate(shared_genes)}
gene_expr  = {}
for gene in SELECTED_GENES:
    if gene not in shared_idx:
        print(f'  WARNING: {gene} not in shared genes — skipping.')
        continue
    gi = shared_idx[gene]
    expr_tran19 = _normalize(x_tran19[:, [gi]], depths_tran19).ravel()
    expr_xenium  = _normalize(x_xenium[:,  [gi]], depths_xenium).ravel()
    gene_expr[gene] = np.concatenate([expr_tran19, expr_xenium])[sub_idx]

# --- Save updated xenium h5ad ---
xenium.write_h5ad(OUT_XENIUM_H5AD)
print(f'Saved updated xenium h5ad → {OUT_XENIUM_H5AD}')

# --- UMAP joint figure: source / subtype / leiden ---
print('Plotting joint UMAP (source / subtype / leiden)...')
source_labels  = np.array(['tran19'] * n_tran19 + ['xenium'] * n_xenium)[sub_idx]
subtype_labels = np.concatenate([
    tran19.obs['subtype'].values,
    xenium.obs['tran19_subtype'].values,
])[sub_idx]
leiden_labels  = np.concatenate([
    np.array(['ref'] * n_tran19),
    xenium.obs['leiden'].values,
])[sub_idx]
scatter_2d_categorical_html(
    xp_grid=[umap_coords[sub_idx]],
    cell_metadata={
        'Source':   source_labels,
        'Subtype':  subtype_labels,
        'Leiden':   leiden_labels,
    },
    title='Joint UMAP — tran19 + Xenium',
    out_path=OUT_FIG_UMAP_JOINT,
    xlabel='UMAP1', ylabel='UMAP2',
    ordered_labels=('Source',),
)
print(f'  Saved → {OUT_FIG_UMAP_JOINT}')

# --- UMAP gene expression figure ---
print('Plotting UMAP (gene expression)...')
scatter_2d_categorical_html(
    xp_grid=[umap_coords[sub_idx]],
    cell_metadata=gene_expr,
    title='Joint UMAP — gene expression (tran19 + Xenium)',
    out_path=OUT_FIG_UMAP_GENES,
    xlabel='UMAP1', ylabel='UMAP2',
)
print(f'  Saved → {OUT_FIG_UMAP_GENES}')

# --- Leiden cluster fraction barplot (full xenium data) ---
print('Plotting Leiden cluster fraction barplot...')
leiden_counts = xenium.obs['leiden'].value_counts()
leiden_counts = leiden_counts.reindex(sorted(leiden_counts.index, key=int))
leiden_frac   = leiden_counts / leiden_counts.sum()
fig = go.Figure(go.Bar(
    x=['l' + c for c in leiden_frac.index],
    y=leiden_frac.values,
    marker_color='steelblue',
))
fig.update_layout(
    title='Fraction of Xenium cells per Leiden cluster',
    xaxis_title='Leiden cluster', yaxis_title='Fraction of cells',
    height=450, width=700,
)
fig.write_html(OUT_FIG_LEIDEN_BAR)
print(f'  Saved → {OUT_FIG_LEIDEN_BAR}')

print('Done.')
