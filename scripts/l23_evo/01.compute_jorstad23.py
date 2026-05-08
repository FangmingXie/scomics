"""First-pass computation for Jorstad23 human L2/3 IT snRNA-seq dataset.

Outputs: PCA coords, UMAP coords, PC-covariate correlation matrix, marker genes.
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import scipy.stats
from statsmodels.stats.multitest import multipletests

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_PCA      = os.path.join(OUT_RES_DIR, '01.pca_coords.tsv')
OUT_LOADINGS = os.path.join(OUT_RES_DIR, '01.pca_loadings.tsv')
OUT_UMAP     = os.path.join(OUT_RES_DIR, '01.umap_coords.tsv')
OUT_PC_CORR  = os.path.join(OUT_RES_DIR, '01.pc_covariate_corr.tsv')
OUT_MARKERS  = os.path.join(OUT_RES_DIR, '01.markers.tsv')

# --- config ---
N_HVG          = 2000
N_PCS          = 10
N_NEIGHBORS    = 15
N_PCS_FOR_CORR = 10
MIN_LOG2FC     = 0.5
MIN_FRAC_IN    = 0.25
CLUSTER_COL    = 'WithinArea_cluster'
COVARIATE_COLS = ['donor_id', 'Source']
METADATA_COLS  = [CLUSTER_COL] + COVARIATE_COLS + ['development_stage', 'nCount_RNA']

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- load ---
print('Loading data...')
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

# adata.X is already log-normalized (float32, values ~2–12)
X_norm = adata.X.toarray().astype(np.float32)
print(f'X_norm shape: {X_norm.shape}, dtype: {X_norm.dtype}')

gene_names = adata.var['feature_name'].values if 'feature_name' in adata.var.columns else adata.var_names.values

# --- Part A: HVG + PCA + UMAP ---
print(f'\nSelecting top {N_HVG} HVGs by variance...')
gene_var = X_norm.var(axis=0)
hvg_idx = np.argsort(gene_var)[::-1][:N_HVG]
X_hvg = X_norm[:, hvg_idx]

print('Scaling...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hvg)

print(f'PCA (n={N_PCS})...')
pca_model = PCA(n_components=N_PCS, random_state=0)
X_pca = pca_model.fit_transform(X_scaled)
print(f'  Explained variance ratio: {pca_model.explained_variance_ratio_.round(3)}')

print(f'UMAP (n_neighbors={N_NEIGHBORS})...')
reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, n_components=2, random_state=0)
X_umap = reducer.fit_transform(X_pca)

# save PCA
meta = adata.obs[METADATA_COLS]
pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
pca_df = pd.DataFrame(X_pca, index=adata.obs_names, columns=pc_cols)
for col in METADATA_COLS:
    pca_df[col] = meta[col].values
pca_df.to_csv(OUT_PCA, sep='\t')
print(f'  Saved {OUT_PCA}')

# save gene loadings (genes × PCs): components_.T shape (N_HVG, N_PCS)
hvg_gene_names = gene_names[hvg_idx]
loadings_df = pd.DataFrame(
    pca_model.components_.T,
    index=hvg_gene_names,
    columns=pc_cols,
)
loadings_df.to_csv(OUT_LOADINGS, sep='\t')
print(f'  Saved {OUT_LOADINGS}')

# save UMAP
umap_df = pd.DataFrame(X_umap, index=adata.obs_names, columns=['UMAP1', 'UMAP2'])
for col in METADATA_COLS:
    umap_df[col] = meta[col].values
umap_df.to_csv(OUT_UMAP, sep='\t')
print(f'  Saved {OUT_UMAP}')

# --- Part B: PC-covariate correlations ---
print('\nComputing PC-covariate correlations...')
corr_cols = [CLUSTER_COL] + COVARIATE_COLS
covariate_dummies = pd.get_dummies(adata.obs[corr_cols])
corr_rows = []
for i in range(N_PCS_FOR_CORR):
    pc_vals = X_pca[:, i]
    row = {}
    for col in covariate_dummies.columns:
        r, _ = scipy.stats.pearsonr(pc_vals, covariate_dummies[col].values.astype(float))
        row[col] = r
    corr_rows.append(row)

corr_df = pd.DataFrame(corr_rows, index=[f'PC{i+1}' for i in range(N_PCS_FOR_CORR)])
corr_df.to_csv(OUT_PC_CORR, sep='\t')
print(f'  Saved {OUT_PC_CORR}')

# --- Part C: Marker genes (Wilcoxon one-vs-rest) ---
print('\nFinding marker genes...')
clusters = adata.obs[CLUSTER_COL].values
unique_clusters = sorted(set(clusters))
all_markers = []
for c in unique_clusters:
    print(f'  Cluster {c}...')
    mask_in  = clusters == c
    mask_out = ~mask_in
    X_in  = X_hvg[mask_in]
    X_out = X_hvg[mask_out]
    hvg_gene_names = gene_names[hvg_idx]

    pvals = np.empty(N_HVG)
    log2fc = np.empty(N_HVG)
    frac_in  = np.empty(N_HVG)
    frac_out = np.empty(N_HVG)

    mean_in  = X_in.mean(axis=0)
    mean_out = X_out.mean(axis=0)
    log2fc   = (mean_in - mean_out) / np.log(2)
    frac_in  = (X_in  > 0).mean(axis=0)
    frac_out = (X_out > 0).mean(axis=0)

    for g in range(N_HVG):
        _, pvals[g] = scipy.stats.ranksums(X_in[:, g], X_out[:, g])

    _, fdr, _, _ = multipletests(pvals, method='fdr_bh')

    df = pd.DataFrame({
        'gene':     hvg_gene_names,
        'cluster':  c,
        'log2FC':   log2fc,
        'pval':     pvals,
        'fdr':      fdr,
        'frac_in':  frac_in,
        'frac_out': frac_out,
    })
    # filter
    df = df[(df['log2FC'] >= MIN_LOG2FC) & (df['frac_in'] >= MIN_FRAC_IN)]
    df = df.sort_values('fdr')
    all_markers.append(df)
    print(f'    {len(df)} markers after filtering')

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(OUT_MARKERS, sep='\t', index=False)
print(f'  Saved {OUT_MARKERS}  ({len(markers_df)} total markers)')
print('\nDone.')
