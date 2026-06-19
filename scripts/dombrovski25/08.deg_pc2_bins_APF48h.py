# DEG analysis between PC2 extremes — LPLC2 APF_48h, Dombrovski 2025 fly dataset
# Group A: pc2_bin 00+01 (low PC2)  vs  Group B: pc2_bin 08+09 (high PC2)
# Test: Wilcoxon rank-sum (non-parametric, standard for scRNA-seq)
# Log2FC computed from CP10k+log2(1+x) expression (not z-scores)
# Significance: FDR (Benjamini-Hochberg) < 0.05 and |Log2FC| > 1
# Outputs:
#   local_data/res/dombrovski25_fly/08.deg_pc2_bins_APF48h.parquet  (all genes, ranked by FDR)

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, 'links', 'fly', 'dombrovski25_fly.h5ad')
IN_PCA = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'dombrovski25_fly', '07.lplc2_APF48h_pca.parquet')
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'dombrovski25_fly')
OUT_DEG = os.path.join(RES_DIR, '08.deg_pc2_bins_APF48h.parquet')

CELLTYPE = 'LPLC2'
AGE = 'APF_48h'
GROUP_A_BINS = ['bin00', 'bin01']   # low PC2
GROUP_B_BINS = ['bin08', 'bin09']   # high PC2
FDR_THRESH = 0.05
LOG2FC_THRESH = 1.0

os.makedirs(RES_DIR, exist_ok=True)

# --- load, filter, remove mt genes ---
adata = ad.read_h5ad(INPUT_FILE)
adata = adata[adata.obs['type1'] == CELLTYPE].copy()
mt_mask = np.array([g.lower().startswith('mt:') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
adata = adata[adata.obs['orig.ident'] == AGE].copy()
print(f'{AGE} LPLC2 cells (no mt genes): {adata.shape[0]} × {adata.shape[1]}')

# --- use all non-mt genes, drop constant genes ---
x_raw = np.array(adata.X)
depths = adata.obs['nCount_RNA'].values
var_mask = x_raw.var(axis=0) > 0
x_hvg = x_raw[:, var_mask]
gene_names = np.array(adata.var_names)[var_mask]

# CP10k + log2(1+x) — used for Log2FC (interpretable fold change, not z-score)
depth_col = depths[:, None]
xlog = np.log2(x_hvg / depth_col * 1e4 + 1)

# --- assign groups from script 07 PC2 bins ---
pca_df = pd.read_parquet(IN_PCA)
assert list(pca_df.index) == list(adata.obs_names), 'Cell order mismatch between parquet and adata'
pc2_bin = pca_df['pc2_bin'].values

mask_a = np.isin(pc2_bin, GROUP_A_BINS)
mask_b = np.isin(pc2_bin, GROUP_B_BINS)
print(f'Group A ({GROUP_A_BINS}): {mask_a.sum()} cells')
print(f'Group B ({GROUP_B_BINS}): {mask_b.sum()} cells')

# --- Wilcoxon rank-sum test per gene ---
print(f'Running Wilcoxon tests on {len(gene_names)} genes...')
pvals = np.empty(len(gene_names))
log2fc = np.empty(len(gene_names))

for i in range(len(gene_names)):
    a = xlog[mask_a, i]
    b = xlog[mask_b, i]
    log2fc[i] = b.mean() - a.mean()   # positive = higher in group B (high PC2)
    _, pvals[i] = mannwhitneyu(a, b, alternative='two-sided')

# BH FDR correction
_, fdr, _, _ = multipletests(pvals, method='fdr_bh')

# --- assemble results ---
result_df = pd.DataFrame({
    'gene': gene_names,
    'log2fc': log2fc,
    'pval': pvals,
    'fdr': fdr,
})
result_df['significant'] = (result_df['fdr'] < FDR_THRESH) & (result_df['log2fc'].abs() > LOG2FC_THRESH)
result_df = result_df.sort_values('fdr').reset_index(drop=True)

result_df.to_parquet(OUT_DEG)
print(f'Saved {OUT_DEG}')

n_sig = result_df['significant'].sum()
n_up = ((result_df['fdr'] < FDR_THRESH) & (result_df['log2fc'] > LOG2FC_THRESH)).sum()
n_dn = ((result_df['fdr'] < FDR_THRESH) & (result_df['log2fc'] < -LOG2FC_THRESH)).sum()
print(f'\nSignificant DEGs (FDR<{FDR_THRESH}, |Log2FC|>{LOG2FC_THRESH}): {n_sig}')
print(f'  Up in group B (high PC2): {n_up}')
print(f'  Up in group A (low PC2):  {n_dn}')
print('\nTop 10 by FDR:')
print(result_df[['gene', 'log2fc', 'pval', 'fdr', 'significant']].head(10).to_string(index=False))
