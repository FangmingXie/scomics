# DEG analysis: Pvalb Gaba vs Sst Gaba — P56 mouse VIS cortex, Gao 2025 DevVIS dataset
# Outgroup comparison for Dombrovski 2025 fly LPLC2 analysis
# Group A: Pvalb Gaba  vs  Group B: Sst Gaba
# Test: Wilcoxon rank-sum (non-parametric, standard for scRNA-seq)
# Log2FC computed from CP10k+log2(1+x) expression (not z-scores)
# Significance: FDR (Benjamini-Hochberg) < 0.05 and |Log2FC| > 1
# Outputs:
#   local_data/res/dombrovski25_fly/10.deg_pv_sst_P56_gao25.parquet  (all genes, ranked by FDR)

import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, 'links', 'broad', 'DevVIS_scRNA_processed.h5ad')
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'dombrovski25_fly')
OUT_DEG = os.path.join(RES_DIR, '10.deg_pv_sst_P56_gao25.parquet')

AGE           = 'P56'
GROUP_A_LABEL = 'Pvalb Gaba'
GROUP_B_LABEL = 'Sst Gaba'
FDR_THRESH    = 0.05
LOG2FC_THRESH = 1.0

os.makedirs(RES_DIR, exist_ok=True)

# --- load, filter to P56, remove mt genes ---
adata = ad.read_h5ad(INPUT_FILE)
adata = adata[adata.obs['Age'] == AGE].copy()
mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
print(f'{AGE} cells (no mt- genes): {adata.shape[0]} × {adata.shape[1]}')

# --- dense raw counts, library depth, drop constant genes ---
x_raw = adata.X.toarray()
depths = np.array(adata.X.sum(axis=1)).flatten()
var_mask = x_raw.var(axis=0) > 0
x_raw = x_raw[:, var_mask]
gene_names = np.array(adata.var_names)[var_mask]

# CP10k + log2(1+x) — used for Log2FC (interpretable fold change, not z-score)
xlog = np.log2(x_raw / depths[:, None] * 1e4 + 1)

# --- assign groups ---
subclass = adata.obs['subclass_label'].values
mask_a = subclass == GROUP_A_LABEL
mask_b = subclass == GROUP_B_LABEL
print(f'Group A ({GROUP_A_LABEL}): {mask_a.sum()} cells')
print(f'Group B ({GROUP_B_LABEL}): {mask_b.sum()} cells')

# --- Wilcoxon rank-sum test per gene ---
print(f'Running Wilcoxon tests on {len(gene_names)} genes...')
pvals  = np.empty(len(gene_names))
log2fc = np.empty(len(gene_names))

for i in range(len(gene_names)):
    a = xlog[mask_a, i]
    b = xlog[mask_b, i]
    log2fc[i] = b.mean() - a.mean()   # positive = higher in Group B (Sst)
    _, pvals[i] = mannwhitneyu(a, b, alternative='two-sided')

# BH FDR correction
_, fdr, _, _ = multipletests(pvals, method='fdr_bh')

# --- assemble results ---
result_df = pd.DataFrame({
    'gene':   gene_names,
    'log2fc': log2fc,
    'pval':   pvals,
    'fdr':    fdr,
})
result_df['significant'] = (result_df['fdr'] < FDR_THRESH) & (result_df['log2fc'].abs() > LOG2FC_THRESH)
result_df = result_df.sort_values('fdr').reset_index(drop=True)

result_df.to_parquet(OUT_DEG)
print(f'Saved {OUT_DEG}')

n_sig = result_df['significant'].sum()
n_up  = ((result_df['fdr'] < FDR_THRESH) & (result_df['log2fc'] >  LOG2FC_THRESH)).sum()
n_dn  = ((result_df['fdr'] < FDR_THRESH) & (result_df['log2fc'] < -LOG2FC_THRESH)).sum()
print(f'\nSignificant DEGs (FDR<{FDR_THRESH}, |Log2FC|>{LOG2FC_THRESH}): {n_sig}')
print(f'  Up in Group B ({GROUP_B_LABEL}): {n_up}')
print(f'  Up in Group A ({GROUP_A_LABEL}): {n_dn}')
print('\nTop 10 by FDR:')
print(result_df[['gene', 'log2fc', 'pval', 'fdr', 'significant']].head(10).to_string(index=False))
