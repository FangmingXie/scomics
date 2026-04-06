import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'local_data', 'raw', 'morcom26_cux2mice', 'P26_ENs.h5ad')
OUT_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26_cux2mice')
OUT_FILE     = os.path.join(OUT_DIR, '13.ENL23CTX_vs_ENL2mix_sig.tsv')
OUT_ALL_FILE = os.path.join(OUT_DIR, '13.ENL23CTX_vs_ENL2mix_all.tsv')

# --- config ---
CELLTYPE_COL   = 'celltype'
SAMPLE_COL     = 'samples'
GROUP_A        = 'EN-L2-3-CTX'   # coded as 0
GROUP_B        = 'EN-L2-mix'     # coded as 1
FDR_THRESH     = 0.05
MIN_EXPR_FRAC  = 0.1

# --- load & subset ---
print(f'Loading {INPUT_FILE}')
adata = ad.read_h5ad(INPUT_FILE)
print(f'  Full dataset: {adata.shape[0]} cells x {adata.shape[1]} genes')

mask = adata.obs[CELLTYPE_COL].isin([GROUP_A, GROUP_B])
adata = adata[mask].copy()
print(f'  Subset: {adata.shape[0]} cells')
print(f'  Cell type counts: {adata.obs[CELLTYPE_COL].value_counts().to_dict()}')

# --- normalize: CP10k + log1p ---
X = adata.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)
counts_per_cell = X.sum(axis=1, keepdims=True)
logcpm = np.log1p(X / counts_per_cell * 1e4)

# --- pre-filter genes ---
expr_frac = (logcpm > 0).mean(axis=0)
gene_mask = expr_frac >= MIN_EXPR_FRAC
filtered_genes = adata.var_names[gene_mask].tolist()
logcpm = logcpm[:, gene_mask]
print(f'  Genes after filtering (>= {MIN_EXPR_FRAC:.0%} cells expressed): {len(filtered_genes)}')

# --- celltype encoding: GROUP_A=0, GROUP_B=1 ---
obs = adata.obs[[CELLTYPE_COL, SAMPLE_COL]].copy()
obs['celltype_code'] = (obs[CELLTYPE_COL] == GROUP_B).astype(int)
obs = obs.rename(columns={SAMPLE_COL: 'sample'})

# --- LMM loop (sample as random effect) ---
print('Running LMM for each gene...')
results = []
n_genes = len(filtered_genes)
for i, gene in enumerate(filtered_genes):
    if i % 500 == 0:
        print(f'  {i}/{n_genes}')
    df = obs.copy()
    df['expr'] = logcpm[:, i]
    coef, pval = np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for method in ['lbfgs', 'powell', 'nm']:
            try:
                fit = smf.mixedlm('expr ~ celltype_code', data=df, groups=df['sample']).fit(
                    reml=True, method=method, disp=False)
                c = fit.params['celltype_code']
                p = fit.pvalues['celltype_code']
                if not (np.isnan(c) or np.isnan(p)):
                    coef, pval = c, p
                    break
            except Exception:
                continue
    results.append({'gene': gene, 'coef': coef, 'pval': pval})

print(f'  {n_genes}/{n_genes} done')

# --- compute log2FC and FDR ---
# positive log2FC = higher in GROUP_B (EN-L2-mix)
res_df = pd.DataFrame(results)
res_df['log2FC'] = res_df['coef'] / np.log(2)

valid = res_df['pval'].notna()
_, fdr, _, _ = multipletests(res_df.loc[valid, 'pval'], method='fdr_bh')
res_df['fdr'] = np.nan
res_df.loc[valid, 'fdr'] = fdr

sig = res_df[res_df['fdr'] < FDR_THRESH].sort_values('fdr').reset_index(drop=True)
print(f'  Significant genes (FDR<{FDR_THRESH}): {len(sig)}')

# --- save ---
os.makedirs(OUT_DIR, exist_ok=True)
res_df[['gene', 'log2FC', 'pval', 'fdr']].sort_values('fdr').to_csv(OUT_ALL_FILE, sep='\t', index=False)
print(f'Saved {OUT_ALL_FILE}')
sig[['gene', 'log2FC', 'pval', 'fdr']].to_csv(OUT_FILE, sep='\t', index=False)
print(f'Saved {OUT_FILE}')
print(f'  Note: positive log2FC = higher in {GROUP_B}; negative = higher in {GROUP_A}')
print('Done.')
