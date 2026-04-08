import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# warnings.filterwarnings('ignore')

# --- file paths ---
PROJECT_ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE          = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'morcom26_cux2mice', 'P26_EN-L2-3-CTX_EN-L4-5-CTX_EN-L2-mix.h5ad')
LABEL_TRANSFER_FILE = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26', '16.label_transfer.tsv')
OUT_DIR             = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26')
OUT_FILE_ALL_TMPL   = os.path.join(OUT_DIR, '21.deg_{label}_all.tsv')
OUT_FILE_SIG_TMPL   = os.path.join(OUT_DIR, '21.deg_{label}_sig.tsv')

# --- config ---
CELLTYPE_COL   = 'celltype'
SAMPLE_COL     = 'samples'
CONDITION_COL  = 'Condition'
KEEP_CELLTYPES = ['EN-L2-3-CTX'] # , 'EN-L2-mix']
PRED_TYPE_COL  = 'pred_type'
PRED_TYPES     = ['L2/3_A', 'L2/3_B', 'L2/3_C']
FDR_THRESH     = 0.05
MIN_EXPR_FRAC  = 0.1
TEST_N_GENES   = None    # set to None to disable test mode

os.makedirs(OUT_DIR, exist_ok=True)


def run_lmm(logcpm, genes, obs):
    """Run LMM (WT vs Null) for each gene; return DataFrame with coef, pval, log2FC, fdr."""
    results = []
    n_genes = len(genes)
    for i, gene in enumerate(genes):
        if i % 100 == 0:
            print(f'    {i}/{n_genes}')
        df = obs.copy()
        df['expr'] = logcpm[:, i]
        try:
            fit = smf.mixedlm('expr ~ condition_code', data=df, groups=df['sample']).fit(
                reml=True, method='lbfgs', disp=False)
            coef = fit.params['condition_code']
            pval = fit.pvalues['condition_code']
        except Exception as e:
            print(f'    WARNING: {gene} failed: {e}')
            coef, pval = np.nan, np.nan
        results.append({'gene': gene, 'coef': coef, 'pval': pval})

    res_df = pd.DataFrame(results)
    res_df['log2FC'] = res_df['coef'] / np.log(2)
    valid = res_df['pval'].notna()
    _, fdr, _, _ = multipletests(res_df.loc[valid, 'pval'], method='fdr_bh')
    res_df['fdr'] = np.nan
    res_df.loc[valid, 'fdr'] = fdr
    return res_df[['gene', 'log2FC', 'pval', 'fdr']]


def save_results(res_df, label):
    out_all = OUT_FILE_ALL_TMPL.format(label=label)
    out_sig = OUT_FILE_SIG_TMPL.format(label=label)
    res_df.sort_values('fdr').to_csv(out_all, sep='\t', index=False)
    sig = res_df[res_df['fdr'] < FDR_THRESH].sort_values('fdr')
    sig.to_csv(out_sig, sep='\t', index=False)
    print(f'  Significant genes (FDR<{FDR_THRESH}): {len(sig)}')
    print(f'  Saved -> {out_all}')
    print(f'  Saved -> {out_sig}')


# --- load ---
print(f'Loading {INPUT_FILE}')
adata = ad.read_h5ad(INPUT_FILE)
print(f'  Full dataset: {adata.shape[0]} cells x {adata.shape[1]} genes')

print(f'Loading label transfer: {LABEL_TRANSFER_FILE}')
lt = pd.read_csv(LABEL_TRANSFER_FILE, sep='\t', index_col=0)

# --- filter to EN-L2-3-CTX , then to L2/3 pred_types ---
mask_ct = adata.obs[CELLTYPE_COL].isin(KEEP_CELLTYPES)
adata = adata[mask_ct].copy()
print(f'  After celltype filter: {adata.shape[0]} cells')

# align label transfer to adata
common_cells = adata.obs_names.intersection(lt.index)
adata = adata[common_cells].copy()
lt = lt.loc[common_cells]

adata.obs[PRED_TYPE_COL] = lt[PRED_TYPE_COL].values

mask_pt = adata.obs[PRED_TYPE_COL].isin(PRED_TYPES)
adata = adata[mask_pt].copy()
print(f'  After pred_type filter: {adata.shape[0]} cells')
print(f'  Condition counts: {adata.obs[CONDITION_COL].value_counts().to_dict()}')

# --- normalize: CP10k + log1p ---
X = adata.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)
logcpm_full = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)

# --- shared obs columns ---
obs_full = adata.obs[[CONDITION_COL, SAMPLE_COL, PRED_TYPE_COL]].copy()
obs_full['condition_code'] = (obs_full[CONDITION_COL] == 'Null').astype(int)
obs_full = obs_full.rename(columns={SAMPLE_COL: 'sample'})

# =============================================================================
# OVERALL (all pred_types combined)
# =============================================================================
print('\n--- Overall (all L2/3 subtypes combined) ---')
expr_frac = (logcpm_full > 0).mean(axis=0)
gene_mask = expr_frac >= MIN_EXPR_FRAC
if TEST_N_GENES:
    gene_indices = np.where(gene_mask)[0][:TEST_N_GENES]
    genes = adata.var_names[gene_indices].tolist()
    logcpm_in = logcpm_full[:, gene_indices]
else:
    genes = adata.var_names[gene_mask].tolist()
    logcpm_in = logcpm_full[:, gene_mask]
print(f'  Genes after filtering: {len(genes)}')
print(f'  Cells: {len(obs_full)}')
res = run_lmm(logcpm_in, genes, obs_full)
save_results(res, 'overall')

# =============================================================================
# PER PRED_TYPE
# =============================================================================
for pt in PRED_TYPES:
    label = pt.replace('/', '')  # e.g. L2/3_A -> L23_A
    print(f'\n--- {pt} ---')
    mask = adata.obs[PRED_TYPE_COL] == pt
    logcpm_sub = logcpm_full[mask]
    obs_sub = obs_full[mask.values]
    print(f'  Cells: {len(obs_sub)}  |  Condition: {obs_sub[CONDITION_COL].value_counts().to_dict()}')

    expr_frac = (logcpm_sub > 0).mean(axis=0)
    gene_mask = expr_frac >= MIN_EXPR_FRAC
    if TEST_N_GENES:
        gene_indices = np.where(gene_mask)[0][:TEST_N_GENES]
        genes = adata.var_names[gene_indices].tolist()
        logcpm_in = logcpm_sub[:, gene_indices]
    else:
        genes = adata.var_names[gene_mask].tolist()
        logcpm_in = logcpm_sub[:, gene_mask]
    print(f'  Genes after filtering: {len(genes)}')
    res = run_lmm(logcpm_in, genes, obs_sub)
    save_results(res, label)

print('\nDone.')
