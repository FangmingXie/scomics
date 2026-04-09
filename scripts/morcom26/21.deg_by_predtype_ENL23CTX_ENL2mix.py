import os
import logging
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

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
FDR_THRESH       = 0.05
LOG2FC_THRESH       = np.log2(1.5)  # minimum |log2FC| for significance
PREFILT_LOG2FC_FRAC = 0.5           # pre-filter: skip genes with naive |log2FC| < LOG2FC_THRESH * this
MIN_EXPR_FRAC    = 0.1    # min fraction of cells expressing gene within a sample
MIN_SAMPLE_FRAC  = 0.5    # min fraction of samples that must pass MIN_EXPR_FRAC
MIN_SIGMA        = 1e-2   # min per-gene std (logCPM); near-constant genes are excluded
TEST_N_GENES     = None  # set to None to disable test mode
LMM_METHODS      = ['lbfgs', 'bfgs', 'cg', 'nm']  # fallback methods if convergence fails

os.makedirs(OUT_DIR, exist_ok=True)


def make_gene_mask(logcpm, sample_series):
    """Return (gene_mask, sigma) after filtering by per-sample expression fraction and sigma.

    Keeps genes expressed (>0) in >=MIN_EXPR_FRAC cells in >=MIN_SAMPLE_FRAC of samples,
    and with per-gene std >= MIN_SIGMA. Logs per-sample expression fractions.
    """
    sigma = logcpm.std(axis=0)  # (n_genes,) in logCPM space

    samples = sample_series.unique()
    n_samples = len(samples)
    per_sample_fracs = []
    for s in samples:
        mask = (sample_series == s).values
        frac = (logcpm[mask] > 0).mean(axis=0)  # shape (n_genes,)
        per_sample_fracs.append(frac)
        log.info('  sample %s: mean expr_frac=%.3f  n_cells=%d', s, frac.mean(), mask.sum())
    per_sample_fracs = np.stack(per_sample_fracs)  # (n_samples, n_genes)
    n_passing = (per_sample_fracs >= MIN_EXPR_FRAC).sum(axis=0)
    min_samples = int(np.ceil(n_samples * MIN_SAMPLE_FRAC))
    gene_mask = (n_passing >= min_samples) & (sigma >= MIN_SIGMA)
    return gene_mask, sigma[gene_mask]


def run_lmm(logcpm, genes, sigma, obs):
    """Run LMM (WT vs Null) for each gene; return DataFrame with coef, pval, log2FC, fdr.

    Expects pre-computed sigma (per-gene std in logCPM space) from make_gene_mask.
    Genes are z-scored before fitting; coef is rescaled by sigma to recover log2FC.
    Genes with naive |log2FC| < LOG2FC_THRESH * PREFILT_LOG2FC_FRAC are skipped.
    """
    null_mask = obs['condition_code'].values == 1
    naive_log2fc = (logcpm[null_mask].mean(0) - logcpm[~null_mask].mean(0)) / np.log(2)
    prefilt_mask = np.abs(naive_log2fc) >= LOG2FC_THRESH * PREFILT_LOG2FC_FRAC
    n_before = len(genes)
    genes  = [g for g, keep in zip(genes, prefilt_mask) if keep]
    logcpm = logcpm[:, prefilt_mask]
    sigma  = sigma[prefilt_mask]
    log.info('Pre-filter by naive log2FC: %d -> %d genes', n_before, len(genes))

    logcpm_z = (logcpm - logcpm.mean(axis=0)) / sigma

    results = []
    df = pd.concat([obs.reset_index(drop=True),
                    pd.DataFrame(logcpm_z, columns=genes)], axis=1)
    for i, gene in enumerate(tqdm(genes, desc='LMM', unit='gene')):
        try:
            model = smf.mixedlm(f'Q("{gene}") ~ condition_code', data=df, groups=df['sample'])
            fit = None
            for method in LMM_METHODS:
                fit = model.fit(reml=True, method=method, disp=False)
                if fit.converged:
                    break
                log.debug('%s did not converge with %s, trying next method', gene, method)
            coef = fit.params['condition_code']
            pval = fit.pvalues['condition_code']
        except Exception as e:
            log.warning('%s failed: %s', gene, e)
            coef, pval = np.nan, np.nan
        results.append({'gene': gene, 'coef': coef, 'pval': pval, 'sigma': sigma[i]})

    res_df = pd.DataFrame(results)
    res_df['log2FC'] = res_df['coef'] * res_df['sigma'] / np.log(2)
    valid = res_df['pval'].notna()
    _, fdr, _, _ = multipletests(res_df.loc[valid, 'pval'], method='fdr_bh')
    res_df['fdr'] = np.nan
    res_df.loc[valid, 'fdr'] = fdr
    return res_df[['gene', 'log2FC', 'sigma', 'pval', 'fdr']]


def save_results(res_df, label):
    out_all = OUT_FILE_ALL_TMPL.format(label=label)
    out_sig = OUT_FILE_SIG_TMPL.format(label=label)
    res_df.sort_values('fdr').to_csv(out_all, sep='\t', index=False)
    sig = res_df[(res_df['fdr'] < FDR_THRESH) & (res_df['log2FC'].abs() > LOG2FC_THRESH)].sort_values('fdr')
    sig.to_csv(out_sig, sep='\t', index=False)
    log.info('Significant genes (FDR<%s): %d', FDR_THRESH, len(sig))
    log.info('Saved -> %s', out_all)
    log.info('Saved -> %s', out_sig)


# --- load ---
log.info('Loading %s', INPUT_FILE)
adata = ad.read_h5ad(INPUT_FILE)
log.info('Full dataset: %d cells x %d genes', adata.shape[0], adata.shape[1])

log.info('Loading label transfer: %s', LABEL_TRANSFER_FILE)
lt = pd.read_csv(LABEL_TRANSFER_FILE, sep='\t', index_col=0)

# --- filter to EN-L2-3-CTX , then to L2/3 pred_types ---
mask_ct = adata.obs[CELLTYPE_COL].isin(KEEP_CELLTYPES)
adata = adata[mask_ct].copy()
log.info('After celltype filter: %d cells', adata.shape[0])

# align label transfer to adata
common_cells = adata.obs_names.intersection(lt.index)
adata = adata[common_cells].copy()
lt = lt.loc[common_cells]

adata.obs[PRED_TYPE_COL] = lt[PRED_TYPE_COL].values

mask_pt = adata.obs[PRED_TYPE_COL].isin(PRED_TYPES)
adata = adata[mask_pt].copy()
log.info('After pred_type filter: %d cells', adata.shape[0])
log.info('Condition counts: %s', adata.obs[CONDITION_COL].value_counts().to_dict())

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
log.info('--- Overall (all L2/3 subtypes combined) ---')
gene_mask, sigma = make_gene_mask(logcpm_full, obs_full['sample'])
if TEST_N_GENES:
    gene_indices = np.where(gene_mask)[0][:TEST_N_GENES]
    genes = adata.var_names[gene_indices].tolist()
    logcpm_in = logcpm_full[:, gene_indices]
    sigma_in = sigma[:TEST_N_GENES]
else:
    genes = adata.var_names[gene_mask].tolist()
    logcpm_in = logcpm_full[:, gene_mask]
    sigma_in = sigma
log.info('Genes after filtering: %d', len(genes))
log.info('Cells: %d', len(obs_full))
res = run_lmm(logcpm_in, genes, sigma_in, obs_full)
save_results(res, 'overall')

# =============================================================================
# PER PRED_TYPE
# =============================================================================
for pt in PRED_TYPES:
    label = pt.replace('/', '')  # e.g. L2/3_A -> L23_A
    log.info('--- %s ---', pt)
    mask = adata.obs[PRED_TYPE_COL] == pt
    logcpm_sub = logcpm_full[mask]
    obs_sub = obs_full[mask.values]
    log.info('Cells: %d  |  Condition: %s', len(obs_sub), obs_sub[CONDITION_COL].value_counts().to_dict())

    gene_mask, sigma = make_gene_mask(logcpm_sub, obs_sub['sample'])
    if TEST_N_GENES:
        gene_indices = np.where(gene_mask)[0][:TEST_N_GENES]
        genes = adata.var_names[gene_indices].tolist()
        logcpm_in = logcpm_sub[:, gene_indices]
        sigma_in = sigma[:TEST_N_GENES]
    else:
        genes = adata.var_names[gene_mask].tolist()
        logcpm_in = logcpm_sub[:, gene_mask]
        sigma_in = sigma
    log.info('Genes after filtering: %d', len(genes))
    res = run_lmm(logcpm_in, genes, sigma_in, obs_sub)
    save_results(res, label)

log.info('Done.')
