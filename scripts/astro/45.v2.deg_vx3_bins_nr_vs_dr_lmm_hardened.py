"""NR-vs-DR DEG analysis in 4 equal-size VX3 bins — HARDENED cell-level LMM.

Same design as 45.deg_vx3_bins_nr_vs_dr.py (cells binned into 4 equal-size
quantile bins along VX3 within each sample; LMM expr ~ NR/DR + (1|sample) per
bin), but with robustness hardening so that unidentifiable / degenerate per-gene
fits no longer leak into the results as huge-fold-change, p~=1 (or sign-flipped)
"hits". There are 7 NR and 8 DR mice, so condition is a well-replicated
between-subjects factor; the random intercept by sample is the correct model.
The instability seen in v1 came from per-gene singular/non-converged fits, which
this script detects and excludes.

Hardening vs v1:
  1. Stricter expression filter (MIN_EXPR_FRAC 0.10 -> 0.25) to drop ultra-sparse
     genes whose coefficients are wild and unstable.
  2. Require the LMM to actually converge (prefer a converged optimizer; flag if
     none converge) instead of taking the first method that returns a number.
  3. Flag unstable fits: huge fixed-effect SE (non-identifiable, p~=1) and
     singular random-effect variance (collapsed to OLS -> pseudoreplication,
     anticonservative p).
  4. Flag sign mismatch between the LMM coefficient and the raw NR-vs-DR mean
     difference (the v1 Rfx4 bin-1/bin-3 artifact).
  5. Report raw per-condition means and raw log2FC alongside the model log2FC.
  6. Compute BH-FDR only over the trustworthy ('ok') gene set per bin.

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/26.combined_labels.parquet
  local_data/res/astro/41.cheng22_nr_harmony_arch12.h5ad
Outputs:
  local_data/res/astro/45.v2.deg_vx3_bin{1..4}_NR_vs_DR_all.tsv
  local_data/res/astro/45.v2.deg_vx3_bin{1..4}_NR_vs_DR_sig.tsv
"""

import os
import io
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import statsmodels.formula.api as smf
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.utils import norm

INPUT_H5AD         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_COMBINED_LABELS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
IN_NR_H5AD         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.cheng22_nr_harmony_arch12.h5ad')
OUT_DIR            = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
LOG_FILE           = os.path.join(OUT_DIR, '45.v2.deg_vx3_bins_nr_vs_dr_lmm_hardened.log')

LABELED_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
MIN_CELLS     = 50
N_BINS        = 4
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(1.5)

# --- hardening parameters ---
MIN_EXPR_FRAC   = 0.25    # stricter than v1 (0.10); drop ultra-sparse genes
MAX_SE          = 3.0     # fixed-effect SE (log units) above this => non-identifiable
RE_VAR_REL_MIN  = 1e-3    # random-effect var / residual var below this => singular RE
SIGN_EPS        = 0.05    # ignore sign mismatch when raw effect is negligible (log2 units)

os.makedirs(OUT_DIR, exist_ok=True)

# --- logging: write everything to console AND a log file ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('deg_vx3_v2')


class TqdmToLogger(io.StringIO):
    """File-like object that redirects tqdm bar updates into the logger,
    so the progress bar also lands in the log file (as periodic lines)."""

    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger, self.level, self.buf = logger, level, ''

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf)


tqdm_out = TqdmToLogger(log)
log.info(f'Logging to {LOG_FILE}')

# --- load NR embedding (script 41) ---
log.info(f'Loading NR embedding from {IN_NR_H5AD}')
adata_nr = ad.read_h5ad(IN_NR_H5AD)
nr_barcodes  = adata_nr.obs_names.values
vx_load      = np.array(adata_nr.varm['VX_loadings'], dtype=np.float64)
hvg_names    = adata_nr.var_names.values
vx_scores_nr = np.array(adata_nr.obsm['X_vx'], dtype=np.float64)
pca_mean     = np.array(adata_nr.X.toarray() if hasattr(adata_nr.X, 'toarray')
                        else adata_nr.X, dtype=np.float64).mean(axis=0)
samples_nr   = adata_nr.obs['Sample'].values
log.info(f'  {len(nr_barcodes)} NR cells, {len(hvg_names)} HVGs, {vx_load.shape[1]} VX dims')

# --- load arch labels ---
log.info(f'Loading arch labels from {IN_COMBINED_LABELS}')
df_combined = pd.read_parquet(IN_COMBINED_LABELS)
labels_c22  = df_combined[df_combined['dataset'] == 'cheng22'].reset_index(drop=True)

# --- load raw data ---
log.info(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
log.info(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()

ages = adata.obs['Age'].values
age_counts = pd.Series(ages).value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
ages = adata.obs['Age'].values
log.info(f'  {adata.shape[0]} cells after MT + age filter')

x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float64)
depths = x.sum(axis=1)
gene_names = adata.var_names.tolist()

hvg_col_idx = np.array([np.where(adata.var_names == g)[0][0] for g in hvg_names])

# --- normalize ---
log.info('Normalizing all cells...')
xn_all = norm(x[:, hvg_col_idx], depths)

# --- identify NR and DR indices ---
labeled_mask = np.isin(ages, LABELED_AGES)
assert len(labels_c22) == labeled_mask.sum()
labeled_idx = np.where(labeled_mask)[0]

all_barcodes = adata.obs_names.values
barcode_to_idx = {b: i for i, b in enumerate(all_barcodes)}
nr_idx_in_adata = np.array([barcode_to_idx[b] for b in nr_barcodes])

dr_mask_in_labeled = np.isin(ages[labeled_mask], DR_AGES)
dr_idx_in_adata    = labeled_idx[dr_mask_in_labeled]
samples_dr         = adata.obs['Sample'].values[dr_idx_in_adata]
log.info(f'  NR cells: {len(nr_idx_in_adata)}, DR cells: {len(dr_idx_in_adata)}')

# --- regress library size from DR cells ---
log.info('Regressing library size out of DR cells...')
xn_dr = xn_all[dr_idx_in_adata].copy()
log_depth_dr = np.log(depths[dr_idx_in_adata]).reshape(-1, 1)
reg_dr = LinearRegression().fit(log_depth_dr, xn_dr)
xn_dr = xn_dr - reg_dr.predict(log_depth_dr)

# --- project DR cells onto NR VX space ---
log.info('Projecting DR cells onto NR VX space...')
vx_scores_dr = (xn_dr - pca_mean) @ vx_load

# --- build combined arrays ---
n_nr = len(vx_scores_nr)
n_dr = len(vx_scores_dr)
vx3_combined    = np.concatenate([vx_scores_nr[:, 2], vx_scores_dr[:, 2]])
condition_code  = np.array([0] * n_nr + [1] * n_dr)
sample_combined = np.concatenate([samples_nr, samples_dr])
combined_idx    = np.concatenate([nr_idx_in_adata, dr_idx_in_adata])

# --- bin within each sample ---
log.info(f'Binning into {N_BINS} equal-size bins within each sample...')
bin_labels = np.full(len(vx3_combined), -1, dtype=int)
for sample in np.unique(sample_combined):
    mask = sample_combined == sample
    bin_labels[mask] = pd.qcut(vx3_combined[mask], q=N_BINS, labels=False, duplicates='drop')

for b in range(N_BINS):
    bmask = bin_labels == b
    nr_b = (bmask & (condition_code == 0)).sum()
    dr_b = (bmask & (condition_code == 1)).sum()
    vx3_range = (vx3_combined[bmask].min(), vx3_combined[bmask].max())
    log.info(f'  Bin {b+1}: {bmask.sum()} cells (NR={nr_b}, DR={dr_b}), VX3 [{vx3_range[0]:.3f}, {vx3_range[1]:.3f}]')

# --- CP10k + log1p for all combined cells ---
x_comb       = x[combined_idx]
depths_comb  = depths[combined_idx]
counts_per_cell = depths_comb.reshape(-1, 1)
logcpm_all   = np.log1p(x_comb / counts_per_cell * 1e4).astype(np.float32)


def fit_lmm_hardened(df_gene):
    """Fit expr ~ condition_code + (1|sample); prefer a converged optimizer.

    Returns dict(coef, se, pval, re_var, scale, converged) or None if no
    optimizer produced a finite fixed-effect estimate.
    """
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for method in ['lbfgs', 'powell', 'nm']:
            try:
                fit = smf.mixedlm(
                    'expr ~ condition_code', data=df_gene, groups=df_gene['sample']
                ).fit(reml=True, method=method, disp=False)
            except Exception:
                continue
            coef = fit.params.get('condition_code', np.nan)
            se   = fit.bse.get('condition_code', np.nan)
            pval = fit.pvalues.get('condition_code', np.nan)
            if not np.isfinite(coef) or not np.isfinite(se) or not np.isfinite(pval):
                continue
            rec = dict(
                coef=float(coef), se=float(se), pval=float(pval),
                re_var=float(fit.cov_re.iloc[0, 0]), scale=float(fit.scale),
                converged=bool(getattr(fit, 'converged', False)),
            )
            if rec['converged']:
                return rec            # prefer a genuinely converged fit
            if best is None:
                best = rec            # fall back to first finite-but-unconverged fit
    return best


def classify(rec, log2fc, raw_log2fc):
    """Assign a robustness status to a finished LMM fit."""
    if rec is None:
        return 'no_fit'
    if not rec['converged']:
        return 'not_converged'
    if not np.isfinite(rec['se']) or rec['se'] > MAX_SE:
        return 'unstable_se'
    re_rel = rec['re_var'] / rec['scale'] if rec['scale'] > 0 else 0.0
    if re_rel < RE_VAR_REL_MIN:
        return 'singular_re'
    if abs(raw_log2fc) > SIGN_EPS and np.sign(log2fc) != np.sign(raw_log2fc):
        return 'sign_mismatch'
    return 'ok'


# --- DEG loop per bin ---
for b in range(N_BINS):
    bmask    = bin_labels == b
    bin_idx  = np.where(bmask)[0]
    n_cells  = len(bin_idx)
    log.info(f'=== Bin {b+1}/{N_BINS}: {n_cells} cells ===')

    logcpm_bin = logcpm_all[bin_idx]
    cond_bin   = condition_code[bin_idx]
    obs_bin = pd.DataFrame({
        'condition_code': cond_bin,
        'sample': sample_combined[bin_idx],
    })

    # gene filter
    expr_frac = (logcpm_bin > 0).mean(axis=0)
    gene_mask = expr_frac >= MIN_EXPR_FRAC
    filtered_genes = [g for g, m in zip(gene_names, gene_mask) if m]
    logcpm_filt = logcpm_bin[:, gene_mask]
    n_genes = len(filtered_genes)
    log.info(f'  Genes after filtering (>={MIN_EXPR_FRAC:.0%} expressed): {n_genes}')

    # raw per-condition means (vectorized)
    nr_cells = cond_bin == 0
    dr_cells = cond_bin == 1
    mean_nr_all = logcpm_filt[nr_cells].mean(axis=0)
    mean_dr_all = logcpm_filt[dr_cells].mean(axis=0)
    raw_log2fc_all = (mean_dr_all - mean_nr_all) / np.log(2)

    # LMM loop
    log.info('  Running hardened LMM...')
    results = []
    for i, gene in enumerate(tqdm(filtered_genes, desc=f'Bin {b+1} LMM', unit='gene',
                                  file=tqdm_out, mininterval=30.0)):
        df_gene = obs_bin.copy()
        df_gene['expr'] = logcpm_filt[:, i]
        rec = fit_lmm_hardened(df_gene)
        raw_log2fc = float(raw_log2fc_all[i])
        log2fc = rec['coef'] / np.log(2) if rec is not None else np.nan
        status = classify(rec, log2fc, raw_log2fc)
        results.append({
            'gene': gene,
            'log2FC': log2fc,
            'raw_log2FC': raw_log2fc,
            'mean_NR': float(mean_nr_all[i]),
            'mean_DR': float(mean_dr_all[i]),
            'pval': rec['pval'] if rec is not None else np.nan,
            'se': rec['se'] if rec is not None else np.nan,
            're_var_rel': (rec['re_var'] / rec['scale']) if (rec is not None and rec['scale'] > 0) else np.nan,
            'converged': bool(rec['converged']) if rec is not None else False,
            'status': status,
        })
    log.info(f'    {n_genes}/{n_genes} done')

    res_df = pd.DataFrame(results)

    # BH-FDR computed ONLY over trustworthy ('ok') genes
    ok = res_df['status'] == 'ok'
    res_df['fdr'] = np.nan
    if ok.sum() > 0:
        _, fdr, _, _ = multipletests(res_df.loc[ok, 'pval'], method='fdr_bh')
        res_df.loc[ok, 'fdr'] = fdr

    status_counts = res_df['status'].value_counts().to_dict()
    log.info(f'  Status: {status_counts}')

    sig = res_df[ok & (res_df['fdr'] < FDR_THRESH) & (res_df['log2FC'].abs() > LOG2FC_THRESH)] \
        .sort_values('fdr').reset_index(drop=True)
    log.info(f'  Significant ok genes (FDR<{FDR_THRESH}, |log2FC|>log2(1.5)): {len(sig)}')

    cols = ['gene', 'log2FC', 'raw_log2FC', 'mean_NR', 'mean_DR',
            'pval', 'fdr', 'se', 're_var_rel', 'converged', 'status']
    out_all = os.path.join(OUT_DIR, f'45.v2.deg_vx3_bin{b+1}_NR_vs_DR_all.tsv')
    out_sig = os.path.join(OUT_DIR, f'45.v2.deg_vx3_bin{b+1}_NR_vs_DR_sig.tsv')
    res_df[cols].sort_values('fdr').to_csv(out_all, sep='\t', index=False)
    sig[cols].to_csv(out_sig, sep='\t', index=False)
    log.info(f'  Saved {out_all}')
    log.info(f'  Saved {out_sig}')

log.info('Done.')
