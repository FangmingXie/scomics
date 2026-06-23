"""NR-vs-DR DEG analysis in 4 equal-size VX3 bins.

Cells (NR + DR) are binned into 4 equal-size quantile bins along VX3
independently within each sample, then an LMM (expr ~ NR/DR + (1|sample))
is run per bin to detect spatially graded visual-dependent gene changes.

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/26.combined_labels.parquet
  local_data/res/astro/41.cheng22_nr_harmony_arch12.h5ad
Outputs:
  local_data/res/astro/45.deg_vx3_bin{1..4}_NR_vs_DR_all.tsv
  local_data/res/astro/45.deg_vx3_bin{1..4}_NR_vs_DR_sig.tsv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.utils import norm

INPUT_H5AD         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_COMBINED_LABELS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
IN_NR_H5AD         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.cheng22_nr_harmony_arch12.h5ad')
OUT_DIR            = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')

LABELED_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
MIN_CELLS     = 50
N_BINS        = 4
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(1.5)
MIN_EXPR_FRAC = 0.1

os.makedirs(OUT_DIR, exist_ok=True)

# --- load NR embedding (script 41) ---
print(f'Loading NR embedding from {IN_NR_H5AD}')
adata_nr = ad.read_h5ad(IN_NR_H5AD)
nr_barcodes  = adata_nr.obs_names.values
vx_load      = np.array(adata_nr.varm['VX_loadings'], dtype=np.float64)
hvg_names    = adata_nr.var_names.values
vx_scores_nr = np.array(adata_nr.obsm['X_vx'], dtype=np.float64)
pca_mean     = np.array(adata_nr.X.toarray() if hasattr(adata_nr.X, 'toarray')
                        else adata_nr.X, dtype=np.float64).mean(axis=0)
samples_nr   = adata_nr.obs['Sample'].values
print(f'  {len(nr_barcodes)} NR cells, {len(hvg_names)} HVGs, {vx_load.shape[1]} VX dims')

# --- load arch labels ---
print(f'Loading arch labels from {IN_COMBINED_LABELS}')
df_combined = pd.read_parquet(IN_COMBINED_LABELS)
labels_c22  = df_combined[df_combined['dataset'] == 'cheng22'].reset_index(drop=True)

# --- load raw data ---
print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
print(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()

ages = adata.obs['Age'].values
age_counts = pd.Series(ages).value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
ages = adata.obs['Age'].values
print(f'  {adata.shape[0]} cells after MT + age filter')

x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float64)
depths = x.sum(axis=1)
gene_names = adata.var_names.tolist()

hvg_col_idx = np.array([np.where(adata.var_names == g)[0][0] for g in hvg_names])

# --- normalize ---
print('Normalizing all cells...')
xn_all = norm(x[:, hvg_col_idx], depths)

# --- identify NR and DR indices ---
labeled_mask = np.isin(ages, LABELED_AGES)
assert len(labels_c22) == labeled_mask.sum()

arch_labels_all = labels_c22['archetype'].values
labeled_idx = np.where(labeled_mask)[0]

all_barcodes = adata.obs_names.values
barcode_to_idx = {b: i for i, b in enumerate(all_barcodes)}
nr_idx_in_adata = np.array([barcode_to_idx[b] for b in nr_barcodes])

dr_mask_in_labeled = np.isin(ages[labeled_mask], DR_AGES)
dr_idx_in_adata    = labeled_idx[dr_mask_in_labeled]
samples_dr         = adata.obs['Sample'].values[dr_idx_in_adata]
print(f'  NR cells: {len(nr_idx_in_adata)}, DR cells: {len(dr_idx_in_adata)}')

# --- regress library size from DR cells ---
print('Regressing library size out of DR cells...')
xn_dr = xn_all[dr_idx_in_adata].copy()
log_depth_dr = np.log(depths[dr_idx_in_adata]).reshape(-1, 1)
reg_dr = LinearRegression().fit(log_depth_dr, xn_dr)
xn_dr = xn_dr - reg_dr.predict(log_depth_dr)

# --- project DR cells onto NR VX space ---
print('Projecting DR cells onto NR VX space...')
vx_scores_dr = (xn_dr - pca_mean) @ vx_load

# --- build combined arrays ---
n_nr = len(vx_scores_nr)
n_dr = len(vx_scores_dr)
vx3_combined    = np.concatenate([vx_scores_nr[:, 2], vx_scores_dr[:, 2]])
condition_code  = np.array([0] * n_nr + [1] * n_dr)
sample_combined = np.concatenate([samples_nr, samples_dr])
combined_idx    = np.concatenate([nr_idx_in_adata, dr_idx_in_adata])

# --- bin within each sample ---
print(f'Binning into {N_BINS} equal-size bins within each sample...')
bin_labels = np.full(len(vx3_combined), -1, dtype=int)
for sample in np.unique(sample_combined):
    mask = sample_combined == sample
    bin_labels[mask] = pd.qcut(vx3_combined[mask], q=N_BINS, labels=False, duplicates='drop')

for b in range(N_BINS):
    bmask = bin_labels == b
    nr_b = (bmask & (condition_code == 0)).sum()
    dr_b = (bmask & (condition_code == 1)).sum()
    vx3_range = (vx3_combined[bmask].min(), vx3_combined[bmask].max())
    print(f'  Bin {b+1}: {bmask.sum()} cells (NR={nr_b}, DR={dr_b}), VX3 [{vx3_range[0]:.3f}, {vx3_range[1]:.3f}]')

# --- CP10k + log1p for all combined cells ---
x_comb       = x[combined_idx]
depths_comb  = depths[combined_idx]
counts_per_cell = depths_comb.reshape(-1, 1)
logcpm_all   = np.log1p(x_comb / counts_per_cell * 1e4).astype(np.float32)

# --- DEG loop per bin ---
for b in range(N_BINS):
    bmask    = bin_labels == b
    bin_idx  = np.where(bmask)[0]
    n_cells  = len(bin_idx)
    print(f'\n=== Bin {b+1}/{N_BINS}: {n_cells} cells ===')

    logcpm_bin = logcpm_all[bin_idx]
    obs_bin = pd.DataFrame({
        'condition_code': condition_code[bin_idx],
        'sample': sample_combined[bin_idx],
    })

    # gene filter
    expr_frac = (logcpm_bin > 0).mean(axis=0)
    gene_mask = expr_frac >= MIN_EXPR_FRAC
    filtered_genes = [g for g, m in zip(gene_names, gene_mask) if m]
    logcpm_filt = logcpm_bin[:, gene_mask]
    n_genes = len(filtered_genes)
    print(f'  Genes after filtering (>={MIN_EXPR_FRAC:.0%} expressed): {n_genes}')

    # LMM loop
    print('  Running LMM...')
    results = []
    for i, gene in enumerate(filtered_genes):
        if i % 500 == 0:
            print(f'    {i}/{n_genes}', flush=True)
        df_gene = obs_bin.copy()
        df_gene['expr'] = logcpm_filt[:, i]
        coef, pval = np.nan, np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for method in ['lbfgs', 'powell', 'nm']:
                try:
                    fit = smf.mixedlm(
                        'expr ~ condition_code', data=df_gene, groups=df_gene['sample']
                    ).fit(reml=True, method=method, disp=False)
                    c = fit.params['condition_code']
                    p = fit.pvalues['condition_code']
                    if not (np.isnan(c) or np.isnan(p)):
                        coef, pval = c, p
                        break
                except Exception:
                    continue
        results.append({'gene': gene, 'coef': coef, 'pval': pval})
    print(f'    {n_genes}/{n_genes} done')

    res_df = pd.DataFrame(results)
    res_df['log2FC'] = res_df['coef'] / np.log(2)
    valid = res_df['pval'].notna()
    _, fdr, _, _ = multipletests(res_df.loc[valid, 'pval'], method='fdr_bh')
    res_df['fdr'] = np.nan
    res_df.loc[valid, 'fdr'] = fdr

    sig = res_df[(res_df['fdr'] < FDR_THRESH) & (res_df['log2FC'].abs() > LOG2FC_THRESH)].sort_values('fdr').reset_index(drop=True)
    print(f'  Significant genes (FDR<{FDR_THRESH}, |log2FC|>log2(1.5)): {len(sig)}')

    out_all = os.path.join(OUT_DIR, f'45.deg_vx3_bin{b+1}_NR_vs_DR_all.tsv')
    out_sig = os.path.join(OUT_DIR, f'45.deg_vx3_bin{b+1}_NR_vs_DR_sig.tsv')
    res_df[['gene', 'log2FC', 'pval', 'fdr']].sort_values('fdr').to_csv(out_all, sep='\t', index=False)
    sig[['gene', 'log2FC', 'pval', 'fdr']].to_csv(out_sig, sep='\t', index=False)
    print(f'  Saved {out_all}')
    print(f'  Saved {out_sig}')

print('\nDone.')
