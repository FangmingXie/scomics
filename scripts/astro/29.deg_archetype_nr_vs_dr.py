# DEG analysis per archetype: NR vs DR in cheng22 cells.
# Uses archetype labels from 26.combined_labels.parquet (script 26 output).
# LMM: expr ~ condition_code + (1|sample), where NR=0, DR=1.
# P28_dl cells are excluded. One TSV per archetype (all genes + sig genes).

import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# --- file paths ---
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARQUET_COMBINED  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
INPUT_CHENG22     = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
OUT_DIR           = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')

# --- config ---
CHENG22_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']  # same subset as script 26
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
FDR_THRESH    = 0.05
MIN_EXPR_FRAC = 0.1
MAX_GENES     = None  # set to None for full run

os.makedirs(OUT_DIR, exist_ok=True)

# --- load combined parquet and extract cheng22 rows ---
print('Loading combined parquet...')
df_combined = pd.read_parquet(PARQUET_COMBINED)
df_cheng22_meta = df_combined[df_combined['dataset'] == 'cheng22'].copy()
print(f'  cheng22 cells in parquet: {len(df_cheng22_meta)}')

# --- load cheng22 h5ad (same subset as script 26) ---
print(f'Loading {INPUT_CHENG22}...')
adata_full = ad.read_h5ad(INPUT_CHENG22)
adata_cheng22 = adata_full[adata_full.obs['Age'].isin(CHENG22_AGES)].copy()
print(f'  cheng22 cells in h5ad: {adata_cheng22.shape[0]}')

if len(df_cheng22_meta) != adata_cheng22.shape[0]:
    raise ValueError(
        f'Cell count mismatch: parquet has {len(df_cheng22_meta)}, h5ad has {adata_cheng22.shape[0]}'
    )

# --- assign cell barcodes to parquet rows (positional alignment with script 26) ---
df_cheng22_meta = df_cheng22_meta.copy()
df_cheng22_meta['cell_barcode'] = adata_cheng22.obs_names.values

# --- subset to NR/DR ages only (exclude P28_dl) ---
nr_dr_ages = NR_AGES + DR_AGES
keep = df_cheng22_meta['age'].isin(nr_dr_ages)
df_meta = df_cheng22_meta[keep].copy()
df_meta['condition_code'] = df_meta['age'].isin(DR_AGES).astype(int)
df_meta['sample'] = df_meta['donor_name']
print(f'  NR/DR cells: {len(df_meta)}  (NR={df_meta["condition_code"].eq(0).sum()}, DR={df_meta["condition_code"].eq(1).sum()})')

# --- extract raw counts for NR/DR cells ---
nr_dr_barcodes = df_meta['cell_barcode'].values
adata_nr_dr = adata_cheng22[nr_dr_barcodes].copy()

X = adata_nr_dr.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)

# CP10k + log1p normalization
counts_per_cell = X.sum(axis=1, keepdims=True)
logcpm = np.log1p(X / counts_per_cell * 1e4)
gene_names = adata_nr_dr.var_names.tolist()

print(f'  Genes: {len(gene_names)}')

# --- per-archetype DEG loop ---
archetypes = sorted(df_meta['archetype'].unique())
print(f'\nArchetypes: {archetypes}')

for arch in archetypes:
    arch_mask = df_meta['archetype'] == arch
    arch_idx  = np.where(arch_mask.values)[0]
    n_cells   = len(arch_idx)
    print(f'\n=== {arch}: {n_cells} cells ===')

    logcpm_arch = logcpm[arch_idx]
    obs_arch    = df_meta.iloc[arch_idx][['condition_code', 'sample']].reset_index(drop=True)

    # gene filter
    expr_frac = (logcpm_arch > 0).mean(axis=0)
    gene_mask = expr_frac >= MIN_EXPR_FRAC
    filtered_genes = [g for g, m in zip(gene_names, gene_mask) if m]
    logcpm_filt    = logcpm_arch[:, gene_mask]
    if MAX_GENES is not None:
        filtered_genes = filtered_genes[:MAX_GENES]
        logcpm_filt    = logcpm_filt[:, :MAX_GENES]
    n_genes = len(filtered_genes)
    print(f'  Genes after filtering (>={MIN_EXPR_FRAC:.0%} expressed): {n_genes}')

    # LMM loop
    print(f'  Running LMM...')
    results = []
    for i, gene in enumerate(filtered_genes):
        if i % 10 == 0:
            print(f'    {i}/{n_genes}', flush=True)
        df_gene = obs_arch.copy()
        df_gene['expr'] = logcpm_filt[:, i]
        coef, pval = np.nan, np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for method in ['lbfgs', 'powell', 'nm']:
                try:
                    fit = smf.mixedlm('expr ~ condition_code', data=df_gene, groups=df_gene['sample']).fit(
                        reml=True, method=method, disp=False)
                    c = fit.params['condition_code']
                    p = fit.pvalues['condition_code']
                    if not (np.isnan(c) or np.isnan(p)):
                        coef, pval = c, p
                        break
                except Exception:
                    continue
        results.append({'gene': gene, 'coef': coef, 'pval': pval})
    print(f'    {n_genes}/{n_genes} done')

    # FDR
    res_df = pd.DataFrame(results)
    res_df['log2FC'] = res_df['coef'] / np.log(2)
    valid = res_df['pval'].notna()
    _, fdr, _, _ = multipletests(res_df.loc[valid, 'pval'], method='fdr_bh')
    res_df['fdr'] = np.nan
    res_df.loc[valid, 'fdr'] = fdr

    sig = res_df[res_df['fdr'] < FDR_THRESH].sort_values('fdr').reset_index(drop=True)
    print(f'  Significant genes (FDR<{FDR_THRESH}): {len(sig)}')

    out_all = os.path.join(OUT_DIR, f'29.deg_{arch}_NR_vs_DR_all.tsv')
    out_sig = os.path.join(OUT_DIR, f'29.deg_{arch}_NR_vs_DR_sig.tsv')
    res_df[['gene', 'log2FC', 'pval', 'fdr']].sort_values('fdr').to_csv(out_all, sep='\t', index=False)
    sig[['gene', 'log2FC', 'pval', 'fdr']].to_csv(out_sig, sep='\t', index=False)
    print(f'  Saved {out_all}')
    print(f'  Saved {out_sig}')

print('\nDone.')
