"""NR-vs-DR DEG analysis in 4 equal-size VX3 bins — PSEUDOBULK (v3 prototype).

Motivation
----------
v2 (cell-level LMM, hardened) showed that in VX3 bins 1-3 ~78% of genes hit a
statsmodels MixedLM variance-boundary degeneracy (random-intercept variance pinned
at 0 + exploded fixed-effect SE), even though effect sizes, cell counts, and
between-mouse variance (ICC ~0.016) are matched across all four bins. That is a
numerical pathology of the per-gene mixed model, not a biological difference.

Pseudobulk sidesteps it entirely: aggregate raw counts per (sample x bin), giving
one profile per mouse per bin (7 NR + 8 DR mice). The NR-vs-DR contrast is then a
simple two-group comparison at the mouse level — the correct unit of replication —
with no per-gene random-effect boundary to hit.

No DESeq2 / edgeR / rpy2 / R are available in this env, so this prototype uses two
simple, dependency-free tests on log2(CPM+1) pseudobulk values:
  - Welch's t-test  (unequal variance)  -> primary
  - Mann-Whitney U / Wilcoxon rank-sum  -> robustness check
Both are reported; significance is called on the t-test. (For a publication-grade
result, prefer pseudobulk + DESeq2/edgeR with dispersion shrinkage.)

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/26.combined_labels.parquet
  local_data/res/astro/41.cheng22_nr_harmony_arch12.h5ad
Outputs:
  local_data/res/astro/45.v3.deg_vx3_bin{1..4}_NR_vs_DR_all.tsv
  local_data/res/astro/45.v3.deg_vx3_bin{1..4}_NR_vs_DR_sig.tsv
  local_data/res/astro/45.v3.deg_vx3_bins_nr_vs_dr_pseudobulk.log
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.utils import norm

INPUT_H5AD         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_COMBINED_LABELS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
IN_NR_H5AD         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.cheng22_nr_harmony_arch12.h5ad')
OUT_DIR            = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
LOG_FILE           = os.path.join(OUT_DIR, '45.v3.deg_vx3_bins_nr_vs_dr_pseudobulk.log')

LABELED_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
MIN_CELLS     = 50
N_BINS        = 4
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(1.5)

# --- pseudobulk parameters ---
MIN_PB_CELLS    = 10    # drop a (sample x bin) pseudobulk profile with fewer cells
MIN_DET_FRAC    = 0.5   # keep a gene if detected (count>0) in >= this frac of samples in BOTH groups
CPM_SCALE       = 1e6

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('deg_vx3_v3')

# =============================================================================
# Step 1 — recompute bins (identical pipeline to scripts 45 / 45.v2)
# =============================================================================
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

log.info(f'Loading arch labels from {IN_COMBINED_LABELS}')
df_combined = pd.read_parquet(IN_COMBINED_LABELS)
labels_c22  = df_combined[df_combined['dataset'] == 'cheng22'].reset_index(drop=True)

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
gene_names = np.array(adata.var_names)

# adata.X is log1p(CP10k) (used only for the VX embedding/binning). Pseudobulk must
# aggregate RAW counts (counts are additive; log-normalized values are not), which
# live in adata.raw. CPM + log2 normalization is then applied at the pseudobulk level.
assert adata.raw is not None, 'adata.raw (raw counts) is required for pseudobulk'
raw_counts = adata.raw[:, adata.var_names].X
raw_counts = raw_counts.toarray() if sp.issparse(raw_counts) else np.asarray(raw_counts)
raw_counts = np.rint(raw_counts).astype(np.int64)   # already integer; rint guards float dtype
log.info(f'  raw counts from adata.raw: shape {raw_counts.shape}, max {raw_counts.max()}, '
         f'per-cell median depth {int(np.median(raw_counts.sum(axis=1)))}')

hvg_col_idx = np.array([np.where(adata.var_names == g)[0][0] for g in hvg_names])

log.info('Normalizing all cells...')
xn_all = norm(x[:, hvg_col_idx], depths)

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

log.info('Regressing library size out of DR cells...')
xn_dr = xn_all[dr_idx_in_adata].copy()
log_depth_dr = np.log(depths[dr_idx_in_adata]).reshape(-1, 1)
reg_dr = LinearRegression().fit(log_depth_dr, xn_dr)
xn_dr = xn_dr - reg_dr.predict(log_depth_dr)

log.info('Projecting DR cells onto NR VX space...')
vx_scores_dr = (xn_dr - pca_mean) @ vx_load

n_nr = len(vx_scores_nr)
n_dr = len(vx_scores_dr)
vx3_combined    = np.concatenate([vx_scores_nr[:, 2], vx_scores_dr[:, 2]])
condition_code  = np.array([0] * n_nr + [1] * n_dr)
sample_combined = np.concatenate([samples_nr, samples_dr])
combined_idx    = np.concatenate([nr_idx_in_adata, dr_idx_in_adata])

log.info(f'Binning into {N_BINS} equal-size bins within each sample...')
bin_labels = np.full(len(vx3_combined), -1, dtype=int)
for sample in np.unique(sample_combined):
    mask = sample_combined == sample
    bin_labels[mask] = pd.qcut(vx3_combined[mask], q=N_BINS, labels=False, duplicates='drop')

# raw counts for the combined NR+DR cells (all genes)
x_comb = raw_counts[combined_idx]   # raw integer counts for pseudobulk aggregation

# map each sample to its condition (a sample is entirely NR or entirely DR)
sample_to_cond = {}
for s, c in zip(sample_combined, condition_code):
    sample_to_cond[s] = c

# =============================================================================
# Step 2 — pseudobulk DEG per bin (Welch t-test + Wilcoxon rank-sum)
# =============================================================================
for b in range(N_BINS):
    log.info(f'=== Bin {b + 1}/{N_BINS} ===')
    bmask = bin_labels == b

    # build pseudobulk: one summed-count profile per sample present in this bin
    pb_counts, pb_cond, pb_names, pb_ncells = [], [], [], []
    for s in np.unique(sample_combined[bmask]):
        smask = bmask & (sample_combined == s)
        ncell = int(smask.sum())
        if ncell < MIN_PB_CELLS:
            log.info(f'    dropping pseudobulk {s} (only {ncell} cells)')
            continue
        pb_counts.append(x_comb[smask].sum(axis=0))
        pb_cond.append(sample_to_cond[s])
        pb_names.append(s)
        pb_ncells.append(ncell)
    pb_counts = np.vstack(pb_counts)                 # (n_samples, n_genes)
    pb_cond   = np.array(pb_cond)
    nr_rows   = pb_cond == 0
    dr_rows   = pb_cond == 1
    n_nr_s, n_dr_s = int(nr_rows.sum()), int(dr_rows.sum())
    log.info(f'  pseudobulk samples: NR={n_nr_s}, DR={n_dr_s}; total cells={sum(pb_ncells)}')

    # CPM + log2(CPM+1) per pseudobulk sample
    lib = pb_counts.sum(axis=1, keepdims=True)
    logcpm = np.log2(pb_counts / lib * CPM_SCALE + 1.0)   # (n_samples, n_genes)

    # gene filter: detected in >= MIN_DET_FRAC of samples in BOTH groups
    det_nr = (pb_counts[nr_rows] > 0).mean(axis=0)
    det_dr = (pb_counts[dr_rows] > 0).mean(axis=0)
    gene_mask = (det_nr >= MIN_DET_FRAC) & (det_dr >= MIN_DET_FRAC)
    genes_b = gene_names[gene_mask]
    A = logcpm[nr_rows][:, gene_mask]    # (n_nr_s, n_genes_kept)
    B = logcpm[dr_rows][:, gene_mask]    # (n_dr_s, n_genes_kept)
    log.info(f'  genes after pseudobulk detection filter: {gene_mask.sum()} / {len(gene_names)}')

    mean_nr = A.mean(axis=0)
    mean_dr = B.mean(axis=0)
    log2fc  = mean_dr - mean_nr          # + = up in DR

    # Welch t-test (DR vs NR) — vectorized across genes
    t_res = ss.ttest_ind(B, A, axis=0, equal_var=False)
    pval_t = np.asarray(t_res.pvalue, dtype=np.float64)
    # Mann-Whitney U / Wilcoxon rank-sum — vectorized across genes
    w_res = ss.mannwhitneyu(B, A, axis=0, alternative='two-sided')
    pval_w = np.asarray(w_res.pvalue, dtype=np.float64)

    res_df = pd.DataFrame({
        'gene': genes_b,
        'log2FC': log2fc,
        'mean_NR': mean_nr,
        'mean_DR': mean_dr,
        'pval_t': pval_t,
        'pval_w': pval_w,
        'n_NR': n_nr_s,
        'n_DR': n_dr_s,
    })

    for src, dst in [('pval_t', 'fdr_t'), ('pval_w', 'fdr_w')]:
        res_df[dst] = np.nan
        valid = res_df[src].notna()
        if valid.sum() > 0:
            _, fdr, _, _ = multipletests(res_df.loc[valid, src], method='fdr_bh')
            res_df.loc[valid, dst] = fdr

    # significance called on the t-test
    sig = res_df[(res_df['fdr_t'] < FDR_THRESH) & (res_df['log2FC'].abs() > LOG2FC_THRESH)] \
        .sort_values('fdr_t').reset_index(drop=True)
    n_sig_w = int(((res_df['fdr_w'] < FDR_THRESH) & (res_df['log2FC'].abs() > LOG2FC_THRESH)).sum())
    log.info(f'  significant (t-test, FDR<{FDR_THRESH}, |log2FC|>log2(1.5)): {len(sig)} '
             f'(up DR={(sig.log2FC > 0).sum()}, up NR={(sig.log2FC < 0).sum()}); '
             f'Wilcoxon would call {n_sig_w}')

    cols = ['gene', 'log2FC', 'mean_NR', 'mean_DR', 'pval_t', 'fdr_t', 'pval_w', 'fdr_w', 'n_NR', 'n_DR']
    out_all = os.path.join(OUT_DIR, f'45.v3.deg_vx3_bin{b + 1}_NR_vs_DR_all.tsv')
    out_sig = os.path.join(OUT_DIR, f'45.v3.deg_vx3_bin{b + 1}_NR_vs_DR_sig.tsv')
    res_df[cols].sort_values('fdr_t').to_csv(out_all, sep='\t', index=False)
    sig[cols].to_csv(out_sig, sep='\t', index=False)
    log.info(f'  Saved {out_all}')
    log.info(f'  Saved {out_sig}')

log.info('Done.')
