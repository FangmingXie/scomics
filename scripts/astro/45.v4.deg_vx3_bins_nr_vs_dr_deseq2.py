"""NR-vs-DR DEG analysis in 4 equal-size VX3 bins — PSEUDOBULK + DESeq2 (v4).

The calibrated version of the v3 pseudobulk prototype. Aggregate raw counts per
(sample x bin) -> one profile per mouse per bin (7 NR + 8 DR), then run DESeq2
(via pydeseq2) per bin with negative-binomial GLM, dispersion shrinkage, Wald
test, Cook's/independent filtering, and apeglm-style LFC shrinkage.

Why this over v2/v3:
  - v2 (cell-level LMM): bins 1-3 collapsed via a statsmodels variance-boundary
    numerical artifact (SE ~1e6), not real biology.
  - v3 (pseudobulk + t-test/Wilcoxon): fixes identifiability but the unmoderated
    t-test is anticonservative at n=7/8 and Wilcoxon has ~zero power.
  - v4 (pseudobulk + DESeq2): moderates per-gene dispersion across genes ->
    calibrated p-values/FDR at low replicate count, and shrinks noisy LFCs.

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/26.combined_labels.parquet
  local_data/res/astro/41.cheng22_nr_harmony_arch12.h5ad
Outputs:
  local_data/res/astro/45.v4.deg_vx3_bin{1..4}_NR_vs_DR_all.tsv
  local_data/res/astro/45.v4.deg_vx3_bin{1..4}_NR_vs_DR_sig.tsv
  local_data/res/astro/45.v4.deg_vx3_bins_nr_vs_dr_deseq2.log
"""

import os
import io
import sys
import logging
import contextlib
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.utils import norm

INPUT_H5AD         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_COMBINED_LABELS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
IN_NR_H5AD         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.cheng22_nr_harmony_arch12.h5ad')
OUT_DIR            = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
LOG_FILE           = os.path.join(OUT_DIR, '45.v4.deg_vx3_bins_nr_vs_dr_deseq2.log')

LABELED_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
MIN_CELLS     = 50
N_BINS        = 4
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(1.5)

# --- pseudobulk / DESeq2 parameters ---
MIN_PB_CELLS  = 10    # drop a (sample x bin) pseudobulk profile with fewer cells
MIN_GENE_SUM  = 10    # DESeq2 pre-filter: drop genes with total count < this in the bin

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('deg_vx3_v4')

# =============================================================================
# Step 1 — recompute bins (identical pipeline to scripts 45 / 45.v2 / 45.v3)
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

# adata.X is NORMALIZED (used for the VX embedding/binning); DESeq2 needs RAW counts,
# which live in adata.raw. Pull raw counts aligned to the same cells and gene order.
assert adata.raw is not None, 'adata.raw (raw counts) is required for DESeq2'
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
sample_combined = np.concatenate([samples_nr, adata.obs['Sample'].values[dr_idx_in_adata]])
combined_idx    = np.concatenate([nr_idx_in_adata, dr_idx_in_adata])

log.info(f'Binning into {N_BINS} equal-size bins within each sample...')
bin_labels = np.full(len(vx3_combined), -1, dtype=int)
for sample in np.unique(sample_combined):
    mask = sample_combined == sample
    bin_labels[mask] = pd.qcut(vx3_combined[mask], q=N_BINS, labels=False, duplicates='drop')

# raw counts for combined NR+DR cells (all genes) — true counts from adata.raw
x_comb = raw_counts[combined_idx]

sample_to_cond = {s: ('DR' if c == 1 else 'NR') for s, c in zip(sample_combined, condition_code)}

# =============================================================================
# Step 2 — pseudobulk DESeq2 per bin
# =============================================================================
for b in range(N_BINS):
    log.info(f'=== Bin {b + 1}/{N_BINS} ===')
    bmask = bin_labels == b

    pb_counts, pb_cond, pb_names = [], [], []
    for s in np.unique(sample_combined[bmask]):
        smask = bmask & (sample_combined == s)
        if int(smask.sum()) < MIN_PB_CELLS:
            log.info(f'    dropping pseudobulk {s} (only {int(smask.sum())} cells)')
            continue
        pb_counts.append(x_comb[smask].sum(axis=0))
        pb_cond.append(sample_to_cond[s])
        pb_names.append(s)
    pb_counts = np.vstack(pb_counts)
    n_nr_s = pb_cond.count('NR')
    n_dr_s = pb_cond.count('DR')
    log.info(f'  pseudobulk samples: NR={n_nr_s}, DR={n_dr_s}')

    counts_df = pd.DataFrame(pb_counts, index=pb_names, columns=gene_names)
    # DESeq2 pre-filter: drop very low-count genes
    keep = counts_df.sum(axis=0) >= MIN_GENE_SUM
    counts_df = counts_df.loc[:, keep]
    metadata = pd.DataFrame({'condition': pb_cond}, index=pb_names)
    log.info(f'  genes after low-count pre-filter (>= {MIN_GENE_SUM} total): {counts_df.shape[1]} / {len(gene_names)}')

    # run DESeq2 (suppress pydeseq2's verbose stdout)
    with contextlib.redirect_stdout(io.StringIO()):
        dds = DeseqDataSet(
            counts=counts_df, metadata=metadata,
            design='~condition', ref_level=['condition', 'NR'],
        )
        dds.deseq2()
        ds = DeseqStats(dds, contrast=['condition', 'DR', 'NR'])
        ds.summary()
        res = ds.results_df.copy()                 # DR-vs-NR: MLE LFC + Wald p/padj
        log2fc_mle = res['log2FoldChange'].copy()
        # apeglm-style shrinkage on the fitted coefficient (whichever level is the
        # reference); align the sign to the DR-vs-NR contrast.
        try:
            ds.lfc_shrink(coeff='condition[T.DR]')
            log2fc_shrink = ds.results_df['log2FoldChange'].copy()
        except KeyError:
            ds.lfc_shrink(coeff='condition[T.NR]')
            log2fc_shrink = -ds.results_df['log2FoldChange'].copy()

    res_df = pd.DataFrame({
        'gene': res.index.values,
        'log2FC': log2fc_shrink.values,            # shrunk (primary, for ranking/threshold)
        'log2FC_mle': log2fc_mle.values,           # unshrunk MLE
        'baseMean': res['baseMean'].values,
        'lfcSE': res['lfcSE'].values,
        'stat': res['stat'].values,
        'pval': res['pvalue'].values,
        'fdr': res['padj'].values,                 # DESeq2 BH + independent filtering
        'n_NR': n_nr_s,
        'n_DR': n_dr_s,
    })

    sig = res_df[(res_df['fdr'] < FDR_THRESH) & (res_df['log2FC'].abs() > LOG2FC_THRESH)] \
        .sort_values('fdr').reset_index(drop=True)
    log.info(f'  significant (DESeq2, FDR<{FDR_THRESH}, |log2FC_shrink|>log2(1.5)): {len(sig)} '
             f'(up DR={(sig.log2FC > 0).sum()}, up NR={(sig.log2FC < 0).sum()})')

    cols = ['gene', 'log2FC', 'log2FC_mle', 'baseMean', 'lfcSE', 'stat', 'pval', 'fdr', 'n_NR', 'n_DR']
    out_all = os.path.join(OUT_DIR, f'45.v4.deg_vx3_bin{b + 1}_NR_vs_DR_all.tsv')
    out_sig = os.path.join(OUT_DIR, f'45.v4.deg_vx3_bin{b + 1}_NR_vs_DR_sig.tsv')
    res_df.sort_values('fdr')[cols].to_csv(out_all, sep='\t', index=False)
    sig[cols].to_csv(out_sig, sep='\t', index=False)
    log.info(f'  Saved {out_all}')
    log.info(f'  Saved {out_sig}')

log.info('Done.')
