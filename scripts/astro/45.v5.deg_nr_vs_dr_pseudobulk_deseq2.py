"""NR-vs-DR DEG analysis — whole-population PSEUDOBULK + DESeq2 (v5).

Simplified counterpart of v4: no VX3 axis, no binning, all archetypes (Arch1-4).
Aggregate raw counts per SAMPLE (one pseudobulk profile per mouse, pooling all of
its cells regardless of archetype), then run a single DESeq2 (via pydeseq2) with
negative-binomial GLM, dispersion shrinkage, Wald test, Cook's/independent
filtering, and apeglm-style LFC shrinkage.

Why this over v4:
  - v4 projects cells onto a VX3 axis built from the Arch1+Arch2-only NR embedding
    (41.cheng22_nr_harmony_arch12.h5ad) and runs a separate DESeq2 per VX3 quantile
    bin. This v5 drops that machinery entirely: every NR/DR-age cell is an Arch1-4
    astrocyte, so one pseudobulk per sample across all cells gives a clean,
    whole-population dark-rearing DEG result.

Reads:
  links/astro/cheng22_astro.h5ad
Outputs:
  local_data/res/astro/45.v5.deg_nr_vs_dr_all.tsv
  local_data/res/astro/45.v5.deg_nr_vs_dr_sig.tsv
  local_data/res/astro/45.v5.deg_nr_vs_dr_pseudobulk_deseq2.log
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
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_H5AD = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
OUT_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_ALL    = os.path.join(OUT_DIR, '45.v5.deg_nr_vs_dr_all.tsv')
OUT_SIG    = os.path.join(OUT_DIR, '45.v5.deg_nr_vs_dr_sig.tsv')
LOG_FILE   = os.path.join(OUT_DIR, '45.v5.deg_nr_vs_dr_pseudobulk_deseq2.log')

NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
MIN_CELLS     = 50    # drop ages with fewer cells than this
MIN_PB_CELLS  = 10    # drop a sample's pseudobulk profile with fewer cells
MIN_GENE_SUM  = 10    # DESeq2 pre-filter: drop genes with total count < this
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(1.5)

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('deg_nr_vs_dr_v5')

# =============================================================================
# Step 1 — load, QC-filter, subset to NR/DR cells, pull raw counts
# =============================================================================
log.info(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
log.info(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()

ages = adata.obs['Age'].values
age_counts = pd.Series(ages).value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
log.info(f'  {adata.shape[0]} cells after MT + age filter')

# keep only NR/DR ages (all such cells are Arch1-4 astrocytes — all archetypes used)
nr_dr_mask = np.isin(adata.obs['Age'].values, NR_AGES + DR_AGES)
adata = adata[nr_dr_mask].copy()
ages = adata.obs['Age'].values
samples = adata.obs['Sample'].values
log.info(f'  {adata.shape[0]} NR/DR cells across {len(np.unique(samples))} samples')

# DESeq2 needs RAW counts (adata.X is normalized); pull them from adata.raw.
assert adata.raw is not None, 'adata.raw (raw counts) is required for DESeq2'
raw_counts = adata.raw[:, adata.var_names].X
raw_counts = raw_counts.toarray() if sp.issparse(raw_counts) else np.asarray(raw_counts)
raw_counts = np.rint(raw_counts).astype(np.int64)   # already integer; rint guards float dtype
gene_names = np.array(adata.var_names)
log.info(f'  raw counts from adata.raw: shape {raw_counts.shape}, max {raw_counts.max()}, '
         f'per-cell median depth {int(np.median(raw_counts.sum(axis=1)))}')

# map each sample -> condition by its age (each sample is wholly NR or DR)
sample_to_cond = {}
for s in np.unique(samples):
    s_ages = np.unique(ages[samples == s])
    is_dr = np.isin(s_ages, DR_AGES).all()
    is_nr = np.isin(s_ages, NR_AGES).all()
    assert is_dr ^ is_nr, f'sample {s} mixes NR/DR ages: {s_ages}'
    sample_to_cond[s] = 'DR' if is_dr else 'NR'

# =============================================================================
# Step 2 — pseudobulk per sample (pool all cells), single DESeq2 NR vs DR
# =============================================================================
pb_counts, pb_cond, pb_names = [], [], []
for s in np.unique(samples):
    smask = samples == s
    if int(smask.sum()) < MIN_PB_CELLS:
        log.info(f'  dropping pseudobulk {s} (only {int(smask.sum())} cells)')
        continue
    pb_counts.append(raw_counts[smask].sum(axis=0))
    pb_cond.append(sample_to_cond[s])
    pb_names.append(s)
pb_counts = np.vstack(pb_counts)
n_nr_s = pb_cond.count('NR')
n_dr_s = pb_cond.count('DR')
log.info(f'pseudobulk samples: NR={n_nr_s}, DR={n_dr_s}')

counts_df = pd.DataFrame(pb_counts, index=pb_names, columns=gene_names)
keep = counts_df.sum(axis=0) >= MIN_GENE_SUM   # DESeq2 pre-filter: drop low-count genes
counts_df = counts_df.loc[:, keep]
metadata = pd.DataFrame({'condition': pb_cond}, index=pb_names)
log.info(f'genes after low-count pre-filter (>= {MIN_GENE_SUM} total): '
         f'{counts_df.shape[1]} / {len(gene_names)}')

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
log.info(f'significant (DESeq2, FDR<{FDR_THRESH}, |log2FC_shrink|>log2(1.5)): {len(sig)} '
         f'(up DR={(sig.log2FC > 0).sum()}, up NR={(sig.log2FC < 0).sum()})')

cols = ['gene', 'log2FC', 'log2FC_mle', 'baseMean', 'lfcSE', 'stat', 'pval', 'fdr', 'n_NR', 'n_DR']
res_df.sort_values('fdr')[cols].to_csv(OUT_ALL, sep='\t', index=False)
sig[cols].to_csv(OUT_SIG, sep='\t', index=False)
log.info(f'Saved {OUT_ALL}')
log.info(f'Saved {OUT_SIG}')
log.info('Done.')
