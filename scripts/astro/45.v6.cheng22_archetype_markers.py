"""One-vs-rest archetype markers computed directly on cheng22 NR astrocytes.

Alternative to the P56/gao25 marker table (script 35): compute archetype-identity markers
on the ACTUAL cheng22 cells and archetype labels used downstream, so the numbering matches
and the markers are on-target. Uses the NR (P28/P38) Arch1-4 cells from the 41.v3 embedding
(the reference state; DR is the perturbation and is held out of the marker definition).

Method (same statistics as script 35): normalize raw counts to log2(CP10k + 1); for each
archetype k, one-vs-rest Wilcoxon (scipy ranksums) of that archetype's cells vs all other
NR cells; log2FC = mean_in - mean_out (log space); frac_in / frac_out = fraction expressing;
BH-FDR across tested genes. Only genes detected in >= FRAC_IN_MIN of the archetype's cells
are tested. The table is saved unfiltered by gene class (downstream viz applies any
technical-gene filter).

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/41.v3.cheng22_nr_harmony_arch1234.h5ad
Outputs:
  local_data/res/astro/45.v6.cheng22_archetype_markers.tsv
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_H5AD  = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_NR_H5AD  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.v3.cheng22_nr_harmony_arch1234.h5ad')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_MARKERS = os.path.join(OUT_RES_DIR, '45.v6.cheng22_archetype_markers.tsv')

N_ARCH       = 4
FRAC_IN_MIN  = 0.10    # only test genes detected in >= this fraction of the archetype's cells

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- NR cell set + archetype labels (from the 41.v3 embedding) ---
print(f'Loading NR cells from {IN_NR_H5AD}')
adata_nr = ad.read_h5ad(IN_NR_H5AD)
nr_barcodes = adata_nr.obs_names.values
nr_arch     = adata_nr.obs['archetype'].values
print(f'  {len(nr_barcodes)} NR cells; archetypes: {pd.Series(nr_arch).value_counts().to_dict()}')

# --- raw counts (all genes, mt removed) for those NR cells ---
print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
gene_names = np.array(adata.var_names)

assert adata.raw is not None, 'adata.raw (raw counts) is required'
raw = adata.raw[:, adata.var_names].X
raw = raw.toarray() if sp.issparse(raw) else np.asarray(raw)
raw = np.rint(raw).astype(np.float32)

barcode_to_idx = {b: i for i, b in enumerate(adata.obs_names.values)}
nr_idx = np.array([barcode_to_idx[b] for b in nr_barcodes])
raw_nr = raw[nr_idx]

# --- normalize: log2(CP10k + 1) ---
depths = raw_nr.sum(axis=1, keepdims=True)
depths = np.where(depths == 0, 1, depths)
Xnorm = np.log2(raw_nr / depths * 1e4 + 1)
n_genes = Xnorm.shape[1]
print(f'  Xnorm: {Xnorm.shape}')

# --- one-vs-rest Wilcoxon per archetype ---
all_markers = []
for k in range(1, N_ARCH + 1):
    arch = f'Arch{k}'
    in_mask = nr_arch == arch
    X_in, X_out = Xnorm[in_mask], Xnorm[~in_mask]
    log2fc   = X_in.mean(axis=0) - X_out.mean(axis=0)      # already log2
    frac_in  = (X_in > 0).mean(axis=0)
    frac_out = (X_out > 0).mean(axis=0)

    cand = np.where(frac_in >= FRAC_IN_MIN)[0]             # only test plausibly-expressed genes
    pvals = np.full(n_genes, np.nan)
    for g in cand:
        _, pvals[g] = scipy.stats.ranksums(X_in[:, g], X_out[:, g])
    fdr = np.full(n_genes, np.nan)
    fdr[cand] = multipletests(pvals[cand], method='fdr_bh')[1]

    df = pd.DataFrame({
        'gene':      gene_names[cand],
        'archetype': f'archetype_{k}',
        'log2FC':    log2fc[cand],
        'pval':      pvals[cand],
        'fdr':       fdr[cand],
        'frac_in':   frac_in[cand],
        'frac_out':  frac_out[cand],
    }).sort_values('log2FC', ascending=False)
    all_markers.append(df)
    n_sig = int(((df['fdr'] < 0.05) & (df['log2FC'] > 0)).sum())
    print(f'  {arch}: {len(cand)} genes tested, {n_sig} enriched significant (FDR<0.05, log2FC>0)')

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(OUT_MARKERS, sep='\t', index=False)
print(f'Saved {OUT_MARKERS}  ({len(markers_df)} rows)')
print('Done.')
