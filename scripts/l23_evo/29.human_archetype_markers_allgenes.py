"""Redefine Jorstad23 human L2/3 IT archetype markers over the FULL gene universe.

Script 25 computed archetype markers by Wilcoxon over only the 2000 HVGs, which makes
a biased, under-powered background for downstream enrichment (script 28). This script
reproduces script 25's archetype/top-cell assignment EXACTLY, but runs the conservative
one-vs-each Wilcoxon over the proper universe: all *expressed & non-uniform* genes
(nonzero total expression and nonzero variance across the L2/3 cells) — the human analog
of the mouse `(X.sum>0) & (X.var>0)` rule.

The archetype labels archetype_1..4 are reproducible because `pcha` orders archetypes by
PC1; this is validated against the saved 25.human_pcha_aa.tsv (fail-fast).

Reads:
  local_data/res/l23_evo/05.varimax_coords.tsv
  local_data/res/l23_evo/25.human_pcha_aa.tsv          (validation only)
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  local_data/res/l23_evo/29.human_gene_universe.tsv            (expressed & non-uniform genes)
  local_data/res/l23_evo/29.human_archetype_markers_allgenes.tsv
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from scomics.main import SCA

# --- file paths ---
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
IN_VARIMAX  = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
IN_PCHA_AA  = os.path.join(OUT_RES_DIR, '25.human_pcha_aa.tsv')
INPUT_HUMAN = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_UNIVERSE = os.path.join(OUT_RES_DIR, '29.human_gene_universe.tsv')
OUT_MARKERS  = os.path.join(OUT_RES_DIR, '29.human_archetype_markers_allgenes.tsv')

# --- parameters (identical to script 25) ---
CLUSTER_COL    = 'WithinArea_cluster'
VX_COLS        = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
NOC            = 4
NDIM           = len(VX_COLS) - 1   # 5
N_TOP_CELLS    = 300
FRAC_IN_THRESH = 0.25
FDR_THRESH     = 0.001
LOG2FC_THRESH  = np.log2(1.5)
AA_CORR_MIN    = 0.99               # per-archetype validation vs saved PCHA (fail-fast)

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- fit PCHA exactly as script 25 ---
vx_df = pd.read_csv(IN_VARIMAX, sep='\t', index_col=0)
xn    = vx_df[VX_COLS].values
types = vx_df[CLUSTER_COL].values

print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)

# validate archetype labeling matches script 25 (same hull, same PC1 ordering)
aa_now   = sca.aa.T  # (NOC, NDIM) in PCHA space
aa_saved = pd.read_csv(IN_PCHA_AA, sep='\t', index_col=0).values  # (NOC, NDIM)
for k in range(NOC):
    r = np.corrcoef(aa_now[k], aa_saved[k])[0, 1]
    assert r > AA_CORR_MIN, (
        f'archetype_{k+1} PCHA coords diverge from 25.human_pcha_aa.tsv (corr={r:.3f} '
        f'< {AA_CORR_MIN}); refit did not reproduce script 25 archetypes')
print(f'PCHA archetypes reproduce script 25 (per-archetype corr > {AA_CORR_MIN}).')

# back-project archetype coords to VX space, pick top cells (identical to script 25)
aa_vx     = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_   # (NOC, len(VX_COLS))
dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(NOC)], axis=1)
top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(NOC)]
print(f'Top {N_TOP_CELLS} cells selected per archetype.')

# --- load full expression, define expressed & non-uniform gene universe ---
print('Loading h5ad...')
adata = ad.read_h5ad(INPUT_HUMAN)
gene_names = (adata.var['feature_name'].values
              if 'feature_name' in adata.var.columns else adata.var_names.values)
assert len(set(gene_names)) == len(gene_names), 'feature_name has duplicate symbols'

X = adata.X  # log-normalized, sparse (cells x genes)
gsum = np.asarray(X.sum(axis=0)).ravel()
gmean = np.asarray(X.mean(axis=0)).ravel()
gsq = np.asarray(X.multiply(X).mean(axis=0)).ravel() if sp.issparse(X) else (X ** 2).mean(axis=0)
gvar = gsq - gmean ** 2
keep = (gsum > 0) & (gvar > 1e-12)
uni_idx = np.where(keep)[0]
uni_names = gene_names[uni_idx]
print(f'gene universe: {keep.sum()} expressed & non-uniform genes '
      f'(dropped {(~keep).sum()} of {keep.size})')
pd.DataFrame({'gene': uni_names}).to_csv(OUT_UNIVERSE, sep='\t', index=False)
print(f'Saved {OUT_UNIVERSE}')

# densify universe genes for the top cells only (memory-light: ~1200 x 20k)
Xs = X[:, uni_idx].tocsr() if sp.issparse(X) else X[:, uni_idx]
X_top = [np.asarray(Xs[top_cells[k]].todense() if sp.issparse(Xs) else Xs[top_cells[k]],
                    dtype=np.float32) for k in range(NOC)]
n_uni = len(uni_names)

# --- conservative one-vs-each Wilcoxon per archetype (identical method to script 25) ---
print('Running conservative one-vs-each Wilcoxon per archetype over the full universe...')
all_markers = []
for k in range(NOC):
    archetype_label = f'archetype_{k + 1}'
    print(f'  {archetype_label}...')
    X_in    = X_top[k]
    frac_in = (X_in > 0).mean(axis=0)

    log2fc_mat, pval_mat, frac_out_list = [], [], []
    for j in range(NOC):
        if j == k:
            continue
        X_out = X_top[j]
        log2fc_j = (X_in.mean(axis=0) - X_out.mean(axis=0)) / np.log(2)
        pvals_j  = np.array([scipy.stats.ranksums(X_in[:, g], X_out[:, g])[1]
                             for g in range(n_uni)])
        log2fc_mat.append(log2fc_j)
        pval_mat.append(pvals_j)
        frac_out_list.append((X_out > 0).mean(axis=0))

    # worst-case across pairwise comparisons
    log2fc   = np.stack(log2fc_mat).min(axis=0)
    pvals    = np.stack(pval_mat).max(axis=0)
    frac_out = np.stack(frac_out_list).mean(axis=0)
    pvals    = np.nan_to_num(pvals, nan=1.0)  # all-zero gene pairs -> p=1

    _, fdr, _, _ = multipletests(pvals, method='fdr_bh')

    df = pd.DataFrame({
        'gene': uni_names, 'archetype': archetype_label,
        'log2FC': log2fc, 'pval': pvals, 'fdr': fdr,
        'frac_in': frac_in, 'frac_out': frac_out,
    })
    df = df[(df['frac_in'] >= FRAC_IN_THRESH) & (df['fdr'] < FDR_THRESH) & (df['log2FC'] > LOG2FC_THRESH)]
    df = df.sort_values('log2FC', ascending=False)
    all_markers.append(df)
    print(f'    {len(df)} markers after filtering')

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(OUT_MARKERS, sep='\t', index=False)
print(f'Saved {OUT_MARKERS}  ({len(markers_df)} total markers)')
