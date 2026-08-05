"""Heatmap of top NR-vs-DR DEGs (rows) × per (sample × VX1 bin) pseudobulks (cols).

Rows: for each of the 4 VX1 bins, the top TOP_N significant DEGs by FDR (both
directions; significant = FDR < FDR_THRESH & |log2FC_shrink| > LOG2FC_THRESH) from the
45.v6 DESeq2 tables, unioned across bins and de-duplicated (a gene keeps the lowest bin
it was selected in). Rows are grouped by that source bin.

Columns: one pseudobulk per (sample, VX1 bin) over the same Arch1-4 NR + all-DR cells as
the 45.v6 DEG pipeline. Expression = log1p(CP10k) of the summed raw counts, then z-scored
PER GENE across all columns. Columns are grouped by bin, and within each bin ordered NR
then DR by age (P28, P38, P28_dr, P38_dr). No clustering.

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/41.v3.cheng22_nr_harmony_arch1234.h5ad
  local_data/res/astro/45.v6.deg_vx1_bin{1..4}_NR_vs_DR_all.tsv
Outputs:
  local_data/fig/astro/45.v6.heatmap_topdeg_vx1_bins_samplebin_pseudobulk.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.utils import norm

INPUT_H5AD   = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_NR_H5AD   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.v3.cheng22_nr_harmony_arch1234.h5ad')
DEG_ALL_TMPL = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '45.v6.deg_vx1_bin{b}_NR_vs_DR_all.tsv')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_PDF      = os.path.join(FIG_DIR, '45.v6.heatmap_topdeg_vx1_bins_samplebin_pseudobulk.pdf')

LABELED_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
MIN_CELLS     = 50
N_BINS        = 4
MIN_PB_CELLS  = 10             # drop a (sample x bin) pseudobulk with fewer cells (matches DEG)

# --- row (gene) selection thresholds ---
TOP_N         = 20             # top DEGs per bin (by FDR), both directions
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(1.5)

# --- column ordering: NR ages first, then DR ages ---
AGE_ORDER = {'P28': 0, 'P38': 1, 'P28_dr': 2, 'P38_dr': 3}

COLOR_NR = '#1f77b4'
COLOR_DR = '#d62728'
VMIN, VMAX = -2.5, 2.5         # z-score color clip

os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# Step 1 — recompute the 4 VX1 bins (identical pipeline to script 45.v6)
# =============================================================================
print(f'Loading NR embedding from {IN_NR_H5AD}')
adata_nr = ad.read_h5ad(IN_NR_H5AD)
nr_barcodes  = adata_nr.obs_names.values
vx_load      = np.array(adata_nr.varm['VX_loadings'], dtype=np.float64)
hvg_names    = adata_nr.var_names.values
vx_scores_nr = np.array(adata_nr.obsm['X_vx'], dtype=np.float64)
pca_mean     = np.array(adata_nr.X.toarray() if hasattr(adata_nr.X, 'toarray')
                        else adata_nr.X, dtype=np.float64).mean(axis=0)
samples_nr   = adata_nr.obs['Sample'].values

print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)

mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()

ages = adata.obs['Age'].values
age_counts = pd.Series(ages).value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
ages = adata.obs['Age'].values

x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float64)
depths = x.sum(axis=1)
gene_names = np.array(adata.var_names)

assert adata.raw is not None, 'adata.raw (raw counts) is required'
raw_counts = adata.raw[:, adata.var_names].X
raw_counts = raw_counts.toarray() if sp.issparse(raw_counts) else np.asarray(raw_counts)
raw_counts = np.rint(raw_counts).astype(np.int64)

hvg_col_idx = np.array([np.where(adata.var_names == g)[0][0] for g in hvg_names])
xn_all = norm(x[:, hvg_col_idx], depths)

labeled_mask = np.isin(ages, LABELED_AGES)
labeled_idx  = np.where(labeled_mask)[0]

barcode_to_idx  = {b: i for i, b in enumerate(adata.obs_names.values)}
nr_idx_in_adata = np.array([barcode_to_idx[b] for b in nr_barcodes])

dr_mask_in_labeled = np.isin(ages[labeled_mask], DR_AGES)
dr_idx_in_adata    = labeled_idx[dr_mask_in_labeled]

# regress library size out of DR cells, then project onto the NR VX space
xn_dr = xn_all[dr_idx_in_adata].copy()
log_depth_dr = np.log(depths[dr_idx_in_adata]).reshape(-1, 1)
reg_dr = LinearRegression().fit(log_depth_dr, xn_dr)
xn_dr = xn_dr - reg_dr.predict(log_depth_dr)
vx_scores_dr = (xn_dr - pca_mean) @ vx_load

n_nr = len(vx_scores_nr)
n_dr = len(vx_scores_dr)
vx1_combined    = np.concatenate([vx_scores_nr[:, 0], vx_scores_dr[:, 0]])
condition_code  = np.array([0] * n_nr + [1] * n_dr)
sample_combined = np.concatenate([samples_nr, adata.obs['Sample'].values[dr_idx_in_adata]])
combined_idx    = np.concatenate([nr_idx_in_adata, dr_idx_in_adata])
age_combined    = ages[combined_idx]

bin_labels = np.full(len(vx1_combined), -1, dtype=int)
for sample in np.unique(sample_combined):
    mask = sample_combined == sample
    bin_labels[mask] = pd.qcut(vx1_combined[mask], q=N_BINS, labels=False, duplicates='drop')

# raw counts for the combined NR+DR cells (all genes), for pseudobulk aggregation
raw_comb = raw_counts[combined_idx]

# =============================================================================
# Step 2 — select rows: top TOP_N significant DEGs by FDR per bin (union, dedup)
# =============================================================================
gene_to_bin = {}   # gene -> source bin (lowest bin it was selected in), preserves order
for b in range(N_BINS):
    deg = pd.read_csv(DEG_ALL_TMPL.format(b=b + 1), sep='\t')
    sig = deg[(deg['fdr'] < FDR_THRESH) & (deg['fdr'].notna())
              & (deg['log2FC'].abs() > LOG2FC_THRESH)]
    top = sig.nsmallest(TOP_N, 'fdr')
    for g in top['gene']:
        if g not in gene_to_bin:          # keep the lowest bin; group rows by it
            gene_to_bin[g] = b
    print(f'  Bin {b + 1}: {len(sig)} significant, took top {len(top)}')

# order rows by source bin, then by FDR within that source bin
row_bin_of = dict(gene_to_bin)
row_genes = []
for b in range(N_BINS):
    deg = pd.read_csv(DEG_ALL_TMPL.format(b=b + 1), sep='\t').set_index('gene')
    genes_b = [g for g, bb in row_bin_of.items() if bb == b]
    row_genes.extend(sorted(genes_b, key=lambda g: deg.loc[g, 'fdr']))
print(f'  Total unique DEG rows: {len(row_genes)}')

gene_col_of = {g: np.where(gene_names == g)[0][0] for g in row_genes}
assert all(g in gene_col_of for g in row_genes), 'selected DEG missing from gene_names'
row_gene_idx = np.array([gene_col_of[g] for g in row_genes])

# =============================================================================
# Step 3 — build (sample x bin) pseudobulk columns: log1p(CP10k), ordered
# =============================================================================
cond_of_sample = {s: ('DR' if c == 1 else 'NR') for s, c in zip(sample_combined, condition_code)}
age_of_sample  = dict(zip(sample_combined, age_combined))
uniq_samples   = np.unique(sample_combined)
ordered_samples = sorted(
    uniq_samples,
    key=lambda s: (0 if cond_of_sample[s] == 'NR' else 1, AGE_ORDER[age_of_sample[s]], s),
)

col_ids, col_sample, col_bin, col_cond = [], [], [], []
expr_cols = []   # each: log1p(CP10k) over the selected genes
for b in range(N_BINS):
    for s in ordered_samples:
        cmask = (sample_combined == s) & (bin_labels == b)
        if int(cmask.sum()) < MIN_PB_CELLS:
            continue
        pb = raw_comb[cmask][:, row_gene_idx].sum(axis=0).astype(np.float64)
        total = raw_comb[cmask].sum()                     # total counts over ALL genes
        cp10k = pb / total * 1e4
        expr_cols.append(np.log1p(cp10k))                 # ln(1 + CP10k)
        col_ids.append(f'{s}|b{b + 1}')
        col_sample.append(s)
        col_bin.append(b)
        col_cond.append(cond_of_sample[s])

expr = np.vstack(expr_cols).T                             # genes x columns
print(f'  Pseudobulk columns: {expr.shape[1]} (of max {N_BINS * len(ordered_samples)})')

# --- z-score per gene across all columns ---
mu = expr.mean(axis=1, keepdims=True)
sd = expr.std(axis=1, keepdims=True)
zero_var = (sd[:, 0] == 0)
if zero_var.any():
    print(f'  WARNING: {int(zero_var.sum())} genes have zero variance across columns -> set to 0')
sd[sd == 0] = 1.0
z = (expr - mu) / sd

data = pd.DataFrame(z, index=row_genes, columns=col_ids)

# =============================================================================
# Step 4 — heatmap (rows grouped by source bin, cols grouped by bin then NR/DR)
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
bin_palette = sns.color_palette('Set2', N_BINS)
bin_color   = {b: bin_palette[b] for b in range(N_BINS)}
cond_color  = {'NR': COLOR_NR, 'DR': COLOR_DR}

col_colors = pd.DataFrame({
    'VX1 bin':   [bin_color[b] for b in col_bin],
    'condition': [cond_color[c] for c in col_cond],
}, index=col_ids)
row_colors = pd.Series([bin_color[row_bin_of[g]] for g in row_genes],
                       index=row_genes, name='VX1 bin')

g = sns.clustermap(
    data, row_cluster=False, col_cluster=False,
    cmap='RdBu_r', center=0, vmin=VMIN, vmax=VMAX,
    col_colors=col_colors, row_colors=row_colors,
    xticklabels=col_sample, yticklabels=True,
    figsize=(0.22 * data.shape[1] + 6, 0.16 * data.shape[0] + 4),
    cbar_kws={'label': 'z-score  log1p(CP10k)'},
    dendrogram_ratio=(0.01, 0.01), colors_ratio=(0.012, 0.02),
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=6, rotation=90)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
g.ax_heatmap.set_xlabel('sample (grouped by VX1 bin, then NR/DR by age)')
g.ax_heatmap.set_ylabel('top DEGs (grouped by source VX1 bin)')

# legends for the bin and condition color strips
bin_handles  = [mpatches.Patch(color=bin_color[b], label=f'Bin {b + 1}') for b in range(N_BINS)]
cond_handles = [mpatches.Patch(color=cond_color[c], label=c) for c in ['NR', 'DR']]
leg1 = g.ax_heatmap.legend(handles=bin_handles, title='VX1 bin', frameon=False,
                           loc='upper left', bbox_to_anchor=(1.06, 1.0), fontsize=7, title_fontsize=8)
g.ax_heatmap.add_artist(leg1)
g.ax_heatmap.legend(handles=cond_handles, title='condition', frameon=False,
                    loc='upper left', bbox_to_anchor=(1.06, 0.80), fontsize=7, title_fontsize=8)

g.fig.suptitle(f'Top {TOP_N}/bin NR-vs-DR DEGs (FDR<{FDR_THRESH}, |log2FC|>log2(1.5)) '
               f'× (sample × VX1 bin) pseudobulk, z-scored per gene', y=1.01, fontsize=11)
g.savefig(OUT_PDF, bbox_inches='tight')
plt.close(g.fig)
print(f'Saved {OUT_PDF}')
print('Done.')
