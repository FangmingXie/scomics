"""Heatmap of top NR-vs-DR DEGs (rows) × per (sample × VX3 bin) pseudobulks (cols).

VX3 / Arch1+Arch2 counterpart of 45.v6.viz_heatmap_topdeg_vx1_bins.py.

Rows: the union of all genes that are significant up-regulated (log2FC > +LOG2FC_THRESH)
or down-regulated (log2FC < -LOG2FC_THRESH) NR-vs-DR DEGs (FDR < FDR_THRESH) in ANY VX3
bin, from the 45.v4 DESeq2 tables. Genes are organized in two blocks — up-regulated (up
in DR) then down-regulated (up in NR); a gene's direction is taken from the bin where it
is most significant. Within each block, rows are ordered by hierarchical clustering (ward
linkage, optimal leaf ordering) of the z-scored per (sample × bin) profiles. NOT by bin.

Columns: one pseudobulk per (sample, VX3 bin) over the same cells as the 45.v4 DEG
pipeline — the NR reference is Arch1+Arch2 only (41.cheng22_nr_harmony_arch12.h5ad); DR is
ALL DR-labeled astrocytes (all archetypes) projected onto that VX space. Expression =
log1p(CP10k) of the summed raw counts, then z-scored PER GENE across all columns. Columns
are grouped by condition (NR then DR), and within each condition ordered by VX3 bin (then
age: P28, P38 for NR; P28_dr, P38_dr for DR). No clustering.

Two panels share the rows and z-score scale: (A) the full per (sample × bin) matrix, and
(B) a simpler summary = the mean z across samples for each condition × bin (2 × 4 = 8
columns).

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/41.cheng22_nr_harmony_arch12.h5ad
  local_data/res/astro/45.v4.deg_vx3_bin{1..4}_NR_vs_DR_all.tsv
Outputs:
  local_data/fig/astro/45.v4.heatmap_topdeg_vx3_bins_samplebin_pseudobulk.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.utils import norm

INPUT_H5AD   = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_NR_H5AD   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.cheng22_nr_harmony_arch12.h5ad')
DEG_ALL_TMPL = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '45.v4.deg_vx3_bin{b}_NR_vs_DR_all.tsv')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_PDF      = os.path.join(FIG_DIR, '45.v4.heatmap_topdeg_vx3_bins_samplebin_pseudobulk.pdf')

LABELED_AGES  = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES       = ['P28', 'P38']
DR_AGES       = ['P28_dr', 'P38_dr']
MIN_CELLS     = 50
N_BINS        = 4
MIN_PB_CELLS  = 10             # drop a (sample x bin) pseudobulk with fewer cells (matches DEG)

# --- row (gene) selection thresholds (stringent union across bins) ---
FDR_THRESH    = 0.05
LOG2FC_THRESH = np.log2(3)     # up = log2FC > +log2(3), down = log2FC < -log2(3)
LOG2FC_LABEL  = 'log2(3)'      # for figure text

# --- column ordering: NR ages first, then DR ages ---
AGE_ORDER = {'P28': 0, 'P38': 1, 'P28_dr': 2, 'P38_dr': 3}

COLOR_NR = '#1f77b4'
COLOR_DR = '#d62728'
VMIN, VMAX = -2.5, 2.5         # z-score color clip

os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# Step 1 — recompute the 4 VX3 bins (identical pipeline to script 45.v4)
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
vx3_combined    = np.concatenate([vx_scores_nr[:, 2], vx_scores_dr[:, 2]])
condition_code  = np.array([0] * n_nr + [1] * n_dr)
sample_combined = np.concatenate([samples_nr, adata.obs['Sample'].values[dr_idx_in_adata]])
combined_idx    = np.concatenate([nr_idx_in_adata, dr_idx_in_adata])
age_combined    = ages[combined_idx]

bin_labels = np.full(len(vx3_combined), -1, dtype=int)
for sample in np.unique(sample_combined):
    mask = sample_combined == sample
    bin_labels[mask] = pd.qcut(vx3_combined[mask], q=N_BINS, labels=False, duplicates='drop')

# raw counts for the combined NR+DR cells (all genes), for pseudobulk aggregation
raw_comb = raw_counts[combined_idx]

# =============================================================================
# Step 2 — select rows: union of all genes significant (FDR < FDR_THRESH,
#          |log2FC| > LOG2FC_THRESH) in ANY bin; direction from the most
#          significant qualifying bin. Organize up- then down-regulated.
# =============================================================================
gene_dir = {}   # gene -> 'up' (up in DR) or 'dn' (up in NR)
gene_fdr = {}   # gene -> smallest qualifying FDR across bins (for ordering)
for b in range(N_BINS):
    deg = pd.read_csv(DEG_ALL_TMPL.format(b=b + 1), sep='\t')
    sig = deg[(deg['fdr'] < FDR_THRESH) & (deg['fdr'].notna())
              & (deg['log2FC'].abs() > LOG2FC_THRESH)]
    for g, lfc, fdr in zip(sig['gene'], sig['log2FC'], sig['fdr']):
        if g not in gene_fdr or fdr < gene_fdr[g]:
            gene_fdr[g] = fdr
            gene_dir[g] = 'up' if lfc > 0 else 'dn'
    print(f'  Bin {b + 1}: {len(sig)} significant '
          f'(up={int((sig["log2FC"] > 0).sum())}, down={int((sig["log2FC"] < 0).sum())})')

# organize: up-regulated block then down-regulated block (within-block order set by
# hierarchical clustering of the z-profiles below, in Step 3)
up_genes = sorted([g for g in gene_dir if gene_dir[g] == 'up'], key=lambda g: gene_fdr[g])
dn_genes = sorted([g for g in gene_dir if gene_dir[g] == 'dn'], key=lambda g: gene_fdr[g])
row_genes  = up_genes + dn_genes
row_dir_of = gene_dir
n_up = len(up_genes)
print(f'  Union DEG rows: {len(row_genes)} ({len(up_genes)} up, {len(dn_genes)} down)')

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

# column order: condition (NR then DR) -> VX3 bin -> age/sample
samples_by_cond = {c: [s for s in ordered_samples if cond_of_sample[s] == c]
                   for c in ['NR', 'DR']}
col_ids, col_sample, col_bin, col_cond = [], [], [], []
expr_cols = []   # each: log1p(CP10k) over the selected genes
for cond in ['NR', 'DR']:
    for b in range(N_BINS):
        for s in samples_by_cond[cond]:
            cmask = (sample_combined == s) & (bin_labels == b)
            if int(cmask.sum()) < MIN_PB_CELLS:
                continue
            pb = raw_comb[cmask][:, row_gene_idx].sum(axis=0).astype(np.float64)
            total = raw_comb[cmask].sum()                 # total counts over ALL genes
            cp10k = pb / total * 1e4
            expr_cols.append(np.log1p(cp10k))             # ln(1 + CP10k)
            col_ids.append(f'{s}|b{b + 1}')
            col_sample.append(s)
            col_bin.append(b)
            col_cond.append(cond)

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

# --- order rows within each direction block by hierarchical clustering of z-profiles ---
def cluster_leaf_order(mat):
    """Ward-linkage leaf order (optimal ordering) of the rows of `mat`; identity if < 3."""
    if mat.shape[0] < 3:
        return np.arange(mat.shape[0])
    return leaves_list(linkage(mat, method='ward', optimal_ordering=True))


up_slice, dn_slice = np.arange(n_up), np.arange(n_up, len(row_genes))
row_order = np.concatenate([up_slice[cluster_leaf_order(z[up_slice])],
                            dn_slice[cluster_leaf_order(z[dn_slice])]])
z = z[row_order]
row_genes = [row_genes[i] for i in row_order]
print('  Rows ordered by ward hierarchical clustering within up/down blocks')

# --- collapse to condition × bin means (mean z across samples): the simpler panel ---
col_bin_arr, col_cond_arr = np.array(col_bin), np.array(col_cond)
avg_cols, avg_bin, avg_cond, avg_labels = [], [], [], []
for cond in ['NR', 'DR']:
    for b in range(N_BINS):
        sel = (col_cond_arr == cond) & (col_bin_arr == b)
        if not sel.any():
            continue
        avg_cols.append(z[:, sel].mean(axis=1))
        avg_bin.append(b)
        avg_cond.append(cond)
        avg_labels.append(f'{cond} b{b + 1}')
zavg = np.vstack(avg_cols).T                              # genes x (condition × bin)
print(f'  Averaged columns (condition × bin): {zavg.shape[1]}')

# =============================================================================
# Step 4 — two-panel heatmap: (A) per sample × bin, (B) mean over samples per
#          condition × bin. Shared rows and z-score scale.
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
bin_palette = sns.color_palette('Set2', N_BINS)
bin_rgb  = {b: tuple(bin_palette[b]) for b in range(N_BINS)}
cond_rgb = {c: mcolors.to_rgb(col) for c, col in {'NR': COLOR_NR, 'DR': COLOR_DR}.items()}
dir_rgb  = {'up': mcolors.to_rgb(COLOR_DR), 'dn': mcolors.to_rgb(COLOR_NR)}  # up in DR / up in NR

# column annotation strips (row 0 = condition, row 1 = VX3 bin)
annA = np.stack([[cond_rgb[c] for c in col_cond], [bin_rgb[b] for b in col_bin]])
annB = np.stack([[cond_rgb[c] for c in avg_cond], [bin_rgb[b] for b in avg_bin]])
row_strip = np.array([[dir_rgb[row_dir_of[g]]] for g in row_genes])   # (nrows, 1, 3)

nrows, nA, nB = len(row_genes), z.shape[1], zavg.shape[1]
fig = plt.figure(figsize=(2.0 + 0.15 * nA + 0.35 * nB + 2.5, 0.17 * nrows + 1.8))
gs = fig.add_gridspec(2, 6, width_ratios=[0.5, nA, 1.6, nB * 2.4, 0.8, 0.35],
                      height_ratios=[1.5, nrows], hspace=0.02, wspace=0.06)

axA_ann = fig.add_subplot(gs[0, 1])
axB_ann = fig.add_subplot(gs[0, 3])
ax_row  = fig.add_subplot(gs[1, 0])
axA     = fig.add_subplot(gs[1, 1])
axB     = fig.add_subplot(gs[1, 3])
ax_cbar = fig.add_subplot(gs[1, 5])

# column annotations
for axn, ann in [(axA_ann, annA), (axB_ann, annB)]:
    axn.imshow(ann, aspect='auto')
    axn.set_xticks([])
    axn.set_yticks([])
axA_ann.set_yticks([0, 1])
axA_ann.set_yticklabels(['condition', 'VX3 bin'], fontsize=7)
axA_ann.tick_params(length=0)

# row (gene) color strip + gene labels
ax_row.imshow(row_strip, aspect='auto')
ax_row.set_xticks([])
ax_row.set_yticks(range(nrows))
ax_row.set_yticklabels(row_genes, fontsize=5)
ax_row.tick_params(length=0)
ax_row.set_ylabel('union DEGs (up-regulated then down-regulated)', fontsize=8)

# heatmaps
im = axA.imshow(z, aspect='auto', cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
axA.set_yticks([])
axA.set_xticks(range(nA))
axA.set_xticklabels(col_sample, fontsize=5, rotation=90)
axA.set_xlabel('sample (NR then DR, by VX3 bin)', fontsize=8)
axA.set_title('per sample × bin', fontsize=9)

# thin vertical lines at each (condition, bin) group boundary
for i in range(1, nA):
    if col_bin[i] != col_bin[i - 1] or col_cond[i] != col_cond[i - 1]:
        axA.axvline(i - 0.5, color='black', lw=0.5)

axB.imshow(zavg, aspect='auto', cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
axB.set_yticks([])
axB.set_xticks(range(nB))
axB.set_xticklabels(avg_labels, fontsize=7, rotation=90)
axB.set_xlabel('mean over samples', fontsize=8)
axB.set_title('condition × bin mean', fontsize=9)

fig.colorbar(im, cax=ax_cbar, label='z-score  log1p(CP10k)')

# legends: column bin & condition strips + row direction strip
bin_handles  = [mpatches.Patch(color=bin_rgb[b], label=f'Bin {b + 1}') for b in range(N_BINS)]
cond_handles = [mpatches.Patch(color=cond_rgb[c], label=c) for c in ['NR', 'DR']]
dir_handles  = [mpatches.Patch(color=dir_rgb['up'], label='up in DR'),
                mpatches.Patch(color=dir_rgb['dn'], label='up in NR')]
fig.legend(handles=bin_handles, title='VX3 bin (col)', frameon=False,
           loc='upper left', bbox_to_anchor=(0.99, 0.92), fontsize=7, title_fontsize=8)
fig.legend(handles=cond_handles, title='condition (col)', frameon=False,
           loc='upper left', bbox_to_anchor=(0.99, 0.68), fontsize=7, title_fontsize=8)
fig.legend(handles=dir_handles, title='DEG direction (row)', frameon=False,
           loc='upper left', bbox_to_anchor=(0.99, 0.50), fontsize=7, title_fontsize=8)

fig.suptitle(f'Union up/down NR-vs-DR DEGs (FDR<{FDR_THRESH}, |log2FC|>{LOG2FC_LABEL}) '
             f'× VX3-bin pseudobulk, z-scored per gene', y=0.99, fontsize=11)
fig.savefig(OUT_PDF, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT_PDF}')
print('Done.')
