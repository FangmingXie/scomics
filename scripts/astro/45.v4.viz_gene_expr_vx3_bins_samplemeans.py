"""Per-sample mean gene expression across the 4 VX3 bins, NR vs DR (matplotlib PDF).

Variant of 45.v4.viz_gene_expr_vx3_bins.py: instead of per-cell boxplots, each sample's
mean log2(CP10k+1) expression is a faint dot (alpha=0.3) and the overall mean (mean of
the per-sample means) is a solid dot; the overall means are connected by a line across
bins, one line per condition. Recomputes the SAME VX3 bins as the v4 DEG pipeline.
One panel per gene in GENES, side by side in a single figure. A bin marked with '*'
is a significant NR-vs-DR DEG in that bin per the v4 DESeq2 tables (FDR < 0.05 and
|log2FC_shrink| > 1).

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/41.cheng22_nr_harmony_arch12.h5ad
  local_data/res/astro/45.v4.deg_vx3_bin{1..4}_NR_vs_DR_all.tsv
Outputs:
  local_data/fig/astro/45.v4.gene_expr_vx3_bins_NR_vs_DR_samplemeans.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.utils import norm

GENES = ['Rfx4', 'Igfbp2']   # one panel per gene

INPUT_H5AD   = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_NR_H5AD   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.cheng22_nr_harmony_arch12.h5ad')
DEG_ALL_TMPL = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '45.v4.deg_vx3_bin{b}_NR_vs_DR_all.tsv')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_PDF      = os.path.join(FIG_DIR, '45.v4.gene_expr_vx3_bins_NR_vs_DR_samplemeans.pdf')

LABELED_AGES = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES      = ['P28', 'P38']
DR_AGES      = ['P28_dr', 'P38_dr']
MIN_CELLS    = 50
N_BINS       = 4
FDR_THRESH   = 0.05          # DESeq2 significance threshold for the '*' marker
LOG2FC_SIG   = np.log2(1.5)  # |log2FC_shrink| threshold for the '*' marker

COLOR_NR  = '#1f77b4'   # matches "up in NR" (blue) in the volcano
COLOR_DR  = '#d62728'   # matches "up in DR" (red) in the volcano
DODGE     = 0.0         # NR/DR share the exact same x per bin
JITTER    = 0.03        # per-sample dot horizontal jitter

os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# Step 1 — recompute the 4 VX3 bins (identical pipeline to scripts 45 / 45.v2-v4)
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

# raw counts (adata.X is normalized for the VX embedding); used for the displayed expression
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

bin_labels = np.full(len(vx3_combined), -1, dtype=int)
for sample in np.unique(sample_combined):
    mask = sample_combined == sample
    bin_labels[mask] = pd.qcut(vx3_combined[mask], q=N_BINS, labels=False, duplicates='drop')

# =============================================================================
# Step 2 — per-cell log2(CP10k+1) expression, then per-sample means per bin (per gene)
# =============================================================================
depths_raw = raw_counts.sum(axis=1)


def gene_sample_means(gene):
    """Per-sample and overall (mean of per-sample) mean log2(CP10k+1) per (bin, condition)."""
    gene_idx = np.where(gene_names == gene)[0]
    assert len(gene_idx) == 1, f'gene {gene!r} not found uniquely in var_names'
    expr_gene = np.log2(1 + raw_counts[:, gene_idx[0]] / depths_raw * 1e4)   # log2 CP10k per cell
    df = pd.DataFrame({
        'bin':       bin_labels,
        'condition': np.where(condition_code == 1, 'DR', 'NR'),
        'sample':    sample_combined,
        'expr':      expr_gene[combined_idx],
    })
    sample_means = df.groupby(['bin', 'condition', 'sample'], observed=True)['expr'].mean().reset_index()
    overall_means = sample_means.groupby(['bin', 'condition'], observed=True)['expr'].mean().reset_index()
    return sample_means, overall_means


# --- v4 DESeq2 tables per bin (gene-indexed); bin file b+1 <-> bin label b ---
deg_tables = {b: pd.read_csv(DEG_ALL_TMPL.format(b=b + 1), sep='\t').set_index('gene')
              for b in range(N_BINS)}


def bin_significant(gene, b):
    """True if `gene` is a significant NR-vs-DR DEG in bin `b` (FDR < FDR_THRESH and
    |log2FC_shrink| > LOG2FC_SIG). NaN FDR (DESeq2 filtered) or absent gene -> False."""
    df = deg_tables[b]
    if gene not in df.index:
        return False
    fdr, lfc = df.loc[gene, 'fdr'], df.loc[gene, 'log2FC']
    if pd.isna(fdr):
        return False
    return bool((fdr < FDR_THRESH) and (abs(lfc) > LOG2FC_SIG))


# =============================================================================
# Step 3 — plot: one panel per gene; per-sample dots (alpha 0.3) + overall-mean
#          solid dots + lines. x = bins (Bin 4 -> Bin 1). '*' = significant DEG.
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
bin_plot_order = list(range(N_BINS))[::-1]          # bin index 3,2,1,0 -> Bin 4..Bin 1
bin_to_x = {b: i for i, b in enumerate(bin_plot_order)}
cond_dodge = {'NR': -DODGE, 'DR': DODGE}
cond_color = {'NR': COLOR_NR, 'DR': COLOR_DR}

rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, len(GENES), figsize=(6.5 * len(GENES), 5), squeeze=False)
axes = axes[0]

for gi, gene in enumerate(GENES):
    ax = axes[gi]
    sample_means, overall_means = gene_sample_means(gene)
    print(f'{gene}:')
    print(overall_means.pivot(index='bin', columns='condition', values='expr'))

    for cond in ['NR', 'DR']:
        # per-sample faint dots
        sm = sample_means[sample_means['condition'] == cond]
        xs = np.array([bin_to_x[b] for b in sm['bin']]) + cond_dodge[cond] \
            + rng.uniform(-JITTER, JITTER, len(sm))
        ax.scatter(xs, sm['expr'].values, color=cond_color[cond], alpha=0.3, s=30,
                   linewidths=0, zorder=2)
        # overall-mean solid dots + connecting line (ordered along the x-axis)
        om = overall_means[overall_means['condition'] == cond].set_index('bin').reindex(bin_plot_order)
        x_om = np.array([bin_to_x[b] for b in bin_plot_order]) + cond_dodge[cond]
        ax.plot(x_om, om['expr'].values, '-', color=cond_color[cond], lw=1.5, zorder=3)
        ax.scatter(x_om, om['expr'].values, color=cond_color[cond], s=70, zorder=4,
                   edgecolors='white', linewidths=0.8, label=cond)

    # significance markers: '*' above bins that are significant NR-vs-DR DEGs
    bin_max = sample_means.groupby('bin', observed=True)['expr'].max()
    yl = ax.get_ylim()
    span = yl[1] - yl[0]
    star_bins = [b for b in bin_plot_order if bin_significant(gene, b)]
    for b in star_bins:
        ax.text(bin_to_x[b], bin_max[b] + 0.03 * span, '*', ha='center', va='bottom',
                fontsize=16, fontweight='bold', color='black', zorder=5)
    if star_bins:
        ax.set_ylim(top=yl[1] + 0.10 * span)   # headroom so stars aren't clipped

    ax.set_xticks(range(N_BINS))
    ax.set_xticklabels([f'Bin {b + 1}' for b in bin_plot_order])
    ax.set_xlabel('VX3 bin')
    if gi == 0:
        ax.set_ylabel('log2(CP10k + 1)')
    ax.set_title(gene)
    if gi == len(GENES) - 1:
        ax.legend(title='condition', frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    sns.despine(ax=ax)

fig.suptitle(f'per-sample mean expression across VX3 bins (NR vs DR)   '
             f'* = DESeq2 FDR<{FDR_THRESH} & |log2FC|>log2(1.5)')
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT_PDF}')
print('Done.')
