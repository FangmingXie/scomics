"""Per-sample mean gene expression across the 4 VX3 bins, NR vs DR (matplotlib PDF).

Variant of 45.v4.viz_gene_expr_vx3_bins.py: instead of per-cell boxplots, each sample's
mean log2(CP10k+1) expression is a faint dot (alpha=0.3) and the overall mean (mean of
the per-sample means) is a solid dot; the overall means are connected by a line across
bins, one line per condition. Recomputes the SAME VX3 bins as the v4 DEG pipeline.

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/41.cheng22_nr_harmony_arch12.h5ad
Outputs:
  local_data/fig/astro/45.v4.{GENE}_expr_vx3_bins_NR_vs_DR_samplemeans.pdf
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

GENE = 'Rfx4'

INPUT_H5AD = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_NR_H5AD = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.cheng22_nr_harmony_arch12.h5ad')
FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_PDF    = os.path.join(FIG_DIR, f'45.v4.{GENE}_expr_vx3_bins_NR_vs_DR_samplemeans.pdf')

LABELED_AGES = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES      = ['P28', 'P38']
DR_AGES      = ['P28_dr', 'P38_dr']
MIN_CELLS    = 50
N_BINS       = 4

COLOR_NR  = '#1f77b4'   # matches "up in NR" (blue) in the volcano
COLOR_DR  = '#d62728'   # matches "up in DR" (red) in the volcano
DODGE     = 0.12        # horizontal NR/DR separation per bin
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
# Step 2 — per-cell log2(CP10k+1) expression, then per-sample means per bin
# =============================================================================
gene_idx = np.where(gene_names == GENE)[0]
assert len(gene_idx) == 1, f'gene {GENE!r} not found uniquely in var_names'
gene_idx = gene_idx[0]

depths_raw = raw_counts.sum(axis=1)
expr_gene  = np.log2(1 + raw_counts[:, gene_idx] / depths_raw * 1e4)   # log2 CP10k per cell

df = pd.DataFrame({
    'bin':       bin_labels,
    'condition': np.where(condition_code == 1, 'DR', 'NR'),
    'sample':    sample_combined,
    'expr':      expr_gene[combined_idx],
})
sample_means = df.groupby(['bin', 'condition', 'sample'], observed=True)['expr'].mean().reset_index()
overall_means = sample_means.groupby(['bin', 'condition'], observed=True)['expr'].mean().reset_index()
print(overall_means.pivot(index='bin', columns='condition', values='expr'))

# =============================================================================
# Step 3 — plot: per-sample dots (alpha 0.3) + overall-mean solid dots + lines
#          x = bins (Bin 4 -> Bin 1)
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
bin_plot_order = list(range(N_BINS))[::-1]          # bin index 3,2,1,0 -> Bin 4..Bin 1
bin_to_x = {b: i for i, b in enumerate(bin_plot_order)}
cond_dodge = {'NR': -DODGE, 'DR': DODGE}
cond_color = {'NR': COLOR_NR, 'DR': COLOR_DR}

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(7, 5))

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

ax.set_xticks(range(N_BINS))
ax.set_xticklabels([f'Bin {b + 1}' for b in bin_plot_order])
ax.set_xlabel('VX3 bin')
ax.set_ylabel('log2(CP10k + 1)')
ax.set_title(f'{GENE} per-sample mean expression across VX3 bins (NR vs DR)')
ax.legend(title='condition', frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT_PDF}')
print('Done.')
