"""
Plot expression of a selected gene in L2/3 excitatory neurons (EN-L2_3-IT) over
developmental time in the Wang25 developing-human-cortex MERFISH dataset.

Expression is quantified as log2(CP10k+1). The MERFISH matrix (adata.X) is stored as
log1p of raw counts on the 300-gene panel (i.e. X = ln(1 + raw_count), no size
normalization); verified empirically that expm1(X) returns exact integers. We therefore
recover raw counts as round(expm1(X)) and recompute log2(CP10k+1) using the per-cell
300-gene-panel depth (the only depth available; full-transcriptome counts are not stored).

All 5 postconceptional ages in the dataset are used (154, 222, 235, 267, 507 days),
pooling cells across brain regions. NOTE: developmental age is confounded with brain
region here (154->V1/PFC, 222->BA17, 235->BA10, 267->MFG, 507->BA17), so the trajectory
mixes occipital and frontal cortex.

Plot: line of per-age mean +/- SEM. All figure components are vectorized and editable
(pdf.fonttype = 42).

To plot a different gene, edit the GENE constant below.
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42     # editable vector text in PDF
plt.rcParams['svg.fonttype'] = 'none'  # editable vector text in SVG

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'wang25_human_merfish_host.h5ad')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')

GENE       = 'FOS'              # selected gene (example)
KEEP_TYPE  = 'EN-L2_3-IT'       # L2/3 excitatory neurons
TYPE_COL   = 'type'
AGE_COL    = 'Estimated_postconceptional_age_in_days'
REGION_COL = 'Region_details'
CP_TARGET  = 1e4                # CP10k
DODGE_DAYS = 7                  # horizontal offset to separate regions sharing an age

# Human-readable developmental group per age (days), for x-tick labels.
AGE_GROUP = {154: '2nd tri', 222: '3rd tri', 235: '3rd tri', 267: 'infancy', 507: 'infancy'}

# Collapse detailed regions into two macro-regions (points/lines colored by macro-region
# to make the age/region confound explicit).
REGION_TO_GROUP = {
    'V1':                   'Visual cortex',
    'BA17':                 'Visual cortex',
    'PFC':                  'PFC',
    'BA10':                 'PFC',
    'Middle frontal gyrus': 'PFC',
}
GROUP_ORDER  = ['Visual cortex', 'PFC']
GROUP_COLORS = {'Visual cortex': '#4C72B0', 'PFC': '#C44E52'}

OUT_PDF = os.path.join(OUT_FIG_DIR, f'47.wang25_merfish_{GENE}_expr_over_age.pdf')
OUT_TSV = os.path.join(OUT_RES_DIR, f'47.wang25_merfish_{GENE}_expr_over_age.tsv')


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    print(f'Loading {IN_H5AD}')
    adata = ad.read_h5ad(IN_H5AD)

    # Fail-fast: gene must be on the panel.
    if GENE not in adata.var_names:
        raise ValueError(f"Gene {GENE!r} not found in the {adata.n_vars}-gene MERFISH panel")

    # Subset: L2/3 excitatory neurons (all brain regions / all ages).
    adata = adata[(adata.obs[TYPE_COL] == KEEP_TYPE).values].copy()
    if adata.n_obs == 0:
        raise ValueError(f"No {KEEP_TYPE} cells found")

    print(f'{adata.n_obs} {KEEP_TYPE} cells (all regions)')
    print('Per-age cell counts:')
    print(adata.obs[AGE_COL].value_counts().sort_index().to_string())

    # Recover raw counts (X = ln(1 + raw_count)) and compute log2(CP10k+1).
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    counts = np.rint(np.expm1(X))                        # exact integer raw counts
    depth = counts.sum(axis=1, keepdims=True)            # per-cell 300-gene-panel depth
    if np.any(depth == 0):
        raise ValueError("Found cells with zero total counts on the panel; cannot normalize")
    log2cp10k = np.log2(counts / depth * CP_TARGET + 1.0)

    gi = list(adata.var_names).index(GENE)
    expr = log2cp10k[:, gi]                              # (n_cells,)

    region = adata.obs[REGION_COL].astype(str).values
    group = np.array([REGION_TO_GROUP[r] for r in region])
    df = pd.DataFrame({'age': adata.obs[AGE_COL].values, 'group': group, 'expr': expr})

    # Per-(age, macro-region) statistics -- one plotted point per sample.
    grp = df.groupby(['age', 'group'], observed=True)['expr']
    stats = pd.DataFrame({
        'n_cells': grp.size(),
        'mean': grp.mean(),
        'std': grp.std(ddof=1),
    }).reset_index().sort_values(['group', 'age']).reset_index(drop=True)
    stats['sem'] = stats['std'] / np.sqrt(stats['n_cells'])

    # Min-max normalize the per-sample means to [0, 1] (SEM scaled by the same range).
    mmin, mmax = stats['mean'].min(), stats['mean'].max()
    rng = mmax - mmin
    if rng == 0:
        raise ValueError("All per-(age, region) means are equal; cannot min-max normalize")
    stats['mean_norm'] = (stats['mean'] - mmin) / rng
    stats['sem_norm'] = stats['sem'] / rng

    stats.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'\nPer-(age, macro-region) {GENE} log2(CP10k+1):')
    print(stats.to_string(index=False))
    print(f'Wrote {OUT_TSV}')

    # Small horizontal dodge so macro-regions sharing an age (154d) don't overlap.
    x_off = {}
    for age, sub in stats.groupby('age', observed=True):
        grps = [g for g in GROUP_ORDER if g in set(sub['group'])]
        offsets = (np.arange(len(grps)) - (len(grps) - 1) / 2.0) * DODGE_DAYS
        for g, off in zip(grps, offsets):
            x_off[(age, g)] = float(age) + off

    # Plot: one line per macro-region connecting its per-age dots (mean +/- SEM).
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for g in GROUP_ORDER:
        sub = stats[stats['group'] == g].sort_values('age')
        if sub.empty:
            continue
        xs = [x_off[(a, g)] for a in sub['age']]
        ax.errorbar(xs, sub['mean_norm'].values, yerr=sub['sem_norm'].values, fmt='-o',
                    color=GROUP_COLORS[g], markersize=6, capsize=3,
                    linewidth=1.5, elinewidth=1.2, label=g)

    ages = sorted(df['age'].unique())
    ax.set_xticks(ages)
    ax.set_xticklabels([f'{int(a)} ({AGE_GROUP.get(int(a), "")})' for a in ages],
                       rotation=90, fontsize=7)
    ax.set_xlabel('Postconceptional age (days)')
    ax.set_ylabel(f'{GENE}  log2(CP10k+1), min-max norm.')
    ax.set_title(f'{GENE} in {KEEP_TYPE}', fontsize=10)
    ax.legend(title='Region', fontsize=8, title_fontsize=8, frameon=False, loc='best')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PDF}')


if __name__ == '__main__':
    main()
