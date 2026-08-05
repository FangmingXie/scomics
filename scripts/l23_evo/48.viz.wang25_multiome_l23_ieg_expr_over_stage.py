"""
Plot expression of a list of genes (immediate-early / activity-regulated genes) in
L2/3 excitatory neurons (EN-L2_3-IT) over developmental time, using the Wang25
developing-human-cortex 10x multiome GEX dataset (V1, L2/3 subset). One page per gene,
saved into a single multi-page PDF.

This dataset is full-transcriptome (35k genes) with raw integer counts in adata.X, all
from visual cortex (V1) L2/3 IT neurons across 14 postconceptional ages (98 -> 5353 days,
second trimester -> adolescence). Because it is single-region, there is no age/region
confound -- a single per-age line is plotted (unlike the MERFISH script 47).

Cells are pooled by developmental stage (Group: second trimester -> third trimester ->
infancy -> adolescence), giving one dot per stage per gene. Expression is quantified as
log2(CP10k+1) computed directly from raw counts (per-cell depth = total counts across all
genes), then the per-stage means are min-max normalized to [0, 1] per gene so temporal
patterns are comparable across pages.

To change the gene set, edit GENES below.
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['pdf.fonttype'] = 42     # editable vector text in PDF
plt.rcParams['svg.fonttype'] = 'none'  # editable vector text in SVG

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'wang25_human_multiome_gex_v1_l23.h5ad')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')

GENES = ['FOS', 'FOSL2', 'FOSB', 'JUNB', 'EGR1', 'EGR2', 'EGR3', 'EGR4', 'ATF6']

KEEP_TYPE = 'EN-L2_3-IT'       # L2/3 excitatory neurons
TYPE_COL  = 'Type'
GROUP_COL = 'Group'
CP_TARGET = 1e4                # CP10k

# Developmental stages in temporal order (obs values -> display labels).
STAGE_ORDER  = ['Second_trimester', 'Third_trimester', 'Infancy', 'Adolescence']
STAGE_LABELS = {'Second_trimester': 'Second trimester',
                'Third_trimester': 'Third trimester',
                'Infancy': 'Infancy',
                'Adolescence': 'Adolescence'}

OUT_PDF = os.path.join(OUT_FIG_DIR, '48.wang25_multiome_l23_ieg_expr_over_stage.pdf')
OUT_TSV = os.path.join(OUT_RES_DIR, '48.wang25_multiome_l23_ieg_expr_over_stage.tsv')


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    print(f'Loading {IN_H5AD}')
    adata = ad.read_h5ad(IN_H5AD)

    # Fail-fast: all requested genes must be present.
    missing = [g for g in GENES if g not in adata.var_names]
    if missing:
        raise ValueError(f"Genes not found in dataset: {missing}")

    # Subset to L2/3 excitatory neurons (dataset is already this type + V1, so a no-op guard).
    adata = adata[(adata.obs[TYPE_COL] == KEEP_TYPE).values].copy()
    if adata.n_obs == 0:
        raise ValueError(f"No {KEEP_TYPE} cells found")

    # Developmental stage per cell, ordered temporally; keep only stages present.
    stage = adata.obs[GROUP_COL].astype(str).values
    stages = [s for s in STAGE_ORDER if s in set(stage)]
    unknown = set(stage) - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"Unknown developmental stage(s) not in STAGE_ORDER: {unknown}")
    print(f'{adata.n_obs} {KEEP_TYPE} cells across {len(stages)} stages: {stages}')

    # Per-cell sequencing depth (total counts across all genes); X holds raw counts.
    X = adata.X
    depth = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    if np.any(depth == 0):
        raise ValueError("Found cells with zero total counts; cannot normalize")

    all_stats = []
    with PdfPages(OUT_PDF) as pdf:
        for gene in GENES:
            gi = adata.var_names.get_loc(gene)
            col = X[:, gi]
            col = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
            expr = np.log2(col / depth * CP_TARGET + 1.0)      # log2(CP10k+1)

            df = pd.DataFrame({'stage': stage, 'expr': expr})
            grp = df.groupby('stage', observed=True)['expr']
            stats = pd.DataFrame({
                'n_cells': grp.size(),
                'mean': grp.mean(),
                'std': grp.std(ddof=1),
            }).reindex(stages).reset_index().rename(columns={'index': 'stage'})
            stats['sem'] = stats['std'] / np.sqrt(stats['n_cells'])

            # Min-max normalize per-stage means to [0, 1] per gene (SEM scaled by same range).
            mmin, mmax = stats['mean'].min(), stats['mean'].max()
            rng = mmax - mmin
            if rng == 0:
                raise ValueError(f"All per-stage means equal for {gene}; cannot min-max normalize")
            stats['mean_norm'] = (stats['mean'] - mmin) / rng
            stats['sem_norm'] = stats['sem'] / rng
            stats.insert(0, 'gene', gene)
            all_stats.append(stats)

            # Plot: one dot per stage, connected line, mean +/- SEM (normalized).
            xpos = np.arange(len(stats))
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.errorbar(xpos, stats['mean_norm'].values, yerr=stats['sem_norm'].values,
                        fmt='-o', color='#4C72B0', markersize=6, capsize=3,
                        linewidth=1.5, elinewidth=1.2)

            ax.set_xticks(xpos)
            ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in stats['stage']],
                               rotation=45, ha='right', fontsize=8)
            ax.set_xlim(-0.4, len(stats) - 0.6)
            ax.set_xlabel('Developmental stage')
            ax.set_ylabel(f'{gene}  log2(CP10k+1), min-max norm.')
            ax.set_title(f'{gene} in {KEEP_TYPE} (V1)', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f'  plotted {gene}')

    out = pd.concat(all_stats, ignore_index=True)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_PDF}')
    print(f'Wrote {OUT_TSV}')


if __name__ == '__main__':
    main()
