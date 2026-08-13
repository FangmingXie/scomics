"""Individual gene expression over human developmental stage (Wang25 10x multiome GEX, V1).

Same figure layout and readout as scripts/it_evo/25, but each line is a single gene instead
of a mouse-archetype marker set:

  - Panel 1 (L2/3):  EN-L2_3-IT cells
  - Panel 2 (L4):    EN-L4-IT cells
  - Panel 3 (L5 IT): EN-L5-IT cells
  - Panel 4 (L6 IT): EN-L6-IT cells

Per-cell score is the gene's log2(CP10k+1). Cells are pooled by developmental stage (second
trimester -> third trimester -> infancy -> adolescence), V1 only, and each gene's per-stage
mean +/- SEM is max-scaled across stages (divided by its largest per-stage mean), so 1 marks
the peak stage and 0 marks no expression; relative magnitudes are preserved. The raw
per-stage means are written to the output TSV.

Reads:
  links/it_evo/wang25_human_multiome_gex_it.h5ad
Outputs:
  local_data/fig/it_evo/25b.wang25_multiome_it_gene_expr_over_stage.pdf
  local_data/res/it_evo/25b.wang25_multiome_it_gene_expr_over_stage.tsv
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
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'it_evo', 'wang25_human_multiome_gex_it.h5ad')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')

OUT_PDF = os.path.join(OUT_FIG_DIR, '25b.wang25_multiome_it_gene_expr_over_stage.pdf')
OUT_TSV = os.path.join(OUT_RES_DIR, '25b.wang25_multiome_it_gene_expr_over_stage.tsv')

# Genes shown as separate lines in every panel.
GENES = ['RFX3', 'SORCS3', 'JDP2']

# Panels: (label, human Type).
PANELS = [
    ('L2/3',  'EN-L2_3-IT'),
    ('L4',    'EN-L4-IT'),
    ('L5 IT', 'EN-L5-IT'),
    ('L6 IT', 'EN-L6-IT'),
]
# Per-line colors/markers (one per gene).
LINE_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
LINE_MARKERS = ['o', 's', '^', 'D', 'v']

KEEP_REGION = 'V1'             # matches script 25 (V1 only)
TYPE_COL    = 'Type'
REGION_COL  = 'Region'
GROUP_COL   = 'Group'
CP_TARGET   = 1e4              # CP10k

# Developmental stages in temporal order (obs values -> display labels).
STAGE_ORDER  = ['Second_trimester', 'Third_trimester', 'Infancy', 'Adolescence']
STAGE_LABELS = {'Second_trimester': 'Second trimester',
                'Third_trimester': 'Third trimester',
                'Infancy': 'Infancy',
                'Adolescence': 'Adolescence'}


def gene_stage_stats(gene, X, depth, stage, stages, name2idx):
    """Per-stage mean+/-SEM (and stage-level max-scaling) of one gene's log2(CP10k+1)."""
    score = np.log2(X[:, name2idx[gene]] / depth * CP_TARGET + 1.0)   # (n_cells,)

    df = pd.DataFrame({'stage': stage, 'score': score})
    grp = df.groupby('stage', observed=True)['score']
    stats = pd.DataFrame({
        'n_cells': grp.size(),
        'mean': grp.mean(),
        'std': grp.std(ddof=1),
    }).reindex(stages).reset_index().rename(columns={'index': 'stage'})
    stats['sem'] = stats['std'] / np.sqrt(stats['n_cells'])

    mmax = stats['mean'].max()
    if mmax <= 0:
        raise ValueError(f"Max per-stage mean is {mmax} for {gene}; cannot max-scale")
    stats['mean_norm'] = stats['mean'] / mmax
    stats['sem_norm'] = stats['sem'] / mmax
    return stats


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    print(f'Loading {IN_H5AD}')
    adata = ad.read_h5ad(IN_H5AD)
    missing = [g for g in GENES if g not in set(adata.var_names)]
    if missing:
        raise ValueError(f"Gene(s) not found in {IN_H5AD}: {missing}")

    fig, axes = plt.subplots(1, len(PANELS), figsize=(2.9 * len(PANELS), 3.9), squeeze=False)
    all_stats = []

    for col, (label, human_type) in enumerate(PANELS):
        # --- cells: this IT type, V1 only ---
        sub = adata[((adata.obs[TYPE_COL] == human_type)
                     & (adata.obs[REGION_COL] == KEEP_REGION)).values].copy()
        if sub.n_obs == 0:
            raise ValueError(f"No {human_type} cells in region {KEEP_REGION}")

        stage = sub.obs[GROUP_COL].astype(str).values
        unknown = set(stage) - set(STAGE_ORDER)
        if unknown:
            raise ValueError(f"Unknown developmental stage(s) not in STAGE_ORDER: {unknown}")
        stages = [s for s in STAGE_ORDER if s in set(stage)]
        print(f'[{label}] {sub.n_obs} {human_type} cells ({KEEP_REGION}) across '
              f'{len(stages)} stages: {stages}')

        X = sub.X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        X = X.astype(np.float64)
        depth = X.sum(axis=1)
        if np.any(depth == 0):
            raise ValueError(f"Found {human_type} cells with zero total counts; cannot normalize")
        name2idx = {g: i for i, g in enumerate(sub.var_names)}

        xpos = np.arange(len(stages))
        ax = axes[0, col]
        n_by_stage = None

        for i, gene in enumerate(GENES):
            stats = gene_stage_stats(gene, X, depth, stage, stages, name2idx)
            n_by_stage = stats['n_cells'].values
            stats.insert(0, 'panel', label)
            stats.insert(1, 'human_type', human_type)
            stats.insert(2, 'gene', gene)
            all_stats.append(stats)
            print(f'  {gene}: mean log2(CP10k+1) '
                  f'{stats["mean"].min():.3f}-{stats["mean"].max():.3f} across stages')

            ax.errorbar(xpos, stats['mean_norm'].values, yerr=stats['sem_norm'].values,
                        fmt='-' + LINE_MARKERS[i % len(LINE_MARKERS)],
                        color=LINE_COLORS[i % len(LINE_COLORS)], markersize=5, capsize=2.5,
                        linewidth=1.5, elinewidth=1.0, label=gene)

        ticklabels = [f'{STAGE_LABELS.get(s, s)}\n(n={n})' for s, n in zip(stages, n_by_stage)]
        ax.set_xlim(-0.4, len(stages) - 0.6)
        ax.set_xticks(xpos)
        ax.set_xticklabels(ticklabels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Developmental stage')
        ax.set_title(f'{label} — {human_type}', fontsize=11)
        ax.legend(fontsize=8, frameon=False, title='gene', title_fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col == 0:
            ax.set_ylabel('gene expression\n(max-scaled across stages, peak = 1)', fontsize=8)

    # [0,1] by construction — share limits across panels.
    for ax in axes[0]:
        ax.set_ylim(0, 1.12)

    fig.suptitle('Gene expression over human developmental stage — '
                 f'Wang25 multiome ({KEEP_REGION})', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)

    out = pd.concat(all_stats, ignore_index=True)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_PDF}')
    print(f'Wrote {OUT_TSV} ({len(out)} rows)')


if __name__ == '__main__':
    main()
