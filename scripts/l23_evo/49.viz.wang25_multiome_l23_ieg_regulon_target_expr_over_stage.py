"""
Like script 48, but instead of each TF's OWN expression, plot the mean expression of the
TF's REGULON TARGET GENES across developmental stages, in L2/3 excitatory neurons
(EN-L2_3-IT) of the Wang25 developing-human-cortex 10x multiome GEX dataset (V1).

For each TF we use its activating (+/+) SCENIC+ eRegulon from the Wang25 SuppTable13 table
(`27.human_wang25_regulon_targets.tsv`) -- the same regulon source used by script 44's
human panel and by script 42. The per-cell regulon score is computed as in script 42: each
target gene's log2(CP10k+1) is first min-max normalized to [0, 1] across cells (using its
own NORM_PCTILE percentiles, so highly-expressed genes don't dominate), THEN averaged
across the regulon's target genes present in the dataset (target sets are intersected with
the gene panel; coverage is printed). Per-cell depth = total counts across all genes.

Cells are then pooled by developmental stage (second trimester -> third trimester ->
infancy -> adolescence), giving one dot per stage per TF. The per-stage means are min-max
normalized to [0, 1] per regulon so temporal patterns are comparable across panels. All TFs
are shown as a single figure (4 columns x 2 rows) in one PDF.

TFs without a (+/+) regulon in the table (e.g. FOSB) are skipped with a log message.

Reads:
  links/l23_evo/wang25_human_multiome_gex_v1_l23.h5ad
  local_data/res/l23_evo/27.human_wang25_regulon_targets.tsv
Outputs:
  local_data/fig/l23_evo/49.wang25_multiome_l23_ieg_regulon_target_expr_over_stage.pdf
  local_data/res/l23_evo/49.wang25_multiome_l23_ieg_regulon_target_expr_over_stage.tsv
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
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'wang25_human_multiome_gex_v1_l23.h5ad')
IN_REG      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo', '27.human_wang25_regulon_targets.tsv')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')

# TFs whose regulons we score. FOSB has no (+/+) regulon in the table and is auto-skipped,
# leaving 8 plotted regulons -> a 4x2 grid.
TFS = ['FOS', 'FOSL2', 'FOSB', 'JUN', 'JUNB', 'EGR1', 'EGR2', 'EGR3', 'EGR4']
NCOLS, NROWS = 4, 2            # panel grid

DIRECTION = '+/+'              # activating regulons only (matches scripts 42/44)
KEEP_TYPE = 'EN-L2_3-IT'       # L2/3 excitatory neurons
TYPE_COL  = 'Type'
GROUP_COL = 'Group'
CP_TARGET = 1e4                # CP10k
NORM_PCTILE = (0, 99)          # per-gene min-max [0,1] percentiles, before averaging (as script 42)

# Developmental stages in temporal order (obs values -> display labels).
STAGE_ORDER  = ['Second_trimester', 'Third_trimester', 'Infancy', 'Adolescence']
STAGE_LABELS = {'Second_trimester': 'Second trimester',
                'Third_trimester': 'Third trimester',
                'Infancy': 'Infancy',
                'Adolescence': 'Adolescence'}

OUT_PDF = os.path.join(OUT_FIG_DIR, '49.wang25_multiome_l23_ieg_regulon_target_expr_over_stage.pdf')
OUT_TSV = os.path.join(OUT_RES_DIR, '49.wang25_multiome_l23_ieg_regulon_target_expr_over_stage.tsv')


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    # --- regulon target sets (+/+ only): TF -> set of target symbols ---
    reg = pd.read_csv(IN_REG, sep='\t')
    reg = reg[reg['regulation_direction'] == DIRECTION]
    tf_targets = {tf: set(reg.loc[reg['regulon'] == f'{tf}_{DIRECTION}', 'Gene']) for tf in TFS}

    print(f'Loading {IN_H5AD}')
    adata = ad.read_h5ad(IN_H5AD)

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

    # Per-cell depth (total counts across all genes); X holds raw counts.
    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float64)
    depth = X.sum(axis=1)
    if np.any(depth == 0):
        raise ValueError("Found cells with zero total counts; cannot normalize")

    name2idx = {g: i for i, g in enumerate(adata.var_names)}

    all_stats, results = [], []   # results: (tf, stats, n_used) for each plotted regulon
    for tf in TFS:
        targets = tf_targets[tf]
        if not targets:
            print(f'  {tf}: no {DIRECTION} regulon in table — skipped')
            continue
        used = [g for g in sorted(targets) if g in name2idx]
        if not used:
            print(f'  {tf}: 0/{len(targets)} targets in panel — skipped')
            continue

        # Per-cell score: min-max [0,1] each target gene across cells (NORM_PCTILE),
        # then average across the regulon's target genes (script-42 method).
        cols = [name2idx[g] for g in used]
        l2 = np.log2(X[:, cols] / depth[:, None] * CP_TARGET + 1.0)   # (n_cells, n_targets)
        lo, hi = np.percentile(l2, NORM_PCTILE, axis=0)
        rng_g = np.where(hi > lo, hi - lo, 1.0)
        norm = np.clip((l2 - lo) / rng_g, 0.0, 1.0)                   # (n_cells, n_targets)
        score = norm.mean(axis=1)                                     # (n_cells,)

        df = pd.DataFrame({'stage': stage, 'score': score})
        grp = df.groupby('stage', observed=True)['score']
        stats = pd.DataFrame({
            'n_cells': grp.size(),
            'mean': grp.mean(),
            'std': grp.std(ddof=1),
        }).reindex(stages).reset_index().rename(columns={'index': 'stage'})
        stats['sem'] = stats['std'] / np.sqrt(stats['n_cells'])

        # Min-max normalize per-stage means to [0, 1] per regulon (SEM scaled by same range).
        mmin, mmax = stats['mean'].min(), stats['mean'].max()
        rng = mmax - mmin
        if rng == 0:
            raise ValueError(f"All per-stage means equal for {tf}; cannot min-max normalize")
        stats['mean_norm'] = (stats['mean'] - mmin) / rng
        stats['sem_norm'] = stats['sem'] / rng
        stats.insert(0, 'TF', tf)
        stats.insert(1, 'n_targets_total', len(targets))
        stats.insert(2, 'n_targets_used', len(used))
        all_stats.append(stats)
        results.append((tf, stats, len(used)))
        print(f'  {tf}: {len(used)}/{len(targets)} targets used')

    if not results:
        raise ValueError("No regulons plotted; nothing written")
    if len(results) > NROWS * NCOLS:
        raise ValueError(f"{len(results)} regulons exceed the {NROWS}x{NCOLS} grid")

    # --- single figure: NROWS x NCOLS grid, one panel per regulon ---
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(3.4 * NCOLS, 3.3 * NROWS), squeeze=False)
    axflat = axes.ravel()
    for ax, (tf, stats, n_used) in zip(axflat, results):
        xpos = np.arange(len(stats))
        ax.errorbar(xpos, stats['mean_norm'].values, yerr=stats['sem_norm'].values,
                    fmt='-o', color='#C44E52', markersize=6, capsize=3,
                    linewidth=1.5, elinewidth=1.2)
        ax.set_xticks(xpos)
        ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in stats['stage']],
                           rotation=45, ha='right', fontsize=8)
        ax.set_xlim(-0.4, len(stats) - 0.6)
        ax.set_xlabel('Developmental stage')
        ax.set_ylabel('regulon target activity\n(per-gene [0,1] mean, minmax norm.)', fontsize=8)
        ax.set_title(f'{tf} regulon ({n_used} targets)', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in axflat[len(results):]:
        ax.axis('off')

    fig.suptitle(f'IEG regulon-target activity over developmental stage — {KEEP_TYPE} (V1)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)

    out = pd.concat(all_stats, ignore_index=True)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_PDF}')
    print(f'Wrote {OUT_TSV}')


if __name__ == '__main__':
    main()
