"""
Two-panel version of script 49: regulon-target activity over developmental stage in L2/3
excitatory neurons (EN-L2_3-IT) of the Wang25 developing-human-cortex 10x multiome GEX
dataset (V1), with related TFs overlaid as separate lines in one panel.

  - Panel 1 (AP-1):  FOS, FOSL2, JUN, JUNB, JUND
  - Panel 2 (EGR):   EGR1, EGR2, EGR3, EGR4

For each TF, the per-cell regulon score uses the activating (+/+) SCENIC+ eRegulon from the
Wang25 SuppTable13 table (`27.human_wang25_regulon_targets.tsv`): each target gene's
log2(CP10k+1) is min-max normalized to [0, 1] across cells (NORM_PCTILE percentiles) then
averaged across the regulon's target genes present in the panel (script-42 method).

Cells are pooled by developmental stage (second trimester -> third trimester -> infancy ->
adolescence). Each TF's per-stage means are min-max normalized to [0, 1] across stages so
lines are directly comparable in temporal shape. One figure, 2 panels side by side, in a
single PDF.

Reads:
  links/l23_evo/wang25_human_multiome_gex_v1_l23.h5ad
  local_data/res/l23_evo/27.human_wang25_regulon_targets.tsv
Outputs:
  local_data/fig/l23_evo/50.wang25_multiome_l23_ap1_egr_regulon_target_expr_over_stage.pdf
  local_data/res/l23_evo/50.wang25_multiome_l23_ap1_egr_regulon_target_expr_over_stage.tsv
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

# Panels: (label, list of TFs shown as separate lines).
PANELS = [
    ('AP-1', ['FOS', 'FOSL2', 'JUN', 'JUNB', 'JUND']),
    ('EGR',  ['EGR1', 'EGR2', 'EGR3', 'EGR4']),
]
# Per-line colors/markers (up to 5 lines per panel).
LINE_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
LINE_MARKERS = ['o', 's', '^', 'D', 'v']

DIRECTION = '+/+'              # activating regulons only (matches scripts 42/44/49)
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

OUT_PDF = os.path.join(OUT_FIG_DIR, '50.wang25_multiome_l23_ap1_egr_regulon_target_expr_over_stage.pdf')
OUT_TSV = os.path.join(OUT_RES_DIR, '50.wang25_multiome_l23_ap1_egr_regulon_target_expr_over_stage.tsv')


def regulon_stage_stats(tf, tf_targets, X, depth, stage, stages, name2idx):
    """Per-stage mean+/-SEM (and stage-level min-max) of a TF's regulon-target activity.

    Returns (stats_df, n_targets_used) or None if the TF has no regulon / no panel targets.
    """
    targets = tf_targets.get(tf, set())
    if not targets:
        return None
    used = [g for g in sorted(targets) if g in name2idx]
    if not used:
        return None

    # Per-cell score: min-max [0,1] each target across cells (NORM_PCTILE), then average.
    cols = [name2idx[g] for g in used]
    l2 = np.log2(X[:, cols] / depth[:, None] * CP_TARGET + 1.0)   # (n_cells, n_targets)
    lo, hi = np.percentile(l2, NORM_PCTILE, axis=0)
    rng_g = np.where(hi > lo, hi - lo, 1.0)
    norm = np.clip((l2 - lo) / rng_g, 0.0, 1.0)
    score = norm.mean(axis=1)

    df = pd.DataFrame({'stage': stage, 'score': score})
    grp = df.groupby('stage', observed=True)['score']
    stats = pd.DataFrame({
        'n_cells': grp.size(),
        'mean': grp.mean(),
        'std': grp.std(ddof=1),
    }).reindex(stages).reset_index().rename(columns={'index': 'stage'})
    stats['sem'] = stats['std'] / np.sqrt(stats['n_cells'])

    mmin, mmax = stats['mean'].min(), stats['mean'].max()
    rng = mmax - mmin
    if rng == 0:
        raise ValueError(f"All per-stage means equal for {tf}; cannot min-max normalize")
    stats['mean_norm'] = (stats['mean'] - mmin) / rng
    stats['sem_norm'] = stats['sem'] / rng
    stats.insert(0, 'TF', tf)
    stats.insert(1, 'n_targets_used', len(used))
    return stats, len(used)


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    # --- regulon target sets (+/+ only): TF -> set of target symbols ---
    reg = pd.read_csv(IN_REG, sep='\t')
    reg = reg[reg['regulation_direction'] == DIRECTION]
    all_tfs = [tf for _, tfs in PANELS for tf in tfs]
    tf_targets = {tf: set(reg.loc[reg['regulon'] == f'{tf}_{DIRECTION}', 'Gene']) for tf in all_tfs}

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

    xpos = np.arange(len(stages))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 3.8),
                             squeeze=False, sharey=True)

    all_stats = []
    for col, (ax, (label, tfs)) in enumerate(zip(axes[0], PANELS)):
        for i, tf in enumerate(tfs):
            res = regulon_stage_stats(tf, tf_targets, X, depth, stage, stages, name2idx)
            if res is None:
                print(f'  [{label}] {tf}: no {DIRECTION} regulon / no targets — skipped')
                continue
            stats, n_used = res
            stats.insert(0, 'panel', label)
            all_stats.append(stats)
            print(f'  [{label}] {tf}: {n_used} targets used')

            ax.errorbar(xpos, stats['mean_norm'].values, yerr=stats['sem_norm'].values,
                        fmt='-' + LINE_MARKERS[i % len(LINE_MARKERS)],
                        color=LINE_COLORS[i % len(LINE_COLORS)], markersize=5, capsize=2.5,
                        linewidth=1.5, elinewidth=1.0, label=f'{tf} ({n_used})')

        ax.set_xticks(xpos)
        ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in stages],
                           rotation=45, ha='right', fontsize=8)
        ax.set_xlim(-0.4, len(stages) - 0.6)
        ax.set_xlabel('Developmental stage')
        if col == 0:
            ax.set_ylabel('regulon target activity\n(per-gene [0,1] mean, minmax norm.)', fontsize=8)
        ax.set_title(f'{label} regulons', fontsize=11)
        ax.legend(fontsize=8, frameon=False, title='TF (n targets)', title_fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if not all_stats:
        raise ValueError("No regulons plotted; nothing written")

    fig.suptitle(f'AP-1 & EGR regulon-target activity over developmental stage — {KEEP_TYPE} (V1)',
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
