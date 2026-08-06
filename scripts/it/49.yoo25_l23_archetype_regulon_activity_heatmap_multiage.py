"""Developmental archetype x regulon-activity heatmap (Yoo25 AllAges mouse L2/3 IT).

Multi-age extension of script 48: instead of the two P21 conditions only, this spans six ages
across the developmental window — P14, P14DR, P17, P17DR, P21, P21DR (each normal age paired
with its dark-reared counterpart) — and splits every age into the three L2/3 archetypes A/B/C
(defined in script 34). Each column is thus an (age, archetype) pseudobulk (6 ages x 3
archetypes = 18 columns), exposing how the TF/IEG regulons differ across archetypes, how that
changes over postnatal development, and how dark-rearing perturbs it at each age.

Archetype membership is recomputed here (NOT read from script 34's saved scores, which cover
only P21-normal + cheng22): per-cell A/B/C scores are recomputed for the joint six-age L2/3
pool from script 34's marker gene sets using the identical scoring recipe
(34.follow.two_L23_archetype_scores.py:227-244), so all ages sit on one footing.

Because the earlier ages have fewer cells in some archetypes (smallest assigned group ~129
cells), the per-column pool is the top N_TOP_CELLS=100 cells (script 48 used 300 for two ages).

Pipeline:
  1. Regulon target sets (+/+ only): TF -> set of target symbols (fail-fast if any missing).
  2. Pool: subset AllAges to Subclass == 'L2/3' AND Age in AGES. Counts from adata.raw;
     depth = obs['total_counts'] (validated finite/>0).
  3. Recompute per-cell scores: for each archetype k, markers = script-34 marker rows for
     archetype_{k+1} present in raw.var_names; l2 = log2(counts/depth*1e4 + 1); per-gene 2/98
     percentile clip over the POOL -> [0,1]; score_k = clipped mean over that archetype's markers.
  4. Assign: argmax over the 3 scores -> each cell to A/B/C (disjoint).
  5. 18 pseudobulk columns (age-major, archetypes shown A',B',C'): per (age, k) take the top
     N_TOP_CELLS by score_k among assigned cells, SUM raw counts. Archetypes are relabeled for
     publication (internal A/B/C -> C'/B'/A') and displayed in A',B',C' order.
  6. Regulon activity (as in script 47): CPM = counts/col_total*1e6 -> log1p; min-max scale each
     gene to [0,1] jointly across ALL columns; activity per regulon = mean over targets present.
  7. Per-row max-normalize (each regulon / its own max across all cols); hierarchically cluster
     rows (correlation distance, average linkage) for row order.
  8. RdBu_r heatmap (regulon rows x 18 cols), two-line column labels, per-age dividers (thin
     within a normal/DR pair, bold between developmental days).

Reads:
  links/it/superdupermegaRNA_yoo25_IT_AllAges.h5ad
  local_data/res/it/40.yoo25_L2_3_regulon_targets.tsv
  local_data/res/it/34.follow.two_L23_archetype_markers.tsv
Outputs:
  local_data/fig/it/49.yoo25_l23_archetype_regulon_activity_heatmap_multiage.pdf
  local_data/res/it/49.yoo25_l23_archetype_regulon_activity_heatmap_multiage.tsv
  local_data/res/it/49.yoo25_l23_multiage_archetype_scores.tsv
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, leaves_list
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
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_yoo25_IT_AllAges.h5ad')
IN_REG      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '40.yoo25_L2_3_regulon_targets.tsv')
IN_MARKERS  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '34.follow.two_L23_archetype_markers.tsv')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')

# Regulon rows (row order is set by clustering below; this is just the input set).
SELECTED_REGULONS = ['Nfib', 'Rfx3', 'Meis2', 'Bcl11a', 'Zbtb20', 'Pbx1', 'Satb2', 'Tcf4',
                     'Mef2c', 'Fosl2', 'Egr3', 'Fos', 'Smad3', 'Fosb', 'Egr4', 'Egr2',
                     'Junb', 'Atf6', 'Egr1', 'Etv5', 'Mxi1', 'Irf2', 'Tfdp2', 'Klf9',
                     'Satb1', 'Jdp2', 'Tcf12', 'Nr3c1']
DIRECTION    = '+/+'           # activating regulons only (matches scripts 42/46/47/48)
SUBCLASS_COL = 'Subclass'
KEEP_SUBCLASS = 'L2/3'
AGE_COL      = 'Age'
# column-block order: each normal age immediately followed by its dark-reared counterpart.
AGES         = ['P14', 'P14DR', 'P17', 'P17DR', 'P21', 'P21DR']
DEPTH_COL    = 'total_counts'
# archetype label A/B/C -> marker-file 'archetype' value (script 34 convention)
ARCH_MARKER_LABEL = {'A': 'archetype_1', 'B': 'archetype_2', 'C': 'archetype_3'}
ARCHETYPES   = ['A', 'B', 'C']    # internal keys (marker identity + scoring; unchanged)
# published relabeling of the internal archetypes, and the column display order.
ARCH_RELABEL = {'A': "C'", 'B': "B'", 'C': "A'"}   # internal A/B/C -> displayed label
ARCH_ORDER   = ['C', 'B', 'A']    # internal keys ordered so labels read A', B', C'
N_TOP_CELLS  = 100             # top cells (by archetype score) per (age, archetype) column
SCORE_PCTILE = (2, 98)         # per-gene percentile clip for scoring (script 34)
CP10K        = 1e4             # CP10k target for log2 scoring expression
CP_TARGET    = 1e6             # CPM for pseudobulk activity
CMAP         = 'RdBu_r'        # per-row [0,1] color: blue (low) -> red (high)
CLUSTER_METRIC = 'correlation'  # row distance for hierarchical clustering
CLUSTER_METHOD = 'average'      # linkage method

OUT_PDF        = os.path.join(OUT_FIG_DIR, '49.yoo25_l23_archetype_regulon_activity_heatmap_multiage.pdf')
OUT_TSV        = os.path.join(OUT_RES_DIR, '49.yoo25_l23_archetype_regulon_activity_heatmap_multiage.tsv')
OUT_SCORES_TSV = os.path.join(OUT_RES_DIR, '49.yoo25_l23_multiage_archetype_scores.tsv')

TOP_COL = f'in_top{N_TOP_CELLS}'   # aux-TSV membership flag name


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    # --- regulon target sets (+/+ only): TF -> set of target symbols ---
    reg = pd.read_csv(IN_REG, sep='\t')
    reg = reg[reg['regulation_direction'] == DIRECTION]
    tf_targets = {}
    for tf in SELECTED_REGULONS:
        key = f'{tf}_{DIRECTION}'
        targets = set(reg.loc[reg['regulon'] == key, 'Gene'])
        if not targets:
            raise ValueError(f"No {DIRECTION} regulon '{key}' found in {IN_REG}")
        tf_targets[tf] = targets

    # --- archetype marker gene sets: A/B/C -> list of marker symbols ---
    markers = pd.read_csv(IN_MARKERS, sep='\t')
    arch_markers = {}
    for a in ARCHETYPES:
        genes = markers.loc[markers['archetype'] == ARCH_MARKER_LABEL[a], 'gene'].tolist()
        if not genes:
            raise ValueError(f"No markers for archetype {a} ({ARCH_MARKER_LABEL[a]}) in {IN_MARKERS}")
        arch_markers[a] = genes

    print(f'Loading {IN_H5AD}')
    adata = ad.read_h5ad(IN_H5AD)

    # --- pool: L2/3 AND Age in AGES ---
    ages_all = adata.obs[AGE_COL].astype(str).values
    mask_pool = (adata.obs[SUBCLASS_COL] == KEEP_SUBCLASS).values & np.isin(ages_all, AGES)
    adata = adata[mask_pool].copy()
    if adata.n_obs == 0:
        raise ValueError(f"No {KEEP_SUBCLASS} cells in {AGES} found")
    pool_ages = adata.obs[AGE_COL].astype(str).values
    print(f'Pool = {adata.n_obs} cells ('
          + ' + '.join(f'{a} {int((pool_ages == a).sum())}' for a in AGES) + ')')

    if adata.raw is None:
        raise ValueError("adata.raw is None; raw integer counts required")
    raw_var = list(adata.raw.var_names)
    name2idx = {g: i for i, g in enumerate(raw_var)}
    print(f'Using adata.raw ({adata.raw.shape[1]} genes) as raw-count source')

    # --- depth (validated finite/>0) ---
    depth = adata.obs[DEPTH_COL].values.astype(np.float64)
    if not (np.all(np.isfinite(depth)) and np.all(depth > 0)):
        raise ValueError(f"Invalid depth column '{DEPTH_COL}' (NaN or <=0)")

    # --- dense raw counts for the pool (n_pool x n_genes) ---
    Xraw = adata.raw.X
    Xraw = Xraw.toarray() if sp.issparse(Xraw) else np.asarray(Xraw)
    Xraw = np.asarray(Xraw, dtype=np.float64)

    # --- recompute per-cell A/B/C scores (replicates 34.follow...py:227-244) ---
    # log2 CP10k expression over the whole pool, then per-gene 2/98 percentile clip to [0,1].
    print('Recomputing archetype scores over the six-age pool...')
    scores = np.zeros((adata.n_obs, len(ARCHETYPES)), dtype=np.float64)
    for k, a in enumerate(ARCHETYPES):
        used = [g for g in arch_markers[a] if g in name2idx]
        print(f'  markers found {a}: {len(used)}/{len(arch_markers[a])}')
        if not used:
            raise ValueError(f"No archetype-{a} markers present in raw.var_names")
        cols = [name2idx[g] for g in used]
        l2 = np.log2(Xraw[:, cols] / depth[:, None] * CP10K + 1.0)
        lo, hi = np.percentile(l2, SCORE_PCTILE, axis=0)
        rng = np.where(hi > lo, hi - lo, 1.0)
        scores[:, k] = np.clip((l2 - lo) / rng, 0.0, 1.0).mean(axis=1)

    # --- assign each cell to its argmax archetype (disjoint) ---
    assigned_idx = scores.argmax(axis=1)
    assigned = np.array([ARCHETYPES[i] for i in assigned_idx])

    # --- pseudobulk columns: (age, archetype), age-major, displayed A',B',C' order ---
    columns = [(a, k) for a in AGES for k in ARCH_ORDER]  # internal C,B,A -> shown A',B',C'
    n_genes = adata.raw.shape[1]
    pb = np.zeros((n_genes, len(columns)), dtype=np.float64)  # gene x column
    top_mask = np.zeros(adata.n_obs, dtype=bool)  # cells used in any column (for aux TSV)
    n_assigned = {}
    n_used = {}
    for j, (a, k) in enumerate(columns):
        ki = ARCHETYPES.index(k)
        cell_mask = (pool_ages == a) & (assigned == k)
        cell_idx = np.where(cell_mask)[0]
        n_assigned[(a, k)] = int(cell_idx.size)
        if cell_idx.size < N_TOP_CELLS:
            raise ValueError(f"({a},{k}) has {cell_idx.size} assigned cells < N_TOP_CELLS="
                             f"{N_TOP_CELLS}")
        # top N by this archetype's score
        top = cell_idx[np.argsort(scores[cell_idx, ki])[::-1][:N_TOP_CELLS]]
        top_mask[top] = True
        n_used[(a, k)] = int(top.size)
        pb[:, j] = Xraw[top].sum(axis=0)
        print(f'  {a}_{ARCH_RELABEL[k]} (internal {k}): '
              f'n_assigned={n_assigned[(a, k)]:5d}  n_used={n_used[(a, k)]}')

    # --- normalize per column: CPM -> log1p ---
    col_totals = pb.sum(axis=0)
    if np.any(col_totals == 0):
        zero = [columns[j] for j in np.where(col_totals == 0)[0]]
        raise ValueError(f"Zero-count pseudobulk column(s): {zero}")
    cpm = pb / col_totals[None, :] * CP_TARGET
    logexpr = np.log1p(cpm)

    # --- min-max scale each gene to [0,1] jointly across ALL columns ---
    gmin = logexpr.min(axis=1, keepdims=True)
    gmax = logexpr.max(axis=1, keepdims=True)
    grng = gmax - gmin
    scaled = np.where(grng > 0, (logexpr - gmin) / np.where(grng > 0, grng, 1.0), 0.0)

    # --- regulon activity per column = mean of [0,1] scaled target expression ---
    act_mat = np.zeros((len(SELECTED_REGULONS), len(columns)), dtype=np.float64)
    for i, tf in enumerate(SELECTED_REGULONS):
        used = [g for g in sorted(tf_targets[tf]) if g in name2idx]
        if not used:
            raise ValueError(f"{tf}: 0/{len(tf_targets[tf])} targets present in raw.var_names")
        print(f'  {tf:8s}: {len(used)}/{len(tf_targets[tf])} targets used')
        cidx = [name2idx[g] for g in used]
        act_mat[i, :] = scaled[cidx, :].mean(axis=0)

    # --- per-row max normalize (each regulon / its own max across all cols; peak->1) ---
    rmax = act_mat.max(axis=1, keepdims=True)
    act_norm = np.where(rmax > 0, act_mat / np.where(rmax > 0, rmax, 1.0), 0.0)

    # --- hierarchical clustering of rows on the max-normalized profiles ---
    Z = linkage(act_norm, method=CLUSTER_METHOD, metric=CLUSTER_METRIC)
    row_order = leaves_list(Z)
    row_labels = [SELECTED_REGULONS[i] for i in row_order]
    act_mat = act_mat[row_order, :]
    act_norm = act_norm[row_order, :]
    print(f'Row order ({CLUSTER_METHOD}/{CLUSTER_METRIC} clustering): {row_labels}')

    # --- long TSV (published archetype labels; internal key kept for traceability) ---
    col_labels = [f'{a}_{ARCH_RELABEL[k]}' for (a, k) in columns]
    rows = []
    for i, tf in enumerate(row_labels):
        for j, (a, k) in enumerate(columns):
            rows.append(dict(regulon=tf, row=i, age=a, archetype=ARCH_RELABEL[k],
                             archetype_internal=k, column=col_labels[j],
                             n_assigned=n_assigned[(a, k)], n_used=n_used[(a, k)],
                             activity=float(act_mat[i, j]),
                             activity_rowmaxnorm=float(act_norm[i, j])))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_TSV} ({len(out)} rows)')

    # --- aux per-cell scores TSV (transparency; published labels, A',B',C' score order) ---
    score_by_key = {k: scores[:, ki] for ki, k in enumerate(ARCHETYPES)}
    scores_df = pd.DataFrame({
        'cell': adata.obs_names.values,
        'age': pool_ages,
        **{f'score_{ARCH_RELABEL[k]}': score_by_key[k] for k in ARCH_ORDER},
        'assigned': [ARCH_RELABEL[k] for k in assigned],
        TOP_COL: top_mask,
    })
    scores_df.to_csv(OUT_SCORES_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_SCORES_TSV} ({len(scores_df)} rows)')

    # --- heatmap: regulons (clustered rows) x (age, archetype) columns ---
    ncols = len(columns)
    nrows = len(row_labels)
    n_arch = len(ARCH_ORDER)
    fig, ax = plt.subplots(figsize=(0.55 * ncols + 2.4, 0.42 * nrows + 1.8))
    im = ax.imshow(act_norm, aspect='auto', cmap=CMAP, vmin=0.0, vmax=1.0)

    # two-line column labels: archetype (A'/B'/C') on the axis, age above each triplet
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels([ARCH_RELABEL[k] for (_, k) in columns], fontsize=9)
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_ylabel('regulon')

    # age labels centered over each triplet, above the grid
    for bi, a in enumerate(AGES):
        center = bi * n_arch + (n_arch - 1) / 2.0
        ax.text(center, -0.9, a, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # thin white gridlines between cells
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', length=0)

    # per-age dividers: thin between a normal age and its DR pair, bold between days
    for bi in range(len(AGES) - 1):
        x = (bi + 1) * n_arch - 0.5
        within_pair = AGES[bi + 1] == AGES[bi] + 'DR'
        ax.axvline(x, color='white', linewidth=1.5 if within_pair else 3.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('regulon activity\n(per-row max-normalized)', fontsize=8)

    ax.set_title(f'Archetype x regulon activity over development — {KEEP_SUBCLASS} IT '
                 f'(Yoo25 AllAges)', fontsize=11, pad=24)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PDF}')


if __name__ == '__main__':
    main()
