"""Dark-rearing fold change of enrichment-derived regulon TFs per L2/3 archetype (Yoo25).

Same analysis as script 50 (log2(P21DR / P21) per gene x archetype for L2/3 IT), but the gene
panel and its A/B/C row-blocks are NOT hand-curated from script 48.v2 — they are derived from
the archetype x regulon enrichment computed in script 41
(41.L2_3_regulon_archetype_enrichment.tsv). Each activating (+/+) regulon that is significantly
enriched (log2 OR > LOG2OR_SHOW AND FDR < STAR_FDR in >= 1 archetype, the exact rule script 41
uses to pick shown rows) is assigned to the archetype where its enrichment log2 OR peaks; those
TFs become the plotted "genes", grouped into A'/B'/C' regulon blocks in that order (within a
block, sorted by descending peak log2 OR). This ties each row to the archetype it marks, then
asks how dark-rearing perturbs that TF's own expression within each archetype.

Downstream is identical to script 50: pool P21 + P21DR L2/3, recompute A/B/C archetype
membership (script-34 markers, scoring as in 48.v2 / 49), pseudobulk the top N_TOP_CELLS purest
cells per (age, archetype), CPM normalize, and for each regulon TF g and archetype k:
    log2FC[g, k] = log2( (CPM_P21DR[g, k] + PSEUDO) / (CPM_P21[g, k] + PSEUDO) ).

NOTE ON NORMALIZATION: the fold change is taken directly on pseudobulk CPM (raw expression
magnitude), NOT on the per-gene [0,1] min-max-scaled values used by scripts 48.v2 / 49. That
scaling exists there only to make regulon-activity averages comparable and has no fixed zero
point, so a log2 ratio of scaled values would not be an interpretable fold change. Working
from CPM keeps log2FC = 0 meaning "unchanged by dark-rearing" (what the diverging map centers
on). The only nonlinearity is the +PSEUDO (=1 CPM) pseudocount, which stabilizes the ratio for
low-expression genes.

Three panels are drawn (single PDF):
  - Heatmap: regulon TFs (rows, blocked by enriched archetype) x archetypes A'/B'/C' (cols),
    diverging RdBu_r centered at 0.
  - Boxplot by archetype: per-gene log2FC pooled over genes, one box per archetype A'/B'/C'.
  - Boxplot by regulon block: heatmap values pooled over genes x archetypes within each of the
    three enrichment-derived blocks (A'/B'/C' regulons).

Reads:
  links/it/superdupermegaRNA_yoo25_IT_AllAges.h5ad
  local_data/res/it/34.follow.two_L23_archetype_markers.tsv
  local_data/res/it/41.L2_3_regulon_archetype_enrichment.tsv
Outputs:
  local_data/fig/it/51.yoo25_l23_enriched_regulon_p21dr_vs_p21_foldchange.pdf
  local_data/res/it/51.yoo25_l23_enriched_regulon_p21dr_vs_p21_foldchange.tsv
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
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_yoo25_IT_AllAges.h5ad')
IN_MARKERS  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '34.follow.two_L23_archetype_markers.tsv')
IN_ENRICH   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '41.L2_3_regulon_archetype_enrichment.tsv')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')

# Enrichment-derived panel selection (matches script 41's "shown regulon" rule).
DIRECTION_REG = '+/+'          # activating regulons only
LOG2OR_SHOW   = 2.0            # keep regulon if log2 OR > this ...
STAR_FDR      = 1e-5           # ... AND FDR < this, in >= 1 archetype
# archetype block order top-to-bottom (enrichment arch_letter labels, = heatmap column labels)
BLOCK_ORDER   = ["A'", "B'", "C'"]

SUBCLASS_COL = 'Subclass'
KEEP_SUBCLASS = 'L2/3'
AGE_COL      = 'Age'
AGES         = ['P21', 'P21DR']   # numerator is P21DR, denominator is P21 (see FC below)
DEPTH_COL    = 'total_counts'
# archetype label A/B/C -> marker-file 'archetype' value (script 34 convention)
ARCH_MARKER_LABEL = {'A': 'archetype_1', 'B': 'archetype_2', 'C': 'archetype_3'}
ARCHETYPES   = ['A', 'B', 'C']    # internal keys (marker identity + scoring; unchanged)
# published relabeling of the internal archetypes, and the display order.
ARCH_RELABEL = {'A': "C'", 'B': "B'", 'C': "A'"}   # internal A/B/C -> displayed label
ARCH_ORDER   = ['C', 'B', 'A']    # internal keys ordered so labels read A', B', C'
N_TOP_CELLS  = 300             # top cells (by archetype score) per (age, archetype) column
SCORE_PCTILE = (2, 98)         # per-gene percentile clip for scoring (script 34)
CP10K        = 1e4             # CP10k target for log2 scoring expression
CP_TARGET    = 1e6             # CPM for pseudobulk expression
PSEUDO       = 1.0             # CPM pseudocount added before the log2 ratio
CMAP         = 'RdBu_r'        # diverging map centered at log2FC = 0

OUT_PDF = os.path.join(OUT_FIG_DIR, '51.yoo25_l23_enriched_regulon_p21dr_vs_p21_foldchange.pdf')
OUT_TSV = os.path.join(OUT_RES_DIR, '51.yoo25_l23_enriched_regulon_p21dr_vs_p21_foldchange.tsv')


def derive_regulon_blocks():
    """From script 41's L2/3 enrichment, pick significant +/+ regulons and group their TFs by
    the archetype where log2 OR peaks. Returns (genes, row_groups) where genes is the ordered
    TF list and row_groups is [(block_label, [TFs]), ...] in BLOCK_ORDER."""
    enr = pd.read_csv(IN_ENRICH, sep='\t')
    enr = enr[enr['regulation_direction'] == DIRECTION_REG].copy()
    if enr.empty:
        raise ValueError(f"No {DIRECTION_REG} regulons in {IN_ENRICH}")

    orm = enr.pivot(index='regulon', columns='arch_letter', values='log2_or')
    fdm = enr.pivot(index='regulon', columns='arch_letter', values='fdr')
    tf_of = enr.drop_duplicates('regulon').set_index('regulon')['TF']

    keep = orm.index[((orm > LOG2OR_SHOW) & (fdm < STAR_FDR)).any(axis=1)]
    if len(keep) == 0:
        raise ValueError(f"No regulon passes log2 OR>{LOG2OR_SHOW} AND FDR<{STAR_FDR:g}")

    dominant = orm.loc[keep].idxmax(axis=1)   # archetype (arch_letter) of peak enrichment
    peak = orm.loc[keep].max(axis=1)
    unknown = set(dominant) - set(BLOCK_ORDER)
    if unknown:
        raise ValueError(f"Dominant archetype label(s) not in BLOCK_ORDER: {unknown}")

    genes, row_groups = [], []
    for block in BLOCK_ORDER:
        regs = dominant.index[dominant == block]
        # within a block: descending peak log2 OR
        regs = peak.loc[regs].sort_values(ascending=False).index
        tfs = [tf_of.loc[r] for r in regs]
        if not tfs:
            raise ValueError(f"No enriched regulon assigned to block {block}")
        row_groups.append((f'{block} regulons', tfs))
        genes.extend(tfs)
        print(f'  {block}: {len(tfs)} regulons -> {tfs}')

    if len(genes) != len(set(genes)):
        raise ValueError("Duplicate TF across enriched regulons; panel would double-count rows")
    return genes, row_groups


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    # --- regulon TF panel + A'/B'/C' row-blocks derived from script 41 enrichment ---
    print(f'Deriving regulon panel from {IN_ENRICH}')
    GENES, ROW_GROUPS = derive_regulon_blocks()

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

    # --- pool: L2/3 AND Age in {P21, P21DR} ---
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

    # Fail-fast: all derived regulon TFs must be present.
    missing = [g for g in GENES if g not in name2idx]
    if missing:
        raise ValueError(f"Regulon TFs not found in raw.var_names: {missing}")

    # --- depth (validated finite/>0) ---
    depth = adata.obs[DEPTH_COL].values.astype(np.float64)
    if not (np.all(np.isfinite(depth)) and np.all(depth > 0)):
        raise ValueError(f"Invalid depth column '{DEPTH_COL}' (NaN or <=0)")

    # --- dense raw counts for the pool (n_pool x n_genes) ---
    Xraw = adata.raw.X
    Xraw = Xraw.toarray() if sp.issparse(Xraw) else np.asarray(Xraw)
    Xraw = np.asarray(Xraw, dtype=np.float64)

    # --- recompute per-cell A/B/C scores (replicates 34.follow...py:227-244) ---
    print('Recomputing archetype scores over the P21+P21DR pool...')
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

    # --- pseudobulk columns: (age, archetype), age-major ---
    columns = [(a, k) for a in AGES for k in ARCH_ORDER]
    n_genes = adata.raw.shape[1]
    pb = np.zeros((n_genes, len(columns)), dtype=np.float64)  # gene x column
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
        top = cell_idx[np.argsort(scores[cell_idx, ki])[::-1][:N_TOP_CELLS]]
        n_used[(a, k)] = int(top.size)
        pb[:, j] = Xraw[top].sum(axis=0)
        print(f'  {a}_{ARCH_RELABEL[k]} (internal {k}): '
              f'n_assigned={n_assigned[(a, k)]:5d}  n_used={n_used[(a, k)]}')

    # --- normalize per column: CPM ---
    col_totals = pb.sum(axis=0)
    if np.any(col_totals == 0):
        zero = [columns[j] for j in np.where(col_totals == 0)[0]]
        raise ValueError(f"Zero-count pseudobulk column(s): {zero}")
    cpm = pb / col_totals[None, :] * CP_TARGET

    # --- per-gene, per-archetype log2(P21DR / P21) on CPM (+ pseudocount) ---
    # NOTE: fold change is on raw CPM magnitude, deliberately NOT on 48.v2's [0,1]-scaled
    # values (see module docstring) so log2FC = 0 == "unchanged by dark-rearing".
    col_index = {(a, k): j for j, (a, k) in enumerate(columns)}
    gene_idx = [name2idx[g] for g in GENES]
    fc = np.zeros((len(GENES), len(ARCH_ORDER)), dtype=np.float64)  # gene x archetype (A',B',C')
    for ai, k in enumerate(ARCH_ORDER):
        cpm_p21   = cpm[gene_idx, col_index[('P21', k)]]
        cpm_p21dr = cpm[gene_idx, col_index[('P21DR', k)]]
        fc[:, ai] = np.log2((cpm_p21dr + PSEUDO) / (cpm_p21 + PSEUDO))

    arch_labels = [ARCH_RELABEL[k] for k in ARCH_ORDER]  # published as A',B',C'
    # per-gene block membership (enriched archetype) for the TSV
    gene_block = {g: lab for lab, gs in ROW_GROUPS for g in gs}

    # --- long TSV ---
    rows = []
    for gi, g in enumerate(GENES):
        for ai, k in enumerate(ARCH_ORDER):
            rows.append(dict(
                gene=g, regulon_block=gene_block[g], archetype=ARCH_RELABEL[k],
                archetype_internal=k,
                cpm_P21=float(cpm[gene_idx[gi], col_index[('P21', k)]]),
                cpm_P21DR=float(cpm[gene_idx[gi], col_index[('P21DR', k)]]),
                n_used_P21=n_used[('P21', k)], n_used_P21DR=n_used[('P21DR', k)],
                log2FC_P21DR_over_P21=float(fc[gi, ai])))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_TSV} ({len(out)} rows)')

    # --- figure: heatmap + boxplot-by-archetype + boxplot-by-row-block ---
    vmax = float(np.abs(fc).max())
    vmax = vmax if vmax > 0 else 1.0

    fig, (ax_h, ax_b, ax_r) = plt.subplots(
        1, 3, figsize=(11.6, 0.36 * len(GENES) + 2.0),
        gridspec_kw={'width_ratios': [len(ARCH_ORDER) + 1.2, 3.0, 3.0], 'wspace': 0.5})

    # heatmap: genes x archetypes, diverging, centered at 0
    im = ax_h.imshow(fc, aspect='auto', cmap=CMAP, vmin=-vmax, vmax=vmax)
    ax_h.set_xticks(np.arange(len(arch_labels)))
    ax_h.set_xticklabels(arch_labels, fontsize=10)
    ax_h.set_yticks(np.arange(len(GENES)))
    ax_h.set_yticklabels(GENES, fontsize=8)
    ax_h.set_xlabel('archetype')
    # thin white gridlines between cells
    ax_h.set_xticks(np.arange(-0.5, len(arch_labels), 1), minor=True)
    ax_h.set_yticks(np.arange(-0.5, len(GENES), 1), minor=True)
    ax_h.grid(which='minor', color='white', linewidth=0.5)
    ax_h.tick_params(which='minor', length=0)
    # black dividers between the enrichment-derived row blocks (A'/B'/C' regulons)
    for b in np.cumsum([len(g) for _, g in ROW_GROUPS])[:-1]:
        ax_h.axhline(b - 0.5, color='black', linewidth=1.2)
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.06, pad=0.04)
    cbar.set_label('log2(P21DR / P21)', fontsize=9)
    ax_h.set_title('Dark-rearing fold change\nby archetype', fontsize=11)

    palette = plt.get_cmap('tab10')

    # boxplot 1: distribution of per-gene log2FC, one box per archetype (over heatmap columns)
    data = [fc[:, ai] for ai in range(len(arch_labels))]
    bp = ax_b.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                      medianprops=dict(color='black', linewidth=1.4))
    for pi, patch in enumerate(bp['boxes']):
        patch.set_facecolor(palette(pi))
        patch.set_alpha(0.55)
    # jittered points (deterministic offsets; no RNG per project constraints)
    for ai in range(len(arch_labels)):
        jitter = (np.arange(len(GENES)) - (len(GENES) - 1) / 2.0) / len(GENES) * 0.5
        ax_b.scatter(np.full(len(GENES), ai + 1) + jitter, fc[:, ai],
                     s=16, color='black', alpha=0.7, zorder=3)
    ax_b.axhline(0.0, color='grey', linewidth=1.0, linestyle='--')
    ax_b.set_xticks(np.arange(1, len(arch_labels) + 1))
    ax_b.set_xticklabels(arch_labels, fontsize=10)
    ax_b.set_xlabel('archetype')
    ax_b.set_ylabel('log2(P21DR / P21)', fontsize=9)
    ax_b.set_title(f'By archetype\n(over genes, n={len(GENES)})', fontsize=11)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # boxplot 2: heatmap values grouped by row block (all archetype columns pooled per block)
    gene_pos = {g: i for i, g in enumerate(GENES)}
    row_data = [fc[[gene_pos[g] for g in genes], :].ravel() for _, genes in ROW_GROUPS]
    row_labels_bp = [lab for lab, _ in ROW_GROUPS]
    bp2 = ax_r.boxplot(row_data, widths=0.6, showfliers=False, patch_artist=True,
                       medianprops=dict(color='black', linewidth=1.4))
    for pi, patch in enumerate(bp2['boxes']):
        patch.set_facecolor(palette(pi))
        patch.set_alpha(0.55)
    for gi, vals in enumerate(row_data):
        jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2.0) / max(len(vals), 1) * 0.5
        ax_r.scatter(np.full(len(vals), gi + 1) + jitter, vals,
                     s=16, color='black', alpha=0.7, zorder=3)
    ax_r.axhline(0.0, color='grey', linewidth=1.0, linestyle='--')
    ax_r.set_xticks(np.arange(1, len(row_labels_bp) + 1))
    ax_r.set_xticklabels(row_labels_bp, fontsize=10, rotation=20, ha='right')
    ax_r.set_xlabel('regulon block')
    ax_r.set_ylabel('log2(P21DR / P21)', fontsize=9)
    ax_r.set_title('By regulon block\n(over genes x archetypes)', fontsize=11)
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(False)

    fig.suptitle(f'{KEEP_SUBCLASS} IT dark-rearing response by enriched regulon '
                 f'(Yoo25 AllAges)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PDF}')


if __name__ == '__main__':
    main()
