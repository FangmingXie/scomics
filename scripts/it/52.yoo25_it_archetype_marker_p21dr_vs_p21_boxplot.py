"""Dark-rearing response of each archetype's own marker genes, all four mouse IT subclasses.

Scripts 50 and 51 ask how dark-rearing shifts a *fixed* gene panel (hand-picked IEGs, or
enrichment-derived regulon TFs) within each L2/3 archetype. This script asks the
self-referential version across the whole IT sheet: for every archetype, how does dark-rearing
move *that archetype's own marker genes*, inside *that archetype's own cells*?

One box per (subclass, archetype), 11 in total, laid out left-to-right in laminar-depth order
(L2/3 A'B'C', L4 A'B'C', L5IT A'B', L6IT A'B'C'). For box (subclass S, archetype k):

    genes = markers(k)                    # k's OWN marker set only -- the diagonal
    cells = cells of S assigned to k      # argmax of recomputed archetype scores
    1 point = 1 gene

    log2FC_k(g) = log2( (CPM_P21DR[g, S, k] + PSEUDO) / (CPM_P21[g, S, k] + PSEUDO) )

This is the diagonal of a (marker-set x archetype) grid -- not full columns. Box sizes are
therefore deliberately unequal (they are the per-archetype marker counts: 55..206).

METHOD (steps 1-6 replicate scripts/it/50.yoo25_l23_ieg_p21dr_vs_p21_foldchange.py:99-229,
generalized over a subclass loop):
  1. Read the subclass marker TSV; NOC is inferred from it (L5IT has 2 archetypes, not 3).
  2. Pool Subclass == S AND Age in {P21, P21DR}; depth = obs['total_counts'] (validated > 0).
  3. Recompute per-cell archetype scores over that pooled P21+P21DR set -- the saved
     *_archetype_scores.tsv files cover only cheng22 + yoo25-P21 and contain no DR cells, so
     both conditions must be re-scored together to sit on one footing.
  4. Assign each cell to its argmax archetype (disjoint).
  5. Pseudobulk per (age, archetype): top N_TOP_CELLS purest assigned cells, SUM raw counts.
  6. CPM-normalize each column independently, then take the log2 ratio per marker gene.

NOTE ON NORMALIZATION: the fold change is taken directly on pseudobulk CPM (raw expression
magnitude), NOT on the per-gene [0,1] min-max-scaled values used by scripts 48.v2 / 49. That
scaling has no fixed zero point, so a log2 ratio of scaled values would not be an interpretable
fold change. Working from CPM keeps log2FC = 0 meaning "unchanged by dark-rearing". The only
nonlinearity is the +PSEUDO (=1 CPM) pseudocount, which stabilizes low-expression genes.

ARCHETYPE LABELS are read from the persisted depth-arc table rather than hard-coded (see
scripts/ARCHETYPE_MAPPING.md). Its `old_letter` is the internal PCHA letter (A = archetype_1,
B = archetype_2, ...), `new_letter` is the published primed label, and its global `arc_rank`
(0-10) is exactly the laminar-depth box order used here. L2/3, L4 and L5IT are reversals;
L6IT is the identity. Color follows the *displayed* label (A' -> C0, B' -> C1, C' -> C2).

Reads:
  links/it/superdupermegaRNA_yoo25_IT_AllAges.h5ad
  local_data/res/it/33.follow.two_L4_archetype_markers.tsv
  local_data/res/it/34.follow.two_L23_archetype_markers.tsv
  local_data/res/it/35.follow.two_L5IT_archetype_markers.tsv
  local_data/res/it/36.follow.two_L6IT_archetype_markers.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv
Outputs:
  local_data/fig/it/52.yoo25_it_archetype_marker_p21dr_vs_p21_boxplot.pdf
  local_data/res/it/52.yoo25_it_archetype_marker_p21dr_vs_p21_boxplot.tsv
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42      # editable vector text in PDF
plt.rcParams['svg.fonttype'] = 'none'  # editable vector text in SVG

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
OUT_RES_DIR  = RES_DIR

IN_H5AD    = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_yoo25_IT_AllAges.h5ad')
IN_ARCMAP  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo',
                          '15.mouse_IT_joint_archetype_arc_order.tsv')

OUT_PDF = os.path.join(OUT_FIG_DIR, '52.yoo25_it_archetype_marker_p21dr_vs_p21_boxplot.pdf')
OUT_TSV = os.path.join(OUT_RES_DIR, '52.yoo25_it_archetype_marker_p21dr_vs_p21_boxplot.tsv')

# Per-subclass config. `token` keys into the depth-arc table; `markers` is the script-33/34/35/36
# marker TSV. NOC is NOT listed -- it is inferred from the marker file (L5IT has 2 archetypes).
SUBCLASSES = [
    dict(subclass='L2/3', token='L23',  markers='34.follow.two_L23_archetype_markers.tsv'),
    dict(subclass='L4',   token='L4',   markers='33.follow.two_L4_archetype_markers.tsv'),
    dict(subclass='L5IT', token='L5IT', markers='35.follow.two_L5IT_archetype_markers.tsv'),
    dict(subclass='L6IT', token='L6IT', markers='36.follow.two_L6IT_archetype_markers.tsv'),
]

ARCHETYPE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']  # archetype_1 -> A, archetype_2 -> B, ...

SUBCLASS_COL = 'Subclass'
AGE_COL      = 'Age'
AGES         = ['P21', 'P21DR']   # numerator is P21DR, denominator is P21
DEPTH_COL    = 'total_counts'     # yoo25's n_counts is all-NaN; total_counts is the valid depth

N_TOP_CELLS  = 100      # top cells (by archetype score) per (age, archetype) pseudobulk column.
                        # 100 not 300 (scripts 50/51): the smallest (age, archetype) group is 134
                        # (L6IT P21 archetype_1), so 300 is infeasible for L5IT / L6IT.
SCORE_PCTILE = (2, 98)  # per-gene percentile clip for archetype scoring (script 34 recipe)
CP10K        = 1e4      # CP10k target for the log2 scoring expression
CP_TARGET    = 1e6      # CPM for pseudobulk expression
PSEUDO       = 1.0      # CPM pseudocount added before the log2 ratio

# Color follows the DISPLAYED (primed) label, never the internal key -- ARCHETYPE_MAPPING.md.
ARCH_COLORS = {"A'": 'C0', "B'": 'C1', "C'": 'C2', "D'": 'C3'}


def load_arch_map():
    """Read the persisted depth-arc table into {token: {old_letter: (new_letter, arc_rank)}}.

    Mirrors the `relabel_for` idiom used by scripts/it_evo/{12,14,18c,18d,32} (quoted in
    scripts/ARCHETYPE_MAPPING.md) -- the mapping is read, never hard-coded.
    """
    arc = pd.read_csv(IN_ARCMAP, sep='\t')
    need = {'token', 'old_letter', 'new_letter', 'arc_rank'}
    missing = need - set(arc.columns)
    if missing:
        raise ValueError(f"{IN_ARCMAP} missing column(s): {sorted(missing)}")
    amap = {}
    for token, sub in arc.groupby('token'):
        amap[token] = {r.old_letter: (r.new_letter, int(r.arc_rank)) for r in sub.itertuples()}
    return amap


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    print(f'Reading archetype label map from {IN_ARCMAP}')
    arch_map = load_arch_map()

    # --- per-subclass marker sets, keyed by internal letter; validate against the arc table ---
    cfgs = []
    for cfg in SUBCLASSES:
        path = os.path.join(RES_DIR, cfg['markers'])
        mk = pd.read_csv(path, sep='\t')
        arch_vals = sorted(mk['archetype'].unique())          # archetype_1, archetype_2, ...
        expect = [f'archetype_{i + 1}' for i in range(len(arch_vals))]
        if arch_vals != expect:
            raise ValueError(f"{path}: archetype values {arch_vals} are not a 1..k run")
        # internal letter A/B/C... <- archetype_1/2/3...  (the arc table's `old_letter`)
        letters = [ARCHETYPE_LETTERS[i] for i in range(len(arch_vals))]
        markers = {ARCHETYPE_LETTERS[i]: mk.loc[mk['archetype'] == a, 'gene'].tolist()
                   for i, a in enumerate(arch_vals)}

        token = cfg['token']
        if token not in arch_map:
            raise ValueError(f"token '{token}' absent from {IN_ARCMAP}")
        # Fail-fast: a silent key mismatch here would mislabel every box for this subclass.
        if set(arch_map[token]) != set(letters):
            raise ValueError(f"{cfg['subclass']}: arc-table letters {sorted(arch_map[token])} != "
                             f"marker-file letters {letters}")

        cfgs.append(dict(cfg, letters=letters, markers=markers,
                         relabel={k: arch_map[token][k][0] for k in letters},
                         rank={k: arch_map[token][k][1] for k in letters}))
        shown = ', '.join(f'{k}({len(markers[k])}) -> {arch_map[token][k][0]}'
                          f' rank {arch_map[token][k][1]}' for k in letters)
        print(f"  {cfg['subclass']:5s} NOC={len(letters)}  {shown}")

    print(f'\nLoading {IN_H5AD}')
    adata_all = ad.read_h5ad(IN_H5AD)
    if adata_all.raw is None:
        raise ValueError('adata.raw is None; raw integer counts required')
    print(f'  {adata_all.n_obs} cells x {adata_all.n_vars} genes '
          f'(raw {adata_all.raw.shape[1]} genes)')

    rows = []
    for cfg in cfgs:
        S, letters = cfg['subclass'], cfg['letters']
        print(f'\n=== {S} ===')

        # --- pool: this subclass AND Age in {P21, P21DR} ---
        ages_all = adata_all.obs[AGE_COL].astype(str).values
        mask = (adata_all.obs[SUBCLASS_COL] == S).values & np.isin(ages_all, AGES)
        adata = adata_all[mask].copy()
        if adata.n_obs == 0:
            raise ValueError(f'No {S} cells in {AGES} found')
        pool_ages = adata.obs[AGE_COL].astype(str).values
        print(f'  pool = {adata.n_obs} cells ('
              + ' + '.join(f'{a} {int((pool_ages == a).sum())}' for a in AGES) + ')')

        raw_var = list(adata.raw.var_names)
        name2idx = {g: i for i, g in enumerate(raw_var)}
        # Fail-fast: every marker gene must be present in the yoo25 gene space.
        missing = sorted({g for k in letters for g in cfg['markers'][k] if g not in name2idx})
        if missing:
            raise ValueError(f'{S}: {len(missing)} marker genes absent from raw.var_names, '
                             f'e.g. {missing[:5]}')

        depth = adata.obs[DEPTH_COL].values.astype(np.float64)
        if not (np.all(np.isfinite(depth)) and np.all(depth > 0)):
            raise ValueError(f"{S}: invalid depth column '{DEPTH_COL}' (NaN or <=0)")

        Xraw = adata.raw.X
        Xraw = Xraw.toarray() if sp.issparse(Xraw) else np.asarray(Xraw)
        Xraw = np.asarray(Xraw, dtype=np.float64)

        # --- recompute per-cell archetype scores over the pooled P21+P21DR cells ---
        # (replicates 34.follow.two_L23_archetype_scores.py:227-244, as in scripts 50/51)
        scores = np.zeros((adata.n_obs, len(letters)), dtype=np.float64)
        for ki, k in enumerate(letters):
            cols = [name2idx[g] for g in cfg['markers'][k]]
            l2 = np.log2(Xraw[:, cols] / depth[:, None] * CP10K + 1.0)
            lo, hi = np.percentile(l2, SCORE_PCTILE, axis=0)
            rng = np.where(hi > lo, hi - lo, 1.0)
            scores[:, ki] = np.clip((l2 - lo) / rng, 0.0, 1.0).mean(axis=1)

        # --- assign each cell to its argmax archetype (disjoint) ---
        assigned = np.array([letters[i] for i in scores.argmax(axis=1)])

        # --- pseudobulk one column per (age, archetype): top-N purest, SUM raw counts ---
        columns = [(a, k) for a in AGES for k in letters]
        pb = np.zeros((adata.raw.shape[1], len(columns)), dtype=np.float64)
        n_used = {}
        for j, (a, k) in enumerate(columns):
            ki = letters.index(k)
            cell_idx = np.where((pool_ages == a) & (assigned == k))[0]
            if cell_idx.size < N_TOP_CELLS:
                raise ValueError(f'{S} ({a},{k}) has {cell_idx.size} assigned cells < '
                                 f'N_TOP_CELLS={N_TOP_CELLS}')
            top = cell_idx[np.argsort(scores[cell_idx, ki])[::-1][:N_TOP_CELLS]]
            n_used[(a, k)] = int(top.size)
            pb[:, j] = Xraw[top].sum(axis=0)
            print(f'  {a:5s} {cfg["relabel"][k]} (internal {k}): '
                  f'n_assigned={cell_idx.size:5d}  n_used={top.size}')

        # --- CPM-normalize each column independently ---
        col_totals = pb.sum(axis=0)
        if np.any(col_totals == 0):
            zero = [columns[j] for j in np.where(col_totals == 0)[0]]
            raise ValueError(f'{S}: zero-count pseudobulk column(s): {zero}')
        cpm = pb / col_totals[None, :] * CP_TARGET
        col_index = {c: j for j, c in enumerate(columns)}

        # --- the diagonal: k's own markers, measured in k's own cells ---
        for k in letters:
            genes = cfg['markers'][k]
            gi = [name2idx[g] for g in genes]
            c_p21 = cpm[gi, col_index[('P21', k)]]
            c_p21dr = cpm[gi, col_index[('P21DR', k)]]
            fc = np.log2((c_p21dr + PSEUDO) / (c_p21 + PSEUDO))
            for n, g in enumerate(genes):
                rows.append(dict(
                    subclass=S, archetype=cfg['relabel'][k], archetype_internal=k,
                    arch_rank=cfg['rank'][k], gene=g,
                    cpm_P21=float(c_p21[n]), cpm_P21DR=float(c_p21dr[n]),
                    log2FC_P21DR_over_P21=float(fc[n]),
                    n_used_P21=n_used[('P21', k)], n_used_P21DR=n_used[('P21DR', k)]))

    out = pd.DataFrame(rows).sort_values(['arch_rank', 'gene']).reset_index(drop=True)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'\nWrote {OUT_TSV} ({len(out)} rows)')

    # -----------------------------------------------------------------------
    # Figure: one box per (subclass, archetype), ordered by laminar-depth arc_rank
    # -----------------------------------------------------------------------
    boxes = (out.groupby(['arch_rank', 'subclass', 'archetype'], sort=True)
                .size().reset_index(name='n').sort_values('arch_rank'))
    data, labels, colors, subclasses = [], [], [], []
    for r in boxes.itertuples():
        vals = out.loc[out['arch_rank'] == r.arch_rank, 'log2FC_P21DR_over_P21'].values
        data.append(vals)
        labels.append(f'{r.archetype}\nn={r.n}')
        colors.append(ARCH_COLORS[r.archetype])
        subclasses.append(r.subclass)
    npos = len(data)
    print(f'  {npos} boxes: ' + ', '.join(f'{s} {l.splitlines()[0]}'
                                          for s, l in zip(subclasses, labels)))

    fig, ax = plt.subplots(figsize=(1.0 * npos + 2.0, 4.6))
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.4))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    # jittered points (deterministic offsets; no RNG per project constraints)
    for xi, vals in enumerate(data):
        n = len(vals)
        jitter = (np.arange(n) - (n - 1) / 2.0) / n * 0.55
        ax.scatter(np.full(n, xi + 1) + jitter, vals,
                   s=4, color='black', alpha=0.35, linewidths=0, zorder=3, rasterized=True)

    ax.axhline(0.0, color='grey', linewidth=1.0, linestyle='--', zorder=1)
    ax.set_xticks(np.arange(1, npos + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('log2(P21DR / P21)', fontsize=10)
    ax.set_xlim(0.4, npos + 0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # subclass blocks: dividers between them, label centred under each
    trans = ax.get_xaxis_transform()
    starts = [i for i in range(npos) if i == 0 or subclasses[i] != subclasses[i - 1]]
    for i in starts[1:]:
        ax.axvline(i + 0.5, color='0.75', linewidth=0.8, linestyle=':', zorder=0)
    for bi, i in enumerate(starts):
        j = starts[bi + 1] if bi + 1 < len(starts) else npos
        ax.text((i + j + 1) / 2.0, -0.16, subclasses[i], transform=trans,
                ha='center', va='top', fontsize=11, fontweight='bold')

    ax.set_title("Dark-rearing response of each archetype's own marker genes\n"
                 f'(Yoo25 mouse IT; top {N_TOP_CELLS} purest cells per age x archetype)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Wrote {OUT_PDF}')


if __name__ == '__main__':
    main()
