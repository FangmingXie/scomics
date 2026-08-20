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

Two lower barplot panels decompose each box into the count of genes significantly moved by
dark-rearing (FDR < FDR_THRESH AND |log2FC| > t), up drawn above zero and down below. Their bars
are a strict subset of the points in the box directly above -- same genes, same cells, same fold
changes. One panel per threshold in LOG2FC_PANELS: t=1.0 isolates the few large movers (mostly
0-5 per archetype, so relative bar heights there are noise-dominated and should be read as
"L2/3 B' versus everything else", not as a ranking), while t=0.5 is permissive enough to
discriminate among the quieter archetypes.

Note that FDR barely binds at t=1.0: with N_TOP_CELLS vs N_TOP_CELLS cells the rank test has
ample power for 2-fold effects, so the log2FC cut does nearly all the filtering there. The
FDR cut carries more weight at t=0.5.

SIGNIFICANCE: the pseudobulk has one column per (age, archetype), so it carries no replicate
structure and cannot itself yield a p-value. Significance is therefore taken per cell: a
two-sided Wilcoxon rank-sum (Mann-Whitney U) on per-cell CP10k log2 expression, P21DR vs P21,
over the *same* top-N purest cells that formed the pseudobulk columns, with BH-FDR applied
across that archetype's marker genes. This matches how scripts 33-36 derive the marker sets
themselves. Keeping the test on the same cells that produced the fold change is what makes the
bars an exact decomposition of the boxes; the rank test is invariant to the CP10k/log2 transform.

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
  local_data/fig/it/52.yoo25_it_archetype_marker_p21dr_vs_p21_boxplot.pdf  (3 panels)
  local_data/res/it/52.yoo25_it_archetype_marker_p21dr_vs_p21_boxplot.tsv
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats
import anndata as ad
from statsmodels.stats.multitest import multipletests

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

# Significance cuts for the lower barplot panels: BH-FDR (shared) x one |log2FC| cut per panel.
FDR_THRESH    = 0.05          # BH-FDR, computed within each archetype's own marker set
LOG2FC_PANELS = (1.0, 0.5)    # one barplot panel per cut, in order top-to-bottom

# Color follows the DISPLAYED (primed) label, never the internal key -- ARCHETYPE_MAPPING.md.
ARCH_COLORS = {"A'": 'C0', "B'": 'C1', "C'": 'C2', "D'": 'C3'}
# Up/down colors reuse the volcano convention of scripts/viz.py:save_volcano_pdf.
COLOR_UP = '#d62728'
COLOR_DN = '#1f77b4'


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
        n_used, top_idx = {}, {}
        for j, (a, k) in enumerate(columns):
            ki = letters.index(k)
            cell_idx = np.where((pool_ages == a) & (assigned == k))[0]
            if cell_idx.size < N_TOP_CELLS:
                raise ValueError(f'{S} ({a},{k}) has {cell_idx.size} assigned cells < '
                                 f'N_TOP_CELLS={N_TOP_CELLS}')
            top = cell_idx[np.argsort(scores[cell_idx, ki])[::-1][:N_TOP_CELLS]]
            n_used[(a, k)] = int(top.size)
            top_idx[(a, k)] = top
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

            # per-cell Wilcoxon rank-sum over the SAME top-N cells that formed the pseudobulk,
            # so the bars below are an exact decomposition of the boxes above
            i21, i21dr = top_idx[('P21', k)], top_idx[('P21DR', k)]
            e21 = np.log2(Xraw[np.ix_(i21, gi)] / depth[i21, None] * CP10K + 1.0)
            e21dr = np.log2(Xraw[np.ix_(i21dr, gi)] / depth[i21dr, None] * CP10K + 1.0)
            pval = scipy.stats.mannwhitneyu(e21dr, e21, axis=0, alternative='two-sided').pvalue
            if not np.all(np.isfinite(pval)):
                raise ValueError(f'{S} {k}: non-finite Wilcoxon p-value')
            fdr = multipletests(pval, method='fdr_bh')[1]
            sig = {}
            for t in LOG2FC_PANELS:
                sig[t] = ((fdr < FDR_THRESH) & (fc > t), (fdr < FDR_THRESH) & (fc < -t))
            print(f'  {cfg["relabel"][k]} markers n={len(genes):3d}  ' + '  '.join(
                f'|log2FC|>{t:g}: up={int(u.sum()):3d} down={int(d.sum()):3d}'
                for t, (u, d) in sig.items()) + f'  (FDR<{FDR_THRESH})')

            for n, g in enumerate(genes):
                rows.append(dict(
                    subclass=S, archetype=cfg['relabel'][k], archetype_internal=k,
                    arch_rank=cfg['rank'][k], gene=g,
                    cpm_P21=float(c_p21[n]), cpm_P21DR=float(c_p21dr[n]),
                    log2FC_P21DR_over_P21=float(fc[n]),
                    pval=float(pval[n]), fdr=float(fdr[n]),
                    **{f'sig_{d}_lfc{t:g}': bool(m[n])
                       for t, (u, dn) in sig.items()
                       for d, m in (('up', u), ('down', dn))},
                    n_used_P21=n_used[('P21', k)], n_used_P21DR=n_used[('P21DR', k)]))

    out = pd.DataFrame(rows).sort_values(['arch_rank', 'gene']).reset_index(drop=True)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'\nWrote {OUT_TSV} ({len(out)} rows)')

    # -----------------------------------------------------------------------
    # Figure: boxplot of per-gene log2FC (top) + one significant up/down count panel
    # per LOG2FC_PANELS threshold (below), all sharing the laminar-depth arc_rank x-axis
    # -----------------------------------------------------------------------
    counts = {f'{d}{t:g}': (f'sig_{d}_lfc{t:g}', 'sum')
              for t in LOG2FC_PANELS for d in ('up', 'down')}
    boxes = (out.groupby(['arch_rank', 'subclass', 'archetype'], sort=True)
                .agg(n=('gene', 'size'), **counts).reset_index().sort_values('arch_rank'))
    data, labels, colors, subclasses = [], [], [], []
    for r in boxes.itertuples():
        vals = out.loc[out['arch_rank'] == r.arch_rank, 'log2FC_P21DR_over_P21'].values
        data.append(vals)
        labels.append(f'{r.archetype}\nn={r.n}')
        colors.append(ARCH_COLORS[r.archetype])
        subclasses.append(r.subclass)
    npos = len(data)
    print(f'  {npos} boxes: ' + ', '.join(f'{s_} {l.splitlines()[0]}'
                                          for s_, l in zip(subclasses, labels)))

    npanel = len(LOG2FC_PANELS)
    fig, axes = plt.subplots(
        1 + npanel, 1, sharex=True, figsize=(1.0 * npos + 2.0, 5.1 + 1.6 * npanel),
        gridspec_kw={'height_ratios': [3.0] + [1.5] * npanel, 'hspace': 0.10})
    ax, bar_axes = axes[0], axes[1:]

    # ---- top panel: per-gene log2FC, one box per (subclass, archetype) ----
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
    # guides at each cut the panels below count against; heavier line = stricter cut.
    # labelled just inside the right spine (y in data coords, x in axes coords)
    ytrans = ax.get_yaxis_transform()
    for pi, t in enumerate(LOG2FC_PANELS):
        for y in (t, -t):
            ax.axhline(y, color='grey', linestyle=':', zorder=1,
                       linewidth=0.9 - 0.25 * pi)
        ax.text(0.997, t, f'{t:g}', transform=ytrans, fontsize=7, color='grey',
                va='bottom', ha='right')
    ax.set_ylabel('log2(P21DR / P21)', fontsize=10)
    ax.set_xlim(0.4, npos + 0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Dark-rearing response of each archetype's own marker genes\n"
                 f'(Yoo25 mouse IT; top {N_TOP_CELLS} purest cells per age x archetype)',
                 fontsize=11)

    # ---- one count panel per threshold: up above zero, down below ----
    x = np.arange(1, npos + 1)
    for pi, (ax_b, t) in enumerate(zip(bar_axes, LOG2FC_PANELS)):
        n_up = boxes[f'up{t:g}'].values.astype(int)
        n_dn = boxes[f'down{t:g}'].values.astype(int)
        ax_b.bar(x, n_up, width=0.6, color=COLOR_UP, alpha=0.85,
                 label=f'up (log2FC > {t:g})')
        ax_b.bar(x, -n_dn, width=0.6, color=COLOR_DN, alpha=0.85,
                 label=f'down (log2FC < -{t:g})')
        ax_b.axhline(0.0, color='black', linewidth=0.8)
        for xi in range(npos):
            if n_up[xi]:
                ax_b.text(x[xi], n_up[xi], str(n_up[xi]), ha='center', va='bottom', fontsize=7)
            if n_dn[xi]:
                ax_b.text(x[xi], -n_dn[xi], str(n_dn[xi]), ha='center', va='top', fontsize=7)
        ax_b.set_ylabel('significant genes', fontsize=10)
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        # legend top-right: tallest bars are down bars on the left, so this stays clear of them
        ax_b.legend(frameon=False, fontsize=8, loc='upper right', ncol=2)
        # symmetric limits, with headroom for the count labels and the legend
        lim = max(int(max(n_up.max(), n_dn.max())), 1) * 1.55
        ax_b.set_ylim(-lim, lim)
        # y ticks as positive counts on both sides (sign encodes direction, not a negative count)
        yt = [v for v in ax_b.get_yticks() if -lim <= v <= lim]
        ax_b.set_yticks(yt)
        ax_b.set_yticklabels([f'{abs(int(v))}' for v in yt], fontsize=9)
        ax_b.set_ylim(-lim, lim)
        ax_b.text(0.005, 0.97, f'FDR<{FDR_THRESH:g}, |log2FC|>{t:g}', transform=ax_b.transAxes,
                  fontsize=8, va='top', ha='left', color='0.35')

    # ---- x tick labels and subclass blocks on the bottom panel only ----
    ax_last = bar_axes[-1]
    ax_last.set_xticks(x)
    ax_last.set_xticklabels(labels, fontsize=9)
    starts = [i for i in range(npos) if i == 0 or subclasses[i] != subclasses[i - 1]]
    for a_ in axes:
        for i in starts[1:]:
            a_.axvline(i + 0.5, color='0.75', linewidth=0.8, linestyle=':', zorder=0)
    trans = ax_last.get_xaxis_transform()
    for bi, i in enumerate(starts):
        j = starts[bi + 1] if bi + 1 < len(starts) else npos
        ax_last.text((i + j + 1) / 2.0, -0.24, subclasses[i], transform=trans,
                     ha='center', va='top', fontsize=11, fontweight='bold')

    fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Wrote {OUT_PDF}')


if __name__ == '__main__':
    main()
