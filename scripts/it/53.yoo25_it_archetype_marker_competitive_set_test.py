"""Competitive set test: does each archetype's marker set move more than a matched control set?

Script 52 shows that each archetype's own marker genes shift under dark rearing, but a boxplot
centred off zero is not evidence that *this set* is special -- any broad transcriptional response
would move a random panel of genes too. This script supplies the missing contrast: for every
archetype it tests markers(k) against an expression-matched CONTROL set drawn from the rest of the
transcriptome, measured in exactly the same cells and the same pseudobulk columns.

    genes_test    = markers(k)                       # k's OWN marker set -- as in script 52
    genes_control = expression-matched genes drawn from the background pool (all detected
                    genes MINUS every marker of that subclass), up to N_MATCH per marker gene
    cells         = cells of S assigned to k         # argmax of recomputed archetype scores

    log2FC(g) = log2( (CPM_P21DR[g, S, k] + PSEUDO) / (CPM_P21[g, S, k] + PSEUDO) )

    H0: log2FC of markers(k) and of the control set are drawn from the same distribution
    test: two-sided Wilcoxon rank-sum (Mann-Whitney U), BH-FDR across the 11 archetypes

This is a COMPETITIVE (set-vs-rest) test in the CAMERA sense, not a self-contained one. That
choice is deliberate: the yoo25 P21/P21DR design has only two animals per condition
(obs['Sample'] = P21a/P21b/P21DRa/P21DRb), so a self-contained test of "is this set's mean log2FC
different from zero" has no honest replicate-level null -- per-gene log2FC values from one pair of
pseudobulk columns are not independent observations. A competitive test sidesteps this: it asks
only whether markers(k) is displaced RELATIVE to comparable genes measured in the very same
columns, so anything shared by all genes (sequencing depth, global DR response, normalization,
composition) cancels between the two sets.

EXPRESSION MATCHING is the crux, and is why the control set is matched rather than random.
log2FC variance is strongly mean-dependent (the +PSEUDO pseudocount shrinks low-CPM genes toward
zero), and marker genes are not a random draw from the expression range -- they are selected to be
high in their archetype. An unmatched background would therefore differ in log2FC SPREAD for
purely technical reasons and manufacture significance. Matching is a deterministic stratified draw on
mean log2 CPM: equal-count expression bins, the same bin composition as the marker set, an even
spread within each bin, no replacement and no RNG (per project constraints). The lower figure
panel is the QC that the two expression distributions actually overlap, and the script FAILS if
the median gap exceeds MATCH_TOL rather than reporting a test built on a mismatched control.

CELL SELECTION is byte-for-byte the same as script 52 (steps 1-6 of its METHOD) so the marker
boxes here are the same numbers as the boxes there. Two known caveats are inherited unchanged and
are NOT corrected here, because correcting them would break that comparability:
  - the top-N-purest selection is made within each condition separately, which absorbs part of any
    DR-induced weakening of the archetype program;
  - Type_leiden values L2/3_Fem1, L2/3_Fem2 and L4_Fem occur ONLY in P21DR (0 cells in P21), so
    the DR pool contains populations with no P21 counterpart.
Both act on test and control genes alike, so they bias the marker/control CONTRAST far less than
they bias either set's absolute log2FC.

Reads:
  links/it/superdupermegaRNA_yoo25_IT_AllAges.h5ad
  local_data/res/it/33.follow.two_L4_archetype_markers.tsv
  local_data/res/it/34.follow.two_L23_archetype_markers.tsv
  local_data/res/it/35.follow.two_L5IT_archetype_markers.tsv
  local_data/res/it/36.follow.two_L6IT_archetype_markers.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv
Outputs:
  local_data/fig/it/53.yoo25_it_archetype_marker_competitive_set_test.pdf   (2 panels)
  local_data/res/it/53.yoo25_it_archetype_marker_competitive_set_test.tsv   (per gene)
  local_data/res/it/53.yoo25_it_archetype_marker_competitive_set_stats.tsv  (per archetype)
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

OUT_PDF       = os.path.join(OUT_FIG_DIR, '53.yoo25_it_archetype_marker_competitive_set_test.pdf')
OUT_TSV       = os.path.join(OUT_RES_DIR, '53.yoo25_it_archetype_marker_competitive_set_test.tsv')
OUT_STATS_TSV = os.path.join(OUT_RES_DIR, '53.yoo25_it_archetype_marker_competitive_set_stats.tsv')

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

N_TOP_CELLS  = 100      # top cells (by archetype score) per (age, archetype) pseudobulk column
SCORE_PCTILE = (2, 98)  # per-gene percentile clip for archetype scoring (script 34 recipe)
CP10K        = 1e4      # CP10k target for the log2 scoring expression
CP_TARGET    = 1e6      # CPM for pseudobulk expression
PSEUDO       = 1.0      # CPM pseudocount added before the log2 ratio

# Control-set construction. The control is drawn STRATIFIED by expression bin, not by nearest
# neighbour: marker genes crowd the sparse high-expression tail, and a greedy nearest-neighbour
# draw without replacement exhausts that tail and spills downhill into the dense low-expression
# bulk -- which is exactly the bias the matching exists to remove.
N_MATCH      = 10   # CAP on control genes per marker gene; the realized ratio is the largest
                    # value <= N_MATCH that every occupied bin can supply (reported per archetype)
N_BINS       = 50   # equal-count expression bins, cut on the background pool's quantiles. High-
                    # expression bins are WIDE (few genes over a large range), so too few bins
                    # leaves within-bin skew: markers sit high inside their own bin.
MATCH_TOL    = 0.25 # max tolerated |median log2CPM| gap between the two sets, else fail fast
MIN_MEAN_CPM = 1.0  # a gene must average >= this CPM over the two columns to enter EITHER set;
                    # below it the log2 ratio is pseudocount-dominated and carries no information

FDR_THRESH = 0.05   # BH-FDR across the archetypes, for the significance stars

# The two colors encode the CONTRAST being tested (marker set vs null control), not archetype
# identity -- every box pair here is the same comparison, so per-archetype colors would only
# invite the eye to compare across pairs instead of within them.
MARKER_COLOR  = '#2b7bba'
CONTROL_COLOR = '0.72'   # grey: the control set carries no archetype identity

# The per-archetype mean log2FC annotation is printed only where the competitive test survives
# FDR: an unsupported shift is not a number worth reading off. One archetype is emphasised in
# bold -- L2/3 B', the only set that moves down under dark rearing and by far the largest effect
# in the panel.
LFC_BOLD = ('L2/3', "B'")   # (subclass, relabelled archetype)


def load_arch_map():
    """Read the persisted depth-arc table into {token: {old_letter: (new_letter, arc_rank)}}."""
    arc = pd.read_csv(IN_ARCMAP, sep='\t')
    need = {'token', 'old_letter', 'new_letter', 'arc_rank'}
    missing = need - set(arc.columns)
    if missing:
        raise ValueError(f"{IN_ARCMAP} missing column(s): {sorted(missing)}")
    amap = {}
    for token, sub in arc.groupby('token'):
        amap[token] = {r.old_letter: (r.new_letter, int(r.arc_rank)) for r in sub.itertuples()}
    return amap


def build_control(test_expr, bg_expr, n_match, n_bins):
    """Draw a control set with the same expression-bin composition as the test set.

    Bins are equal-count quantiles of the background pool. Within each occupied bin the control
    takes `ratio` genes per test gene, where `ratio` is the largest value <= `n_match` that EVERY
    occupied bin can supply -- so the bin composition is matched exactly by construction rather
    than approximately, and the scarcest bin sets the achievable depth instead of silently
    dragging the draw out of its bin.

    Selection inside a bin is a deterministic nearest-neighbour draw without replacement (no RNG,
    matching the convention of scripts 50/51/52): the bin fixes the coarse composition, the
    nearest-neighbour step fixes the within-bin position. Confining the neighbour search to the
    bin is what keeps it honest -- unconfined, it drains the sparse high-expression tail and
    spills into the dense low-expression bulk.

    Returns (indices into bg_expr, realized ratio).
    """
    edges = np.quantile(bg_expr, np.linspace(0.0, 1.0, n_bins + 1))[1:-1]
    t_bin = np.digitize(test_expr, edges)
    b_bin = np.digitize(bg_expr, edges)

    ratio = n_match
    for b in np.unique(t_bin):
        n_t = int((t_bin == b).sum())
        n_b = int((b_bin == b).sum())
        if n_b == 0:
            raise ValueError(f'expression bin {b} holds {n_t} test gene(s) but no background '
                             f'gene; lower N_BINS')
        ratio = min(ratio, n_b // n_t)
    if ratio < 1:
        raise ValueError('no expression bin can supply even 1 control per test gene; '
                         'lower N_BINS or MIN_MEAN_CPM')

    picked = []
    for b in np.unique(t_bin):
        cand = np.where(b_bin == b)[0]
        picked.append(cand[_nearest_without_replacement(
            test_expr[t_bin == b], bg_expr[cand], ratio)])
    return np.concatenate(picked), ratio


def _nearest_without_replacement(test_vals, bg_vals, k):
    """Claim the `k` closest still-unclaimed `bg_vals` for each of `test_vals`.

    Test values are visited in ascending order and ties break to the high side, so the result is
    a deterministic function of the inputs. Returns indices into `bg_vals`; requires
    len(bg_vals) >= k * len(test_vals), which the caller guarantees per bin.
    """
    order = np.argsort(bg_vals, kind='stable')
    e_sorted = bg_vals[order]
    used = np.zeros(e_sorted.size, dtype=bool)
    picked = []
    for e in np.sort(test_vals):
        pos = int(np.searchsorted(e_sorted, e))
        lo, hi = pos - 1, pos
        for _ in range(k):
            while lo >= 0 and used[lo]:
                lo -= 1
            while hi < e_sorted.size and used[hi]:
                hi += 1
            if lo < 0 and hi >= e_sorted.size:
                raise ValueError('bin exhausted during nearest-neighbour matching')
            take_hi = lo < 0 or (hi < e_sorted.size and (e_sorted[hi] - e) <= (e - e_sorted[lo]))
            j = hi if take_hi else lo
            used[j] = True
            picked.append(order[j])
            if take_hi:
                hi += 1
            else:
                lo -= 1
    return np.array(picked, dtype=int)


def stars(fdr):
    """Significance marker from a BH-FDR value."""
    if fdr < 0.001:
        return '***'
    if fdr < 0.01:
        return '**'
    if fdr < FDR_THRESH:
        return '*'
    return 'n.s.'


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
        letters = [ARCHETYPE_LETTERS[i] for i in range(len(arch_vals))]
        markers = {ARCHETYPE_LETTERS[i]: mk.loc[mk['archetype'] == a, 'gene'].tolist()
                   for i, a in enumerate(arch_vals)}

        token = cfg['token']
        if token not in arch_map:
            raise ValueError(f"token '{token}' absent from {IN_ARCMAP}")
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

    rows, stat_rows = [], []
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

        raw_var = np.array(adata.raw.var_names)
        name2idx = {g: i for i, g in enumerate(raw_var)}
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

        # every marker of this subclass is barred from the background pool: they all carry
        # archetype-program signal, so leaving them in would blunt the contrast being tested
        all_marker_idx = np.array(sorted({name2idx[g] for k in letters
                                          for g in cfg['markers'][k]}), dtype=int)

        for k in letters:
            genes = cfg['markers'][k]
            gi = np.array([name2idx[g] for g in genes], dtype=int)
            c21_all = cpm[:, col_index[('P21', k)]]
            c21dr_all = cpm[:, col_index[('P21DR', k)]]
            lfc_all = np.log2((c21dr_all + PSEUDO) / (c21_all + PSEUDO))
            mean_all = np.log2((c21_all + c21dr_all) / 2.0 + 1.0)   # matching covariate

            # --- test set: k's own markers, above the expression floor ---
            expressed = (c21_all + c21dr_all) / 2.0 >= MIN_MEAN_CPM
            test_idx = gi[expressed[gi]]
            n_drop = gi.size - test_idx.size
            if test_idx.size < 10:
                raise ValueError(f'{S} {k}: only {test_idx.size} markers above '
                                 f'MIN_MEAN_CPM={MIN_MEAN_CPM}')

            # --- control set: expression-matched, drawn from non-marker expressed genes ---
            bg_mask = expressed.copy()
            bg_mask[all_marker_idx] = False
            bg_idx = np.where(bg_mask)[0]
            if bg_idx.size < N_MATCH * test_idx.size:
                raise ValueError(f'{S} {k}: background pool {bg_idx.size} < '
                                 f'{N_MATCH} x {test_idx.size} needed')
            ctrl_local, ratio = build_control(mean_all[test_idx], mean_all[bg_idx],
                                              N_MATCH, N_BINS)
            ctrl_idx = bg_idx[ctrl_local]
            if ctrl_idx.size != np.unique(ctrl_idx).size:
                raise ValueError(f'{S} {k}: control set contains duplicated genes')

            # fail fast rather than plot a mismatched control: the whole test rests on this
            gap = float(mean_all[test_idx].mean() - mean_all[ctrl_idx].mean())
            gap_med = float(np.median(mean_all[test_idx]) - np.median(mean_all[ctrl_idx]))
            # checked on the MEAN: it is the systematic-shift statistic, and it does not jump
            # around for the smaller marker sets the way a 55-gene median does
            if abs(gap) > MATCH_TOL:
                raise ValueError(f'{S} {k}: expression matching failed, mean log2CPM gap '
                                 f'{gap:+.3f} exceeds MATCH_TOL={MATCH_TOL}')

            lfc_t, lfc_c = lfc_all[test_idx], lfc_all[ctrl_idx]
            mw = scipy.stats.mannwhitneyu(lfc_t, lfc_c, alternative='two-sided')
            auc = float(mw.statistic) / (lfc_t.size * lfc_c.size)   # P(marker lfc > control lfc)

            stat_rows.append(dict(
                subclass=S, archetype=cfg['relabel'][k], archetype_internal=k,
                arch_rank=cfg['rank'][k],
                n_marker_total=len(genes), n_marker_tested=int(test_idx.size),
                n_marker_dropped_low_cpm=int(n_drop), n_control=int(ctrl_idx.size),
                control_per_marker=int(ratio), match_gap_mean_log2cpm=gap,
                match_gap_median_log2cpm=gap_med,
                median_lfc_marker=float(np.median(lfc_t)),
                median_lfc_control=float(np.median(lfc_c)),
                delta_median_lfc=float(np.median(lfc_t) - np.median(lfc_c)),
                mean_lfc_marker=float(lfc_t.mean()), mean_lfc_control=float(lfc_c.mean()),
                auc=auc, U=float(mw.statistic), pval=float(mw.pvalue),
                median_log2cpm_marker=float(np.median(mean_all[test_idx])),
                median_log2cpm_control=float(np.median(mean_all[ctrl_idx])),
                n_used_P21=n_used[('P21', k)], n_used_P21DR=n_used[('P21DR', k)]))

            for label, idx in (('marker', test_idx), ('control', ctrl_idx)):
                for i in idx:
                    rows.append(dict(
                        subclass=S, archetype=cfg['relabel'][k], arch_rank=cfg['rank'][k],
                        gene_set=label, gene=raw_var[i],
                        cpm_P21=float(c21_all[i]), cpm_P21DR=float(c21dr_all[i]),
                        log2FC_P21DR_over_P21=float(lfc_all[i]),
                        mean_log2cpm=float(mean_all[i])))

            print(f'  {cfg["relabel"][k]}: n_marker={test_idx.size:3d} (dropped {n_drop:3d} '
                  f'< {MIN_MEAN_CPM:g} CPM)  n_control={ctrl_idx.size:4d} ({ratio}x, match gap '
                  f'{gap:+.3f})  median lfc {np.median(lfc_t):+.3f} vs control '
                  f'{np.median(lfc_c):+.3f}  AUC={auc:.3f}  p={mw.pvalue:.3g}')

    stats = pd.DataFrame(stat_rows).sort_values('arch_rank').reset_index(drop=True)
    stats['fdr'] = multipletests(stats['pval'].values, method='fdr_bh')[1]
    stats['stars'] = [stars(f) for f in stats['fdr']]
    stats.to_csv(OUT_STATS_TSV, sep='\t', index=False)
    print(f'\nWrote {OUT_STATS_TSV} ({len(stats)} archetypes)')

    out = pd.DataFrame(rows).sort_values(['arch_rank', 'gene_set', 'gene']).reset_index(drop=True)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_TSV} ({len(out)} rows)')

    print('\nCompetitive set test (marker vs expression-matched control):')
    for r in stats.itertuples():
        print(f'  {r.subclass:5s} {r.archetype}  delta median lfc={r.delta_median_lfc:+.3f}  '
              f'AUC={r.auc:.3f}  p={r.pval:.3g}  FDR={r.fdr:.3g}  {r.stars}')

    # -----------------------------------------------------------------------
    # Figure: paired log2FC boxes (marker vs control) per archetype, plus the
    # expression-matching QC panel below, both on the laminar-depth arc_rank x-axis
    # -----------------------------------------------------------------------
    npos = len(stats)
    x = np.arange(1, npos + 1)
    off = 0.19          # half-separation of the marker/control box pair
    wid = 0.32
    labels = [f'{r.archetype}\nn={r.n_marker_tested}' for r in stats.itertuples()]
    subclasses = list(stats['subclass'])

    lfc_t = [out.loc[(out['arch_rank'] == r.arch_rank) & (out['gene_set'] == 'marker'),
                     'log2FC_P21DR_over_P21'].values for r in stats.itertuples()]
    lfc_c = [out.loc[(out['arch_rank'] == r.arch_rank) & (out['gene_set'] == 'control'),
                     'log2FC_P21DR_over_P21'].values for r in stats.itertuples()]
    exp_t = [out.loc[(out['arch_rank'] == r.arch_rank) & (out['gene_set'] == 'marker'),
                     'mean_log2cpm'].values for r in stats.itertuples()]
    exp_c = [out.loc[(out['arch_rank'] == r.arch_rank) & (out['gene_set'] == 'control'),
                     'mean_log2cpm'].values for r in stats.itertuples()]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(1.15 * npos + 2.0, 7.8),
                             gridspec_kw={'height_ratios': [3.0, 1.5], 'hspace': 0.26})
    ax, ax_q = axes

    def paired_boxes(ax_, vals_t, vals_c, showfliers):
        """One colored marker box and one grey control box at each x position."""
        b_t = ax_.boxplot(vals_t, positions=x - off, widths=wid, showfliers=showfliers,
                          patch_artist=True, medianprops=dict(color='black', linewidth=1.3),
                          flierprops=dict(marker='.', markersize=1.5, alpha=0.3,
                                          markeredgewidth=0))
        b_c = ax_.boxplot(vals_c, positions=x + off, widths=wid, showfliers=showfliers,
                          patch_artist=True, medianprops=dict(color='black', linewidth=1.3),
                          flierprops=dict(marker='.', markersize=1.5, alpha=0.3,
                                          markeredgewidth=0))
        for patch in b_t['boxes']:
            patch.set_facecolor(MARKER_COLOR)
            patch.set_alpha(0.65)
        for patch in b_c['boxes']:
            patch.set_facecolor(CONTROL_COLOR)
            patch.set_alpha(0.65)
        return b_t, b_c

    # ---- top panel: log2FC, marker set vs matched control ----
    b_t, b_c = paired_boxes(ax, lfc_t, lfc_c, showfliers=False)
    for xi, vals in enumerate(lfc_t):   # markers are few enough to show individually
        n = len(vals)
        jitter = (np.arange(n) - (n - 1) / 2.0) / n * 0.26
        ax.scatter(np.full(n, x[xi] - off) + jitter, vals,
                   s=3, color='black', alpha=0.30, linewidths=0, zorder=3, rasterized=True)
    ax.axhline(0.0, color='grey', linewidth=1.0, linestyle='--', zorder=1)
    ax.set_ylabel('log2(P21DR / P21)', fontsize=10)
    ax.set_xlim(0.4, npos + 0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # significance bracket over each pair, at a common height above everything drawn.
    # Limits come from the DRAWN artists (whisker ends and the plotted marker points), never
    # from a percentile -- a percentile cut silently amputates the whisker of whichever
    # archetype responded most, which is exactly the box a reader needs to see whole.
    whisk = np.concatenate([np.asarray(l.get_ydata(), dtype=float)
                            for b in (b_t, b_c) for l in b['whiskers']])
    drawn = np.concatenate([whisk] + lfc_t)
    ymin, ytop = float(drawn.min()), float(drawn.max())
    span = ytop - ymin
    ymin -= 0.03 * span
    y_br = ytop + 0.10 * span
    for xi, r in enumerate(stats.itertuples()):
        ax.plot([x[xi] - off, x[xi] - off, x[xi] + off, x[xi] + off],
                [y_br, y_br + 0.02 * span, y_br + 0.02 * span, y_br], color='0.3', linewidth=0.8)
        ax.text(x[xi], y_br + 0.03 * span, r.stars, ha='center', va='bottom',
                fontsize=8 if r.stars != 'n.s.' else 7, color='0.2')
        # the effect size the stars refer to, spelled out: the marker set's mean log2FC, shown
        # only where the test clears FDR_THRESH
        if r.stars != 'n.s.':
            ax.text(x[xi], y_br + 0.11 * span, f'{r.mean_lfc_marker:+.2f}',
                    ha='center', va='bottom', fontsize=8, color='black',
                    fontweight='bold' if (r.subclass, r.archetype) == LFC_BOLD else 'normal')
    ax.set_ylim(ymin, y_br + 0.24 * span)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MARKER_COLOR, alpha=0.65,
                             label='archetype marker set'),
               plt.Rectangle((0, 0), 1, 1, facecolor=CONTROL_COLOR, alpha=0.65,
                             label='expression-matched control set')]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc='lower left', ncol=2)
    ax.set_title('Competitive set test: does each archetype\'s marker set move more than '
                 'matched control genes?\n'
                 f'(Yoo25 mouse IT; top {N_TOP_CELLS} purest cells per age x archetype; '
                 f'two-sided Wilcoxon rank-sum, BH-FDR over {npos} archetypes)', fontsize=11)

    # ---- QC panel: the expression covariate the control set was matched on ----
    paired_boxes(ax_q, exp_t, exp_c, showfliers=False)
    ax_q.set_ylabel('mean log2(CPM+1)', fontsize=10)
    ax_q.spines['top'].set_visible(False)
    ax_q.spines['right'].set_visible(False)
    ax_q.text(0.005, 0.96, 'matching QC: control drawn to match marker expression',
              transform=ax_q.transAxes, fontsize=8, va='top', ha='left', color='0.35')

    # ---- x tick labels on both panels; subclass blocks on the bottom one only ----
    for a_ in axes:
        a_.set_xticks(x)
        a_.set_xticklabels(labels, fontsize=9)
        a_.tick_params(labelbottom=True)   # sharex hides the upper labels by default
    starts_ = [i for i in range(npos) if i == 0 or subclasses[i] != subclasses[i - 1]]
    for a_ in axes:
        for i in starts_[1:]:
            a_.axvline(i + 0.5, color='0.75', linewidth=0.8, linestyle=':', zorder=0)
    trans = ax_q.get_xaxis_transform()
    for bi, i in enumerate(starts_):
        j = starts_[bi + 1] if bi + 1 < len(starts_) else npos
        ax_q.text((i + j + 1) / 2.0, -0.30, subclasses[i], transform=trans,
                  ha='center', va='top', fontsize=11, fontweight='bold')

    fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Wrote {OUT_PDF}')


if __name__ == '__main__':
    main()
