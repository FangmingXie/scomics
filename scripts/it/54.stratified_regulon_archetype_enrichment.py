"""Expression-stratified enrichment of archetype markers in SCENIC+ regulons (mouse IT).

41 scores each (archetype marker set M_a, regulon target set T_r) pair with a one-sided
Fisher exact test over a universe U of ~16.5k expressed genes. That null assumes M and T are
exchangeable draws from U. They are not -- both are strongly biased toward highly expressed
genes -- so the statistic is partly a function of a modelling choice: shrinking U from 16544
to the 3795 genes that appear in some regulon moves 41's log2 OR by a near-constant +1.59
(IQR [+1.58, +1.62]). Because 41b/41c/41d threshold on ABSOLUTE values (STAR_LOG2OR = 3.0,
ramp [0, 6.5]), that offset silently sets how much of the figure lights up.

This script keeps the same universe but replaces the null. Draw a pseudo-marker set M* by
sampling, within each expression stratum b, exactly m_b = |M & b| genes uniformly without
replacement from the n_b = |U & b| genes in that stratum, independently across strata. Then
with t_b = |T & b|:

    |M* & T| = sum_b |M*_b & T|,   |M*_b & T| ~ Hypergeometric(n_b, t_b, m_b), independent

so the null is a CONVOLUTION OF PER-STRATUM HYPERGEOMETRICS and is available in closed form
-- no Monte Carlo, no seed, no draw count, and exact tail probabilities with no 1/(B+1)
floor:

    exp   = sum_b m_b t_b / n_b
    var   = sum_b m_b (t_b/n_b) (1 - t_b/n_b) (n_b - m_b) / (n_b - 1)
    pmf   = conv_b hypergeom.pmf(0..min(m_b,t_b); n_b, t_b, m_b)
    p     = sum_{k >= x} pmf[k]

41's test is the ONE-STRATUM special case of this (n_1 = N, m_1 = |M|, t_1 = |T| gives
exactly the Fisher null), so this is a strict generalisation rather than a rival method, and
`log2_or` is carried through unchanged for side-by-side comparison.

Why it fixes the universe sensitivity: exp = sum over g in T of the local marker density at
g's expression level. Adding low-expression genes to U puts them in strata where m_b/n_b ~ 0,
so they contribute ~0 to both the observed overlap and the expectation, instead of inflating
N. Measured on L2/3, log2_enr moves +0.199 between the two universes against log2_or's
+1.590 -- 8x more stable.

Discreteness is the one real wrinkle: ~46% of pairs have overlap 0, where the exact one-sided
p is 1.0 exactly. That makes BH conservative, so `p_mid` (mid-p) is the BH input while the
strict `p_strat` is kept in the output.

Reads (per layer, unchanged from 41):
  local_data/res/it/3X.follow.two_*_archetype_markers.tsv   (via 41.LAYERS)
  local_data/res/it/3X.harmony.two_*_coords.tsv
  local_data/res/it/40.yoo25_<layer>_regulon_targets.tsv
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  links/it/superdupermegaRNA_yoo25_IT_P21.h5ad
Outputs:
  local_data/res/it/54.<layer>_stratified_enrichment.tsv     (panel 1: native regulons)
  local_data/res/it/54.l23set_stratified_enrichment.tsv      (panel 2: L2/3 set everywhere)
  local_data/res/it/54.validation_universe_robustness.tsv
  local_data/res/it/54.validation_bin_sensitivity.tsv
  local_data/fig/it/54.log2or_vs_log2enr.pdf
"""

import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import anndata as ad
import scipy.stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
# fonttype 42 embeds TrueType so the PDF keeps real text objects; the default (3) writes
# Type 3 glyph procedures, which Illustrator/Inkscape open as uneditable outlines
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

SCRIPT_41 = os.path.join(SCRIPTS_DIR, 'it', '41.regulon_archetype_enrichment.py')
INPUT_L23_REGULONS = os.path.join(RES_DIR, '40.yoo25_L2_3_regulon_targets.tsv')
OUT_LONG_TMPL = os.path.join(RES_DIR, '54.{layer}_stratified_enrichment.tsv')
OUT_L23SET = os.path.join(RES_DIR, '54.l23set_stratified_enrichment.tsv')
OUT_ROBUSTNESS = os.path.join(RES_DIR, '54.validation_universe_robustness.tsv')
OUT_BIN_SENS = os.path.join(RES_DIR, '54.validation_bin_sensitivity.tsv')
OUT_PDF = os.path.join(FIG_DIR, '54.log2or_vs_log2enr.pdf')

# 20 quantile strata of mean log2-CP10k expression. Both choices are non-critical: over all
# 417 L2/3 pairs the median log2(obs/exp) is -0.205 / -0.226 / -0.238 at 10 / 20 / 40 bins
# and -0.226 / -0.192 / -0.275 for mean / detection-rate / variance, against +0.438 for
# 41's single stratum. The 2-D expression x variance grid is deliberately NOT offered: the
# two covariates are strongly correlated, so it yields empty strata and buys nothing.
N_BINS = 20
BIN_COVARIATE = 'mean'
# validation sweeps (written to OUT_BIN_SENS, not knobs)
BIN_SENS_NBINS = [10, 20, 40]
BIN_SENS_COVARIATES = ['mean', 'detect', 'var']
# the closed form is checked against brute-force resampling on one pair per layer; this is a
# unit test for the convolution indexing, not a way of computing anything
MC_CHECK_DRAWS = 20000
MC_CHECK_RTOL_MEAN = 0.05
MC_CHECK_RTOL_TAIL = 0.10
# mirrors 41b.MASK_MIN_OVERLAP -- the shared-gene floor below which 41b/41d gray a cell out.
# Defined here rather than imported so 54 does not pull plotly in through 41b; the two must
# be kept in step, which is why the comparison figure labels it explicitly.
MASK_MIN_OVERLAP = 5

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_41():
    """Import 41 as a module (its filename is not a valid identifier) for its helpers."""
    spec = importlib.util.spec_from_file_location('script41', SCRIPT_41)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- null

def make_strata(stats, universe_order, n_bins, covariate):
    """Quantile strata of `covariate` over the universe, as an int code per gene.

    Ranked before qcut: the raw values have heavy ties at the low end, on which qcut
    produces unequal and sometimes empty bins.
    """
    v = stats.loc[universe_order, covariate].rank(method='first')
    bins = pd.qcut(v, n_bins, labels=False).values.astype(np.int64)
    counts = np.bincount(bins, minlength=n_bins)
    assert (counts > 0).all(), \
        f'empty stratum with n_bins={n_bins} on {covariate}: counts={counts.tolist()}'
    return bins, counts


def stratified_null(mb, tb, nb, need_pmf=True):
    """Exact null of |M* & T| under stratum-wise resampling of the marker side.

    Returns (mean, sd, pmf). `pmf` is the convolution of the per-stratum hypergeometrics;
    strata with no markers or no targets contribute a point mass at 0 and are skipped.
    """
    assert (mb <= nb).all() and (tb <= nb).all(), 'stratum count exceeds stratum size'
    frac = tb / nb
    mean = float(np.sum(mb * frac))
    var = float(np.sum(mb * frac * (1.0 - frac) * (nb - mb) / np.maximum(nb - 1, 1)))
    if not need_pmf:
        return mean, np.sqrt(var), None

    pmf = np.array([1.0], dtype=np.float64)
    for b in np.nonzero((mb > 0) & (tb > 0))[0]:
        k = np.arange(min(mb[b], tb[b]) + 1)
        pmf = np.convolve(pmf, scipy.stats.hypergeom.pmf(k, nb[b], tb[b], mb[b]))
    assert abs(pmf.sum() - 1.0) < 1e-9, \
        f'convolved pmf lost mass: sum={pmf.sum():.12f} (repeated convolution underflow?)'
    return mean, np.sqrt(var), pmf


def tail_and_midp(pmf, x):
    """One-sided P(X >= x) and its mid-p variant P(X > x) + 0.5 P(X == x)."""
    if x >= len(pmf):
        return 0.0, 0.0
    upper = float(pmf[x:].sum())
    return upper, float(upper - 0.5 * pmf[x])


# --------------------------------------------------------------------------- per layer

def score_layer(layer, bins, counts, M_idx, T_idx, arch_labels, need_pmf=True):
    """All (archetype, regulon) pairs for one layer under one stratification."""
    n_bins = len(counts)
    mb_by_a = {a: np.bincount(bins[idx], minlength=n_bins) for a, idx in M_idx.items()}
    rows = []
    for r, t_idx in T_idx.items():
        tb = np.bincount(bins[t_idx], minlength=n_bins)
        t_set = set(t_idx.tolist())
        for a in arch_labels:
            x = len(t_set & set(M_idx[a].tolist()))
            mean, sd, pmf = stratified_null(mb_by_a[a], tb, counts, need_pmf=need_pmf)
            row = dict(layer=layer, archetype=a, regulon=r, overlap=x, exp=mean, null_sd=sd,
                       log2_enr=float(np.log2((x + 0.5) / (mean + 0.5))),
                       z=float((x - mean) / sd) if sd > 0 else np.nan)
            if need_pmf:
                row['p_strat'], row['p_mid'] = tail_and_midp(pmf, x)
            rows.append(row)
    return pd.DataFrame(rows)


def mc_check(bins, counts, m_idx, t_idx, rng):
    """Brute-force the stratified resampling for one pair; guards the convolution indexing."""
    n_bins = len(counts)
    mb = np.bincount(bins[m_idx], minlength=n_bins)
    tb = np.bincount(bins[t_idx], minlength=n_bins)
    mean, sd, pmf = stratified_null(mb, tb, counts)
    order = np.argsort(bins, kind='stable')
    starts = np.searchsorted(bins[order], np.arange(n_bins + 1))
    t_set = set(t_idx.tolist())
    x_ref = len(t_set & set(m_idx.tolist()))
    draws = np.empty(MC_CHECK_DRAWS, dtype=np.int64)
    for i in range(MC_CHECK_DRAWS):
        tot = 0
        for b in range(n_bins):
            if mb[b] == 0:
                continue
            pool = order[starts[b]:starts[b + 1]]
            tot += len(t_set.intersection(rng.choice(pool, mb[b], replace=False).tolist()))
        draws[i] = tot
    tail_exact, _ = tail_and_midp(pmf, x_ref)
    tail_mc = float((draws >= x_ref).mean())
    print(f'    MC check (B={MC_CHECK_DRAWS}): mean {mean:.4f} vs {draws.mean():.4f}, '
          f'sd {sd:.4f} vs {draws.std():.4f}, P(X>={x_ref}) {tail_exact:.5f} vs {tail_mc:.5f}')
    assert abs(mean - draws.mean()) <= MC_CHECK_RTOL_MEAN * max(mean, 1e-6), \
        f'closed-form mean {mean:.4f} disagrees with resampling {draws.mean():.4f}'
    assert abs(tail_exact - tail_mc) <= MC_CHECK_RTOL_TAIL * max(tail_exact, 1e-3) + 0.005, \
        f'closed-form tail {tail_exact:.5f} disagrees with resampling {tail_mc:.5f}'


def build_layer(cfg, m41, adatas):
    """Load one layer's universe, marker sets and regulon sets as index arrays over U."""
    layer, noc = cfg['layer'], cfg['noc']
    print(f'\n=== {layer} ===')
    coords = pd.read_csv(os.path.join(RES_DIR, cfg['coords']), sep='\t', index_col=0)
    universe, stats = m41.reconstruct_gene_universe(coords.index.values, cfg['subclass_val'],
                                                    adatas)
    U = np.array(sorted(universe))
    gi = {g: i for i, g in enumerate(U)}
    stats = stats.loc[U]

    markers = pd.read_csv(os.path.join(RES_DIR, cfg['markers']), sep='\t')
    arch_labels = [f'archetype_{k + 1}' for k in range(noc)]
    M_raw = {a: set(markers.loc[markers['archetype'] == a, 'gene']) & universe
             for a in arch_labels}
    M_idx = {a: np.array(sorted(gi[g] for g in s), dtype=np.int64) for a, s in M_raw.items()}

    reg = pd.read_csv(os.path.join(RES_DIR, cfg['regulons']), sep='\t')
    reg_meta = reg.drop_duplicates('regulon').set_index('regulon')[['TF', 'regulation_direction']]
    T_raw = {r: set(g['Gene']) & universe for r, g in reg.groupby('regulon')}
    n_total = len(T_raw)
    T_raw = {r: t for r, t in T_raw.items() if len(t) >= m41.MIN_REGULON_GENES}
    T_idx = {r: np.array(sorted(gi[g] for g in s), dtype=np.int64) for r, s in T_raw.items()}
    print(f'    regulons: {len(T_idx)} kept (>= {m41.MIN_REGULON_GENES} in-universe targets) '
          f'of {n_total}; universe N={len(U)}')
    return dict(layer=layer, U=U, gi=gi, stats=stats, arch_labels=arch_labels,
                M_raw=M_raw, M_idx=M_idx, T_raw=T_raw, T_idx=T_idx, reg_meta=reg_meta,
                cfg=cfg)


def decorate(long, L, t_sizes, reg_meta, m41):
    """Attach set sizes, 41's single-stratum statistics, BH-FDR and display labels.

    Shared by both panels so the native and L2/3-set tables cannot drift apart.
    """
    N = len(L['U'])
    long['n_markers'] = long['archetype'].map({a: len(s) for a, s in L['M_raw'].items()})
    long['n_targets'] = long['regulon'].map(t_sizes)
    long['universe'] = N
    long['n_bins'] = N_BINS
    long['bin_covariate'] = BIN_COVARIATE
    long['TF'] = long['regulon'].map(reg_meta['TF'])
    long['regulation_direction'] = long['regulon'].map(reg_meta['regulation_direction'])

    # 41's single-stratum statistics, recomputed here so both live in one table
    long['log2_or'] = [m41.log2_odds_ratio(x, m, t, N) for x, m, t in
                       zip(long['overlap'], long['n_markers'], long['n_targets'])]
    long['pval'] = [scipy.stats.fisher_exact([[x, m - x], [t - x, N - m - t + x]],
                                             alternative='greater')[1]
                    for x, m, t in zip(long['overlap'], long['n_markers'], long['n_targets'])]
    long['fdr'] = multipletests(long['pval'].values, method='fdr_bh')[1]
    # mid-p is the BH input: ~46% of pairs have overlap 0, where the strict one-sided p is
    # exactly 1.0, and that point mass makes plain BH conservative
    long['fdr_strat'] = multipletests(long['p_mid'].values, method='fdr_bh')[1]

    cfg = L['cfg']
    col_label = cfg.get('arch_relabel') or {a: m41.ARCHETYPE_LETTERS[k]
                                            for k, a in enumerate(L['arch_labels'])}
    long['arch_letter'] = long['archetype'].map(col_label)

    cols = ['layer', 'archetype', 'arch_letter', 'regulon', 'TF', 'regulation_direction',
            'overlap', 'n_markers', 'n_targets', 'universe', 'n_bins', 'bin_covariate',
            'exp', 'null_sd', 'log2_enr', 'z', 'p_strat', 'p_mid', 'fdr_strat',
            'log2_or', 'pval', 'fdr']
    return long[cols]


def enrich_layer(L, m41, rng):
    """Panel 1 for one layer: each subclass's own regulons vs its own archetype markers."""
    bins, counts = make_strata(L['stats'], L['U'], N_BINS, BIN_COVARIATE)
    long = score_layer(L['layer'], bins, counts, L['M_idx'], L['T_idx'], L['arch_labels'])
    long = decorate(long, L, {r: len(s) for r, s in L['T_raw'].items()}, L['reg_meta'], m41)

    out = OUT_LONG_TMPL.format(layer=L['layer'])
    long.to_csv(out, sep='\t', index=False)
    n_zero = int((long['overlap'] == 0).sum())
    print(f'    wrote -> {out} ({len(long)} pairs; {n_zero} with overlap 0 '
          f'({n_zero / len(long):.1%}), where p_strat == 1 exactly)')
    print(f'    median log2_or={long["log2_or"].median():+.3f}  '
          f'median log2_enr={long["log2_enr"].median():+.3f}  '
          f'(41 carries a positive median as systematic inflation)')

    # one convolution-vs-resampling check per layer, on the pair with the largest overlap
    top = long.sort_values('overlap', ascending=False).iloc[0]
    mc_check(bins, counts, L['M_idx'][top['archetype']], L['T_idx'][top['regulon']], rng)
    return long


def enrich_l23_set(Ls, m41):
    """Panel 2: the L2/3 regulon target sets tested against EVERY subclass's markers.

    Mirrors 41b.enrich_l23_set_in_layer, but under this script's stratified null. Each layer
    keeps its own universe, strata and marker sets; only the regulon source is held fixed, so
    a row reads across subclasses as one gene set meeting four different marker programmes.
    """
    print('\n=== L2/3 regulon set, applied to every subclass ===')
    reg = pd.read_csv(INPUT_L23_REGULONS, sep='\t')
    reg_meta = reg.drop_duplicates('regulon').set_index('regulon')[['TF', 'regulation_direction']]
    T_all = {r: set(g['Gene']) for r, g in reg.groupby('regulon')}
    print(f'  L2/3 regulon source: {len(T_all)} regulons from {INPUT_L23_REGULONS}')

    longs = []
    for L in Ls:
        universe = set(L['U'])
        gi = L['gi']
        T_raw = {r: t & universe for r, t in T_all.items()}
        T_raw = {r: t for r, t in T_raw.items() if len(t) >= m41.MIN_REGULON_GENES}
        T_idx = {r: np.array(sorted(gi[g] for g in t), dtype=np.int64) for r, t in T_raw.items()}
        bins, counts = make_strata(L['stats'], L['U'], N_BINS, BIN_COVARIATE)
        long = score_layer(L['layer'], bins, counts, L['M_idx'], T_idx, L['arch_labels'])
        long = decorate(long, L, {r: len(t) for r, t in T_raw.items()}, reg_meta, m41)
        print(f"  {L['layer']:5s}: {len(T_idx)} L2/3 regulons kept "
              f'(>= {m41.MIN_REGULON_GENES} targets in this universe) of {len(T_all)}; '
              f'{len(long)} pairs')
        longs.append(long)

    out = pd.concat(longs, ignore_index=True)
    out.to_csv(OUT_L23SET, sep='\t', index=False)
    print(f'  wrote -> {OUT_L23SET} ({len(out)} pairs)')
    return out


# --------------------------------------------------------------------------- validation

def universe_robustness(L, m41):
    """Recompute both statistics under U = expressed genes and U = union of regulon targets.

    The prediction under test: log2_or shifts by ~1.6 between the two, log2_enr by far less.
    """
    rows = []
    U_full = set(L['U'])
    U_targ = set().union(*L['T_raw'].values()) & U_full
    for name, universe in [('expressed', U_full), ('target_union', U_targ)]:
        U = np.array(sorted(universe))
        gi = {g: i for i, g in enumerate(U)}
        stats = L['stats'].loc[U]
        bins, counts = make_strata(stats, U, N_BINS, BIN_COVARIATE)
        M_idx = {a: np.array(sorted(gi[g] for g in s & universe), dtype=np.int64)
                 for a, s in L['M_raw'].items()}
        T_idx = {r: np.array(sorted(gi[g] for g in s & universe), dtype=np.int64)
                 for r, s in L['T_raw'].items()}
        T_idx = {r: t for r, t in T_idx.items() if len(t) >= m41.MIN_REGULON_GENES}
        df = score_layer(L['layer'], bins, counts, M_idx, T_idx, L['arch_labels'],
                         need_pmf=False)
        N = len(U)
        df['n_markers'] = df['archetype'].map({a: len(i) for a, i in M_idx.items()})
        df['n_targets'] = df['regulon'].map({r: len(i) for r, i in T_idx.items()})
        df['log2_or'] = [m41.log2_odds_ratio(x, m, t, N) for x, m, t in
                         zip(df['overlap'], df['n_markers'], df['n_targets'])]
        df['universe_kind'] = name
        df['universe_n'] = N
        rows.append(df.set_index(['archetype', 'regulon']))

    a, b = rows
    j = a.join(b, lsuffix='_full', rsuffix='_targ', how='inner')
    out = []
    for st in ['log2_or', 'log2_enr']:
        d = j[f'{st}_full'] - j[f'{st}_targ']
        rho = j[[f'{st}_full', f'{st}_targ']].corr('spearman').iloc[0, 1]
        out.append(dict(layer=L['layer'], statistic=st, n_pairs=len(j),
                        universe_full=len(U_full), universe_target_union=len(U_targ),
                        median_shift=d.median(), q25_shift=d.quantile(.25),
                        q75_shift=d.quantile(.75), spearman=rho))
        print(f'    {st:9s} universe shift: median {d.median():+.3f} '
              f'[{d.quantile(.25):+.2f}, {d.quantile(.75):+.2f}]  spearman={rho:.4f}')
    return pd.DataFrame(out)


def bin_sensitivity(L):
    """log2_enr under every (n_bins, covariate) combination; a diagnostic, not a knob."""
    rows = []
    for cov in BIN_SENS_COVARIATES:
        for nb in BIN_SENS_NBINS:
            bins, counts = make_strata(L['stats'], L['U'], nb, cov)
            df = score_layer(L['layer'], bins, counts, L['M_idx'], L['T_idx'],
                             L['arch_labels'], need_pmf=False)
            rows.append(dict(layer=L['layer'], covariate=cov, n_bins=nb,
                             median_log2_enr=df['log2_enr'].median(),
                             q25=df['log2_enr'].quantile(.25), q75=df['log2_enr'].quantile(.75)))
    # 41's single stratum, for reference
    bins = np.zeros(len(L['U']), dtype=np.int64)
    df = score_layer(L['layer'], bins, np.array([len(L['U'])]), L['M_idx'], L['T_idx'],
                     L['arch_labels'], need_pmf=False)
    rows.append(dict(layer=L['layer'], covariate='none', n_bins=1,
                     median_log2_enr=df['log2_enr'].median(),
                     q25=df['log2_enr'].quantile(.25), q75=df['log2_enr'].quantile(.75)))
    out = pd.DataFrame(rows)
    span = out[out['covariate'] != 'none']['median_log2_enr']
    print(f'    bin sensitivity: median log2_enr spans {span.min():+.3f}..{span.max():+.3f} '
          f'across {len(span)} schemes; 1 stratum (= 41) gives '
          f'{out[out["covariate"] == "none"]["median_log2_enr"].iloc[0]:+.3f}')
    return out


def plot_comparison(longs):
    """log2_or vs log2_enr per layer, marking the pairs 41b/41d already gray out."""
    fig, axes = plt.subplots(1, len(longs), figsize=(3.5 * len(longs) + 1.0, 3.9),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, long in zip(axes, longs):
        thin = long['overlap'] < MASK_MIN_OVERLAP
        ax.axhline(0, color='0.85', lw=0.8, zorder=0)
        ax.axvline(0, color='0.85', lw=0.8, zorder=0)
        ax.scatter(long.loc[thin, 'log2_or'], long.loc[thin, 'log2_enr'], s=12,
                   facecolor='none', edgecolor='0.70', linewidths=0.6, zorder=2,
                   label=f'overlap < {MASK_MIN_OVERLAP} (masked in 41b/41d)')
        sc = ax.scatter(long.loc[~thin, 'log2_or'], long.loc[~thin, 'log2_enr'],
                        s=18, c=long.loc[~thin, 'overlap'], cmap='viridis',
                        norm=matplotlib.colors.LogNorm(), zorder=3, label='overlap >= 5')
        lim = [min(long['log2_or'].min(), long['log2_enr'].min()) - 0.4,
               max(long['log2_or'].max(), long['log2_enr'].max()) + 0.4]
        ax.plot(lim, lim, color='black', lw=0.8, ls='--', zorder=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(long['layer'].iloc[0], fontsize=10)
        ax.set_xlabel('41  log2 odds ratio', fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('0.6')
    axes[0].set_ylabel('54  log2 enrichment\n(expression-stratified)', fontsize=9)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.018, pad=0.01, aspect=26)
    cbar.set_label('overlap (genes)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    axes[0].legend(loc='upper left', fontsize=7, frameon=False)
    # y above 1 keeps the two-line suptitle clear of the per-panel titles; bbox_inches='tight'
    # then crops back to it
    fig.suptitle('Expression stratification demotes small-overlap pairs that 41 scores as strong\n'
                 'points below the diagonal: enrichment the single-stratum null overstates',
                 fontsize=10, y=1.13)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved {OUT_PDF}')


# --------------------------------------------------------------------------- main

def main():
    m41 = load_41()
    rng = np.random.default_rng(0)

    print('Loading h5ad inputs once...')
    adatas = {d['tag']: ad.read_h5ad(d['path']) for d in m41.DATASETS}
    for d in m41.DATASETS:
        print(f"  {d['tag']:8s}: {adatas[d['tag']].n_obs} cells, {adatas[d['tag']].n_vars} genes")

    Ls, longs, robust, sens = [], [], [], []
    for cfg in m41.LAYERS:
        L = build_layer(cfg, m41, adatas)
        Ls.append(L)
        longs.append(enrich_layer(L, m41, rng))
        robust.append(universe_robustness(L, m41))
        sens.append(bin_sensitivity(L))

    enrich_l23_set(Ls, m41)

    pd.concat(robust, ignore_index=True).to_csv(OUT_ROBUSTNESS, sep='\t', index=False)
    print(f'\n  wrote -> {OUT_ROBUSTNESS}')
    pd.concat(sens, ignore_index=True).to_csv(OUT_BIN_SENS, sep='\t', index=False)
    print(f'  wrote -> {OUT_BIN_SENS}')
    plot_comparison(longs)

    all_long = pd.concat(longs, ignore_index=True)
    rho = all_long[['log2_or', 'log2_enr']].corr('spearman').iloc[0, 1]
    print(f'\n  across all {len(all_long)} pairs: spearman(log2_or, log2_enr) = {rho:.4f}')
    print(f'  significant (fdr_strat<0.05, log2_enr>1, overlap>={MASK_MIN_OVERLAP}): '
          f'{int(((all_long["fdr_strat"] < 0.05) & (all_long["log2_enr"] > 1) & (all_long["overlap"] >= MASK_MIN_OVERLAP)).sum())}')


if __name__ == '__main__':
    main()
