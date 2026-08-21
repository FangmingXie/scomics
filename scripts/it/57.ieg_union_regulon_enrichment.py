"""The `(IEGs)` super-regulon: the union of the eight IEG TF target sets, scored like a regulon.

54b/54d/54e draw one row per TF. The eight immediate-early TFs

    Fos  Fosb  Fosl2  Junb  Egr1  Egr2  Egr3  Egr4

behave as one programme rather than eight independent ones -- 56 shows their regulons are the
part of the L2/3 B' result that survives swapping the regulon catalogue -- but no row in any
existing table represents the programme itself. This script builds that row: the UNION of the
eight activating target sets, scored against the archetype markers by exactly the machinery
54 uses for a real regulon, and written in the same long-table schema so 54d/54e can append it.

The union is taken over target sets, not over the per-TF results. Those are different things:
the eight regulons overlap heavily, so summing their overlaps would count shared genes many
times, and a union of 8 x |T| genes has a correspondingly larger expected overlap `exp` --
which the stratified null charges for automatically. That is the whole reason to score the
union rather than combine per-TF statistics after the fact.

WHY THIS IS A SEPARATE SCRIPT AND NOT A REGULON ADDED TO 54/55

Adding `(IEGs)` to 54's regulon dict would be a two-line change, and would be wrong twice over:

  1. BH-FDR in 54 is computed per layer over every pair in that layer's table. Injecting the
     union would shift `fdr_strat` for all ~420 pairs of every layer, in both catalogues,
     changing committed results -- including the tables 55's consolidation gate checks against
     42-45 and the significance calls 54b/54c/56 already report.
  2. The union is a SUMMARY of eight rows that are already in that family, not a ninth
     independent test. Multiple-testing correction over a family containing both a set and
     its own summary does not mean anything.

So the committed tables are left untouched and the union is scored separately. Its FDR is
still made comparable: `fdr_strat` here is BH over that layer's committed p-values POOLED with
the union's, and the script asserts that pooling does not flip the significance of any
committed cell (adding 2-3 p-values to a family of ~420 shifts the BH thresholds by well under
1%). Without that, a black outline would mean a different evidence bar on the union row than
on every row above it in the same figure.

Both of 54d's panels are produced, since a row missing from panel 2 would read as untested:

  native   each layer's own IEG regulons, unioned, against that layer's markers
  l23set   the L2/3 IEG union, applied to every layer (54's panel-2 construction)

Regulons are taken from whichever of the eight TFs the catalogue actually carries for that
layer; gao25 L6IT has only some. The contributing TFs are written into every row rather than
being silently dropped, and printed per layer.

Reads (via 54/55, which own these paths):
  local_data/res/it/3X.follow.two_*_archetype_markers.tsv, 3X.harmony.two_*_coords.tsv
  local_data/res/it/40.yoo25_<layer>_regulon_targets.tsv          (yoo25)
  links/it/regulon_gene_table_gao25_v1<token>.csv                 (gao25)
  links/it/superdupermegaRNA_*.h5ad
  local_data/res/it/5{4,5}.<layer>_stratified_enrichment.tsv      (pooled-BH reference)
  local_data/res/it/5{4,5}.l23set_stratified_enrichment.tsv
Outputs:
  local_data/res/it/57.yoo25_iegunion_stratified_enrichment.tsv
  local_data/res/it/57.gao25_iegunion_stratified_enrichment.tsv
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import anndata as ad
from statsmodels.stats.multitest import multipletests

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')

SCRIPT_54 = os.path.join(SCRIPTS_DIR, 'it', '54.stratified_regulon_archetype_enrichment.py')
SCRIPT_55 = os.path.join(SCRIPTS_DIR, 'it', '55.gao25_stratified_regulon_archetype_enrichment.py')
OUT_TMPL = os.path.join(RES_DIR, '57.{tag}_iegunion_stratified_enrichment.tsv')

# The eight classic immediate-early TFs: AP-1 (Fos/Fosb/Fosl2/Junb) and Egr1-4. This is
# 41b.SELECTED_TFS minus Atf6 and Smad3, which are stress-responsive but neither immediate nor
# early -- the same strict core 56 reports its IEG fractions on. Asserted against 41b in main()
# so the two definitions cannot drift.
IEG_TFS = ['Fos', 'Fosb', 'Fosl2', 'Junb', 'Egr1', 'Egr2', 'Egr3', 'Egr4']
SIGN = '+/+'                    # activating only, as everywhere else in this family
UNION_TF = '(IEGs)'             # the display name; parenthesised so it cannot collide with a
UNION_REGULON = f'{UNION_TF}_{SIGN}'                                  # real gene symbol
# Pooling the union's p-values into a committed layer's BH family barely moves that family's
# BH thresholds, but "barely" is not "not at all": BH is a step-up procedure, so adding a very
# small p-value can raise the cutoff and let one more committed cell through. Every such cell
# is printed, and the run fails if they ever amount to more than MAX_FLIP_FRAC of the family --
# past that the union row's outline would no longer mean what the outlines above it mean.
STAR_FDR = 0.05
MAX_FLIP_FRAC = 0.01

os.makedirs(RES_DIR, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def union_targets(reg, where):
    """Union of the activating target sets of whichever IEG TFs this table carries."""
    sub = reg[(reg['TF'].isin(IEG_TFS)) & (reg['regulation_direction'] == SIGN)]
    used = sorted(set(sub['TF']), key=IEG_TFS.index)
    assert used, f'{where}: none of {IEG_TFS} has a {SIGN} regulon'
    genes = set(sub['Gene'])
    per_tf = sub.groupby('TF')['Gene'].nunique().reindex(used)
    print(f'    {where}: {len(used)}/{len(IEG_TFS)} TFs -> {len(genes)} union targets '
          f'from {int(per_tf.sum())} with repeats ({", ".join(used)})')
    return genes, used


def score_union(m54, m41, L, t_genes, used, panel, primed):
    """One layer x one union gene set: 54's scoring, decoration and long-table schema."""
    universe = set(L['U'])
    t_set = t_genes & universe
    assert len(t_set) >= m41.MIN_REGULON_GENES, \
        f"{L['layer']}/{panel}: union has {len(t_set)} in-universe targets, below " \
        f'{m41.MIN_REGULON_GENES}'
    t_idx = np.array(sorted(L['gi'][g] for g in t_set), dtype=np.int64)

    bins, counts = m54.make_strata(L['stats'], L['U'], m54.N_BINS, m54.BIN_COVARIATE)
    long = m54.score_layer(L['layer'], bins, counts, L['M_idx'], {UNION_REGULON: t_idx},
                           L['arch_labels'])
    reg_meta = pd.DataFrame({'TF': [UNION_TF], 'regulation_direction': [SIGN]},
                            index=[UNION_REGULON])
    # decorate() runs BH over whatever it is handed; here that is only the union's own rows,
    # so its `fdr_strat` and `fdr` are placeholders -- pool_fdr() replaces them below.
    long = m54.decorate(long, L, {UNION_REGULON: len(t_set)}, reg_meta, m41)
    long['panel'] = panel
    long['union_tfs'] = ','.join(used)
    long['n_union_tfs'] = len(used)
    # 54's `arch_letter` is the fit-order label; every figure in this family instead shows
    # 41b's published depth-arc letter, and for L4 and L5IT the two orderings are REVERSED
    # (L5IT archetype_1 is `A` here and `B'` there). Both are carried so a row read out of
    # this table cannot be matched to the wrong column of 54d/54e.
    long['arch_primed'] = long['archetype'].map(primed[L['layer']])
    return long


def pool_fdr(long, committed, where):
    """Re-derive the union's BH-FDR inside the committed layer's testing family.

    The union's raw p-values are correct on their own; what is not comparable is a BH
    correction over 3 tests placed next to rows corrected over ~420. So BH is run over the
    committed p-values with the union's appended, and only the union's results are kept.
    """
    n_before = len(committed)
    pooled = np.concatenate([committed['p_mid'].values, long['p_mid'].values])
    fdr = multipletests(pooled, method='fdr_bh')[1]
    was = committed['fdr_strat'].values < STAR_FDR
    now = fdr[:n_before] < STAR_FDR
    flipped = committed[was != now]
    if len(flipped):
        for _, r in flipped.iterrows():
            print(f"      note: {where} {r['regulon']} x {r['archetype']} "
                  f"{'gains' if not r['fdr_strat'] < STAR_FDR else 'loses'} significance when "
                  f"the union joins its BH family ({r['fdr_strat']:.4f} committed)")
    assert len(flipped) <= MAX_FLIP_FRAC * n_before, \
        f'{where}: pooling the union moved {len(flipped)} of {n_before} committed ' \
        f'significance calls (> {MAX_FLIP_FRAC:.0%}); the union row can no longer share an ' \
        'outline rule with the rows above it'
    long = long.copy()
    long['n_committed_flipped'] = len(flipped)
    long['fdr_strat'] = fdr[n_before:]
    long['fdr'] = multipletests(np.concatenate([committed['pval'].values,
                                                long['pval'].values]),
                                method='fdr_bh')[1][n_before:]
    long['fdr_family_n'] = n_before + len(long)
    return long


def run_catalogue(m54, m41, adatas, dcfg, committed_native_tmpl, committed_l23set, tag, primed):
    print(f'\n########## {tag} ##########')
    l23_union, l23_used = union_targets(dcfg['load_l23'](), f'{tag} L2/3 source (panel 2)')

    committed_l23 = pd.read_csv(committed_l23set, sep='\t')
    out = []
    for cfg in m41.LAYERS:
        L = m54.build_layer(cfg, m41, adatas, dcfg)
        layer = L['layer']

        nat_union, nat_used = union_targets(dcfg['load_regulons'](cfg), f'{layer} native')
        nat = score_union(m54, m41, L, nat_union, nat_used, 'native', primed)
        nat = pool_fdr(nat, pd.read_csv(committed_native_tmpl.format(layer=layer), sep='\t'),
                       f'{tag}/{layer}/native')

        l23 = score_union(m54, m41, L, l23_union, l23_used, 'l23set', primed)
        l23 = pool_fdr(l23, committed_l23[committed_l23['layer'] == layer],
                       f'{tag}/{layer}/l23set')

        for panel, df in [('native', nat), ('l23set', l23)]:
            for _, r in df.iterrows():
                print(f"    {panel:7s} {r['arch_primed']:3s}  overlap {int(r['overlap']):4d}"
                      f"/{int(r['n_markers']):4d}  |T|={int(r['n_targets']):4d}  "
                      f"exp={r['exp']:7.2f}  log2_enr={r['log2_enr']:+.2f}  "
                      f"fdr={r['fdr_strat']:.2e}")
        out += [nat, l23]

    long = pd.concat(out, ignore_index=True)
    path = OUT_TMPL.format(tag=tag)
    long.to_csv(path, sep='\t', index=False)
    print(f'  wrote -> {path} ({len(long)} rows)')
    return long


def load_union_long(path, panel, primed, m41b):
    """This script's rows for one panel, keyed by 54d/54e's column labels.

    Lives here rather than in the figure scripts so the four of them (54d, 54e and the 55x
    entry points) share one definition of what the `(IEGs)` row is. The returned frame carries
    the full long-table schema, so 54b.to_matrices and 54e.layer_matrices consume it exactly
    as they consume a real regulon.
    """
    assert os.path.exists(path), f'missing {path}; run 57 first'
    long = pd.read_csv(path, sep='\t')
    long = long[long['panel'] == panel]
    assert len(long), f'{path}: no rows for panel {panel!r}'
    assert (long['TF'] == UNION_TF).all(), f'{path}: unexpected TF in the union table'
    return m41b.to_col(long, primed)


def main():
    m54 = load_module(SCRIPT_54, 'script54')
    m55 = load_module(SCRIPT_55, 'script55')
    m41 = m54.load_41()
    m41b = load_module(os.path.join(SCRIPTS_DIR, 'it',
                                    '41b.selected_regulon_archetype_enrichment.py'), 'script41b')
    assert set(IEG_TFS) < set(m41b.SELECTED_TFS), \
        f'IEG_TFS must be a subset of 41b.SELECTED_TFS; extra: {set(IEG_TFS) - set(m41b.SELECTED_TFS)}'
    assert SIGN == m41b.SIGN, f'sign mismatch with 41b: {SIGN} vs {m41b.SIGN}'

    print('Loading h5ad inputs once...')
    adatas = {d['tag']: ad.read_h5ad(d['path']) for d in m41.DATASETS}

    primed = m41b.load_primed_labels()
    run_catalogue(m54, m41, adatas, m54.CFG_YOO25, m54.OUT_LONG_TMPL, m54.OUT_L23SET,
                  'yoo25', primed)
    run_catalogue(m54, m41, adatas, m55.build_cfg(), m55.OUT_LONG_TMPL, m55.OUT_L23SET,
                  'gao25', primed)


if __name__ == '__main__':
    main()
