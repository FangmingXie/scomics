"""54's expression-stratified enrichment, run on the gao25 V1 regulon catalogue.

42-45 are the gao25 analogue of 41: four near-identical scripts (309-312 lines each, 1239
in total), one per layer, each a copy of 41 with the regulon table swapped and its own verbatim
copy of `reconstruct_gene_universe`. They use the same one-stratum Fisher null 41 does, so they
inherit both inflations 54 documents -- the null ignores that markers and regulon targets are
each biased toward highly expressed genes, and the Haldane-Anscombe odds ratio overstates the
enrichment ratio at small counts.

Only the REGULON SOURCE differs between 41 and 42-45. The gene universe, the archetype marker
sets and therefore the expression strata are identical to 54's -- same h5ads, same coords, same
3X.follow.*_archetype_markers.tsv. So this script is not a reimplementation: it supplies a
gao25 regulon loader and output paths, and hands them to 54.main(), which does the rest. All
four layers run from 41's LAYERS in one pass, replacing 42-45's four-way split.

The correction transfers. Swapping the universe from the expressed-gene set to the union of
each layer's regulon targets moves the two statistics by:

    layer   log2_or   log2_enr
    L4       +0.939     -0.008
    L2/3     +0.984     +0.049
    L5IT     +0.972     +0.094
    L6IT     +0.944     -0.038

10-120x more stable, as on yoo25. Note the naive offset is about HALF yoo25's (+0.94 vs
+1.46..+2.09) because gao25's regulons cover a different share of the universe -- a reason for
55b to re-derive its thresholds rather than inherit 54b's, not a reason to skip the correction.

REGULON IDENTITY DIFFERS BETWEEN THE TWO CATALOGUES, and this script deliberately preserves the
difference. 40 (yoo25) keeps a regulon's DIRECT targets when it exists in both direct and
extended forms, falling back to extended otherwise. 42-45 (gao25) ignore `is_extended` entirely
and union direct + extended for the same TF + sign. `load_regulons_gao25` below keeps 42-45's
union rule so that 55 is directly comparable to the naive gao25 results it supersedes -- but it
means `Rfx3_+/+` is a DIFFERENT GENE SET in the 54 and 55 families, which silently weakens any
yoo25-vs-gao25 comparison by regulon name. Aligning the rule is a separate, deliberate change.

VERIFY_AGAINST_NAIVE re-checks, after the run, that this script's internally recomputed log2_or
reproduces 42-45's committed tables. That is the gate on the consolidation: it proves the
four-into-one merge changed the null and nothing else.

Reads (per layer):
  local_data/res/it/3X.follow.two_*_archetype_markers.tsv   (via 41.LAYERS)
  local_data/res/it/3X.harmony.two_*_coords.tsv
  links/it/regulon_gene_table_gao25_v1<l23|l4|l5|l6>.csv    (TSV despite the extension)
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  links/it/superdupermegaRNA_yoo25_IT_P21.h5ad
  local_data/res/it/4X.gao25_<layer>_regulon_archetype_enrichment.tsv   (verification only)
Outputs:
  local_data/res/it/55.<layer>_stratified_enrichment.tsv      (panel 1: native regulons)
  local_data/res/it/55.l23set_stratified_enrichment.tsv       (panel 2: L2/3 set everywhere)
  local_data/res/it/55.validation_universe_robustness.tsv
  local_data/res/it/55.validation_bin_sensitivity.tsv
  local_data/fig/it/55.log2or_vs_log2enr.pdf
"""

import os
import importlib.util

import numpy as np
import pandas as pd

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
LINKS_IT = os.path.join(PROJECT_ROOT, 'links', 'it')

SCRIPT_54 = os.path.join(SCRIPTS_DIR, 'it', '54.stratified_regulon_archetype_enrichment.py')
INPUT_REGULON_TMPL = os.path.join(LINKS_IT, 'regulon_gene_table_gao25_v1{token}.csv')
# 42-45's committed naive tables, for the consolidation gate
INPUT_NAIVE_TMPL = os.path.join(RES_DIR, '{script}.gao25_{layer}_regulon_archetype_enrichment.tsv')

OUT_LONG_TMPL = os.path.join(RES_DIR, '55.{layer}_stratified_enrichment.tsv')
OUT_L23SET = os.path.join(RES_DIR, '55.l23set_stratified_enrichment.tsv')
OUT_ROBUSTNESS = os.path.join(RES_DIR, '55.validation_universe_robustness.tsv')
OUT_BIN_SENS = os.path.join(RES_DIR, '55.validation_bin_sensitivity.tsv')
OUT_PDF = os.path.join(FIG_DIR, '55.log2or_vs_log2enr.pdf')

# 41's layer key -> the token in the gao25 filename, and the 42-45 script that covered it
LAYER_TOKEN = {'L2_3': 'l23', 'L4': 'l4', 'L5IT': 'l5', 'L6IT': 'l6'}
NAIVE_SCRIPT = {'L2_3': '42', 'L4': '43', 'L5IT': '44', 'L6IT': '45'}
# gao25 carries only these two sign patterns; kept for parity with 42-45
KEEP_DIRECTIONS = {'+/+', '-/+'}
VERIFY_AGAINST_NAIVE = True
VERIFY_ATOL = 1e-9

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_54():
    spec = importlib.util.spec_from_file_location('script54', SCRIPT_54)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_gao25(token):
    """One gao25 eRegulon table, reshaped to 40/41's four columns.

    Lifted from 42:161-165 unchanged, including its union of direct and extended targets --
    see the module docstring on why that difference from 40 is preserved rather than fixed.
    """
    path = INPUT_REGULON_TMPL.format(token=token)
    assert os.path.exists(path), f'missing gao25 regulon table {path}'
    reg = pd.read_csv(path, sep='\t')          # tab-separated despite the .csv extension
    reg['regulation_direction'] = reg['TF2G_sign'] + '/' + reg['R2G_sign']
    reg = reg[reg['regulation_direction'].isin(KEEP_DIRECTIONS)].copy()
    reg['regulon'] = reg['TF'] + '_' + reg['regulation_direction']
    return reg.drop_duplicates(subset=['regulon', 'Gene'])


def load_regulons_gao25(cfg):
    """54.build_layer's regulon hook: this layer's gao25 catalogue."""
    return _read_gao25(LAYER_TOKEN[cfg['layer']])


def verify_against_naive(m54):
    """Gate: 55's recomputed log2_or must reproduce 42-45's committed tables.

    55 folds four scripts into one and swaps the null. Only the null was meant to change, and
    log2_or is the statistic both families compute, so any disagreement here means the loader
    or the universe diverged and every stratified number downstream is suspect.
    """
    print('\n=== verification against the naive 42-45 tables ===')
    worst = 0.0
    for layer, script in NAIVE_SCRIPT.items():
        naive_path = INPUT_NAIVE_TMPL.format(script=script, layer=layer)
        assert os.path.exists(naive_path), \
            f'missing {naive_path}; run scripts/it/{script}.*.py first, or set ' \
            f'VERIFY_AGAINST_NAIVE = False'
        naive = pd.read_csv(naive_path, sep='\t').set_index(['archetype', 'regulon'])
        mine = pd.read_csv(OUT_LONG_TMPL.format(layer=layer), sep='\t')
        mine = mine.set_index(['archetype', 'regulon'])

        assert set(naive.index) == set(mine.index), \
            (f'{layer}: pair sets differ -- naive has {len(set(naive.index) - set(mine.index))} '
             f'pairs 55 lacks, 55 has {len(set(mine.index) - set(naive.index))} the naive lacks')
        j = naive.join(mine, lsuffix='_naive', rsuffix='_new')
        for col in ['overlap', 'n_markers', 'n_targets', 'universe']:
            assert (j[f'{col}_naive'] == j[f'{col}_new']).all(), f'{layer}: {col} differs'
        d = float(np.abs(j['log2_or_naive'] - j['log2_or_new']).max())
        worst = max(worst, d)
        print(f'  {layer:5s}: {len(j):4d} pairs, counts identical, '
              f'max |delta log2_or| = {d:.3g}')
        assert d <= VERIFY_ATOL, \
            f'{layer}: log2_or diverged from {naive_path} by {d:.3g} (> {VERIFY_ATOL})'
    print(f'  PASS -- worst |delta log2_or| across all four layers: {worst:.3g}')


def main():
    m54 = load_54()
    cfg = dict(
        tag='gao25',
        load_regulons=load_regulons_gao25,
        # gao25's L2/3 catalogue is the fixed regulon source for panel 2
        load_l23=lambda: _read_gao25(LAYER_TOKEN['L2_3']),
        l23_source=INPUT_REGULON_TMPL.format(token=LAYER_TOKEN['L2_3']),
        out_long_tmpl=OUT_LONG_TMPL,
        out_l23set=OUT_L23SET,
        out_robustness=OUT_ROBUSTNESS,
        out_bin_sens=OUT_BIN_SENS,
        out_pdf=OUT_PDF,
    )
    m54.main(cfg)
    if VERIFY_AGAINST_NAIVE:
        verify_against_naive(m54)


if __name__ == '__main__':
    main()
