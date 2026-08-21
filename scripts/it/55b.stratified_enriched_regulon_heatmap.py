"""54b's two-panel enrichment heatmap, on the gao25 V1 regulon catalogue.

The gao25 entry point for 54b. 42-45 (the naive gao25 family) were four copies of 41 with
the regulon table swapped, 1239 lines in total; repeating that here would have meant copying
54b's ~250 lines of plotting per figure, four times over, and guaranteeing the two
families drift apart. Instead 54b takes a catalogue config and this file supplies it.

Rows, statistic, thresholds and palette are 54b's, unchanged. That is deliberate and
checked, not assumed: gao25's unmasked activating log2_enr spans -1.495..3.197, inside 54b's
[-1.5, 3.5] ramp, and its selection curve is flat from 0.5 to 1.25 (58/58/57/55 cells), so
STAR_LOG2ENR is nearly non-binding for gao25. Sharing the scale is what lets the yoo25 and gao25
figures be compared by colour.

Reads:
  local_data/res/it/55.<layer>_stratified_enrichment.tsv    (panel 1)
  local_data/res/it/55.l23set_stratified_enrichment.tsv     (panel 2)
Outputs:
  local_data/res/it/55b.enriched_regulon_selection.tsv
  local_data/res/it/55b.enriched_native_log2enr.tsv / _fdr.tsv
  local_data/res/it/55b.enriched_l23set_log2enr.tsv / _fdr.tsv
  local_data/fig/it/55b.stratified_enriched_regulon_heatmap.html
"""

import os
import importlib.util

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

PARENT_SCRIPT = os.path.join(SCRIPTS_DIR, 'it', '54b.stratified_enriched_regulon_heatmap.py')
INPUT_NATIVE_TMPL = os.path.join(RES_DIR, '55.{layer}_stratified_enrichment.tsv')
INPUT_L23SET = os.path.join(RES_DIR, '55.l23set_stratified_enrichment.tsv')
OUT_SELECTION = os.path.join(RES_DIR, '55b.enriched_regulon_selection.tsv')
OUT_NATIVE = {'log2_enr': os.path.join(RES_DIR, '55b.enriched_native_log2enr.tsv'),
              'fdr_strat': os.path.join(RES_DIR, '55b.enriched_native_fdr.tsv')}
OUT_L23SET = {'log2_enr': os.path.join(RES_DIR, '55b.enriched_l23set_log2enr.tsv'),
              'fdr_strat': os.path.join(RES_DIR, '55b.enriched_l23set_fdr.tsv')}
OUT_HTML = os.path.join(FIG_DIR, '55b.stratified_enriched_regulon_heatmap.html')

CFG_GAO25 = dict(
    tag='gao25',
    label='mouse IT subclasses',
    native_tmpl=INPUT_NATIVE_TMPL,
    l23set=INPUT_L23SET,
    selection=OUT_SELECTION,
    out_selection=OUT_SELECTION,
    out_native=OUT_NATIVE,
    out_l23set=OUT_L23SET,
    out_html=OUT_HTML,)


def main():
    spec = importlib.util.spec_from_file_location('script54b', PARENT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(CFG_GAO25)


if __name__ == '__main__':
    main()
