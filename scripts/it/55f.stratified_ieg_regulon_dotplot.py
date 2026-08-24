"""54f's IEG dot matrix, on the gao25 V1 regulon catalogue.

The gao25 entry point for the pinned-row variant of 54d, standing to 54f exactly as 55d stands
to 54d: same rows, same statistic, same thresholds, same ramp, same FRAC_REF, only the regulon
catalogue swapped. That is what makes the pair worth having — laid side by side, 54f and 55f
ask whether the IEG-over-archetype pattern reproduces in an independently derived catalogue,
which is the question 56 answers numerically.

THE ROW AXIS IS THE SAME EIGHT TFs AS 54f, INCLUDING THE ONES gao25 NEVER CALLED. gao25 has no
activating regulon for Egr3 or Egr4 in any subclass, so those two rows are drawn empty. They
are kept rather than dropped for two reasons: the two figures stay row-aligned, and an empty
row is itself the finding — no regulon was called, which is different from a regulon that was
tested and came back unenriched (drawn) or thin (gray). Read an empty row as "absent from the
catalogue", never as "not enriched"; the run prints the absent list.

Reads:
  local_data/res/it/55.<layer>_stratified_enrichment.tsv
  local_data/res/it/55.l23set_stratified_enrichment.tsv
Outputs:
  local_data/fig/it/55f.stratified_ieg_regulon_dotplot.pdf
"""

import os
import importlib.util

import sys
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

PARENT_SCRIPT = os.path.join(SCRIPTS_DIR, 'it', '54d.stratified_enriched_regulon_dotplot.py')
SCRIPT_54F = os.path.join(SCRIPTS_DIR, 'it', '54f.stratified_ieg_regulon_dotplot.py')
INPUT_NATIVE_TMPL = os.path.join(RES_DIR, '55.{layer}_stratified_enrichment.tsv')
INPUT_L23SET = os.path.join(RES_DIR, '55.l23set_stratified_enrichment.tsv')
OUT_PDF = os.path.join(FIG_DIR, '55f.stratified_ieg_regulon_dotplot.pdf')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    # the row set comes from 54f rather than being restated, so the two figures cannot drift
    m54f = load_module(SCRIPT_54F, 'script54f')
    cfg = dict(
        tag='gao25',
        label='mouse IT subclasses',
        native_tmpl=INPUT_NATIVE_TMPL,
        l23set=INPUT_L23SET,
        selection=None,        # unused: rows are pinned, not read from 55b's selection
        rows=m54f.IEG_TFS,
        row_label='The eight IEG regulons',
        out_pdf=OUT_PDF,)
    load_module(PARENT_SCRIPT, 'script54d').main(cfg)


if __name__ == '__main__':
    main()
