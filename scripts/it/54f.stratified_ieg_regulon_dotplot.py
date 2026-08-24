"""54d's dot matrix restricted to the eight IEG regulons (PDF).

54d draws every regulon that 54b's star criterion selected — 30-odd rows, chosen by the data.
This entry point pins the rows instead: the eight immediate-early TFs (Fos, Fosb, Fosl2, Junb,
Egr1-4), in that fixed order, which is 41b's IEG block minus its Atf6/Smad3 tail and its
non-IEG controls. It is the hypothesis-driven read of the same matrix — "how do the IEG
regulons distribute over the archetypes", asked of all eight whether or not each one clears
the star criterion — where 54d is the discovery read.

Rows are drawn even where a regulon is not enriched, which is the point of pinning them:
54d's row set can only show IEGs that won selection, so a reader cannot tell an IEG that was
tested and came back flat from one that was never in the figure. Here every one of the eight
is either a populated row or visibly absent from a subclass (nothing drawn = SCENIC+ called no
activating regulon for that TF there; see the per-panel counts printed at the end).

Everything else is 54d's and unchanged, deliberately: same statistic, same [COLOR_MIN,
COLOR_MAX] ramp, same mask and star rules, and the same FRAC_REF, so a dot of a given size or
colour here means exactly what it means in 54d and the two figures can be read side by side.
Subsetting rows cannot change any cell — the enrichment tests were run per (regulon,
archetype) cell by 54, and 54b's BH-FDR is computed there, upstream of any row selection.

Reads:
  local_data/res/it/54.<layer>_stratified_enrichment.tsv    (panel 1, via 54b.load_native)
  local_data/res/it/54.l23set_stratified_enrichment.tsv     (panel 2)
Outputs:
  local_data/fig/it/54f.stratified_ieg_regulon_dotplot.pdf
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
INPUT_NATIVE_TMPL = os.path.join(RES_DIR, '54.{layer}_stratified_enrichment.tsv')
INPUT_L23SET = os.path.join(RES_DIR, '54.l23set_stratified_enrichment.tsv')
OUT_PDF = os.path.join(FIG_DIR, '54f.stratified_ieg_regulon_dotplot.pdf')

# 41b.SELECTED_TFS[:8]: the AP-1 arm then the EGR arm, kept in that order rather than sorted
# so the two families read as blocks. Not imported from 41b -- this is the row set of THIS
# figure, and it should change here and only here.
IEG_TFS = ['Fos', 'Fosb', 'Fosl2', 'Junb', 'Egr1', 'Egr2', 'Egr3', 'Egr4']

CFG_IEG = dict(
    tag='yoo25',
    label='mouse IT subclasses',
    native_tmpl=INPUT_NATIVE_TMPL,
    l23set=INPUT_L23SET,
    selection=None,        # unused: rows are pinned below, not read from 54b's selection
    rows=IEG_TFS,
    row_label='The eight IEG regulons',
    out_pdf=OUT_PDF,)


def main():
    spec = importlib.util.spec_from_file_location('script54d', PARENT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(CFG_IEG)


if __name__ == '__main__':
    main()
