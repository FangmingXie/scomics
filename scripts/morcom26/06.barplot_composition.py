import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- file paths ---
SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
H5AD_FILE    = os.path.join(PROJECT_ROOT, 'local_data', 'raw', 'morcom26_cux2mice', 'P26_ENs.h5ad')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '06.barplot_composition.html')

# --- config ---
CELLTYPE_COL    = 'celltype'
SAMPLE_COL      = 'samples'
CONDITION_COL   = 'Condition'
OTHER_MIX_LABEL = 'Other-mix'

sys.path.insert(0, SCRIPTS_DIR)
from viz import stacked_bar_html

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load ---
print(f'Loading {H5AD_FILE}')
adata = ad.read_h5ad(H5AD_FILE)
obs = adata.obs[[CELLTYPE_COL, SAMPLE_COL, CONDITION_COL]].copy()
print(f'  {len(obs)} cells')

# --- merge Other-mix* cell types ---
obs[CELLTYPE_COL] = obs[CELLTYPE_COL].astype(str).apply(
    lambda ct: OTHER_MIX_LABEL if 'Other-mix' in ct else ct
)
print(f'  Cell types after merge: {sorted(obs[CELLTYPE_COL].unique())}')

# --- ordered cell types: alphabetical, Other-mix last ---
celltypes = sorted(ct for ct in obs[CELLTYPE_COL].unique() if ct != OTHER_MIX_LABEL)
celltypes.append(OTHER_MIX_LABEL)

# --- color palette ---
cmap = plt.get_cmap('tab20', len(celltypes))
ct_colors = {ct: mcolors.to_hex(cmap(i)) for i, ct in enumerate(celltypes)}

# --- ordered samples and conditions ---
sample_order    = sorted(obs[SAMPLE_COL].unique())
condition_order = sorted(obs[CONDITION_COL].unique())


def compute_fractions(obs, group_col, group_order):
    """Return DataFrame (group x celltype) of cell type fractions."""
    counts = (obs.groupby([group_col, CELLTYPE_COL], observed=True)
                 .size()
                 .unstack(fill_value=0)
                 .reindex(index=group_order, columns=celltypes, fill_value=0))
    return counts.div(counts.sum(axis=1), axis=0)


sample_frac    = compute_fractions(obs, SAMPLE_COL, sample_order)
condition_frac = compute_fractions(obs, CONDITION_COL, condition_order)

# --- plot ---
stacked_bar_html(
    panel_data=[
        ('Per sample',    sample_order,    sample_frac),
        ('Per condition', condition_order, condition_frac),
    ],
    celltypes=celltypes,
    ct_colors=ct_colors,
    title='Cell type composition',
    out_path=OUT_HTML,
)

print('Done.')
