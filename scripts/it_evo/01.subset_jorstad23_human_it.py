"""Materialize the human Jorstad23 L4 / L5 IT / L6 IT subsets (WithinArea, Layer == 'All').

The existing L2/3 IT file was produced from the full Jorstad23 v1 h5ad by keeping
`WithinArea_subclass == 'L2/3 IT' & Layer == 'All'`. This script applies exactly that
recipe to L4 IT / L5 IT / L6 IT and writes the three new per-subclass h5ads next to it,
so the four subclasses share one provenance. `L6 IT Car3` is excluded — Cheng22 has no
counterpart.

The existing L2/3 file is never rewritten. Instead the same mask is applied for
'L2/3 IT' and its `obs_names` set is asserted equal to the published file's, which
proves the filter is the one that produced it.

All categorical obs columns are `remove_unused_categories()`-ed before writing: in the
full file `WithinArea_cluster` is a 129-level Categorical, and leaving the empty levels
in would make `pd.get_dummies` emit 121 all-zero columns downstream (G6).

Reads:
  links/it_evo/jorstad23_human_v1.h5ad
  links/it_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  /home/qlyu/mydata/data/jorstad23_human/jorstad23_human_WithinArea_L4IT.h5ad
  /home/qlyu/mydata/data/jorstad23_human/jorstad23_human_WithinArea_L5IT.h5ad
  /home/qlyu/mydata/data/jorstad23_human/jorstad23_human_WithinArea_L6IT.h5ad
"""

import os
import gc
import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
INPUT_FULL     = os.path.join(PROJECT_ROOT, 'links', 'it_evo', 'jorstad23_human_v1.h5ad')
INPUT_L23_REF  = os.path.join(PROJECT_ROOT, 'links', 'it_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_DIR        = '/home/qlyu/mydata/data/jorstad23_human'
OUT_L4IT       = os.path.join(OUT_DIR, 'jorstad23_human_WithinArea_L4IT.h5ad')
OUT_L5IT       = os.path.join(OUT_DIR, 'jorstad23_human_WithinArea_L5IT.h5ad')
OUT_L6IT       = os.path.join(OUT_DIR, 'jorstad23_human_WithinArea_L6IT.h5ad')

# --- parameters ---
SUBCLASS_COL = 'WithinArea_subclass'
LAYER_COL    = 'Layer'
LAYER_KEEP   = 'All'
CLUSTER_COL  = 'WithinArea_cluster'
DONOR_COL    = 'donor_id'
SOURCE_COL   = 'Source'

SUBCLASSES = [
    {'token': 'L4',   'human_subclass': 'L4 IT', 'out': OUT_L4IT, 'n_expected': 30455},
    {'token': 'L5IT', 'human_subclass': 'L5 IT', 'out': OUT_L5IT, 'n_expected': 10537},
    {'token': 'L6IT', 'human_subclass': 'L6 IT', 'out': OUT_L6IT, 'n_expected': 5035},
]

L23_SUBCLASS   = 'L2/3 IT'
L23_N_EXPECTED = 47125

os.makedirs(OUT_DIR, exist_ok=True)


def clean_categoricals(adata):
    """Drop unused categories from every categorical obs column (G6)."""
    for col in adata.obs.columns:
        if isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].cat.remove_unused_categories()
    return adata


# --- load the full file backed; never densify 241k x 29k (~28 GB) ---
print(f'Opening {INPUT_FULL} (backed)...')
full = ad.read_h5ad(INPUT_FULL, backed='r')
print(f'  {full.n_obs} cells x {full.n_vars} genes')

obs = full.obs
layer_ok = (obs[LAYER_COL] == LAYER_KEEP).values

# --- L2/3 control: verify the recipe, write nothing ---
print(f'\n--- L2/3 control ({L23_SUBCLASS}) ---')
mask_l23 = ((obs[SUBCLASS_COL] == L23_SUBCLASS).values & layer_ok)
n_l23 = int(mask_l23.sum())
print(f'  mask keeps {n_l23} cells (expected {L23_N_EXPECTED})')
if n_l23 != L23_N_EXPECTED:
    raise ValueError(f'L2/3 control: mask keeps {n_l23} cells, expected {L23_N_EXPECTED}')

l23_ref = ad.read_h5ad(INPUT_L23_REF, backed='r')
ref_names = set(l23_ref.obs_names.values)
new_names = set(obs.index.values[mask_l23])
if ref_names != new_names:
    raise ValueError(
        f'L2/3 control: obs_names mismatch vs {INPUT_L23_REF} — '
        f'{len(new_names - ref_names)} extra, {len(ref_names - new_names)} missing'
    )
print(f'  obs_names set-equal to {INPUT_L23_REF} ({len(ref_names)} cells) — recipe confirmed')
del l23_ref, ref_names, new_names
gc.collect()

# --- write the three new subsets ---
for cfg in SUBCLASSES:
    token, subclass, out_path, n_expected = (
        cfg['token'], cfg['human_subclass'], cfg['out'], cfg['n_expected'])
    print(f'\n--- {token} ({subclass}) ---')

    sc_mask = (obs[SUBCLASS_COL] == subclass).values
    layer_counts = obs.loc[sc_mask, LAYER_COL].value_counts().to_dict()
    print(f'  Layer breakdown before filtering: {layer_counts}')

    mask = sc_mask & layer_ok
    n_keep = int(mask.sum())
    print(f'  keeping {n_keep} cells (Layer == {LAYER_KEEP!r}), '
          f'discarding {int(sc_mask.sum()) - n_keep}')
    if n_keep != n_expected:
        raise ValueError(f'{token}: mask keeps {n_keep} cells, expected {n_expected}')

    print('  Materializing subset...')
    sub = full[mask].to_memory()
    sub = clean_categoricals(sub)

    print(f'  {sub.n_obs} cells x {sub.n_vars} genes  '
          f'clusters={sub.obs[CLUSTER_COL].nunique()}  '
          f'donors={sub.obs[DONOR_COL].nunique()}  '
          f'sources={sub.obs[SOURCE_COL].nunique()}  '
          f'layers={sub.obs[LAYER_COL].nunique()}')

    sub.write_h5ad(out_path)
    print(f'  Saved {out_path}')

    del sub
    gc.collect()

print('\nDone.')
