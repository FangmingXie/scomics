"""Human Jorstad23 IT archetype PCHA scatter, colored by cluster or donor (plots only).

Renders the categorical scatter for each human IT subclass from the cached PCHA record
written by 04.human_jorstad23_it_archetype_scores_loadings.py. Split out of `04` so the
figure can be changed without re-running its one-vs-each Wilcoxon loop over 2000 HVGs and
rewriting the analysis TSVs — the same compute/viz split `03.refine` / `03.refine.viz`
uses. No computation here; rerun `04` if the inputs are missing.

Two metadata toggles are offered, `WithinArea_cluster` and `donor_id`. Donor matters for
this record specifically: it is one of the four factors the `02` variance partition
models, Gate A dropped several VX components because donor dominated them, and L6 IT's
VX9 was a borderline keep (donor R² 0.100 vs cell_type 0.038) whose per-donor overlap was
a named Gate-B check. `WithinArea_cluster` is listed first because
`scatter_categorical_html` shows only the first key's traces on load.

`noc` and `ndim` are read off the cached files rather than hardcoded, so there is nothing
here to keep in sync with `04`'s Gate-B settings.

Reads (per TOKEN in L23 / L4 / L5IT / L6IT):
  local_data/res/it_evo/04.human_<TOKEN>_pcha_xp.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_aa.tsv
  local_data/res/it_evo/02.human_<TOKEN>_varimax_coords.tsv
Outputs:
  local_data/fig/it_evo/04.human_<TOKEN>_archetype_scatter.html
"""

import os
import sys
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokens', nargs='*', default=None,
                    help='subset of tokens to render (default: all four)')
args = parser.parse_args()

# --- file paths ---
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')

SUBCLASSES = [
    {'token': 'L23',  'human_subclass': 'L2/3 IT'},
    {'token': 'L4',   'human_subclass': 'L4 IT'},
    {'token': 'L5IT', 'human_subclass': 'L5 IT'},
    {'token': 'L6IT', 'human_subclass': 'L6 IT'},
]
if args.tokens:
    SUBCLASSES = [c for c in SUBCLASSES if c['token'] in args.tokens]

# --- parameters ---
CLUSTER_COL = 'WithinArea_cluster'
DONOR_COL   = 'donor_id'

os.makedirs(OUT_FIG_DIR, exist_ok=True)

for cfg in SUBCLASSES:
    token    = cfg['token']
    subclass = cfg['human_subclass']

    in_pcha_xp    = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_xp.tsv')
    in_pcha_aa    = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_aa.tsv')
    in_vx_coords  = os.path.join(OUT_RES_DIR, f'02.human_{token}_varimax_coords.tsv')
    out_arch_html = os.path.join(OUT_FIG_DIR, f'04.human_{token}_archetype_scatter.html')

    xp_df = pd.read_csv(in_pcha_xp, sep='\t', index_col=0)
    aa_df = pd.read_csv(in_pcha_aa, sep='\t', index_col=0)
    vx_df = pd.read_csv(in_vx_coords, sep='\t', index_col=0)

    ndim = xp_df.shape[1]
    noc  = len(aa_df)
    print(f'\n--- {token} (human {subclass}): {len(xp_df)} cells, NOC={noc}, NDIM={ndim} ---')

    # align metadata to the coord order; fail fast rather than reindex to NaN
    if set(vx_df.index) != set(xp_df.index):
        raise ValueError(
            f'{token}: cell index mismatch between {in_vx_coords} ({len(vx_df)} cells) and '
            f'{in_pcha_xp} ({len(xp_df)} cells) — the metadata and the embedding must '
            f'cover the same cells.'
        )
    vx_df = vx_df.loc[xp_df.index]

    # same panel layout `04` builds
    panels = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')][:max(ndim - 1, 1)]
    panel_3d = (0, 1, 2, 'PC1', 'PC2', 'PC3') if ndim >= 3 else None

    print(f'  donors: {sorted(vx_df[DONOR_COL].unique())}')
    scatter_categorical_html(
        xp_grid=[xp_df.values],
        # CLUSTER_COL first — scatter_categorical_html shows only the first key on load
        cell_metadata={CLUSTER_COL: vx_df[CLUSTER_COL].values,
                       DONOR_COL:   vx_df[DONOR_COL].values},
        title=f'Jorstad23 human {subclass} — varimax PCHA space (NOC={noc})',
        out_path=out_arch_html,
        panels=panels,
        panel_3d=panel_3d,
        # `04` writes aa transposed as (noc, ndim); arch_vis wants (ndim, noc)
        arch_vis=aa_df.values.T,
    )

print('\nDone.')
