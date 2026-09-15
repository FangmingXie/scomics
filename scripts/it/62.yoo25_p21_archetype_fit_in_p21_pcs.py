"""Fit archetype locations on yoo25 P21 cells alone, inside the P21 PC basis of 61.v2.

61/61.v2 project the developmental series into a PC basis fit on P21, but they had no archetype
VERTICES to show there: the 33-36 vertices in *_pcha_aa.tsv live in the two-dataset Harmony
space and cannot be carried across. 61.v2 stood in top-decile score centroids as anchors, which
are landmarks, not archetypes.

This script fits the real thing. It runs PCHA on the P21 cells only, in the P21 PC coordinates
61.v2 already persisted, and writes the archetype locations back in those same PC units so
61.v2 can simply load and plot them. Nothing here touches the earlier ages: the vertices describe
the ADULT structure, which is exactly the reference the younger ages are being compared against.

Archetype IDENTITY is not assumed from the fit order. `scomics.utils.pcha` sorts vertices by
their first inner-PC coordinate, so the k-th vertex here need not be the k-th archetype of
33-36. Each fitted vertex is instead matched to a 33-36 archetype by the marker scores of its
nearest P21 cells, and the matching is required to be a bijection -- if two vertices claim the
same archetype, the script stops rather than mislabelling the figure.

Procedure (per subclass; mirrors 3X.follow.two_*_archetype_scores.py steps 1-3):
  1) read 61.v2's P21 PC coords, keep Age == P21, take PC1..PC5 as the PCHA feature space
  2) NOC is derived from the 33-36 marker file, never hard-coded (L5IT is 2, the rest are 3)
  3) SCA -> proj_and_pcha(NDIM=5, NOC) -- an inner PCA on the 5 PCs, then PCHA
  4) back-project the vertices out of the inner PCA into the PC1..PC5 feature space, the 33-36
     idiom `aa.T @ pca_.components_[:NDIM] + pca_.mean_`
  5) label each vertex by the mean 33-36 marker score of its N nearest P21 cells; require a
     bijection onto the archetype letters

Reads:
  local_data/res/it/61.v2.yoo25_nr_<token>_pc_coords.tsv
  local_data/res/it/61.v2.yoo25_nr_<token>_archetype_scores.tsv
  local_data/res/it/3X.follow.two_*_archetype_markers.tsv     (NOC only)
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv
Outputs:
  local_data/res/it/62.yoo25_p21_<token>_archetype_coords.tsv
  local_data/res/it/62.yoo25_p21_archetype_fit_summary.tsv
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.main import SCA

# --- file paths ---
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IT_EVO_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IN_ARCMAP     = os.path.join(IT_EVO_DIR, '15.mouse_IT_joint_archetype_arc_order.tsv')
IN_COORDS     = os.path.join(RES_DIR, '61.v2.yoo25_nr_{token}_pc_coords.tsv')
IN_SCORES     = os.path.join(RES_DIR, '61.v2.yoo25_nr_{token}_archetype_scores.tsv')
IN_MARKERS    = os.path.join(RES_DIR, '{stem}_archetype_markers.tsv')
OUT_ARCH      = os.path.join(RES_DIR, '62.yoo25_p21_{token}_archetype_coords.tsv')
OUT_SUMMARY   = os.path.join(RES_DIR, '62.yoo25_p21_archetype_fit_summary.tsv')

# `token` keys into the depth-arc table; `stem` prefixes the cached 33-36 marker TSVs.
SUBCLASSES = [
    dict(subclass='L2/3', token='L23',  stem='34.follow.two_L23'),
    dict(subclass='L4',   token='L4',   stem='33.follow.two_L4'),
    dict(subclass='L5IT', token='L5IT', stem='35.follow.two_L5IT'),
    dict(subclass='L6IT', token='L6IT', stem='36.follow.two_L6IT'),
]

# --- parameters ---
AGE_COL       = 'Age'
TYPE_COL      = 'Type'
REF_AGE       = 'P21'
NDIM          = 5                 # PC1..PC5, matching 33-36's N_ARCH_PCS
DROP_PCS      = []                # use all NDIM PCs directly, per 33-36
ARCH_LETTERS  = ['A', 'B', 'C', 'D', 'E', 'F']   # archetype_1 -> A, archetype_2 -> B, ...
# Vertices are labelled from the marker scores of their nearest cells. 33-36 use a flat 300, but
# L5IT has only 424 P21 cells, so cap at a quarter of the population to keep the label local.
N_TOP_CELLS   = 300
TOP_FRAC_CAP  = 0.25
RANDOM_SEED   = 0


def load_relabel():
    """{token: {old_letter: new_letter}} from the persisted depth arc (read, never hard-coded)."""
    arc = pd.read_csv(IN_ARCMAP, sep='\t')
    missing = {'token', 'old_letter', 'new_letter'} - set(arc.columns)
    if missing:
        raise ValueError(f'{IN_ARCMAP} missing column(s): {sorted(missing)}')
    return {token: dict(zip(sub['old_letter'], sub['new_letter']))
            for token, sub in arc.groupby('token')}


print(f'--- archetype fit on {REF_AGE} alone, inside the {REF_AGE} PC basis of 61.v2 ---')

for cfg in SUBCLASSES:                       # fail fast, before any compute
    for path in (IN_COORDS.format(token=cfg['token']),
                 IN_SCORES.format(token=cfg['token']),
                 IN_MARKERS.format(stem=cfg['stem'])):
        if not os.path.exists(path):
            raise FileNotFoundError(f'{cfg["subclass"]}: missing {path} -- run 61.v2 first')
if not os.path.exists(IN_ARCMAP):
    raise FileNotFoundError(f'missing depth-arc table {IN_ARCMAP}')

relabel_by_token = load_relabel()
np.random.seed(RANDOM_SEED)                  # PCHA initialises randomly
summary_rows = []

for cfg in SUBCLASSES:
    S, token, stem = cfg['subclass'], cfg['token'], cfg['stem']
    print(f'\n=== {S} ===')

    # --- 1. P21 cells in 61.v2's PC basis ---
    coords = pd.read_csv(IN_COORDS.format(token=token), sep='\t', index_col=0)
    coords = coords[coords[AGE_COL].astype(str) == REF_AGE]
    if coords.empty:
        raise ValueError(f'{S}: no {REF_AGE} cells in {IN_COORDS.format(token=token)}')
    pc_cols = [f'PC{i+1}' for i in range(NDIM)]
    xn = coords[pc_cols].values
    types = coords[TYPE_COL].astype(str).values
    print(f'  {coords.shape[0]} {REF_AGE} cells, feature space {pc_cols}')

    # --- 2. NOC derived from the 33-36 marker file, never hard-coded ---
    mk = pd.read_csv(IN_MARKERS.format(stem=stem), sep='\t')
    arch_ids = sorted(mk['archetype'].unique(), key=lambda s: int(s.split('_')[1]))
    letters = [ARCH_LETTERS[int(s.split('_')[1]) - 1] for s in arch_ids]
    noc = len(letters)
    relabel = relabel_by_token[token]
    print(f'  NOC={noc} (from {stem}_archetype_markers.tsv)')

    # --- 3. PCHA in the PC feature space ---
    sca = SCA(xn, types)
    sca.setup_feature_matrix(method='data')
    _, aa, varexpl = sca.proj_and_pcha(NDIM, noc, drop_pcs=DROP_PCS)
    print(f'  PCHA variance explained: {varexpl:.4f}')

    # --- 4. back-project the vertices into the PC1..PC5 feature space (33-36 idiom) ---
    aa_feat = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_      # (noc, NDIM)

    # --- 5. label each vertex by the marker scores of its nearest P21 cells ---
    # pcha() sorts vertices by inner-PC1, so vertex order carries NO archetype identity.
    scores = pd.read_csv(IN_SCORES.format(token=token), sep='\t', index_col=0)
    scores = scores.reindex(coords.index)
    score_cols = [f'score_{L}' for L in letters]
    if scores[score_cols].isna().any().any():
        raise ValueError(f'{S}: {REF_AGE} cells missing from the 61.v2 score table')

    n_top = int(min(N_TOP_CELLS, max(noc * 5, TOP_FRAC_CAP * len(coords))))
    dists = np.stack([np.linalg.norm(xn - aa_feat[k], axis=1) for k in range(noc)], axis=1)
    mean_scores = np.stack([scores[score_cols].values[np.argsort(dists[:, k])[:n_top]].mean(axis=0)
                            for k in range(noc)])                           # (noc vertices, noc letters)

    claim = mean_scores.argmax(axis=1)
    if len(set(claim)) != noc:
        raise ValueError(
            f'{S}: vertex->archetype matching is not a bijection (claims {list(claim)}). '
            f'Mean marker scores of the {n_top} nearest cells per vertex:\n'
            f'{pd.DataFrame(mean_scores, columns=score_cols).round(3)}')
    vertex_letter = [letters[c] for c in claim]
    for k, L in enumerate(vertex_letter):
        print(f'  vertex {k} -> archetype {L} ({relabel[L]}), '
              f'mean score {mean_scores[k, claim[k]]:.3f} over its {n_top} nearest cells')

    # --- save, reordered to the canonical 33-36 letter order ---
    order = [vertex_letter.index(L) for L in letters]
    out = pd.DataFrame(aa_feat[order], index=letters, columns=pc_cols)
    out.index.name = 'archetype'
    out['primed'] = [relabel[L] for L in letters]
    out.to_csv(OUT_ARCH.format(token=token), sep='\t')
    print(f'  Saved {OUT_ARCH.format(token=token)}')

    summary_rows.append(dict(subclass=S, token=token, noc=noc, n_p21=len(coords),
                             ndim=NDIM, n_top_cells=n_top, varexpl=varexpl))

pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, sep='\t', index=False)
print(f'\nSaved {OUT_SUMMARY}')
print('\nDone.')
