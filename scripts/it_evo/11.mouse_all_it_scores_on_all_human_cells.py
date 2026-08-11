"""Every mouse IT archetype score on every human IT cell — the off-diagonal of script 06.

Script 06 pairs one mouse subclass with one human subclass, so it only ever produces the
four diagonal combinations; mouse L4 markers never touch human L2/3 cells. That layout
cannot answer specificity: does a mouse L2/3 archetype score high in human L2/3 *and
nowhere else*, or high everywhere? This computes the full 4 human subclasses x 11 mouse
archetypes grid (L2/3 A,B,C; L4 A,B,C; L5IT A,B; L6IT A,B,C).

Two deliberate differences from 06:

  * Every mouse archetype is scored on every human subclass, not just its own pair.
  * The per-gene 2/98 percentiles are computed **once over all pooled human IT cells**
    rather than within each subclass (06:117-119). 06's per-subclass rescaling normalizes
    away exactly the cross-subclass differences this grid is about; a colorbar shared down
    a grid column requires one scale. Diagonal scores therefore differ slightly from 06's
    numbers — 06 and its outputs are left untouched and this writes its own TSVs.

Pooling is safe because all four human h5ads share an identical `var` in the same order
(asserted below), so one `gene_to_idx` serves all four and the blocks stack without
reindexing.

Reads:
  local_data/res/it_evo/05.mouse_<TOKEN>_archetype_markers.tsv        (all four)
  data/human_mouse_orthologs.tsv
  links/it_evo/jorstad23_human_WithinArea_<HTOKEN>.h5ad               (all four)
  local_data/res/it_evo/04.human_<TOKEN>_pcha_xp.tsv                  (index check)
Outputs:
  local_data/res/it_evo/11.mouse_all_archetype_scores_on_human_<TOKEN>.tsv  (4 files,
      11 score columns each, index = cell barcode)
  local_data/res/it_evo/11.pooled_gene_scale.tsv                      (gene, lo, hi)
  local_data/res/it_evo/11.mouse_all_archetype_columns.tsv            (per score column:
      how many mouse markers it started from and how many genes the score actually used)
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

# --- file paths ---
LINK_DIR       = os.path.join(PROJECT_ROOT, 'links', 'it_evo')
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IN_ORTHOLOGS   = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_GENE_SCALE = os.path.join(OUT_RES_DIR, '11.pooled_gene_scale.tsv')
OUT_COLUMNS    = os.path.join(OUT_RES_DIR, '11.mouse_all_archetype_columns.tsv')

# `mouse_noc` must match script 05's `noc` for the same token (as in 06).
SUBCLASSES = [
    {'token': 'L23',  'human_subclass': 'L2/3 IT', 'h5ad': 'jorstad23_human_WithinArea_L23IT.h5ad',
     'mouse_subclass': 'L2/3', 'mouse_noc': 3},
    {'token': 'L4',   'human_subclass': 'L4 IT',   'h5ad': 'jorstad23_human_WithinArea_L4IT.h5ad',
     'mouse_subclass': 'L4',   'mouse_noc': 3},
    {'token': 'L5IT', 'human_subclass': 'L5 IT',   'h5ad': 'jorstad23_human_WithinArea_L5IT.h5ad',
     'mouse_subclass': 'L5IT', 'mouse_noc': 2},
    {'token': 'L6IT', 'human_subclass': 'L6 IT',   'h5ad': 'jorstad23_human_WithinArea_L6IT.h5ad',
     'mouse_subclass': 'L6IT', 'mouse_noc': 3},
]

# --- parameters ---
GENE_NAME_COL   = 'feature_name'
SCORE_PCTILE_LO = 2
SCORE_PCTILE_HI = 98
ALPHABET        = ['A', 'B', 'C', 'D', 'E', 'F']

os.makedirs(OUT_RES_DIR, exist_ok=True)

# --- 1-to-1 ortholog map (verbatim from 06 / l23_evo/60) ---
print('Loading ortholog table...')
ortho = pd.read_csv(IN_ORTHOLOGS, sep='\t')
ortho = ortho.drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
mouse_to_human = dict(zip(ortho['mouse_symbol'], ortho['human_symbol']))
print(f'  {len(mouse_to_human)} 1-to-1 orthologs')

# --- the 11 score columns: every mouse archetype of every mouse subclass ---
markers = {cfg['token']: pd.read_csv(
    os.path.join(OUT_RES_DIR, f'05.mouse_{cfg["token"]}_archetype_markers.tsv'), sep='\t')
    for cfg in SUBCLASSES}

COLUMNS = [{'key': f'{cfg["token"]}_{ALPHABET[k]}',
            'mouse_token': cfg['token'],
            'mouse_subclass': cfg['mouse_subclass'],
            'archetype': ALPHABET[k],
            'mouse_genes': markers[cfg['token']][
                markers[cfg['token']]['archetype'] == f'archetype_{k+1}']['gene'].values}
           for cfg in SUBCLASSES for k in range(cfg['mouse_noc'])]
print(f'{len(COLUMNS)} mouse archetype columns: {", ".join(c["key"] for c in COLUMNS)}')

# ---------------------------------------------------------------------------
# One pass over the four human h5ads: establish the shared gene index on the first,
# assert the others match, and keep only the union of the 11 marker gene lists.
# ---------------------------------------------------------------------------
gene_names_ref = None
union_genes    = None
blocks         = {}   # token -> (n_cells, n_union) float32
barcodes       = {}   # token -> cell index

for cfg in SUBCLASSES:
    token = cfg['token']
    print(f'\nLoading human {cfg["human_subclass"]} h5ad...')
    adata      = ad.read_h5ad(os.path.join(LINK_DIR, cfg['h5ad']))
    gene_names = adata.var[GENE_NAME_COL].values

    if gene_names_ref is None:
        gene_names_ref = gene_names
        if len(set(gene_names_ref)) != len(gene_names_ref):
            raise ValueError(f'{GENE_NAME_COL} is not unique in {cfg["h5ad"]} — the pooled '
                             f'gene index assumes one column per gene symbol')
        human_gene_set = set(gene_names_ref)

        # map each mouse marker set to human symbols present in the human matrix
        for col in COLUMNS:
            col['human_genes'] = [mouse_to_human[g] for g in col['mouse_genes']
                                  if g in mouse_to_human and mouse_to_human[g] in human_gene_set]
            n_mouse = len(col['mouse_genes'])
            if not col['human_genes']:
                raise ValueError(f'{col["key"]}: none of its {n_mouse} mouse markers mapped '
                                 f'to a human gene present in the matrix')
            print(f'  {col["key"]}: {len(col["human_genes"])}/{n_mouse} mouse markers '
                  f'mapped ({len(col["human_genes"]) / n_mouse:.0%})')

        union_genes = sorted({g for col in COLUMNS for g in col['human_genes']})
        gene_to_idx = {g: i for i, g in enumerate(gene_names_ref)}
        union_cols  = [gene_to_idx[g] for g in union_genes]
        union_pos   = {g: i for i, g in enumerate(union_genes)}
        print(f'  union of the {len(COLUMNS)} gene lists: {len(union_genes)} genes')

        # per-column gene counts, so 12 can title a column without redoing the ortholog
        # mapping — the filter that decides `n_genes_used` lives only here
        pd.DataFrame([{'key': col['key'], 'mouse_token': col['mouse_token'],
                       'mouse_subclass': col['mouse_subclass'],
                       'archetype': col['archetype'],
                       'n_mouse_markers': len(col['mouse_genes']),
                       'n_genes_used': len(col['human_genes'])}
                      for col in COLUMNS]).to_csv(OUT_COLUMNS, sep='\t', index=False)
        print(f'  Saved {OUT_COLUMNS}')
    elif not np.array_equal(gene_names, gene_names_ref):
        raise ValueError(
            f'{cfg["h5ad"]} var[{GENE_NAME_COL}] differs from the first h5ad — the pooled '
            f'percentile scale requires all four human subclasses to share one gene index')

    # G7: slice the sparse columns before densifying
    blocks[token]   = adata.X[:, union_cols].toarray().astype(np.float32)
    barcodes[token] = adata.obs_names.values
    print(f'  {blocks[token].shape[0]} cells x {blocks[token].shape[1]} union genes')

    del adata
    gc.collect()

# --- pooled per-gene scale: 2/98 percentiles over all human IT cells at once ---
pooled = np.vstack([blocks[cfg['token']] for cfg in SUBCLASSES])
print(f'\nPooled matrix: {pooled.shape[0]} cells x {pooled.shape[1]} genes')
lo  = np.percentile(pooled, SCORE_PCTILE_LO, axis=0)
hi  = np.percentile(pooled, SCORE_PCTILE_HI, axis=0)
rng = np.where(hi > lo, hi - lo, 1.0)   # 06's degenerate guard
del pooled
gc.collect()

pd.DataFrame({'gene': union_genes, 'lo': lo, 'hi': hi}).to_csv(
    OUT_GENE_SCALE, sep='\t', index=False)
print(f'Saved {OUT_GENE_SCALE}  ({int((hi <= lo).sum())} degenerate genes)')

# --- score every column on every subclass against that one scale ---
for cfg in SUBCLASSES:
    token      = cfg['token']
    block      = blocks[token]
    in_pcha_xp = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_xp.tsv')
    out_scores = os.path.join(
        OUT_RES_DIR, f'11.mouse_all_archetype_scores_on_human_{token}.tsv')

    scores = np.zeros((block.shape[0], len(COLUMNS)), dtype=np.float32)
    for j, col in enumerate(COLUMNS):
        cols = [union_pos[g] for g in col['human_genes']]
        scores[:, j] = np.clip(
            (block[:, cols] - lo[cols]) / rng[cols], 0, 1).mean(axis=1)

    scores_df = pd.DataFrame(scores, index=barcodes[token],
                             columns=[col['key'] for col in COLUMNS])

    xp_index = pd.read_csv(in_pcha_xp, sep='\t', index_col=0).index
    if not scores_df.index.equals(xp_index):
        raise ValueError(
            f'{token}: cell index mismatch — h5ad has {len(scores_df)} cells, '
            f'{in_pcha_xp} has {len(xp_index)}; the scores and the human embedding must '
            f'cover the same cells in the same order')

    scores_df.to_csv(out_scores, sep='\t')
    print(f'\nSaved {out_scores}')
    print(f'  human {cfg["human_subclass"]} mean score per mouse archetype:')
    print(scores_df.mean().round(3).to_string())

print('\nDone.')
