"""Within-species version of 11/12: every human IT archetype score on every human IT cell.

Script 11 asks whether a *mouse* archetype score is specific to the human subclass it was
matched to. The same question applies inside one species, with no ortholog step in the way:
does a human L2/3 archetype score high in human L2/3 and nowhere else, or is it a generic
IT program? Script 04 cannot answer it — it scores each subclass's archetypes only on that
subclass's own cells, i.e. only the diagonal of this grid.

Compute and figure live in one script, as script 06 does; the mouse counterpart is 14.

Two things follow 11 exactly, for the same reasons:

  * Each gene's 2/98 percentiles are computed **once over all pooled human IT cells**, not
    within each subclass (04:281-283). Per-subclass rescaling would normalize away the
    cross-subclass differences the grid is about, and one colorbar shared down a column
    needs one scale. Diagonal scores therefore differ slightly from 04's numbers; 04 and
    its outputs are untouched and this writes its own TSVs.
  * The grid's columns are ordered and labelled by script 22's human depth arc (A' = most
    superficial within each subclass), not by 04's arbitrary PCHA vertex order. The score
    TSVs keep 04's letters and column order; only the figure moves, and the primes mark the
    difference — `L5 IT A'` is 04's `L5 IT D`. 22's file must be curated=True or this
    script refuses it.
  * Pooling relies on all four human h5ads sharing an identical `var` in the same order,
    asserted below.

`.X` is ln(CPM+1) and the marker `gene` column holds `var['feature_name']` symbols (04:250),
so the marker sets index the matrix directly — no gene-name translation anywhere.

Reads:
  local_data/res/it_evo/04.human_<TOKEN>_archetype_markers.tsv        (all four)
  links/it_evo/jorstad23_human_WithinArea_<HTOKEN>.h5ad               (all four)
  local_data/res/it_evo/04.human_<TOKEN>_pcha_{xp,aa}.tsv             (all four)
  local_data/res/it_evo/22.human_IT_joint_archetype_arc_order.tsv     (depth order + labels)
Outputs:
  local_data/res/it_evo/13.human_all_archetype_scores_on_human_<TOKEN>.tsv  (4 files,
      13 score columns each, index = cell barcode)
  local_data/res/it_evo/13.pooled_gene_scale.tsv                      (gene, lo, hi)
  local_data/fig/it_evo/13.human_all_scores_grid.pdf
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_score_grid_pdf

# --- file paths ---
LINK_DIR       = os.path.join(PROJECT_ROOT, 'links', 'it_evo')
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
OUT_GENE_SCALE = os.path.join(OUT_RES_DIR, '13.pooled_gene_scale.tsv')
IN_ARC_ORDER   = os.path.join(OUT_RES_DIR,
                              '22.human_IT_joint_archetype_arc_order.tsv')
OUT_PDF        = os.path.join(OUT_FIG_DIR, '13.human_all_scores_grid.pdf')

# `noc` must match script 04's noc for the same token.
# ===========================================================================
# `flip` is VISUALIZATION-ONLY — DO NOT PROPAGATE TO ANALYSIS. Carried for the L23 entry
# only, identical to 07 / 10 / 12, so the L2/3 row sits in the orientation of
# local_data/fig/l23_evo/57.human_mouse_b_ca_scores.pdf. It is applied to both cell coords
# (xp) and archetype coords (aa).
#
# The hardcoded `rename` this entry used to carry (archetype_1 -> D' ... archetype_4 -> A')
# is GONE: the labels now come from script 22's depth order, read below. That rename was a
# plain reversal chosen to match a figure's orientation, never derived from anything; 22's
# derivation independently reproduces exactly the same L2/3 mapping, which is a check on
# both, and now the other three subclasses are relabelled on the same footing instead of
# being left at 04's arbitrary vertex order.
# ===========================================================================
SUBCLASSES = [
    {'token': 'L23',  'human_subclass': 'L2/3 IT', 'h5ad': 'jorstad23_human_WithinArea_L23IT.h5ad',
     'noc': 4, 'flip': [-1.0, 1.0]},
    {'token': 'L4',   'human_subclass': 'L4 IT',   'h5ad': 'jorstad23_human_WithinArea_L4IT.h5ad',
     'noc': 3},
    {'token': 'L5IT', 'human_subclass': 'L5 IT',   'h5ad': 'jorstad23_human_WithinArea_L5IT.h5ad',
     'noc': 4},
    {'token': 'L6IT', 'human_subclass': 'L6 IT',   'h5ad': 'jorstad23_human_WithinArea_L6IT.h5ad',
     'noc': 2},
]
# ===========================================================================

# --- parameters ---
GENE_NAME_COL   = 'feature_name'
SCORE_PCTILE_LO = 2
SCORE_PCTILE_HI = 98
SCORE_CMAP      = 'RdBu_r'    # matches 04/12
SCORE_PCTILE    = (5, 95)
ALPHABET        = ['A', 'B', 'C', 'D', 'E', 'F']

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


# --- the 13 score columns: every archetype of every human subclass ---
markers = {cfg['token']: pd.read_csv(
    os.path.join(OUT_RES_DIR, f'04.human_{cfg["token"]}_archetype_markers.tsv'), sep='\t')
    for cfg in SUBCLASSES}

COLUMNS = [{'key': f'{cfg["token"]}_{ALPHABET[k]}',
            'token': cfg['token'],
            'genes': markers[cfg['token']][
                markers[cfg['token']]['archetype'] == f'archetype_{k+1}']['gene'].values}
           for cfg in SUBCLASSES for k in range(cfg['noc'])]
# the whole marker set is used — no ortholog step to lose genes to, unlike 12
print(f'{len(COLUMNS)} human archetype columns: {", ".join(c["key"] for c in COLUMNS)}')

# Figure-only relabelling and reordering: script 22's depth arc, where within a subclass A'
# is the most superficial archetype and the last letter the deepest. 04's PCHA vertex order
# (A/B/C/D) is arbitrary and hides that. The score TSVs written below keep 04's letters and
# 04's column order; only the grid is relabelled and reordered, so the primes mark which is
# which — a column titled `L5 IT A'` is 04's `L5 IT D`.
ARC = pd.read_csv(IN_ARC_ORDER, sep='\t').set_index('key').sort_values('arc_rank')
if not ARC['curated'].all():
    raise ValueError(f'{IN_ARC_ORDER} is marked curated=False — that file is script 22\'s '
                     f'run-1 proposal (the raw angular sort), not a depth record, and must '
                     f'not be used to label a figure')
missing = [col['key'] for col in COLUMNS if col['key'] not in ARC.index]
if missing:
    raise ValueError(f'{IN_ARC_ORDER} has no row for {missing} — it must be the depth order '
                     f'21 wrote for these same {len(COLUMNS)} human archetypes')
print('Figure order (depth): ' +
      ', '.join(f'{ARC.loc[k, "new_label"]} [was {ARC.loc[k, "old_label"]}]'
                for k in ARC.index))

# ---------------------------------------------------------------------------
# One pass over the four h5ads: establish the shared gene index on the first, assert the
# others match, and keep only the union of the 13 marker gene lists.
# ---------------------------------------------------------------------------
gene_names_ref, union_genes = None, None
offsets, barcodes, blocks = {}, {}, []
n_seen = 0

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
        gene_set = set(gene_names_ref)

        for col in COLUMNS:
            missing = [g for g in col['genes'] if g not in gene_set]
            if missing:
                raise ValueError(
                    f'{col["key"]}: {len(missing)} of its {len(col["genes"])} marker genes '
                    f'are absent from the human matrix (e.g. {missing[:5]}) — markers and '
                    f'matrix must share the {GENE_NAME_COL} vocabulary')
            print(f'  {col["key"]} (human {ARC.loc[col["key"], "new_label"]}): '
                  f'{len(col["genes"])} marker genes')

        union_genes = sorted({g for col in COLUMNS for g in col['genes']})
        gene_to_idx = {g: i for i, g in enumerate(gene_names_ref)}
        union_cols  = [gene_to_idx[g] for g in union_genes]
        union_pos   = {g: i for i, g in enumerate(union_genes)}
        print(f'  union of the {len(COLUMNS)} gene lists: {len(union_genes)} genes')
    elif not np.array_equal(gene_names, gene_names_ref):
        raise ValueError(
            f'{cfg["h5ad"]} var[{GENE_NAME_COL}] differs from the first h5ad — the pooled '
            f'percentile scale requires all four subclasses to share one gene index')

    # G7: slice the sparse columns before densifying
    block = adata.X[:, union_cols].toarray().astype(np.float32)
    blocks.append(block)
    barcodes[token] = adata.obs_names.values
    offsets[token]  = (n_seen, n_seen + block.shape[0])
    n_seen += block.shape[0]
    print(f'  {block.shape[0]} cells x {block.shape[1]} union genes')

    del adata
    gc.collect()

# --- pooled per-gene scale: 2/98 percentiles over all human IT cells at once ---
pooled = np.vstack(blocks)
del blocks
gc.collect()
print(f'\nPooled matrix: {pooled.shape[0]} cells x {pooled.shape[1]} genes')
lo  = np.percentile(pooled, SCORE_PCTILE_LO, axis=0)
hi  = np.percentile(pooled, SCORE_PCTILE_HI, axis=0)
rng = np.where(hi > lo, hi - lo, 1.0)   # 04's degenerate guard

pd.DataFrame({'gene': union_genes, 'lo': lo, 'hi': hi}).to_csv(
    OUT_GENE_SCALE, sep='\t', index=False)
print(f'Saved {OUT_GENE_SCALE}  ({int((hi <= lo).sum())} degenerate genes)')

# --- score every column on every subclass against that one scale ---
rows, grid_scores = [], []

for cfg in SUBCLASSES:
    token      = cfg['token']
    start, end = offsets[token]
    block      = pooled[start:end]
    in_pcha_xp = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_xp.tsv')
    in_pcha_aa = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_aa.tsv')
    out_scores = os.path.join(
        OUT_RES_DIR, f'13.human_all_archetype_scores_on_human_{token}.tsv')

    scores = np.zeros((block.shape[0], len(COLUMNS)), dtype=np.float32)
    for j, col in enumerate(COLUMNS):
        cols = [union_pos[g] for g in col['genes']]
        scores[:, j] = np.clip((block[:, cols] - lo[cols]) / rng[cols], 0, 1).mean(axis=1)

    scores_df = pd.DataFrame(scores, index=barcodes[token],
                             columns=[col['key'] for col in COLUMNS])

    xp_df = pd.read_csv(in_pcha_xp, sep='\t', index_col=0)
    aa_df = pd.read_csv(in_pcha_aa, sep='\t', index_col=0)
    if not scores_df.index.equals(xp_df.index):
        raise ValueError(
            f'{token}: cell index mismatch — h5ad has {len(scores_df)} cells, '
            f'{in_pcha_xp} has {len(xp_df)}; the scores and the embedding must cover the '
            f'same cells in the same order')

    scores_df.to_csv(out_scores, sep='\t')
    print(f'\nSaved {out_scores}')
    print(f'  human {cfg["human_subclass"]} mean score per human archetype:')
    print(scores_df.mean().round(3).to_string())

    flip = np.array(cfg.get('flip', [1.0, 1.0]))
    rows.append({'label': f'human {cfg["human_subclass"]}\n({len(xp_df)} cells)',
                 'xp': xp_df[['PC1', 'PC2']].values * flip,
                 'aa': aa_df[['PC1', 'PC2']].values * flip,
                 'aa_labels': [ARC.loc[f'{token}_{ALPHABET[k]}', 'new_letter']
                               for k in range(cfg['noc'])]})
    grid_scores.append([scores_df[k].values for k in ARC.index])

del pooled
gc.collect()

# --- the grid: rows = human embeddings, columns = human archetypes in depth order ---
# gene counts stay keyed by the original column key, so each count remains bound to the
# archetype it describes rather than to a position
n_genes   = {col['key']: len(col['genes']) for col in COLUMNS}
col_names = [f'human {ARC.loc[k, "new_label"]}\n({n_genes[k]} genes)' for k in ARC.index]
diagonal  = {(i, j) for i, cfg in enumerate(SUBCLASSES)
             for j, k in enumerate(ARC.index) if k.rsplit('_', 1)[0] == cfg['token']}
print(f'\nDiagonal (own-subclass) cells outlined: {len(diagonal)}')

save_score_grid_pdf(
    rows, col_names, grid_scores,
    title='Jorstad23 human IT archetype scores across all human IT subclasses',
    out_path=OUT_PDF,
    cmap=SCORE_CMAP, pctile=SCORE_PCTILE,
    highlight=diagonal,
)

print('\nDone.')
