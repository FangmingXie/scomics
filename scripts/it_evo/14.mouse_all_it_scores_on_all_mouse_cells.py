"""Within-species version of 11/12 on the mouse side: every mouse IT archetype score on
every mouse IT cell.

The species-matched companion to 13. Script 05 scores each subclass's archetypes only on
that subclass's own cells — the diagonal of this grid — so it cannot say whether a mouse
archetype is subclass-specific or a program shared across IT. Read together with 11/12,
this also separates two explanations for a mouse archetype scoring broadly across human
subclasses: a genuinely generic IT program looks broad here too, whereas one that is
subclass-specific in mouse and broad in human is a cross-species statement.

Compute and figure live in one script, as script 06 does.

Follows 11/13: each gene's 2/98 percentiles are computed **once over all 11,061 pooled
mouse IT cells** rather than within each subclass (05:283-285), because a colorbar shared
down a grid column needs one scale. 05 is untouched and this writes its own TSVs.

G2 (script 05): cheng22 `.X` is already log1p(CP10k), while `.raw.X` holds integer counts;
markers and scores were built by renormalizing from `.raw`, so this does the same —
log2(CP10k+1) over the `var_names` gene set. Depths are summed on the sparse matrix and
only the union marker columns are densified, which is arithmetically identical to 05's
dense normalize-then-subset but avoids materializing 11061 x 16572.

Reads:
  local_data/res/it_evo/05.mouse_<TOKEN>_archetype_markers.tsv        (all four)
  links/it_evo/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  local_data/res/it_evo/05.mouse_<TOKEN>_pcha_{xp,aa}.tsv             (all four)
Outputs:
  local_data/res/it_evo/14.mouse_all_archetype_scores_on_mouse_<TOKEN>.tsv  (4 files,
      11 score columns each, index = cell barcode)
  local_data/res/it_evo/14.pooled_gene_scale.tsv                      (gene, lo, hi)
  local_data/fig/it_evo/14.mouse_all_scores_grid.pdf
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
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_MOUSE_H5AD  = os.path.join(PROJECT_ROOT, 'links', 'it_evo',
                              'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
OUT_GENE_SCALE = os.path.join(OUT_RES_DIR, '14.pooled_gene_scale.tsv')
OUT_PDF        = os.path.join(OUT_FIG_DIR, '14.mouse_all_scores_grid.pdf')

# `noc` must match script 05's noc for the same token.
# ===========================================================================
# `flip` is VISUALIZATION-ONLY — DO NOT PROPAGATE TO ANALYSIS. Carried for the L23 entry
# only, identical to 08, which puts this record's L2/3 triangle in l23_evo/59's
# orientation. Applied to both cell coords (xp) and archetype coords (aa).
# ===========================================================================
SUBCLASSES = [
    {'token': 'L23',  'mouse_subclass': 'L2/3', 'noc': 3, 'flip': [-1.0, -1.0]},
    {'token': 'L4',   'mouse_subclass': 'L4',   'noc': 3},
    {'token': 'L5IT', 'mouse_subclass': 'L5IT', 'noc': 2},
    {'token': 'L6IT', 'mouse_subclass': 'L6IT', 'noc': 3},
]
# ===========================================================================

# --- parameters ---
SUBCLASS_COL    = 'Subclass'
SCORE_PCTILE_LO = 2
SCORE_PCTILE_HI = 98
SCORE_CMAP      = 'RdBu_r'    # matches 05/12; 08's custom colormaps are for its contrasts
SCORE_PCTILE    = (5, 95)
ALPHABET        = ['A', 'B', 'C', 'D', 'E', 'F']

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- the 11 score columns: every archetype of every mouse subclass ---
markers = {cfg['token']: pd.read_csv(
    os.path.join(OUT_RES_DIR, f'05.mouse_{cfg["token"]}_archetype_markers.tsv'), sep='\t')
    for cfg in SUBCLASSES}

COLUMNS = [{'key': f'{cfg["token"]}_{ALPHABET[k]}',
            'token': cfg['token'],
            'display': f'mouse {cfg["mouse_subclass"]} {ALPHABET[k]}',
            'genes': markers[cfg['token']][
                markers[cfg['token']]['archetype'] == f'archetype_{k+1}']['gene'].values}
           for cfg in SUBCLASSES for k in range(cfg['noc'])]
print(f'{len(COLUMNS)} mouse archetype columns: {", ".join(c["key"] for c in COLUMNS)}')

print(f'\nLoading {IN_MOUSE_H5AD}...')
m_adata_all = ad.read_h5ad(IN_MOUSE_H5AD)
print(f'  {m_adata_all.n_obs} cells x {m_adata_all.n_vars} genes')

gene_names = m_adata_all.var_names.values
if not m_adata_all.var_names.is_unique:
    raise ValueError('mouse var_names are not unique — the gene index assumes one column '
                     'per gene symbol')
gene_set = set(gene_names)
for col in COLUMNS:
    missing = [g for g in col['genes'] if g not in gene_set]
    if missing:
        raise ValueError(
            f'{col["key"]}: {len(missing)} of its {len(col["genes"])} marker genes are '
            f'absent from the mouse matrix (e.g. {missing[:5]})')
    print(f'  {col["key"]} ({col["display"]}): {len(col["genes"])} marker genes')

union_genes = sorted({g for col in COLUMNS for g in col['genes']})
gene_to_idx = {g: i for i, g in enumerate(gene_names)}
union_cols  = [gene_to_idx[g] for g in union_genes]
union_pos   = {g: i for i, g in enumerate(union_genes)}
print(f'  union of the {len(COLUMNS)} gene lists: {len(union_genes)} genes')

# ---------------------------------------------------------------------------
# Per subclass: the cells of that subclass in their embedding's order, normalized from
# .raw (G2) and reduced to the union marker genes.
# ---------------------------------------------------------------------------
offsets, barcodes, xps, aas, blocks = {}, {}, {}, {}, []
n_seen = 0

for cfg in SUBCLASSES:
    token      = cfg['token']
    in_pcha_xp = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_xp.tsv')
    in_pcha_aa = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_aa.tsv')

    xp_df = pd.read_csv(in_pcha_xp, sep='\t', index_col=0)
    aa_df = pd.read_csv(in_pcha_aa, sep='\t', index_col=0)
    if len(aa_df) != cfg['noc']:
        raise ValueError(f'{token}: {in_pcha_aa} has {len(aa_df)} archetypes but noc='
                         f'{cfg["noc"]} — noc must match script 05\'s for this token')

    m_adata = m_adata_all[m_adata_all.obs[SUBCLASS_COL] == cfg['mouse_subclass']]
    if set(m_adata.obs_names) != set(xp_df.index):
        raise ValueError(
            f'{token}: the {m_adata.n_obs} {cfg["mouse_subclass"]} cells in the h5ad are '
            f'not the {len(xp_df)} cells of {in_pcha_xp}')
    m_adata = m_adata[xp_df.index]          # align to the embedding's cell order

    # G2: renormalize from the integer counts, restricted to var_names as script 05 does
    X_raw  = m_adata.raw[:, m_adata.var_names].X
    depths = np.asarray(X_raw.sum(axis=1)).reshape(-1, 1)
    depths[depths == 0] = 1
    block  = np.log2(X_raw[:, union_cols].toarray().astype(np.float32) / depths * 1e4 + 1)

    blocks.append(block)
    barcodes[token] = m_adata.obs_names.values
    xps[token], aas[token] = xp_df, aa_df
    offsets[token] = (n_seen, n_seen + block.shape[0])
    n_seen += block.shape[0]
    print(f'\n{token}: {block.shape[0]} mouse {cfg["mouse_subclass"]} cells x '
          f'{block.shape[1]} union genes')

    del m_adata, X_raw
    gc.collect()

del m_adata_all
gc.collect()

# --- pooled per-gene scale: 2/98 percentiles over all mouse IT cells at once ---
pooled = np.vstack(blocks)
del blocks
gc.collect()
print(f'\nPooled matrix: {pooled.shape[0]} cells x {pooled.shape[1]} genes')
lo  = np.percentile(pooled, SCORE_PCTILE_LO, axis=0)
hi  = np.percentile(pooled, SCORE_PCTILE_HI, axis=0)
rng = np.where(hi > lo, hi - lo, 1.0)   # 05's degenerate guard

pd.DataFrame({'gene': union_genes, 'lo': lo, 'hi': hi}).to_csv(
    OUT_GENE_SCALE, sep='\t', index=False)
print(f'Saved {OUT_GENE_SCALE}  ({int((hi <= lo).sum())} degenerate genes)')

# --- score every column on every subclass against that one scale ---
rows, grid_scores = [], []

for cfg in SUBCLASSES:
    token      = cfg['token']
    start, end = offsets[token]
    block      = pooled[start:end]
    out_scores = os.path.join(
        OUT_RES_DIR, f'14.mouse_all_archetype_scores_on_mouse_{token}.tsv')

    scores = np.zeros((block.shape[0], len(COLUMNS)), dtype=np.float32)
    for j, col in enumerate(COLUMNS):
        cols = [union_pos[g] for g in col['genes']]
        scores[:, j] = np.clip((block[:, cols] - lo[cols]) / rng[cols], 0, 1).mean(axis=1)

    scores_df = pd.DataFrame(scores, index=barcodes[token],
                             columns=[col['key'] for col in COLUMNS])
    scores_df.to_csv(out_scores, sep='\t')
    print(f'\nSaved {out_scores}')
    print(f'  mouse {cfg["mouse_subclass"]} mean score per mouse archetype:')
    print(scores_df.mean().round(3).to_string())

    flip = np.array(cfg.get('flip', [1.0, 1.0]))
    rows.append({'label': f'mouse {cfg["mouse_subclass"]}\n({len(scores_df)} cells)',
                 'xp': xps[token][['PC1', 'PC2']].values * flip,
                 'aa': aas[token][['PC1', 'PC2']].values * flip,
                 'aa_labels': ALPHABET[:cfg['noc']]})
    grid_scores.append([scores[:, j] for j in range(len(COLUMNS))])

del pooled
gc.collect()

# --- the grid: rows = mouse embeddings, columns = mouse archetypes ---
diagonal = {(i, j) for i, cfg in enumerate(SUBCLASSES)
            for j, col in enumerate(COLUMNS) if col['token'] == cfg['token']}
print(f'\nDiagonal (own-subclass) cells outlined: {len(diagonal)}')

save_score_grid_pdf(
    rows, [col['display'] for col in COLUMNS], grid_scores,
    title='Cheng22 mouse IT archetype scores across all mouse IT subclasses',
    out_path=OUT_PDF,
    cmap=SCORE_CMAP, pctile=SCORE_PCTILE,
    highlight=diagonal,
)

print('\nDone.')
