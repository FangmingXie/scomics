"""Mouse Cheng22 archetype scores on the paired human Jorstad23 cells, per IT subclass.

The l23_evo/60 procedure applied to the four subclass pairs: each mouse archetype's marker
set (script 05) is mapped through 1-to-1 human/mouse orthologs and scored on the human
cells of the matching subclass with the same [0,1] per-gene 2/98-percentile min-max clip
and mean over the archetype's ortholog list.

The ortholog block is verbatim from 60, including the order of the two `drop_duplicates`
calls — reversing them changes which pairs survive.

Reads (per TOKEN in L23 / L4 / L5IT / L6IT):
  local_data/res/it_evo/05.mouse_<TOKEN>_archetype_markers.tsv
  data/human_mouse_orthologs.tsv
  links/it_evo/jorstad23_human_WithinArea_<HTOKEN>.h5ad
  local_data/res/it_evo/04.human_<TOKEN>_pcha_{xp,aa}.tsv
Outputs:
  local_data/res/it_evo/06.mouse_<TOKEN>_archetype_scores_on_human_cells.tsv
  local_data/fig/it_evo/06.mouse_<TOKEN>_archetype_scores_on_human_cells.html
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import gene_expr_scatter_html

# --- file paths ---
LINK_DIR     = os.path.join(PROJECT_ROOT, 'links', 'it_evo')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_ORTHOLOGS = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')

# `mouse_noc` must match script 05's `noc` for the same token.
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
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- 1-to-1 ortholog map (verbatim from l23_evo/60) ---
print('Loading ortholog table...')
ortho = pd.read_csv(IN_ORTHOLOGS, sep='\t')
ortho = ortho.drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
mouse_to_human = dict(zip(ortho['mouse_symbol'], ortho['human_symbol']))
print(f'  {len(mouse_to_human)} 1-to-1 orthologs')

for cfg in SUBCLASSES:
    token           = cfg['token']
    human_subclass  = cfg['human_subclass']
    mouse_subclass  = cfg['mouse_subclass']
    noc             = cfg['mouse_noc']
    archetype_names = ALPHABET[:noc]

    in_h5ad          = os.path.join(LINK_DIR, cfg['h5ad'])
    in_mouse_markers = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_archetype_markers.tsv')
    in_pcha_xp       = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_xp.tsv')
    in_pcha_aa       = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_aa.tsv')
    out_scores       = os.path.join(
        OUT_RES_DIR, f'06.mouse_{token}_archetype_scores_on_human_cells.tsv')
    out_score_html   = os.path.join(
        OUT_FIG_DIR, f'06.mouse_{token}_archetype_scores_on_human_cells.html')

    print(f'\n{"=" * 70}\n{token} — mouse {mouse_subclass} archetypes (NOC={noc}) '
          f'scored on human {human_subclass}\n{"=" * 70}')

    markers_df = pd.read_csv(in_mouse_markers, sep='\t')

    # --- load human h5ad — .X is already log-normalized ---
    print('Loading human h5ad...')
    adata = ad.read_h5ad(in_h5ad)
    gene_names     = adata.var[GENE_NAME_COL].values
    human_gene_set = set(gene_names)
    gene_to_idx    = {g: i for i, g in enumerate(gene_names)}
    cell_barcodes  = adata.obs_names.values
    n_cells        = adata.n_obs

    # --- per-archetype human gene lists ---
    all_human_genes = []
    for k in range(noc):
        mouse_genes = markers_df[markers_df['archetype'] == f'archetype_{k+1}']['gene'].values
        human_genes = [mouse_to_human[g] for g in mouse_genes
                       if g in mouse_to_human and mouse_to_human[g] in human_gene_set]
        all_human_genes.append(human_genes)
        n_mouse = len(mouse_genes)
        print(f'  Score {archetype_names[k]}: {len(human_genes)}/{n_mouse} mouse markers '
              f'mapped ({len(human_genes) / n_mouse:.0%})' if n_mouse else
              f'  Score {archetype_names[k]}: 0 mouse markers')

    # --- score computation (slice sparse columns before densifying, G7) ---
    scores = np.zeros((n_cells, noc), dtype=np.float32)
    for k, genes in enumerate(all_human_genes):
        cols = [gene_to_idx[g] for g in genes]
        if not cols:
            print(f'  WARNING: no genes for archetype {archetype_names[k]}')
            continue
        mat = adata.X[:, cols].toarray().astype(np.float32)
        lo  = np.percentile(mat, SCORE_PCTILE_LO, axis=0)
        hi  = np.percentile(mat, SCORE_PCTILE_HI, axis=0)
        rng = np.where(hi > lo, hi - lo, 1.0)
        scores[:, k] = np.clip((mat - lo) / rng, 0, 1).mean(axis=1)
        del mat
        gc.collect()

    pd.DataFrame(scores, index=cell_barcodes,
                 columns=[f'score_{n}' for n in archetype_names]).to_csv(out_scores, sep='\t')
    print(f'Saved {out_scores}')

    # --- visualize on the human PCHA embedding ---
    print('Generating scatter HTML...')
    xp = pd.read_csv(in_pcha_xp, sep='\t', index_col=0).values
    aa = pd.read_csv(in_pcha_aa, sep='\t', index_col=0).values.T
    ndim = xp.shape[1]
    panels = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')][:max(ndim - 1, 1)]
    panel_3d = (0, 1, 2, 'PC1', 'PC2', 'PC3') if ndim >= 3 else None

    gene_vals = {f'Score {n}': scores[:, k] for k, n in enumerate(archetype_names)}
    gene_expr_scatter_html(
        gene_vals=gene_vals, x=xp[:, 0], y=xp[:, 1],
        title=f'Jorstad23 human {human_subclass} — mouse {mouse_subclass} archetype scores',
        out_path=out_score_html, xp=xp, panels=panels,
        panel_3d=panel_3d, aa=aa,
        pctile_low=5, pctile_high=95,
        colorbar_title='archetype score [0–1]',
    )

    del adata
    gc.collect()

print('\nDone.')
