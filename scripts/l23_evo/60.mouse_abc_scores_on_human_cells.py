"""Mouse Cheng22 ABC archetype scores on human Jorstad23 L2/3 IT cells — seeded record.

Copy of script 22 on the seeded inputs: mouse markers now come from the seeded 58.* fit
instead of the unseeded 21.*, and the display embedding from 56.* instead of 09.* (the
latter is cosmetic — 09.pcha_xp.tsv and 56.human_pcha_xp.tsv are byte-identical, since the
human cell embedding is deterministic PCA; only the archetype vertices ever differed, and
by ~0.003). Only the all-ortholog score set is produced here; script 22 and its 22.*
outputs, including the TF- and CAM-restricted sets, are kept as the original record.

Maps each mouse archetype marker gene to its human ortholog, then computes the same
[0,1]-normalized per-cell score on human expression.

Reads:
  local_data/res/l23_evo/58.mouse_archetype_markers.tsv
  data/human_mouse_orthologs.tsv
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
  local_data/res/l23_evo/56.human_pcha_xp.tsv
  local_data/res/l23_evo/56.human_pcha_aa.tsv
Outputs:
  local_data/res/l23_evo/60.mouse_archetype_scores_on_human_cells.tsv
  local_data/fig/l23_evo/60.mouse_archetype_scores_on_human_cells.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import gene_expr_scatter_html

# --- file paths ---
OUT_RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_MOUSE_MARKERS  = os.path.join(OUT_RES_DIR, '58.mouse_archetype_markers.tsv')
IN_ORTHOLOGS      = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
INPUT_HUMAN       = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
IN_PCHA_XP        = os.path.join(OUT_RES_DIR, '56.human_pcha_xp.tsv')
IN_PCHA_AA        = os.path.join(OUT_RES_DIR, '56.human_pcha_aa.tsv')
OUT_SCORES        = os.path.join(OUT_RES_DIR, '60.mouse_archetype_scores_on_human_cells.tsv')
OUT_SCORE_HTML    = os.path.join(OUT_FIG_DIR, '60.mouse_archetype_scores_on_human_cells.html')

# --- parameters ---
ARCHETYPE_NAMES = ['A', 'B', 'C']
SCORE_PCTILE_LO = 2
SCORE_PCTILE_HI = 98
CLUSTER_COL     = 'WithinArea_cluster'

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load mouse markers and build 1-to-1 ortholog map ---
print('Loading mouse markers and ortholog table...')
markers_df = pd.read_csv(IN_MOUSE_MARKERS, sep='\t')

ortho = pd.read_csv(IN_ORTHOLOGS, sep='\t')
ortho = ortho.drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
mouse_to_human = dict(zip(ortho['mouse_symbol'], ortho['human_symbol']))
print(f'  {len(mouse_to_human)} 1-to-1 orthologs')

# --- load human h5ad — .X is already log-normalized ---
print('Loading human h5ad...')
adata = ad.read_h5ad(INPUT_HUMAN)
gene_names = (adata.var['feature_name'].values
              if 'feature_name' in adata.var.columns
              else adata.var_names.values)
X_norm = adata.X.toarray().astype(np.float32)
human_gene_set = set(gene_names)
gene_to_idx    = {g: i for i, g in enumerate(gene_names)}
cell_barcodes  = adata.obs_names.values
n_cells        = X_norm.shape[0]
print(f'  X_norm shape: {X_norm.shape}')

# --- build per-archetype human gene lists ---
all_human_genes = []
for k in range(len(ARCHETYPE_NAMES)):
    mouse_genes = markers_df[markers_df['archetype'] == f'archetype_{k+1}']['gene'].values
    human_genes = [mouse_to_human[g] for g in mouse_genes
                   if g in mouse_to_human and mouse_to_human[g] in human_gene_set]
    all_human_genes.append(human_genes)

# --- score computation ---
print('All-gene scores:')
scores = np.zeros((n_cells, len(ARCHETYPE_NAMES)), dtype=np.float32)
for k, (name, genes) in enumerate(zip(ARCHETYPE_NAMES, all_human_genes)):
    cols = [gene_to_idx[g] for g in genes]
    print(f'  Score {name}: {len(cols)} genes')
    if not cols:
        continue
    mat = X_norm[:, cols]
    lo  = np.percentile(mat, SCORE_PCTILE_LO, axis=0)
    hi  = np.percentile(mat, SCORE_PCTILE_HI, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    scores[:, k] = np.clip((mat - lo) / rng, 0, 1).mean(axis=1)

# --- save ---
pd.DataFrame(scores, index=cell_barcodes,
             columns=[f'score_{n}' for n in ARCHETYPE_NAMES]).to_csv(OUT_SCORES, sep='\t')
print(f'Saved {OUT_SCORES}')

# --- visualize ---
print('Generating scatter HTML...')
xp = pd.read_csv(IN_PCHA_XP, sep='\t', index_col=0).values   # (n_cells, 5)
aa = pd.read_csv(IN_PCHA_AA, sep='\t', index_col=0).values.T  # (5, 4)
panels = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')]

gene_vals = {f'Score {n}': scores[:, k] for k, n in enumerate(ARCHETYPE_NAMES)}
gene_expr_scatter_html(
    gene_vals=gene_vals, x=xp[:, 0], y=xp[:, 1],
    title='Jorstad23 human L2/3 IT — mouse ABC archetype scores',
    out_path=OUT_SCORE_HTML, xp=xp, panels=panels,
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'), aa=aa,
    pctile_low=5, pctile_high=95,
    colorbar_title='archetype score [0–1]',
)
print('Done.')
