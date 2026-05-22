"""Apply mouse Cheng22 ABC archetype scores to human Jorstad23 L2/3 IT cells.

Maps each mouse archetype marker gene to its human ortholog, then computes
the same [0,1]-normalized per-cell score on human expression. Produces three
score sets and HTMLs: all ortholog markers, TF-restricted, and CAM-restricted.

TF list is downloaded from Lambert et al. 2018 (humantfs.ccbr.utoronto.ca) on
first run and cached at links/common/transcription_factors.tsv.

Reads:
  local_data/res/l23_evo/21.mouse_archetype_markers.tsv
  data/human_mouse_orthologs.tsv
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
  local_data/res/l23_evo/09.pcha_xp.tsv
  local_data/res/l23_evo/09.pcha_aa.tsv
  links/common/transcription_factors.tsv   (downloaded on first run)
  links/common/cell_adhesion_molecules.csv
Outputs:
  local_data/res/l23_evo/22.human_archetype_scores.tsv
  local_data/res/l23_evo/22.human_archetype_scores_TF.tsv
  local_data/res/l23_evo/22.human_archetype_scores_CAM.tsv
  local_data/fig/l23_evo/22.human_archetype_scores.html
  local_data/fig/l23_evo/22.human_archetype_scores_TF.html
  local_data/fig/l23_evo/22.human_archetype_scores_CAM.html
"""

import os
import sys
import requests
import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import gene_expr_scatter_html

# --- file paths ---
OUT_RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_MOUSE_MARKERS  = os.path.join(OUT_RES_DIR, '21.mouse_archetype_markers.tsv')
IN_ORTHOLOGS      = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
INPUT_HUMAN       = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
IN_PCHA_XP        = os.path.join(OUT_RES_DIR, '09.pcha_xp.tsv')
IN_PCHA_AA        = os.path.join(OUT_RES_DIR, '09.pcha_aa.tsv')
IN_TF_LIST        = os.path.join(PROJECT_ROOT, 'links', 'common', 'transcription_factors.tsv')
IN_CAM_LIST       = os.path.join(PROJECT_ROOT, 'links', 'common', 'cell_adhesion_molecules.csv')
OUT_SCORES        = os.path.join(OUT_RES_DIR, '22.human_archetype_scores.tsv')
OUT_TF_SCORES     = os.path.join(OUT_RES_DIR, '22.human_archetype_scores_TF.tsv')
OUT_CAM_SCORES    = os.path.join(OUT_RES_DIR, '22.human_archetype_scores_CAM.tsv')
OUT_SCORE_HTML    = os.path.join(OUT_FIG_DIR, '22.human_archetype_scores.html')
OUT_TF_HTML       = os.path.join(OUT_FIG_DIR, '22.human_archetype_scores_TF.html')
OUT_CAM_HTML      = os.path.join(OUT_FIG_DIR, '22.human_archetype_scores_CAM.html')

# --- parameters ---
ARCHETYPE_NAMES = ['A', 'B', 'C']
SCORE_PCTILE_LO = 2
SCORE_PCTILE_HI = 98
CLUSTER_COL     = 'WithinArea_cluster'
TF_URL          = 'https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- download TF list (cached) ---
if not os.path.exists(IN_TF_LIST):
    print(f'Downloading TF list from Lambert et al. 2018...')
    r = requests.get(TF_URL)
    r.raise_for_status()
    with open(IN_TF_LIST, 'w') as f:
        f.write(r.text)
    print(f'  Saved → {IN_TF_LIST}')

# --- load gene-set filters ---
tf_df  = pd.read_csv(IN_TF_LIST, sep='\t')
tf_set = set(tf_df[tf_df['Is TF?'] == 'Yes']['HGNC symbol'].dropna())
print(f'TF set: {len(tf_set)} human TFs (Lambert et al. 2018)')

cam_df  = pd.read_csv(IN_CAM_LIST)
cam_set = set(cam_df['gene_name'].dropna())
print(f'CAM set: {len(cam_set)} mouse CAMs')

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

# TF subset: filter human orthologs to TF set
tf_human_genes = [[g for g in hg if g in tf_set] for hg in all_human_genes]

# CAM subset: filter mouse markers to CAM set, then map to human
cam_human_genes = []
for k in range(len(ARCHETYPE_NAMES)):
    mouse_genes = markers_df[markers_df['archetype'] == f'archetype_{k+1}']['gene'].values
    cam_genes   = [mouse_to_human[g] for g in mouse_genes
                   if g in cam_set and g in mouse_to_human
                   and mouse_to_human[g] in human_gene_set]
    cam_human_genes.append(cam_genes)

# --- score computation helper ---
def _compute_scores(per_arch_human_genes, label):
    print(f'{label}:')
    out = np.zeros((n_cells, len(ARCHETYPE_NAMES)), dtype=np.float32)
    for k, (name, genes) in enumerate(zip(ARCHETYPE_NAMES, per_arch_human_genes)):
        cols = [gene_to_idx[g] for g in genes]
        print(f'  Score {name}: {len(cols)} genes')
        if not cols:
            continue
        mat = X_norm[:, cols]
        lo  = np.percentile(mat, SCORE_PCTILE_LO, axis=0)
        hi  = np.percentile(mat, SCORE_PCTILE_HI, axis=0)
        rng = np.where(hi > lo, hi - lo, 1.0)
        out[:, k] = np.clip((mat - lo) / rng, 0, 1).mean(axis=1)
    return out

scores     = _compute_scores(all_human_genes,  'All-gene scores')
scores_tf  = _compute_scores(tf_human_genes,   'TF scores')
scores_cam = _compute_scores(cam_human_genes,  'CAM scores')

# --- save ---
def _save_scores(arr, path):
    pd.DataFrame(arr, index=cell_barcodes,
                 columns=[f'score_{n}' for n in ARCHETYPE_NAMES]).to_csv(path, sep='\t')
    print(f'Saved {path}')

_save_scores(scores,     OUT_SCORES)
_save_scores(scores_tf,  OUT_TF_SCORES)
_save_scores(scores_cam, OUT_CAM_SCORES)

# --- visualize ---
print('Generating scatter HTMLs...')
xp = pd.read_csv(IN_PCHA_XP, sep='\t', index_col=0).values   # (n_cells, 5)
aa = pd.read_csv(IN_PCHA_AA, sep='\t', index_col=0).values.T  # (5, 4)
panels = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')]

def _scatter(scores_arr, title_suffix, out_path):
    gene_vals = {f'Score {n}': scores_arr[:, k] for k, n in enumerate(ARCHETYPE_NAMES)}
    gene_expr_scatter_html(
        gene_vals=gene_vals, x=xp[:, 0], y=xp[:, 1],
        title=f'Jorstad23 human L2/3 IT — {title_suffix}',
        out_path=out_path, xp=xp, panels=panels,
        panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'), aa=aa,
        pctile_low=5, pctile_high=95,
        colorbar_title='archetype score [0–1]',
    )

_scatter(scores,     'mouse ABC archetype scores',       OUT_SCORE_HTML)
_scatter(scores_tf,  'mouse ABC archetype scores (TF)',  OUT_TF_HTML)
_scatter(scores_cam, 'mouse ABC archetype scores (CAM)', OUT_CAM_HTML)
print('Done.')
