"""Spatial combined archetype score (A + B - C - D) in Wang25 V1 MERFISH.

Restricts to EN-L2_3-IT cells and colors each cell by a combined score:
mean expression of archetype A + B marker genes minus archetype C + D marker
genes, using only genes present in the 300-gene MERFISH panel. Diverging
colormap centered at 0. Vector PDF output.

Expression matrix (adata.X) is already log-normalized.

Reads:
  links/l23_evo/wang25_human_merfish_infant_V1_glut.h5ad
  local_data/res/l23_evo/25.human_archetype_markers.tsv
Outputs:
  local_data/fig/l23_evo/34.wang25_merfish_archetype_combo.pdf
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'wang25_human_merfish_infant_V1_glut.h5ad')
IN_MARKERS  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo', '25.human_archetype_markers.tsv')
OUT_PDF     = os.path.join(OUT_FIG_DIR, '34.wang25_merfish_archetype_combo.pdf')

# --- parameters ---
TYPE_COL   = 'type'
KEEP_TYPE  = 'EN-L2_3-IT'
COORD_KEY  = 'coordinate'
POINT_SIZE = 6
CMAP       = 'RdBu_r'
ARCHETYPES = {'A': 'archetype_1', 'B': 'archetype_2', 'C': 'archetype_3', 'D': 'archetype_4'}
COMBO_SIGN = {'A': +1, 'B': +1, 'C': -1, 'D': -1}   # A + B - C - D

os.makedirs(OUT_FIG_DIR, exist_ok=True)
plt.rcParams['pdf.fonttype'] = 42   # editable vector text

# --- load MERFISH data, subset to target cell type ---
adata = ad.read_h5ad(IN_H5AD)
adata = adata[adata.obs[TYPE_COL] == KEEP_TYPE].copy()
coords = np.asarray(adata.obsm[COORD_KEY])
panel = set(adata.var_names)
print(f'{adata.n_obs} {KEEP_TYPE} cells, {adata.n_vars}-gene panel')

# --- archetype marker genes present in the panel ---
markers = pd.read_csv(IN_MARKERS, sep='\t')
gene_to_col = {g: i for i, g in enumerate(adata.var_names)}
X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)

# --- combined score: signed sum of per-archetype mean expression ---
score = np.zeros(adata.n_obs)
for name, key in ARCHETYPES.items():
    genes = sorted(set(markers[markers['archetype'] == key]['gene']) & panel)
    cols = [gene_to_col[g] for g in genes]
    arch_mean = X[:, cols].mean(axis=1)
    score += COMBO_SIGN[name] * arch_mean
    print(f'  archetype {name} ({COMBO_SIGN[name]:+d}): {len(genes)} panel genes -> {genes}')

# --- single spatial scatter, colored by combined score (9-95 percentile range) ---
vmin, vmax = np.percentile(score, [9, 95])
order = np.argsort(np.abs(score))   # draw strongest scores on top
fig, ax = plt.subplots(figsize=(5.5, 5.5))
sc = ax.scatter(coords[order, 0], coords[order, 1], c=score[order],
                s=POINT_SIZE, cmap=CMAP, vmin=vmin, vmax=vmax,
                linewidths=0, rasterized=True)
ax.set_aspect('equal')
ax.invert_yaxis()   # image convention: origin top-left
ax.set_xlabel('x (µm)')
ax.set_ylabel('y (µm)')
ax.set_title(f'{KEEP_TYPE}: archetype score A + B - C - D\n(Wang25 V1 MERFISH)')
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('A + B - C - D  (mean log-norm expr)')

fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'Saved {OUT_PDF}')
