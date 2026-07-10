"""Spatial scatter of Wang25 human infant V1 MERFISH glutamatergic neurons.

Plots each cell at its MERFISH spatial coordinate, colored by categorical
cell-type label ('type'). Saved as an editable vector PDF.

Reads:
  links/l23_evo/wang25_human_merfish_infant_V1_glut.h5ad
Outputs:
  local_data/fig/l23_evo/32.wang25_merfish_spatial.pdf
"""

import os
import numpy as np
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'wang25_human_merfish_infant_V1_glut.h5ad')
OUT_PDF     = os.path.join(OUT_FIG_DIR, '32.wang25_merfish_spatial.pdf')

# --- parameters ---
LABEL_COL = 'type'
COORD_KEY = 'coordinate'
POINT_SIZE = 4

os.makedirs(OUT_FIG_DIR, exist_ok=True)
plt.rcParams['pdf.fonttype'] = 42   # editable vector text

# --- load ---
adata = ad.read_h5ad(IN_H5AD)
coords = np.asarray(adata.obsm[COORD_KEY])
labels = adata.obs[LABEL_COL].astype('category')
categories = list(labels.cat.categories)
print(f'{adata.n_obs} cells, {len(categories)} {LABEL_COL} categories')

# --- color map: one distinct color per category ---
cmap = plt.get_cmap('tab10' if len(categories) <= 10 else 'tab20')
colors = {c: cmap(i % cmap.N) for i, c in enumerate(categories)}

# --- scatter ---
fig, ax = plt.subplots(figsize=(7, 7))
for c in categories:
    mask = (labels == c).values
    ax.scatter(coords[mask, 0], coords[mask, 1],
               s=POINT_SIZE, color=colors[c], label=c,
               linewidths=0, rasterized=True)

ax.set_aspect('equal')
ax.invert_yaxis()   # image convention: origin top-left
ax.set_xlabel('x (µm)')
ax.set_ylabel('y (µm)')
ax.set_title('Wang25 human infant V1 MERFISH (glutamatergic)')
ax.legend(title=LABEL_COL, markerscale=3, fontsize=8,
          loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'Saved {OUT_PDF}')
