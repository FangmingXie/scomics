"""Spatial expression of human archetype (ABCD) marker genes in Wang25 V1 MERFISH.

Restricts to EN-L2_3-IT cells and, for each human archetype A/B/C/D, colors the
cells by the mean expression of that archetype's marker genes that are present in
the 300-gene MERFISH panel. One spatial panel per archetype. Vector PDF output.

Expression matrix (adata.X) is already log-normalized.

Reads:
  links/l23_evo/wang25_human_merfish_infant_V1_glut.h5ad
  local_data/res/l23_evo/25.human_archetype_markers.tsv
Outputs:
  local_data/fig/l23_evo/33.wang25_merfish_archetype_expr.pdf
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'wang25_human_merfish_infant_V1_glut.h5ad')
IN_MARKERS  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo', '25.human_archetype_markers.tsv')
OUT_PDF     = os.path.join(OUT_FIG_DIR, '33.wang25_merfish_archetype_expr.pdf')

# --- parameters ---
TYPE_COL   = 'type'
KEEP_TYPE  = 'EN-L2_3-IT'
COORD_KEY  = 'coordinate'
POINT_SIZE = 6
CMAP       = sns.color_palette('rocket_r', as_cmap=True)
ARCHETYPES = {'A': 'archetype_1', 'B': 'archetype_2', 'C': 'archetype_3', 'D': 'archetype_4'}

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
arch_genes = {}
for name, key in ARCHETYPES.items():
    genes = set(markers[markers['archetype'] == key]['gene'])
    arch_genes[name] = sorted(genes & panel)
    print(f'  archetype {name}: {len(arch_genes[name])} / {len(genes)} markers in panel -> {arch_genes[name]}')

# --- per-cell mean expression of each archetype's panel genes ---
gene_to_col = {g: i for i, g in enumerate(adata.var_names)}
X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
arch_expr = {}
for name, genes in arch_genes.items():
    cols = [gene_to_col[g] for g in genes]
    arch_expr[name] = X[:, cols].mean(axis=1)

# --- 1x4 spatial scatter, colored by mean archetype-gene expression ---
names = list(ARCHETYPES)
fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.6))

for ax, name in zip(axes, names):
    vals = arch_expr[name]
    order = np.argsort(vals)   # draw high-expression cells on top
    sc = ax.scatter(coords[order, 0], coords[order, 1], c=vals[order],
                    s=POINT_SIZE, cmap=CMAP, linewidths=0, rasterized=True)
    ax.set_aspect('equal')
    ax.invert_yaxis()   # image convention: origin top-left
    ax.set_title(f"Archetype {name}  ({len(arch_genes[name])} genes)")
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('mean log-norm expr')

fig.suptitle(f'{KEEP_TYPE}: archetype marker-gene expression (Wang25 V1 MERFISH)', y=1.02)
fig.subplots_adjust(wspace=0.55)
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'Saved {OUT_PDF}')
