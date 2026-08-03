"""Spatial per-gene expression in Wang25 human infant V1 MERFISH (EN-L2_3-IT).

Based on script 34, but instead of a combined archetype score this colors cells by the
log-normalized expression of individual selected genes, one spatial map per gene in a
multi-page vector PDF. Restricts to EN-L2_3-IT cells (as in scripts 33/34).

Gene symbols are given in mouse casing and upper-cased to match the human MERFISH panel
(Kcnh5 -> KCNH5). Every requested gene is validated against the 300-gene panel up front;
a gene not measured in the panel raises a ValueError (fail-fast, no silent skipping).

Expression matrix (adata.X) is already log-normalized.

Reads:
  links/l23_evo/wang25_human_merfish_infant_V1_glut.h5ad
Outputs:
  local_data/fig/l23_evo/41.wang25_merfish_gene_expr.pdf

Usage:
  python 41.viz.wang25_merfish_gene_expr.py [--genes Kcnh5 Rorb ...]
"""

import os
import argparse
import numpy as np
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_H5AD     = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'wang25_human_merfish_infant_V1_glut.h5ad')
OUT_PDF     = os.path.join(OUT_FIG_DIR, '41.wang25_merfish_gene_expr.pdf')

# --- parameters ---
TYPE_COL   = 'type'
KEEP_TYPE  = 'EN-L2_3-IT'
COORD_KEY  = 'coordinate'
POINT_SIZE = 6
CMAP       = 'RdBu_r'
PCTILE     = (9, 95)
GENES      = ['Kcnh5']   # mouse-cased symbols; upper-cased to the human MERFISH panel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--genes', nargs='+', default=GENES,
                    help='Gene symbols in mouse casing (panel match = upper-cased). Default: %(default)s')
args = parser.parse_args()
genes = list(args.genes)
genes_panel = [g.upper() for g in genes]

os.makedirs(OUT_FIG_DIR, exist_ok=True)
plt.rcParams['pdf.fonttype'] = 42   # editable vector text

# --- load MERFISH data, subset to target cell type ---
adata = ad.read_h5ad(IN_H5AD)
adata = adata[adata.obs[TYPE_COL] == KEEP_TYPE].copy()
coords = np.asarray(adata.obsm[COORD_KEY])
print(f'{adata.n_obs} {KEEP_TYPE} cells, {adata.n_vars}-gene panel')

# --- validate genes against the panel (fail-fast) ---
gene_to_col = {g: i for i, g in enumerate(adata.var_names)}
missing = [g for g in genes_panel if g not in gene_to_col]
if missing:
    raise ValueError(f'Genes not in the {adata.n_vars}-gene MERFISH panel: {missing}')

X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)

# --- one spatial scatter per gene, colored by expression (9-95 percentile range) ---
print(f'Writing {OUT_PDF} ({len(genes)} pages)...')
with PdfPages(OUT_PDF) as pdf:
    for g_in, g_panel in zip(genes, genes_panel):
        expr = X[:, gene_to_col[g_panel]]
        vmin, vmax = np.percentile(expr, PCTILE)
        order = np.argsort(expr)   # draw highest-expressing cells on top

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        sc = ax.scatter(coords[order, 0], coords[order, 1], c=expr[order],
                        s=POINT_SIZE, cmap=CMAP, vmin=vmin, vmax=vmax,
                        linewidths=0, rasterized=True)
        ax.set_aspect('equal')
        ax.invert_yaxis()   # image convention: origin top-left
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_title(f'{KEEP_TYPE}: {g_panel} expression\n(Wang25 V1 MERFISH)')
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(f'{g_panel} (log-norm expr)')

        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f'  {g_panel}: {len(expr)} cells, vmin/vmax {vmin:.2f}/{vmax:.2f}')

print(f'Saved {OUT_PDF}')
print('Done.')
