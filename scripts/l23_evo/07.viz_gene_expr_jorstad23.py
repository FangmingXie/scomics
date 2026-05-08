"""Gene expression scatter on varimax axes — Jorstad23 L2/3 IT.

Same VX2/VX6/VX7 layout as script 06, colored by z-scored expression of top
loading genes (top N_TOP per displayed component, deduplicated).  Dropdown
menu lets you switch between genes.

Reads:
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
  local_data/res/l23_evo/05.varimax_coords.tsv
  local_data/res/l23_evo/05.varimax_loadings.tsv
Output:
  local_data/fig/l23_evo/07.gene_expr_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import zscore

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from viz import gene_expr_scatter_html

INPUT_FILE   = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_VARIMAX   = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
IN_LOADINGS  = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '07.gene_expr_scatter.html')

VX_COLS = ['VX2', 'VX6', 'VX7']   # components to display (match script 06)
N_TOP   = 5                         # top genes per component

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load varimax coords ---
vx_df    = pd.read_csv(IN_VARIMAX,  sep='\t', index_col=0)
load_df  = pd.read_csv(IN_LOADINGS, sep='\t', index_col=0)
vx_coords = vx_df[VX_COLS].values  # (n_cells, 3)

# --- select top genes by |loading| across displayed components ---
seen, genes_ordered = set(), []
for col in VX_COLS:
    for gene in load_df[col].abs().nlargest(N_TOP).index:
        if gene not in seen:
            seen.add(gene)
            genes_ordered.append(gene)

print(f'Top genes to display ({len(genes_ordered)}): {genes_ordered}')

# --- load expression for selected genes ---
print('Loading h5ad...')
adata = ad.read_h5ad(INPUT_FILE)
gene_names = (adata.var['feature_name'].values
              if 'feature_name' in adata.var.columns
              else adata.var_names.values)

gene_vals = {}
for gene in genes_ordered:
    idx = np.where(gene_names == gene)[0]
    if len(idx) == 0:
        raise ValueError(f'Gene {gene} not found in adata')
    expr = np.asarray(adata.X[:, idx[0]].todense()).ravel().astype(np.float32)
    gene_vals[gene] = zscore(expr)

# --- plot ---
gene_expr_scatter_html(
    x=None, y=None,          # unused when panels= is provided
    gene_vals=gene_vals,
    title='Jorstad23 L2/3 IT — VX2 vs VX6 vs VX7',
    out_path=OUT_HTML,
    xp=vx_coords,
    panels=[(0, 1, 'VX2', 'VX6'), (0, 2, 'VX2', 'VX7'), (1, 2, 'VX6', 'VX7')],
    panel_3d=(0, 1, 2, 'VX2', 'VX6', 'VX7'),
)
