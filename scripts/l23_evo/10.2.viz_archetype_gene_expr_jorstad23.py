"""Gene expression scatter for Jorstad23 L2/3 IT archetypes (NOC=5).

Shows top marker genes per archetype (from script 09.2) plus manually specified
genes, plotted in the reprojected VX subspace (PC1/PC2/PC3).

Reads:
  local_data/res/l23_evo/05.varimax_coords.tsv
  local_data/res/l23_evo/09.2.archetype_markers.tsv
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Output:
  local_data/fig/l23_evo/10.2.archetype_gene_expr.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import zscore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from viz import gene_expr_scatter_html

from scomics.main import SCA

# --- file paths ---
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_VARIMAX   = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
IN_MARKERS   = os.path.join(OUT_RES_DIR, '09.2.archetype_markers.tsv')
IN_LOADINGS  = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '10.2.archetype_gene_expr.html')

# --- parameters ---
VX_COLS            = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
NOC                = 5
NDIM               = 5
N_TOP_GENES_VIZ    = 5    # top marker genes per archetype from TSV
N_TOP_GENES_PER_VX = 5    # top genes by |loading| per subtype-informative VX component
N_DOWNSAMPLE       = 5000
CLUSTER_COL        = 'WithinArea_cluster'
EXTRA_GENES        = ['MEIS2', 'SOX5', 'SORCS3', 'TRPC6', 'CHRM2', 'RFX3', 'PCDH15', 'NPAS4']

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- fit PCHA at NOC=5 to get reprojected coords and archetype positions ---
print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
vx_df = pd.read_csv(IN_VARIMAX, sep='\t', index_col=0)
xn    = vx_df[VX_COLS].values
types = vx_df[CLUSTER_COL].values

sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)
# sca.xp: (n_cells, NDIM), sca.aa: (NDIM, NOC)

# --- downsample for plotting ---
n_cells = sca.xp.shape[0]
if N_DOWNSAMPLE is not None and N_DOWNSAMPLE < n_cells:
    rng  = np.random.default_rng(0)
    sidx = rng.choice(n_cells, size=N_DOWNSAMPLE, replace=False)
    xp_plot = sca.xp[sidx]
    print(f'Downsampled {n_cells} → {N_DOWNSAMPLE} cells for plotting.')
else:
    sidx    = np.arange(n_cells)
    xp_plot = sca.xp
    print(f'Using all {n_cells} cells.')

# --- collect genes to display ---
markers_df = pd.read_csv(IN_MARKERS, sep='\t')
seen, genes_ordered = set(), []
for arch in sorted(markers_df['archetype'].unique()):
    for gene in markers_df[markers_df['archetype'] == arch].head(N_TOP_GENES_VIZ)['gene']:
        if gene not in seen:
            seen.add(gene)
            genes_ordered.append(gene)
for gene in EXTRA_GENES:
    if gene not in seen:
        seen.add(gene)
        genes_ordered.append(gene)

load_df = pd.read_csv(IN_LOADINGS, sep='\t', index_col=0)
for col in VX_COLS:
    for gene in load_df[col].abs().nlargest(N_TOP_GENES_PER_VX).index:
        if gene not in seen:
            seen.add(gene)
            genes_ordered.append(gene)

print(f'Genes to display ({len(genes_ordered)}): {genes_ordered}')

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
    gene_vals[gene] = zscore(expr)[sidx]

# --- plot ---
gene_expr_scatter_html(
    x=None, y=None,
    gene_vals=gene_vals,
    title=f'Jorstad23 L2/3 IT — Archetype markers (NOC={NOC}, reprojected VX subspace)',
    out_path=OUT_HTML,
    xp=xp_plot,
    panels=[(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')],
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    aa=sca.aa,
)

print('Done.')
