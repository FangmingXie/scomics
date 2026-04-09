import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# --- file paths ---
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'cheng22_yoo25', 'superdupermegaRNA_yoo25_IT_P21.h5ad')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'yoo25')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '01.boxplot_genes_by_sample.html')

sys.path.insert(0, SCRIPTS_DIR)
from viz import gene_expr_boxplot_html

# --- config ---
GENES            = ['Cux2', 'Cux1']
CELLTYPE_COL     = 'Type'
SAMPLE_COL       = 'Sample'
CONDITION_COL    = 'condition'
KEEP_CELLTYPES   = ['L2/3_A', 'L2/3_B', 'L2/3_C']
CONDITION_COLORS = {'L2/3': '#4C72B0'}

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load ---
print(f'Loading {INPUT_FILE}')
adata = ad.read_h5ad(INPUT_FILE)
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes')

# --- filter ---
mask = adata.obs[CELLTYPE_COL].isin(KEEP_CELLTYPES)
adata = adata[mask].copy()
print(f'  {adata.shape[0]} cells after filtering to L2/3 subtypes')

# --- check genes ---
missing = [g for g in GENES if g not in adata.var_names]
if missing:
    raise ValueError(f'Genes not found: {missing}')

# --- normalize: log2(CP10k + 1) ---
X = adata.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)
depths = adata.obs['total_counts'].values.reshape(-1, 1)
logcpm2 = np.log1p(X / depths * 1e4) / np.log(2)

# --- build plot dataframe ---
df = adata.obs[[SAMPLE_COL]].copy()
df[CONDITION_COL] = 'L2/3'
for gene in GENES:
    gene_idx = list(adata.var_names).index(gene)
    df[gene] = logcpm2[:, gene_idx]

sample_order = sorted(df[SAMPLE_COL].unique())

# --- plot ---
gene_expr_boxplot_html(
    df, GENES, SAMPLE_COL, CONDITION_COL, sample_order, CONDITION_COLORS,
    title='L2/3 cells (yoo25)',
    out_path=OUT_HTML,
)
print('Done.')
