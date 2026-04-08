import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# --- file paths ---
SCRIPTS_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT        = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE          = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'morcom26_cux2mice', 'P26_EN-L2-3-CTX_EN-L4-5-CTX_EN-L2-mix.h5ad')
LABEL_TRANSFER_FILE = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26', '16.label_transfer.tsv')
OUT_FIG_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_HTML            = os.path.join(OUT_FIG_DIR, '22.boxplot_genes_by_sample.html')

sys.path.insert(0, SCRIPTS_DIR)
from viz import gene_expr_boxplot_html

# --- config ---
# GENES          = ['Kcnq5', 'Sema6d', 'Pcdh15', 'Sntg1']

GENES = ['Fkbp5', 'Ldb2', 'Nr3c2', 
         'Meis2', 'Gpc5', 'Gpc6', 
         'Nell1', 'Sorcs3', 'Gm15398', 
         'Grm8', 'Pcdh15', 'Zfp804b', 
         'Rgma', 'Mkx', 'Etl4', 
         'Kcnq5', 'Wls', 'Adra1a',
         ]
CELLTYPE_COL   = 'celltype'
SAMPLE_COL     = 'samples'
CONDITION_COL  = 'Condition'
PRED_TYPE_COL  = 'pred_type'
KEEP_CELLTYPES = ['EN-L2-3-CTX']
PRED_TYPES     = ['L2/3_A', 'L2/3_B', 'L2/3_C']
CONDITION_COLORS = {'WT': '#4C72B0', 'Null': '#DD8452'}

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load ---
print(f'Loading {INPUT_FILE}')
adata = ad.read_h5ad(INPUT_FILE)
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes')

print(f'Loading label transfer: {LABEL_TRANSFER_FILE}')
lt = pd.read_csv(LABEL_TRANSFER_FILE, sep='\t', index_col=0)

# --- filter ---
mask_ct = adata.obs[CELLTYPE_COL].isin(KEEP_CELLTYPES)
adata = adata[mask_ct].copy()
common_cells = adata.obs_names.intersection(lt.index)
adata = adata[common_cells].copy()
lt = lt.loc[common_cells]
adata.obs[PRED_TYPE_COL] = lt[PRED_TYPE_COL].values
mask_pt = adata.obs[PRED_TYPE_COL].isin(PRED_TYPES)
adata = adata[mask_pt].copy()
print(f'  {adata.shape[0]} cells after filtering')

# --- check genes ---
missing = [g for g in GENES if g not in adata.var_names]
if missing:
    raise ValueError(f'Genes not found: {missing}')

# --- normalize: log2(CP10k + 1) ---
X = adata.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)
logcpm2 = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4) / np.log(2)

# --- build plot dataframe ---
df = adata.obs[[SAMPLE_COL, CONDITION_COL]].copy()
for gene in GENES:
    gene_idx = list(adata.var_names).index(gene)
    df[gene] = logcpm2[:, gene_idx]

# sort samples: WT first, then Null, alphabetically within each group
wt_samples   = sorted(df.loc[df[CONDITION_COL] == 'WT',   SAMPLE_COL].unique())
null_samples  = sorted(df.loc[df[CONDITION_COL] == 'Null', SAMPLE_COL].unique())
sample_order  = wt_samples + null_samples

# --- plot ---
gene_expr_boxplot_html(
    df, GENES, SAMPLE_COL, CONDITION_COL, sample_order, CONDITION_COLORS,
    title='EN-L2-3-CTX, L2/3 subtypes',
    out_path=OUT_HTML,
)
print('Done.')
