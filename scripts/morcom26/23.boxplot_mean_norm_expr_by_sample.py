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
OUT_HTML            = os.path.join(OUT_FIG_DIR, '23.boxplot_mean_norm_expr_by_sample.html')
OUT_HTML_CONDITION  = os.path.join(OUT_FIG_DIR, '23.boxplot_mean_norm_expr_by_condition.html')

sys.path.insert(0, SCRIPTS_DIR)
from viz import gene_expr_boxplot_html

# --- config ---
# Each key is the panel label; each value is the gene list for that panel.
GENE_LISTS = {
    'Cux2_+': ['Pcnx2', 'Jmjd1c', 'Tnpo1', 'Cdh20', 'Gramd1b', 'Ank3', 'Igsf9', 'Gdpd5', 'C1galt1', 'Dkk3', 'Glis3', 'Egfem1', 'Pard3b', 'Dbn1', 'Plppr1', 'Arl15', 'Fhod3', 'Tshz3', 'Aff3', 'Cacna1b', 'Dlgap4', 'Cyth3', 'Nrg2', 'Cux2', 'Cux1', 'Tnrc18', 'Ntng2', 'Efcab6', 'Ptprn2', 'Ephb2', 'Adgrl2', 'Pou3f1', 'Ntrk2', 'Setbp1', 'Hs6st3', 'Tenm2', 'Nav2', 'Adam11', 'Ptprs'],
    'Cux2_-': ['Zfp804b', 'Kcnt2', 'Ngf', 'Etv1', 'Cth', 'Sgcd', 'Adcy1', 'Vwc2l', 'Gpc3', 'Atp2b4', 'Cnih3', 'Prlr', 'Grid2', 'Cdh4', 'Xylt1', 'Mpped2', 'Zbtb20', 'Gria1', 'Plcb1', 'Pigk', 'Egflam', 'Col25a1', 'Arid5b', 'Pcdh19', 'Kcnab1', 'Ptpru', 'Cpne8', 'Dpyd', 'St6galnac5', 'Spon1', 'Slit3'],
}

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
all_genes = [g for genes in GENE_LISTS.values() for g in genes]
missing = [g for g in all_genes if g not in adata.var_names]
if missing:
    raise ValueError(f'Genes not found: {missing}')

# --- normalize: log2(CP10k + 1) ---
X = adata.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)
logcpm2 = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4) / np.log(2)

var_names = list(adata.var_names)
wt_mask = (adata.obs[CONDITION_COL] == 'WT').values

# --- build plot dataframe ---
df = adata.obs[[SAMPLE_COL, CONDITION_COL]].copy()

# compute one mean-normalized score per gene list
for panel_name, genes in GENE_LISTS.items():
    gene_idx = [var_names.index(g) for g in genes]
    logcpm2_genes = logcpm2[:, gene_idx]                    # (n_cells, n_genes)
    wt_means = logcpm2_genes[wt_mask].mean(axis=0)          # (n_genes,)
    norm_genes = logcpm2_genes - wt_means[np.newaxis, :]    # (n_cells, n_genes)
    df[panel_name] = norm_genes.mean(axis=1)                # (n_cells,)

# sort samples: WT first, then Null, alphabetically within each group
wt_samples  = sorted(df.loc[df[CONDITION_COL] == 'WT',   SAMPLE_COL].unique())
null_samples = sorted(df.loc[df[CONDITION_COL] == 'Null', SAMPLE_COL].unique())
sample_order = wt_samples + null_samples

# --- plot: by sample ---
gene_expr_boxplot_html(
    df, list(GENE_LISTS.keys()), SAMPLE_COL, CONDITION_COL, sample_order, CONDITION_COLORS,
    title='EN-L2-3-CTX, L2/3 subtypes — mean normalized expression',
    out_path=OUT_HTML,
    yaxis_title='mean normalized expression (log2 diff vs WT)',
)

# --- plot: by condition ---
condition_order = list(CONDITION_COLORS.keys())
gene_expr_boxplot_html(
    df, list(GENE_LISTS.keys()), CONDITION_COL, CONDITION_COL, condition_order, CONDITION_COLORS,
    title='EN-L2-3-CTX, L2/3 subtypes — mean normalized expression',
    out_path=OUT_HTML_CONDITION,
    yaxis_title='mean normalized expression (log2 diff vs WT)',
)
print('Done.')
