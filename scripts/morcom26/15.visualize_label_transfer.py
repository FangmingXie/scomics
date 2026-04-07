import os
import sys
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

# --- file paths ---
SCRIPTS_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT         = os.path.dirname(SCRIPTS_DIR)
REF_FILE             = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'cheng22', 'cheng22.h5ad')
TARGET_FILE          = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'morcom26_cux2mice', 'P26_EN-L2-3-CTX_EN-L2-mix.h5ad')
LABEL_TRANSFER_FILE  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26', '14.label_transfer.tsv')
OUT_FIG_DIR          = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_FRAC_HTML        = os.path.join(OUT_FIG_DIR, '15.fraction_barplot.html')
OUT_3D_HTML          = os.path.join(OUT_FIG_DIR, '15.pca3d_metadata.html')
OUT_3D_CAT_HTML      = os.path.join(OUT_FIG_DIR, '15.pca3d_metadata_cat.html')

# --- config ---
REF_CELLTYPE_COL = 'Type'
REF_DEPTH_COL    = 'n_counts'
TARGET_DEPTH_COL = 'total_counts'
N_PCS            = 20
N_KNN            = 15
PRED_TYPE_COL    = 'pred_type'
PRED_TYPES       = ['L2/3_A', 'L2/3_B', 'L2/3_C']

sys.path.insert(0, SCRIPTS_DIR)
from viz import scatter_html, scatter_categorical_html, stacked_bar_html

os.makedirs(OUT_FIG_DIR, exist_ok=True)


def normalize(X, depths):
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    return np.log1p(X / depths.reshape(-1, 1) * 1e4)


def fraction_table(df, group_col):
    counts = df.groupby([group_col, PRED_TYPE_COL]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=PRED_TYPES, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


# =============================================================================
# COMPUTE
# =============================================================================

# --- load label transfer results and target metadata ---
print(f'Loading label transfer results: {LABEL_TRANSFER_FILE}')
df = pd.read_csv(LABEL_TRANSFER_FILE, sep='\t', index_col=0)

print(f'Loading target metadata: {TARGET_FILE}')
target = ad.read_h5ad(TARGET_FILE)
df['samples']   = target.obs.loc[df.index, 'samples'].values
df['Condition'] = target.obs.loc[df.index, 'Condition'].values
print(f'  {len(df)} cells')

# --- compute fraction tables ---
overall    = df[PRED_TYPE_COL].value_counts(normalize=True).reindex(PRED_TYPES, fill_value=0)
frac_cond     = fraction_table(df, 'Condition')
frac_samp     = fraction_table(df, 'samples')
frac_celltype = fraction_table(df, 'celltype')
conditions    = sorted(df['Condition'].unique())
samples       = sorted(df['samples'].unique())
celltypes     = sorted(df['celltype'].unique())

# --- recompute PCA of morcom26 in reference space ---
print(f'Loading reference for PCA: {REF_FILE}')
ref = ad.read_h5ad(REF_FILE)

shared_genes = ref.var_names.intersection(target.var_names)
print(f'  Shared genes: {len(shared_genes)}')
if len(shared_genes) < 50:
    raise ValueError(f'Only {len(shared_genes)} shared genes — too few')

X_ref    = normalize(ref[:, shared_genes].X,    ref.obs[REF_DEPTH_COL].values)
X_target = normalize(target[:, shared_genes].X, target.obs[TARGET_DEPTH_COL].values)

scaler          = StandardScaler()
X_ref_scaled    = scaler.fit_transform(X_ref)
X_target_scaled = scaler.transform(X_target)

pca          = PCA(n_components=N_PCS, random_state=0)
pca.fit(X_ref_scaled)
X_target_pca = pca.transform(X_target_scaled)   # shape: (7556, 20)
print(f'  PCA done. Shape: {X_target_pca.shape}')

cell_metadata = {
    'pred_type':  df[PRED_TYPE_COL].values,
    'celltype':   df['celltype'].values,
    'Condition':  df['Condition'].values,
    'samples':    df['samples'].values,
    'prob_L2/3_A': df['prob_L2/3_A'].values,
    'prob_L2/3_B': df['prob_L2/3_B'].values,
    'prob_L2/3_C': df['prob_L2/3_C'].values,
}

# =============================================================================
# PLOT
# =============================================================================

# --- fraction barplots ---
print('Plotting cell type fraction barplots...')

overall_df = pd.DataFrame([overall.values], index=['Overall'], columns=PRED_TYPES)
stacked_bar_html(
    panel_data=[
        ('Overall',       ['Overall'], overall_df),
        ('Per condition', conditions,  frac_cond),
        ('Per sample',    samples,     frac_samp),
        ('Per celltype',  celltypes,   frac_celltype),
    ],
    celltypes=PRED_TYPES,
    title='Predicted L2/3 subtype fractions (cheng22 → morcom26)',
    out_path=OUT_FRAC_HTML,
)

# --- 3D PCA with per-category legend ---
print('Building 3D PCA categorical visualization...')
scatter_categorical_html(
    xp_grid=[X_target_pca],
    cell_metadata=cell_metadata,
    title='morcom26 EN-L2-3-CTX & EN-L2-mix — PCA (cheng22 reference space, categorical)',
    out_path=OUT_3D_CAT_HTML,
)

print('Done.')
