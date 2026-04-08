import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

# --- file paths ---
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_FILE      = '/home/qlyu/mydata/project/SingleCellArchetype/local_data/source/cheng22_yoo25/superdupermegaRNA_cheng22_IT_P28NR.h5ad'
TARGET_FILE   = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'morcom26_cux2mice', 'P26_EN-L2-3-CTX_EN-L4-5-CTX_EN-L2-mix.h5ad')
OUT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'morcom26')
OUT_FIG_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'morcom26_cux2mice')
OUT_TSV_FILE  = os.path.join(OUT_RES_DIR, '16.label_transfer.tsv')
OUT_UMAP_HTML = os.path.join(OUT_FIG_DIR, '16.umap_label_transfer.html')
OUT_PCA_HTML  = os.path.join(OUT_FIG_DIR, '16.pca_label_transfer.html')

# --- config ---
REF_CELLTYPE_COL = 'Type'
REF_DEPTH_COL    = 'n_counts'
TARGET_DEPTH_COL = 'total_counts'
N_PCS            = 20
N_KNN            = 15
MARKER_SIZE      = 3
MARKER_OPACITY   = 0.6

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def normalize(X, depths):
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    return np.log1p(X / depths.reshape(-1, 1) * 1e4)


# --- load ---
print(f'Loading reference: {REF_FILE}')
ref = ad.read_h5ad(REF_FILE)
print(f'  {ref.shape[0]} cells x {ref.shape[1]} genes')
print(f'  Labels: {sorted(ref.obs[REF_CELLTYPE_COL].unique())}')

print(f'Loading target: {TARGET_FILE}')
target = ad.read_h5ad(TARGET_FILE)
print(f'  {target.shape[0]} cells x {target.shape[1]} genes')
print(f'  Cell types: {sorted(target.obs["celltype"].unique())}')

# --- shared genes ---
shared_genes = ref.var_names.intersection(target.var_names)
print(f'Shared genes: {len(shared_genes)}')
if len(shared_genes) < 50:
    raise ValueError(f'Only {len(shared_genes)} shared genes — too few for label transfer')

# --- normalize on shared genes ---
X_ref    = normalize(ref[:, shared_genes].X,    ref.obs[REF_DEPTH_COL].values)
X_target = normalize(target[:, shared_genes].X, target.obs[TARGET_DEPTH_COL].values)

# --- scale: fit on reference, apply to target ---
scaler          = StandardScaler()
X_ref_scaled    = scaler.fit_transform(X_ref)
X_target_scaled = scaler.transform(X_target)

# --- PCA: fit on reference, transform both ---
print(f'Fitting PCA (n_components={N_PCS}) on reference...')
pca          = PCA(n_components=N_PCS, random_state=0)
X_ref_pca    = pca.fit_transform(X_ref_scaled)
X_target_pca = pca.transform(X_target_scaled)

# --- kNN label transfer ---
print(f'Training kNN (k={N_KNN}) on reference...')
knn = KNeighborsClassifier(n_neighbors=N_KNN, metric='euclidean')
knn.fit(X_ref_pca, ref.obs[REF_CELLTYPE_COL].values)
pred_labels = knn.predict(X_target_pca)
pred_proba  = knn.predict_proba(X_target_pca)

label_counts = pd.Series(pred_labels).value_counts()
print('Predicted label counts:')
for label, count in label_counts.items():
    print(f'  {label}: {count}')

# --- save TSV ---
prob_cols = {f'prob_{c}': pred_proba[:, i] for i, c in enumerate(knn.classes_)}
df = pd.DataFrame(
    {'pred_type': pred_labels, **prob_cols, 'celltype': target.obs['celltype'].values},
    index=target.obs_names,
)
df.index.name = 'cell'
df.to_csv(OUT_TSV_FILE, sep='\t')
print(f'Saved label transfer results -> {OUT_TSV_FILE}')

# --- color palette for predicted subtypes ---
pred_types = sorted(knn.classes_)
cmap = plt.get_cmap('tab10', len(pred_types))
type_colors = {t: mcolors.to_hex(cmap(i)) for i, t in enumerate(pred_types)}

# --- UMAP visualization ---
X_umap = target.obsm['X_umap']
celltype_vals = target.obs['celltype'].values

fig_umap = go.Figure()
for pt in pred_types:
    mask = pred_labels == pt
    fig_umap.add_trace(go.Scatter(
        x=X_umap[mask, 0],
        y=X_umap[mask, 1],
        mode='markers',
        name=pt,
        text=celltype_vals[mask],
        hovertemplate='%{text}<extra>' + pt + '</extra>',
        marker=dict(size=MARKER_SIZE, color=type_colors[pt], opacity=MARKER_OPACITY),
    ))
fig_umap.update_layout(
    title='morcom26 EN-L2-3-CTX, EN-L4-5-CTX & EN-L2-mix — UMAP colored by predicted IT subtype (cheng22_yoo25)',
    xaxis_title='UMAP1',
    yaxis_title='UMAP2',
    legend=dict(itemsizing='constant'),
    width=800,
    height=700,
)
fig_umap.write_html(OUT_UMAP_HTML)
print(f'Saved UMAP -> {OUT_UMAP_HTML}')

# --- PCA visualization (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3) ---
fig_pca = make_subplots(rows=1, cols=3, subplot_titles=['PC1 vs PC2', 'PC1 vs PC3', 'PC2 vs PC3'])
pc_pairs = [(0, 1), (0, 2), (1, 2)]

for col_idx, (px, py) in enumerate(pc_pairs, start=1):
    for pt in pred_types:
        mask = pred_labels == pt
        fig_pca.add_trace(
            go.Scatter(
                x=X_target_pca[mask, px],
                y=X_target_pca[mask, py],
                mode='markers',
                name=pt,
                showlegend=(col_idx == 1),
                text=celltype_vals[mask],
                hovertemplate='%{text}<extra>' + pt + '</extra>',
                marker=dict(size=MARKER_SIZE, color=type_colors[pt], opacity=MARKER_OPACITY),
            ),
            row=1, col=col_idx,
        )
    fig_pca.update_xaxes(title_text=f'PC{px+1}', row=1, col=col_idx)
    fig_pca.update_yaxes(title_text=f'PC{py+1}', row=1, col=col_idx)

fig_pca.update_layout(
    title='morcom26 EN-L2-3-CTX, EN-L4-5-CTX & EN-L2-mix — PCA colored by predicted IT subtype (cheng22_yoo25)',
    legend=dict(itemsizing='constant'),
    width=1400,
    height=550,
)
fig_pca.write_html(OUT_PCA_HTML)
print(f'Saved PCA -> {OUT_PCA_HTML}')

print('Done.')
