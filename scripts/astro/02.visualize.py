import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from natsort import natsorted

# --- file paths ---
SCRIPTS_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT   = os.path.dirname(SCRIPTS_DIR)
IN_STANDARD    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '01.standard_results.parquet')
IN_P56_PROJ    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '01.p56_projection_results.parquet')
H5AD_FILE      = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_UMAP_HTML  = os.path.join(OUT_FIG_DIR, '02.umap_metadata.html')
OUT_P56_HTML   = os.path.join(OUT_FIG_DIR, '02.p56proj_pca2d_metadata.html')
OUT_FRAC_HTML  = os.path.join(OUT_FIG_DIR, '02.fraction_barplot.html')
OUT_GENE_HTML       = os.path.join(OUT_FIG_DIR, '02.umap_gene_expression.html')
OUT_GENE_P56_HTML   = os.path.join(OUT_FIG_DIR, '02.p56proj_gene_expression.html')

# --- config ---
CLUSTER_COL       = 'cluster_label'
METADATA_COLS_STD = ['Age', 'cluster_label', 'kmeans_cluster', 'age_bin', 'roi', 'sex']
METADATA_COLS_P56 = ['Age', 'cluster_label', 'age_bin', 'roi', 'sex']
GENES             = ['Gfap', 'S100b', 'Mt1']

sys.path.insert(0, SCRIPTS_DIR)
from viz import scatter_2d_categorical_html, stacked_bar_html, gene_expr_scatter_html

os.makedirs(OUT_FIG_DIR, exist_ok=True)


# --- load ---
print(f'Loading {IN_STANDARD}')
std = pd.read_parquet(IN_STANDARD)
print(f'  {len(std)} cells')

print(f'Loading {IN_P56_PROJ}')
p56 = pd.read_parquet(IN_P56_PROJ)

# --- age order (natsort) and numeric age for continuous colormap ---
age_order = natsorted(std['Age'].unique())
print(f'  Ages: {age_order}')


# --- cluster_label order (natsorted) ---
cluster_order = natsorted(std[CLUSTER_COL].unique())

# =============================================================================
# Plot 1: UMAP colored by metadata
# =============================================================================
print('\nPlotting UMAP...')
X_umap = std[['UMAP1', 'UMAP2']].values
scatter_2d_categorical_html(
    xp_grid=[X_umap],
    cell_metadata={col: std[col].values for col in METADATA_COLS_STD},
    title='gao25 astrocytes — UMAP',
    out_path=OUT_UMAP_HTML,
    xlabel='UMAP1', ylabel='UMAP2',
    ordered_labels=('Age', 'age_bin'),
)

# =============================================================================
# Plot 2: P56-anchored PCA (PC1 vs PC2) colored by metadata
# =============================================================================
print('Plotting P56 projection PCA...')
X_p56 = p56[['PC1', 'PC2']].values
scatter_2d_categorical_html(
    xp_grid=[X_p56],
    cell_metadata={col: p56[col].values for col in METADATA_COLS_P56},
    title='gao25 astrocytes — P56-anchored PCA (PC1 vs PC2)',
    out_path=OUT_P56_HTML,
    xlabel='PC1', ylabel='PC2',
    ordered_labels=('Age', 'age_bin'),
)

# =============================================================================
# Plot 3: cluster_label fractions by Age
# =============================================================================
print('Plotting fraction barplot...')


def fraction_table(df, group_col):
    counts = df.groupby([group_col, CLUSTER_COL], observed=False).size().unstack(fill_value=0)
    counts = counts.reindex(columns=cluster_order, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


frac_age = fraction_table(std, 'Age').reindex(age_order)

stacked_bar_html(
    panel_data=[('By Age', age_order, frac_age)],
    celltypes=cluster_order,
    title=f'gao25 astrocytes — {CLUSTER_COL} fractions by Age',
    out_path=OUT_FRAC_HTML,
    panel_width=1000,
)

# =============================================================================
# Plot 4: UMAP colored by gene expression
# =============================================================================
print('Plotting gene expression UMAP...')
adata = ad.read_h5ad(H5AD_FILE)
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes')

common_cells = std.index.intersection(adata.obs_names)
adata = adata[common_cells]
x = std.loc[common_cells, 'UMAP1'].values
y = std.loc[common_cells, 'UMAP2'].values
print(f'  {len(common_cells)} cells after alignment')

missing = [g for g in GENES if g not in adata.var_names]
if missing:
    raise ValueError(f'Genes not found in h5ad: {missing}')

X = adata.X
if sp.issparse(X):
    X = X.toarray()
depths = X.sum(axis=1, keepdims=True)
X_norm = np.log1p(X / depths * 1e4)
gene_stds = X_norm.std(axis=0)
gene_stds[gene_stds == 0] = 1
X_zscore = (X_norm - X_norm.mean(axis=0)) / gene_stds

gene_idx  = {g: list(adata.var_names).index(g) for g in GENES}
gene_vals = {g: X_zscore[:, gene_idx[g]] for g in GENES}

gene_expr_scatter_html(
    x, y, gene_vals,
    title='z-score(log1p(CP10k))',
    out_path=OUT_GENE_HTML,
    xlabel='UMAP1', ylabel='UMAP2',
)

# =============================================================================
# Plot 5: P56-anchored PCA colored by gene expression
# =============================================================================
print('Plotting gene expression P56 PCA...')
common_p56 = p56.index.intersection(adata.obs_names)
x_p56 = p56.loc[common_p56, 'PC1'].values
y_p56 = p56.loc[common_p56, 'PC2'].values
common_cells_list = list(common_cells)
p56_row_idx = [common_cells_list.index(c) for c in common_p56]
gene_vals_p56 = {g: X_zscore[p56_row_idx, gene_idx[g]] for g in GENES}

gene_expr_scatter_html(
    x_p56, y_p56, gene_vals_p56,
    title='z-score(log1p(CP10k))',
    out_path=OUT_GENE_P56_HTML,
    xlabel='PC1', ylabel='PC2',
)

print('\nDone.')
