"""Project P28_dr and P38_dr cells onto the NR Arch1+Arch2 VX1-3 space from script 41.

Projection pipeline:
  1. Re-normalize all cells with same HVGs (z-score uses all-cell statistics, matching script 41).
  2. Regress out library size from DR cells independently.
  3. Center by NR pca_mean (mean of adata_41.X) and project via VX_loadings.
  Harmony correction is not applied (it is sample-specific to NR donors).

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/26.combined_labels.parquet
  local_data/res/astro/41.cheng22_nr_harmony_arch12.h5ad
Outputs:
  local_data/fig/astro/44.vx_categorical.html
  local_data/fig/astro/44.vx_gene_expr.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.utils import norm
from viz import scatter_categorical_html, gene_expr_scatter_html

INPUT_H5AD         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_COMBINED_LABELS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
IN_NR_H5AD         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '41.cheng22_nr_harmony_arch12.h5ad')
OUT_FIG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_CAT            = os.path.join(OUT_FIG_DIR, '44.vx_categorical.html')
OUT_GENE           = os.path.join(OUT_FIG_DIR, '44.vx_gene_expr.html')

LABELED_AGES      = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES           = ['P28', 'P38']
DR_AGES           = ['P28_dr', 'P38_dr']
MIN_CELLS         = 50
EXTRA_GENES       = ['Chrdl1', 'Igfbp2', 'Cdh13', 'Cdh19', 'Gria1', 'Il33']
AVG_GENE_SET      = ['Chrdl1', 'Igfbp2', 'Lef1']
N_TOP             = 5

os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load NR embedding (script 41) ---
print(f'Loading NR embedding from {IN_NR_H5AD}')
adata_nr = ad.read_h5ad(IN_NR_H5AD)
nr_barcodes  = adata_nr.obs_names.values
vx_load      = np.array(adata_nr.varm['VX_loadings'], dtype=np.float64)  # (n_hvg, N_PCS)
hvg_names    = adata_nr.var_names.values
vx_scores_nr = np.array(adata_nr.obsm['X_vx'], dtype=np.float64)
# pca_mean = mean of NR normalized+regressed expression (= pca.mean_ used when fitting PCA)
pca_mean     = np.array(adata_nr.X.toarray() if hasattr(adata_nr.X, 'toarray')
                        else adata_nr.X, dtype=np.float64).mean(axis=0)
arch_labels_nr = adata_nr.obs['archetype'].values
leiden_nr      = adata_nr.obs['leiden'].values
ages_nr        = adata_nr.obs['Age'].values
samples_nr     = adata_nr.obs['Sample'].values
print(f'  {len(nr_barcodes)} NR cells, {len(hvg_names)} HVGs, {vx_load.shape[1]} VX dims')

# --- load arch labels ---
print(f'Loading arch labels from {IN_COMBINED_LABELS}')
df_combined = pd.read_parquet(IN_COMBINED_LABELS)
labels_c22  = df_combined[df_combined['dataset'] == 'cheng22'].reset_index(drop=True)

# --- load raw data ---
print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
print(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

# remove mitochondrial genes (same as script 41)
mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()

# age filter (same as script 41)
ages = adata.obs['Age'].values
age_counts = pd.Series(ages).value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
ages = adata.obs['Age'].values
print(f'  {adata.shape[0]} cells after MT + age filter')

x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float64)
depths = x.sum(axis=1)

# get HVG column indices in the full (MT-filtered) gene set
hvg_col_idx = np.array([np.where(adata.var_names == g)[0][0] for g in hvg_names])

# --- normalize all cells with same HVGs (z-score uses all-cell stats, matching script 41) ---
print('Normalizing all cells...')
xn_all = norm(x[:, hvg_col_idx], depths)   # (all_cells, n_hvg)

# --- identify NR and DR cell indices ---
labeled_mask = np.isin(ages, LABELED_AGES)
assert len(labels_c22) == labeled_mask.sum(), (
    f'Labeled cell count mismatch: parquet {len(labels_c22)} vs adata {labeled_mask.sum()}'
)
arch_labels_all = labels_c22['archetype'].values

labeled_idx = np.where(labeled_mask)[0]

# NR indices: verify they match the 41.h5ad barcodes
all_barcodes = adata.obs_names.values
barcode_to_idx = {b: i for i, b in enumerate(all_barcodes)}
nr_idx_in_adata = np.array([barcode_to_idx[b] for b in nr_barcodes])

# DR indices: P28_dr + P38_dr labeled cells
dr_mask_in_labeled = np.isin(ages[labeled_mask], DR_AGES)
dr_idx_in_adata    = labeled_idx[dr_mask_in_labeled]
arch_labels_dr     = arch_labels_all[dr_mask_in_labeled]
ages_dr            = ages[dr_idx_in_adata]
samples_dr         = adata.obs['Sample'].values[dr_idx_in_adata]
print(f'  DR cells (P28_dr + P38_dr): {len(dr_idx_in_adata)}')
print(f'  DR archetype distribution: {pd.Series(arch_labels_dr).value_counts().to_dict()}')
print(f'  DR age distribution: {pd.Series(ages_dr).value_counts().to_dict()}')

# --- regress out library size from DR cells ---
print('Regressing library size out of DR cells...')
xn_dr = xn_all[dr_idx_in_adata].copy()
log_depth_dr = np.log(depths[dr_idx_in_adata]).reshape(-1, 1)
reg_dr = LinearRegression().fit(log_depth_dr, xn_dr)
xn_dr = xn_dr - reg_dr.predict(log_depth_dr)

# --- project DR cells onto NR VX space ---
print('Projecting DR cells onto NR VX space...')
vx_scores_dr = (xn_dr - pca_mean) @ vx_load   # (n_dr, N_PCS)
print(f'  Projected shape: {vx_scores_dr.shape}')

# --- combine NR + DR for visualization ---
n_nr = len(vx_scores_nr)
n_dr = len(vx_scores_dr)
vx_combined = np.vstack([vx_scores_nr, vx_scores_dr])

origin    = np.array(['NR'] * n_nr + ['DR'] * n_dr)
arch_comb = np.concatenate([arch_labels_nr, arch_labels_dr])
age_comb  = np.concatenate([ages_nr, ages_dr])
samp_comb = np.concatenate([samples_nr, samples_dr])
leid_comb = np.concatenate([leiden_nr, ['NA'] * n_dr])

panels   = [(0, 1, 'VX1', 'VX2'), (0, 2, 'VX1', 'VX3'), (1, 2, 'VX2', 'VX3')]
panel_3d = (0, 1, 2, 'VX1', 'VX2', 'VX3')

# --- categorical HTML ---
print('Building categorical HTML...')
cell_metadata = {
    'origin':    origin,
    'archetype': arch_comb,
    'Age':       age_comb,
    'Sample':    samp_comb,
    'leiden':    leid_comb,
}
scatter_categorical_html(
    xp_grid=[vx_combined],
    cell_metadata=cell_metadata,
    title='cheng22 — DR cells projected onto NR Arch1+Arch2 VX space',
    out_path=OUT_CAT,
    panels=panels,
    panel_3d=panel_3d,
    ordered_labels=['Age'],
)

# --- gene expression: top VX genes + extras ---
load_df = pd.DataFrame(vx_load, index=hvg_names,
                       columns=[f'VX{i+1}' for i in range(vx_load.shape[1])])
seen, ordered_genes = set(), []
for comp in ['VX1', 'VX2', 'VX3']:
    for g in load_df[comp].abs().nlargest(N_TOP).index:
        if g not in seen:
            ordered_genes.append(g)
            seen.add(g)
    print(f'  Top {N_TOP} genes for {comp}: {load_df[comp].abs().nlargest(N_TOP).index.tolist()}')
for g in EXTRA_GENES + AVG_GENE_SET:
    if g not in seen:
        ordered_genes.append(g)
        seen.add(g)
print(f'  Genes for visualization ({len(ordered_genes)}): {ordered_genes}')

# extract log2(CP10k) from raw counts for all combined cells
combined_idx = np.concatenate([nr_idx_in_adata, dr_idx_in_adata])
missing = [g for g in ordered_genes if g not in adata.var_names]
if missing:
    raise ValueError(f'Genes not found: {missing}')
gene_col = [np.where(adata.var_names == g)[0][0] for g in ordered_genes]
x_genes   = x[combined_idx][:, gene_col]
x_lognorm = np.log2(1 + x_genes / depths[combined_idx].reshape(-1, 1) * 1e4)
gene_vals  = {g: x_lognorm[:, i] for i, g in enumerate(ordered_genes)}

avg_label = f'avg({",".join(AVG_GENE_SET)})'
avg_cols  = [ordered_genes.index(g) for g in AVG_GENE_SET]
gene_vals[avg_label] = x_lognorm[:, avg_cols].mean(axis=1)

print('Building gene expression HTML...')
gene_expr_scatter_html(
    x=None, y=None,
    gene_vals=gene_vals,
    title='cheng22 — DR cells projected onto NR Arch1+Arch2 VX space — gene expression',
    out_path=OUT_GENE,
    xp=vx_combined,
    panels=panels,
    panel_3d=panel_3d,
    colorbar_title='log2(CP10k)',
    marker_size=3,
    marker_opacity=0.6,
)

print('Done.')
