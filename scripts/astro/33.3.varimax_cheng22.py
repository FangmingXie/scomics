"""Varimax rotation of PCA components + variance partitioning — cheng22 astrocytes.

Analogous to script 33.2 (gao25 P56 gray-matter) but applied to cheng22:
  - Fits PCA on P28 cells (reference population).
  - Archetype labels (Arch1-4) are sourced from 26.combined_labels.parquet
    (kNN-transferred from gao25 P56 reference).
  - Arch4 (white-matter) cells excluded; analysis covers Arch1-3 gray-matter subtypes only.
  - Variance is partitioned among archetype, donor (Sample), and library_size.

Saves PCA+varimax parameters for downstream algebraic projection.

Reads:
  links/astro/cheng22_astro.h5ad
  local_data/res/astro/26.combined_labels.parquet
Outputs:
  local_data/res/astro/33.3.varimax_coords.tsv          (P28 cells × VX1..VX10 + metadata)
  local_data/res/astro/33.3.varimax_loadings.tsv        (HVGs × VX1..VX10)
  local_data/res/astro/33.3.pca_variance_partition.tsv  (PC1..PC10)
  local_data/res/astro/33.3.variance_partition.tsv      (VX1..VX10)
  local_data/res/astro/33.3.pca_components.tsv          (N_PCS × n_hvg)
  local_data/res/astro/33.3.pca_mean.tsv                (1 × n_hvg)
  local_data/res/astro/33.3.varimax_R.tsv               (N_PCS × N_PCS, reordered)
  local_data/fig/astro/33.3.variance_partition.html     (PCA + VX charts combined)
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg
from scomics.utils import norm

# --- file paths ---
INPUT_H5AD         = os.path.join(PROJECT_ROOT, 'links', 'astro', 'cheng22_astro.h5ad')
IN_COMBINED_LABELS = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '26.combined_labels.parquet')
OUT_RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_FIG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_VX_COORDS      = os.path.join(OUT_RES_DIR, '33.3.varimax_coords.tsv')
OUT_VX_LOADINGS    = os.path.join(OUT_RES_DIR, '33.3.varimax_loadings.tsv')
OUT_PCA_VAR_PARTITION = os.path.join(OUT_RES_DIR, '33.3.pca_variance_partition.tsv')
OUT_VAR_PARTITION  = os.path.join(OUT_RES_DIR, '33.3.variance_partition.tsv')
OUT_PCA_COMPONENTS = os.path.join(OUT_RES_DIR, '33.3.pca_components.tsv')
OUT_PCA_MEAN       = os.path.join(OUT_RES_DIR, '33.3.pca_mean.tsv')
OUT_VARIMAX_R      = os.path.join(OUT_RES_DIR, '33.3.varimax_R.tsv')
OUT_VAR_PART_HTML  = os.path.join(OUT_FIG_DIR, '33.3.variance_partition.html')

N_HVG       = 2000
N_PCS       = 10
MIN_CELLS   = 50
LABELED_AGES = ['P28', 'P28_dl', 'P28_dr', 'P38', 'P38_dr']
NR_AGES      = ['P28']

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def varimax(L, gamma=1.0, max_iter=1000, tol=1e-6):
    """Kaiser varimax rotation of loading matrix L (n_vars × n_factors)."""
    n, p = L.shape
    R = np.eye(p)
    for _ in range(max_iter):
        R_old = R.copy()
        for i in range(p - 1):
            for j in range(i + 1, p):
                Lr = L @ R
                u = Lr[:, i] ** 2 - Lr[:, j] ** 2
                v = 2 * Lr[:, i] * Lr[:, j]
                A = u.sum()
                B = v.sum()
                C = (u ** 2 - v ** 2).sum()
                D = 2 * (u * v).sum()
                theta = 0.25 * np.arctan2(
                    D - gamma * 2 * A * B / n,
                    C - gamma * (A ** 2 - B ** 2) / n,
                )
                c, s = np.cos(theta), np.sin(theta)
                Rij = np.eye(p)
                Rij[i, i] = Rij[j, j] = c
                Rij[i, j] = -s
                Rij[j, i] = s
                R = R @ Rij
        if np.max(np.abs(R - R_old)) < tol:
            break
    return R


def r2(X, y):
    reg = LinearRegression().fit(X, y)
    ss_res = np.sum((y - reg.predict(X)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


# --- load data ---
print(f'Loading {INPUT_H5AD}')
adata = ad.read_h5ad(INPUT_H5AD)
print(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

# remove mitochondrial genes
mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
print(f'  Removed {mt_mask.sum()} mitochondrial genes; {(~mt_mask).sum()} remaining')

# filter to ages with > MIN_CELLS cells
ages = adata.obs['Age'].values
age_counts = pd.Series(ages).value_counts()
valid_ages = age_counts[age_counts > MIN_CELLS].index
adata = adata[np.isin(ages, valid_ages)].copy()
ages = adata.obs['Age'].values
print(f'  Ages with >{MIN_CELLS} cells: {sorted(valid_ages.tolist(), key=lambda a: int(a[1:].split("_")[0]))}')
print(f'  Total cells: {adata.shape[0]}')

x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
depths = x.sum(axis=1)
donors = adata.obs['Sample'].values

# --- load arch labels from 26.combined_labels.parquet ---
print(f'Loading arch labels from {IN_COMBINED_LABELS}')
df_combined = pd.read_parquet(IN_COMBINED_LABELS)
labels_c22 = df_combined[df_combined['dataset'] == 'cheng22'].reset_index(drop=True)

labeled_mask = np.isin(ages, LABELED_AGES)
assert len(labels_c22) == labeled_mask.sum(), (
    f'Labeled cell count mismatch: parquet {len(labels_c22)} vs adata {labeled_mask.sum()}'
)
arch_labels_all = labels_c22['archetype'].values
print(f'  Labeled cells (P28/P28_dl/P28_dr/P38/P38_dr): {labeled_mask.sum()}')

# reference cells: P28 only, within labeled cells
nr_mask_in_labeled = np.isin(ages[labeled_mask], NR_AGES)
nr_indices = np.where(labeled_mask)[0][nr_mask_in_labeled]
arch_labels_nr = arch_labels_all[nr_mask_in_labeled]
print(f'  P28 cells (before Arch4 removal): {nr_mask_in_labeled.sum()}')

# exclude Arch4
non_arch4 = arch_labels_nr != 'Arch4'
nr_gm_indices = nr_indices[non_arch4]
nr_mask = np.zeros(len(ages), dtype=bool)
nr_mask[nr_gm_indices] = True
arch_labels = arch_labels_nr[non_arch4]
print(f'  Removed {(~non_arch4).sum()} Arch4 cells; P28 GM cells: {nr_mask.sum()}')
print(f'  Arch1-3 distribution: {pd.Series(arch_labels).value_counts().to_dict()}')

# --- HVG selection (P28 cells only) ---
hvg_mask = select_hvg(x[nr_mask], depths[nr_mask], N_HVG)
gene_names = adata.var_names.values[hvg_mask]
print(f'  HVGs selected: {hvg_mask.sum()}')

# --- normalize all cells jointly ---
xn = norm(x[:, hvg_mask], depths)
xn_nr = xn[nr_mask]
print(f'  xn shape (all cells): {xn.shape}')

# --- fit PCA on P28 cells ---
print(f'Fitting PCA (N_PCS={N_PCS}) on P28 cells...')
pca = PCA(N_PCS, random_state=0)
pca.fit(xn_nr)
pca_scores_nr = pca.transform(xn_nr)   # (n_nr, N_PCS)
L = pca.components_.T                   # (n_hvg, N_PCS)

# --- PCA variance partitioning ---
pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
X_arch    = pd.get_dummies(arch_labels).values.astype(float)
X_donor   = pd.get_dummies(donors[nr_mask]).values.astype(float)
X_libsize = zscore(depths[nr_mask]).reshape(-1, 1)
X_full    = np.hstack([X_arch, X_donor, X_libsize])

FACTORS = {
    'archetype':    X_arch,
    'donor':        X_donor,
    'library_size': X_libsize,
}

print('Computing PCA variance partitioning...')
pca_vp_rows = []
for i, col in enumerate(pc_cols):
    y = pca_scores_nr[:, i]
    r2_full = r2(X_full, y)
    partial = {}
    for name, X_factor in FACTORS.items():
        X_others = np.hstack([v for k, v in FACTORS.items() if k != name])
        r2_reduced = r2(X_others, y)
        partial[name] = max(r2_full - r2_reduced, 0.0)
    residual = max(1.0 - r2_full, 0.0)
    pca_vp_rows.append({'PC': col, **partial, 'residual': residual})
    print(f"  {col}  R²_full={r2_full:.3f}  "
          f"arch={partial['archetype']:.3f}  "
          f"donor={partial['donor']:.3f}  "
          f"libsize={partial['library_size']:.3f}  "
          f"resid={residual:.3f}")

pca_vp_df = pd.DataFrame(pca_vp_rows).set_index('PC')
pca_vp_df.to_csv(OUT_PCA_VAR_PARTITION, sep='\t')
print(f'Saved {OUT_PCA_VAR_PARTITION}')

# --- varimax rotation ---
print('Running varimax rotation...')
R = varimax(L)
vx_scores   = pca_scores_nr @ R    # (n_nr, N_PCS)
vx_loadings = L @ R                 # (n_hvg, N_PCS)

# reorder by descending variance
vx_var_order = np.argsort(vx_scores.var(axis=0))[::-1]
vx_scores    = vx_scores[:, vx_var_order]
vx_loadings  = vx_loadings[:, vx_var_order]

vx_cols = [f'VX{i+1}' for i in range(N_PCS)]

# save R with reorder baked in
R_save = R[:, vx_var_order]   # (N_PCS, N_PCS)

# --- save VX coords (P28 cells only) ---
vx_df = pd.DataFrame(vx_scores, index=adata.obs_names[nr_mask], columns=vx_cols)
vx_df['archetype']  = arch_labels
vx_df['donor_name'] = donors[nr_mask]
vx_df['Age']        = ages[nr_mask]
vx_df['depth']      = depths[nr_mask]
vx_df.to_csv(OUT_VX_COORDS, sep='\t')
print(f'Saved {OUT_VX_COORDS}')

# --- save VX loadings ---
vx_load_df = pd.DataFrame(vx_loadings, index=gene_names, columns=vx_cols)
vx_load_df.to_csv(OUT_VX_LOADINGS, sep='\t')
print(f'Saved {OUT_VX_LOADINGS}')

# --- save PCA parameters for downstream scripts ---
pd.DataFrame(pca.components_, columns=gene_names,
             index=[f'PC{i+1}' for i in range(N_PCS)]).to_csv(OUT_PCA_COMPONENTS, sep='\t')
pd.DataFrame(pca.mean_.reshape(1, -1), columns=gene_names).to_csv(OUT_PCA_MEAN, sep='\t')
pd.DataFrame(R_save, index=[f'PC{i+1}' for i in range(N_PCS)],
             columns=vx_cols).to_csv(OUT_VARIMAX_R, sep='\t')
print(f'Saved {OUT_PCA_COMPONENTS}, {OUT_PCA_MEAN}, {OUT_VARIMAX_R}')

# --- VX variance partitioning ---
print('Computing VX variance partitioning...')
rows = []
for col in vx_cols:
    y = vx_df[col].values
    r2_full = r2(X_full, y)
    partial = {}
    for name, X_factor in FACTORS.items():
        X_others = np.hstack([v for k, v in FACTORS.items() if k != name])
        r2_reduced = r2(X_others, y)
        partial[name] = max(r2_full - r2_reduced, 0.0)
    residual = max(1.0 - r2_full, 0.0)
    rows.append({'VX': col, **partial, 'residual': residual})
    print(f"  {col}  R²_full={r2_full:.3f}  "
          f"arch={partial['archetype']:.3f}  "
          f"donor={partial['donor']:.3f}  "
          f"libsize={partial['library_size']:.3f}  "
          f"resid={residual:.3f}")

result_df = pd.DataFrame(rows).set_index('VX')
result_df.to_csv(OUT_VAR_PARTITION, sep='\t')
print(f'Saved {OUT_VAR_PARTITION}')

# --- HTML: PCA bar + VX bar + top genes tables ---
COLORS = {
    'archetype':    '#4C72B0',
    'donor':        '#DD8452',
    'library_size': '#C44E52',
    'residual':     '#CCCCCC',
}

pc_var_frac = (pca.explained_variance_ / pca.explained_variance_.sum() * 100).round(1)
pc_x_labels = [f'{col}<br>({pc_var_frac[i]}%)' for i, col in enumerate(pc_cols)]

fig_pca = go.Figure()
for factor, color in COLORS.items():
    fig_pca.add_trace(go.Bar(
        name=factor,
        x=pc_x_labels,
        y=(pca_vp_df[factor] * 100).round(1),
        marker_color=color,
        hovertemplate='%{y:.1f}%<extra></extra>',
    ))
fig_pca.update_layout(
    barmode='stack',
    title='Variance partitioning by PCA component — cheng22 P28 gray-matter astrocytes (no mt genes)',
    xaxis_title='PCA component',
    yaxis_title='Variance explained (%)',
    yaxis=dict(range=[0, 100], tickformat='.1f'),
    legend_title='Factor',
    width=750,
    height=500,
)

vx_var = vx_df[vx_cols].var(axis=0)
vx_var_frac = (vx_var / vx_var.sum() * 100).round(1)
vx_x_labels = [f'{col}<br>({vx_var_frac[col]}%)' for col in vx_cols]

fig_vx = go.Figure()
for factor, color in COLORS.items():
    fig_vx.add_trace(go.Bar(
        name=factor,
        x=vx_x_labels,
        y=(result_df[factor] * 100).round(1),
        marker_color=color,
    ))
fig_vx.update_layout(
    barmode='stack',
    title='Variance partitioning by varimax component — cheng22 P28 gray-matter astrocytes (no mt genes)',
    xaxis_title='Varimax component',
    yaxis_title='Variance explained (%)',
    yaxis=dict(range=[0, 100]),
    legend_title='Factor',
    width=750,
    height=500,
)

N_TOP = 5

pc_load_df = pd.DataFrame(L, index=gene_names, columns=pc_cols)
pc_top_genes = {col: pc_load_df[col].abs().nlargest(N_TOP).index.tolist() for col in pc_cols}
pc_header = '<tr><th>Rank</th>' + ''.join(f'<th>{col}</th>' for col in pc_cols) + '</tr>'
pc_rows_html = []
for rank in range(N_TOP):
    cells = ''.join(
        f'<td>{pc_top_genes[col][rank]}<br>'
        f'<small>({pc_load_df.loc[pc_top_genes[col][rank], col]:.3f})</small></td>'
        for col in pc_cols
    )
    pc_rows_html.append(f'<tr><td>{rank+1}</td>{cells}</tr>')

pc_table_html = f"""
<h3 style="font-family:sans-serif;margin-top:30px">
  Top {N_TOP} genes by |loading| per PCA component
</h3>
<table border="1" cellpadding="4" cellspacing="0"
       style="font-family:monospace;font-size:12px;border-collapse:collapse">
  <thead style="background:#eee">{pc_header}</thead>
  <tbody>{''.join(pc_rows_html)}</tbody>
</table>
"""

top_genes = {col: vx_load_df[col].abs().nlargest(N_TOP).index.tolist() for col in vx_cols}
vx_header = '<tr><th>Rank</th>' + ''.join(f'<th>{col}</th>' for col in vx_cols) + '</tr>'
vx_rows_html = []
for rank in range(N_TOP):
    cells = ''.join(
        f'<td>{top_genes[col][rank]}<br>'
        f'<small>({vx_load_df.loc[top_genes[col][rank], col]:.3f})</small></td>'
        for col in vx_cols
    )
    vx_rows_html.append(f'<tr><td>{rank+1}</td>{cells}</tr>')

vx_table_html = f"""
<h3 style="font-family:sans-serif;margin-top:30px">
  Top {N_TOP} genes by |loading| per varimax component
</h3>
<table border="1" cellpadding="4" cellspacing="0"
       style="font-family:monospace;font-size:12px;border-collapse:collapse">
  <thead style="background:#eee">{vx_header}</thead>
  <tbody>{''.join(vx_rows_html)}</tbody>
</table>
"""

pca_div = fig_pca.to_html(full_html=False, include_plotlyjs='cdn')
vx_div  = fig_vx.to_html(full_html=False, include_plotlyjs=False)
with open(OUT_VAR_PART_HTML, 'w') as f:
    f.write(f'<html><body>{pca_div}{pc_table_html}{vx_div}{vx_table_html}</body></html>')
print(f'Saved {OUT_VAR_PART_HTML}')
print('Done.')
