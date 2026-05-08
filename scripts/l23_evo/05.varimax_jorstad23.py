"""Varimax rotation of PCA components + variance partitioning — Jorstad23 L2/3 IT.

Applies Kaiser varimax rotation to the gene loading matrix (genes × PCs), then
derives the same rotation R for cell scores.  This is the standard approach:
varimax maximizes sparsity of gene loadings, pushing each component to load on
a distinct subset of genes.

Reads:
  local_data/res/l23_evo/01.pca_coords.tsv
  local_data/res/l23_evo/01.pca_loadings.tsv
Outputs:
  local_data/res/l23_evo/05.varimax_coords.tsv
  local_data/res/l23_evo/05.varimax_loadings.tsv
  local_data/res/l23_evo/05.variance_partition.tsv
  local_data/fig/l23_evo/05.variance_partition.html
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import plotly.graph_objects as go

# --- file paths ---
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_PCA        = os.path.join(OUT_RES_DIR, '01.pca_coords.tsv')
IN_LOADINGS   = os.path.join(OUT_RES_DIR, '01.pca_loadings.tsv')
OUT_COORDS    = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
OUT_LOADINGS  = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
OUT_TSV       = os.path.join(OUT_RES_DIR, '05.variance_partition.tsv')
OUT_HTML      = os.path.join(OUT_FIG_DIR, '05.variance_partition.html')

CLUSTER_COL = 'WithinArea_cluster'
DONOR_COL   = 'donor_id'
SOURCE_COL  = 'Source'
LIBSIZE_COL = 'nCount_RNA'
META_COLS   = [CLUSTER_COL, DONOR_COL, SOURCE_COL, 'development_stage', LIBSIZE_COL]
N_PCS       = 10

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


# --- varimax rotation ---
def varimax(L, gamma=1.0, max_iter=1000, tol=1e-6):
    """Kaiser varimax rotation of loading matrix L (n_vars × n_factors).

    Finds orthogonal rotation matrix R that maximises variance of squared
    loadings within each factor (pushes loadings toward 0 or large).
    gamma=1 → orthonormal (Kaiser) varimax.
    Returns R (n_factors × n_factors).
    """
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


# --- R² helper ---
def r2(X, y):
    reg = LinearRegression().fit(X, y)
    ss_res = np.sum((y - reg.predict(X)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


# --- load data ---
pca_df      = pd.read_csv(IN_PCA,      sep='\t', index_col=0)
loadings_df = pd.read_csv(IN_LOADINGS, sep='\t', index_col=0)  # genes × PCs

pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
scores  = pca_df[pc_cols].values          # (n_cells, N_PCS)
L       = loadings_df[pc_cols].values     # (n_genes, N_PCS)

# --- apply varimax to gene loadings ---
print('Running varimax rotation on gene loading matrix...')
R = varimax(L)                            # rotation found from gene loadings
vx_scores   = scores   @ R               # same R applied to cell scores
vx_loadings = L        @ R               # rotated gene loadings
vx_cols     = [f'VX{i+1}' for i in range(N_PCS)]

# --- save rotated coords ---
# --- reorder components by descending variance ---
vx_var_order = np.argsort(vx_scores.var(axis=0))[::-1]
vx_scores    = vx_scores[:, vx_var_order]
vx_loadings  = vx_loadings[:, vx_var_order]

vx_df = pd.DataFrame(vx_scores, index=pca_df.index, columns=vx_cols)
for col in META_COLS:
    vx_df[col] = pca_df[col].values
vx_df.to_csv(OUT_COORDS, sep='\t')
print(f'Saved {OUT_COORDS}')

# --- save rotated loadings ---
vx_load_df = pd.DataFrame(vx_loadings, index=loadings_df.index, columns=vx_cols)
vx_load_df.to_csv(OUT_LOADINGS, sep='\t')
print(f'Saved {OUT_LOADINGS}')

# --- variance partitioning ---
X_type    = pd.get_dummies(pca_df[CLUSTER_COL]).values.astype(float)
X_donor   = pd.get_dummies(pca_df[DONOR_COL]).values.astype(float)
X_source  = pd.get_dummies(pca_df[SOURCE_COL]).values.astype(float)
X_libsize = zscore(pca_df[LIBSIZE_COL].values).reshape(-1, 1)
X_full    = np.hstack([X_type, X_donor, X_source, X_libsize])

FACTORS = {
    'cell_type':    X_type,
    'donor':        X_donor,
    'source':       X_source,
    'library_size': X_libsize,
}

print('Computing variance partitioning...')
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
          f"type={partial['cell_type']:.3f}  "
          f"donor={partial['donor']:.3f}  "
          f"source={partial['source']:.3f}  "
          f"libsize={partial['library_size']:.3f}  "
          f"resid={residual:.3f}")

result_df = pd.DataFrame(rows).set_index('VX')
result_df.to_csv(OUT_TSV, sep='\t')
print(f'Saved {OUT_TSV}')

# --- fraction of total variance per VX component (post-hoc, since varimax breaks ordering) ---
vx_var = vx_df[vx_cols].var(axis=0)
vx_var_frac = (vx_var / vx_var.sum() * 100).round(1)
x_labels = [f'{col}<br>({vx_var_frac[col]}%)' for col in vx_cols]

# --- stacked bar chart ---
COLORS = {
    'cell_type':    '#4C72B0',
    'donor':        '#DD8452',
    'source':       '#55A868',
    'library_size': '#C44E52',
    'residual':     '#CCCCCC',
}

fig = go.Figure()
for factor, color in COLORS.items():
    fig.add_trace(go.Bar(
        name=factor,
        x=x_labels,
        y=(result_df[factor] * 100).round(1),
        marker_color=color,
    ))

fig.update_layout(
    barmode='stack',
    title='Variance partitioning by varimax component — Jorstad23 L2/3 IT',
    xaxis_title='Varimax component',
    yaxis_title='Variance explained (%)',
    yaxis=dict(range=[0, 100]),
    legend_title='Factor',
    width=750,
    height=500,
)
# --- top 5 genes by |loading| per VX component ---
N_TOP = 5
top_genes = {col: vx_load_df[col].abs().nlargest(N_TOP).index.tolist() for col in vx_cols}
header = '<tr><th>Rank</th>' + ''.join(f'<th>{col}</th>' for col in vx_cols) + '</tr>'
rows_html = []
for rank in range(N_TOP):
    cells = ''.join(
        f'<td>{top_genes[col][rank]}<br>'
        f'<small>({vx_load_df.loc[top_genes[col][rank], col]:.3f})</small></td>'
        for col in vx_cols
    )
    rows_html.append(f'<tr><td>{rank+1}</td>{cells}</tr>')

table_html = f"""
<h3 style="font-family:sans-serif;margin-top:30px">
  Top {N_TOP} genes by |loading| per varimax component
</h3>
<table border="1" cellpadding="4" cellspacing="0"
       style="font-family:monospace;font-size:12px;border-collapse:collapse">
  <thead style="background:#eee">{header}</thead>
  <tbody>{''.join(rows_html)}</tbody>
</table>
"""

plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
with open(OUT_HTML, 'w') as f:
    f.write(f'<html><body>{plot_div}{table_html}</body></html>')
print(f'Saved {OUT_HTML}')
print('Done.')
