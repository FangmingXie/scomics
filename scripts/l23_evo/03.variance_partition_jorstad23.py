"""Variance partitioning of PC axes — Jorstad23 human L2/3 IT.

For each PC, computes the fraction of variance explained by cell type, donor,
and lab source using partial R² (each factor's unique contribution after
controlling for all others).

Reads: local_data/res/l23_evo/01.pca_coords.tsv
Outputs:
  local_data/res/l23_evo/03.variance_partition.tsv
  local_data/fig/l23_evo/03.variance_partition.html
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import plotly.graph_objects as go

# --- file paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_PCA       = os.path.join(OUT_RES_DIR, '01.pca_coords.tsv')
OUT_TSV      = os.path.join(OUT_RES_DIR, '03.variance_partition.tsv')
OUT_HTML     = os.path.join(OUT_FIG_DIR, '03.variance_partition.html')

CLUSTER_COL  = 'WithinArea_cluster'
DONOR_COL    = 'donor_id'
SOURCE_COL   = 'Source'
LIBSIZE_COL  = 'nCount_RNA'
N_PCS        = 10

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load PCA coords ---
pca_df = pd.read_csv(IN_PCA, sep='\t', index_col=0)
pc_cols = [f'PC{i+1}' for i in range(N_PCS)]

# --- encode each factor ---
X_type    = pd.get_dummies(pca_df[CLUSTER_COL]).values.astype(float)
X_donor   = pd.get_dummies(pca_df[DONOR_COL]).values.astype(float)
X_source  = pd.get_dummies(pca_df[SOURCE_COL]).values.astype(float)
X_libsize = zscore(pca_df[LIBSIZE_COL].values).reshape(-1, 1)  # continuous, z-scored
X_full    = np.hstack([X_type, X_donor, X_source, X_libsize])

FACTORS = {
    'cell_type':    X_type,
    'donor':        X_donor,
    'source':       X_source,
    'library_size': X_libsize,
}

def r2(X, y):
    """R² of OLS fit of y ~ X (with intercept via centering)."""
    reg = LinearRegression().fit(X, y)
    ss_res = np.sum((y - reg.predict(X)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

# --- compute partial R² for each PC ---
print('Computing variance partitioning...')
rows = []
for pc in pc_cols:
    y = pca_df[pc].values
    r2_full = r2(X_full, y)

    partial = {}
    for name, X_factor in FACTORS.items():
        # reduced model = full minus this factor
        X_others = np.hstack([v for k, v in FACTORS.items() if k != name])
        r2_reduced = r2(X_others, y)
        partial[name] = max(r2_full - r2_reduced, 0.0)   # clip negatives from rounding

    residual = max(1.0 - r2_full, 0.0)
    rows.append({'PC': pc, **partial, 'residual': residual})

    print(f"  {pc}  R²_full={r2_full:.3f}  "
          f"type={partial['cell_type']:.3f}  "
          f"donor={partial['donor']:.3f}  "
          f"source={partial['source']:.3f}  "
          f"libsize={partial['library_size']:.3f}  "
          f"resid={residual:.3f}")

result_df = pd.DataFrame(rows).set_index('PC')
result_df.to_csv(OUT_TSV, sep='\t')
print(f'Saved {OUT_TSV}')

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
        x=pc_cols,
        y=(result_df[factor] * 100).round(1),
        marker_color=color,
    ))

fig.update_layout(
    barmode='stack',
    title='Variance partitioning by PC — Jorstad23 L2/3 IT',
    xaxis_title='PC',
    yaxis_title='Variance explained (%)',
    yaxis=dict(range=[0, 100]),
    legend_title='Factor',
    width=750,
    height=500,
)
fig.write_html(OUT_HTML)
print(f'Saved {OUT_HTML}')
print('Done.')
