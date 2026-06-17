"""Variance partitioning for cheng22 NR (P28+P38) PCA and VX components.

Reads:
  local_data/res/astro/37.cheng22_nr.h5ad
Outputs:
  local_data/res/astro/38.pca_variance_partition.tsv
  local_data/res/astro/38.vx_variance_partition.tsv
  local_data/fig/astro/38.variance_partition.html
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IN_H5AD             = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro', '37.cheng22_nr.h5ad')
OUT_RES_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_FIG_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
OUT_PCA_VAR_PART    = os.path.join(OUT_RES_DIR, '38.pca_variance_partition.tsv')
OUT_VX_VAR_PART     = os.path.join(OUT_RES_DIR, '38.vx_variance_partition.tsv')
OUT_HTML            = os.path.join(OUT_FIG_DIR, '38.variance_partition.html')

N_TOP = 5

COLORS = {
    'archetype':    '#4C72B0',
    'donor':        '#DD8452',
    'library_size': '#C44E52',
    'residual':     '#CCCCCC',
}

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def r2(X, y):
    reg = LinearRegression().fit(X, y)
    ss_res = np.sum((y - reg.predict(X)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def _variance_partition(scores, factors, col_names):
    X_full = np.hstack(list(factors.values()))
    rows = []
    for i, col in enumerate(col_names):
        y = scores[:, i]
        r2_full = r2(X_full, y)
        partial = {}
        for name, X_factor in factors.items():
            X_others = np.hstack([v for k, v in factors.items() if k != name])
            partial[name] = max(r2_full - r2(X_others, y), 0.0)
        residual = max(1.0 - r2_full, 0.0)
        rows.append({'component': col, **partial, 'residual': residual})
        print(f"  {col}  R²={r2_full:.3f}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in partial.items())
              + f"  resid={residual:.3f}")
    return pd.DataFrame(rows).set_index('component')


def top_genes_table(loadings, gene_names, col_names, title):
    load_df = pd.DataFrame(loadings, index=gene_names, columns=col_names)
    top = {col: load_df[col].abs().nlargest(N_TOP).index.tolist() for col in col_names}
    header = '<tr><th>Rank</th>' + ''.join(f'<th>{col}</th>' for col in col_names) + '</tr>'
    body_rows = []
    for rank in range(N_TOP):
        cells = ''.join(
            f'<td>{top[col][rank]}<br><small>({load_df.loc[top[col][rank], col]:.3f})</small></td>'
            for col in col_names
        )
        body_rows.append(f'<tr><td>{rank+1}</td>{cells}</tr>')
    return (
        f'<h3 style="font-family:sans-serif;margin-top:30px">Top {N_TOP} genes by |loading| — {title}</h3>'
        f'<table border="1" cellpadding="4" cellspacing="0"'
        f' style="font-family:monospace;font-size:12px;border-collapse:collapse">'
        f'<thead style="background:#eee">{header}</thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )


def stacked_bar_fig(vp_df, x_labels, title, xaxis_title):
    fig = go.Figure()
    for factor, color in COLORS.items():
        fig.add_trace(go.Bar(
            name=factor,
            x=x_labels,
            y=(vp_df[factor] * 100).round(1),
            marker_color=color,
            hovertemplate='%{y:.1f}%<extra></extra>',
        ))
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title='Variance explained (%)',
        yaxis=dict(range=[0, 100]),
        legend_title='Factor',
        width=750,
        height=500,
    )
    return fig


# --- load ---
print(f'Loading {IN_H5AD}')
adata = ad.read_h5ad(IN_H5AD)
print(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

pca_scores = np.array(adata.obsm['X_pca'], dtype=np.float64)
vx_scores  = np.array(adata.obsm['X_vx'],  dtype=np.float64)
pca_load   = np.array(adata.varm['PCA_loadings'], dtype=np.float64)  # (n_hvg, N_PCS)
vx_load    = np.array(adata.varm['VX_loadings'],  dtype=np.float64)  # (n_hvg, N_PCS)
gene_names = adata.var_names.values

arch_labels = adata.obs['archetype'].values
donors      = adata.obs['Sample'].values
depths      = adata.obs['depth'].values

N_PCS   = pca_scores.shape[1]
pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
vx_cols = [f'VX{i+1}' for i in range(N_PCS)]

# --- covariate matrices ---
X_arch    = pd.get_dummies(arch_labels).values.astype(float)
X_donor   = pd.get_dummies(donors).values.astype(float)
X_libsize = zscore(depths).reshape(-1, 1)

FACTORS = {
    'archetype':    X_arch,
    'donor':        X_donor,
    'library_size': X_libsize,
}

# --- PCA variance partition ---
print('Computing PCA variance partitioning...')
pca_vp = _variance_partition(pca_scores, FACTORS, pc_cols)
pca_vp.to_csv(OUT_PCA_VAR_PART, sep='\t')
print(f'Saved {OUT_PCA_VAR_PART}')

# --- VX variance partition ---
print('Computing VX variance partitioning...')
vx_vp = _variance_partition(vx_scores, FACTORS, vx_cols)
vx_vp.to_csv(OUT_VX_VAR_PART, sep='\t')
print(f'Saved {OUT_VX_VAR_PART}')

# --- x-axis labels with variance fractions ---
pca_var = pca_scores.var(axis=0)
pca_var_frac = (pca_var / pca_var.sum() * 100).round(1)
pc_x_labels = [f'{col}<br>({pca_var_frac[i]}%)' for i, col in enumerate(pc_cols)]

vx_var = vx_scores.var(axis=0)
vx_var_frac = (vx_var / vx_var.sum() * 100).round(1)
vx_x_labels = [f'{col}<br>({vx_var_frac[i]}%)' for i, col in enumerate(vx_cols)]

# --- figures ---
fig_pca = stacked_bar_fig(
    pca_vp, pc_x_labels,
    title='Variance partitioning by PCA component — cheng22 NR (P28+P38) astrocytes',
    xaxis_title='PCA component',
)
fig_vx = stacked_bar_fig(
    vx_vp, vx_x_labels,
    title='Variance partitioning by varimax component — cheng22 NR (P28+P38) astrocytes',
    xaxis_title='Varimax component',
)

pc_table = top_genes_table(pca_load, gene_names, pc_cols, 'PCA components')
vx_table = top_genes_table(vx_load,  gene_names, vx_cols, 'varimax components')

pca_div = fig_pca.to_html(full_html=False, include_plotlyjs='cdn')
vx_div  = fig_vx.to_html(full_html=False, include_plotlyjs=False)
with open(OUT_HTML, 'w') as f:
    f.write(f'<html><body>{pca_div}{pc_table}{vx_div}{vx_table}</body></html>')
print(f'Saved {OUT_HTML}')
print('Done.')
