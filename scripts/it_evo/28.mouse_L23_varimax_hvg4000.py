"""Mouse Cheng22 L2/3 IT — PCA + varimax + variance partitioning at N_HVG = 4000.

Standalone copy of `it/19.cheng22_L23_varimax.py` refit on the top-4000 HVGs instead of the
top-2000 (plan/it_evo/10). Nothing in it/19 is edited: the N=2000 record stays valid by
construction, and the two records coexist because every output here carries an `_hvg4000`
tag.

Deviation from the repository layout, noted deliberately (plan §"New scripts"): this is a
mouse script but lives in `it_evo/` — and writes to `local_data/res/it_evo/` — so the five
scripts of the N=4000 effort (27-31) read as one chain rather than being split across two
sub-projects.

Normalization is unchanged from it/19: cheng22 `.X` is log1p(CP10k), so the integer raw
counts in `.raw` are re-normalized with log2(CP10k + 1). HVG ranking is raw variance of that
matrix (`np.argsort(gene_var)[::-1][:N_HVG]`), NOT scanpy's dispersion-binned selection.

Note that VX NUMBERING IS NOT COMPARABLE TO it/19's. Varimax components are ordered by score
variance, and that order changes with the gene set, so the Gate-A VX selection must be
re-derived from the new `28...vx_variance_partition_hvg4000.tsv` — the it/19-era
`['VX1','VX2','VX5','VX7','VX8','VX9']` is meaningless in this basis.

Reads:
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
Outputs:
  local_data/res/it_evo/28.mouse_L23_varimax_coords_hvg4000.tsv
  local_data/res/it_evo/28.mouse_L23_varimax_loadings_hvg4000.tsv
  local_data/res/it_evo/28.mouse_L23_pc_variance_partition_hvg4000.tsv
  local_data/res/it_evo/28.mouse_L23_vx_variance_partition_hvg4000.tsv
  local_data/fig/it_evo/28.mouse_L23_pc_variance_partition_hvg4000.html
  local_data/fig/it_evo/28.mouse_L23_vx_variance_partition_hvg4000.html
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_MOUSE        = os.path.join(PROJECT_ROOT, 'links', 'it',
                               'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
OUT_VX_COORDS   = os.path.join(OUT_RES_DIR, '28.mouse_L23_varimax_coords_hvg4000.tsv')
OUT_VX_LOADINGS = os.path.join(OUT_RES_DIR, '28.mouse_L23_varimax_loadings_hvg4000.tsv')
OUT_PC_VAR_TSV  = os.path.join(OUT_RES_DIR, '28.mouse_L23_pc_variance_partition_hvg4000.tsv')
OUT_VX_VAR_TSV  = os.path.join(OUT_RES_DIR, '28.mouse_L23_vx_variance_partition_hvg4000.tsv')
OUT_PC_VAR_HTML = os.path.join(OUT_FIG_DIR, '28.mouse_L23_pc_variance_partition_hvg4000.html')
OUT_VX_VAR_HTML = os.path.join(OUT_FIG_DIR, '28.mouse_L23_vx_variance_partition_hvg4000.html')

# --- parameters ---
MOUSE_SUBCLASS = 'L2/3'
CLUSTER_COL    = 'Type'
SAMPLE_COL     = 'Sample'
LIBSIZE_COL    = 'n_counts'
N_HVG          = 4000            # it/19 uses 2000; this is the only analytical change
N_PCS          = 10
MOUSE_CP       = 1e4             # counts-per-10k target, matching it/19...py:204

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


def variance_partition(scores, comp_cols, types, samples, libsize, label):
    """Compute partial R² for Type, sample, library_size per component."""
    X_type    = pd.get_dummies(types).values.astype(float)
    X_sample  = pd.get_dummies(samples).values.astype(float)
    X_libsize = zscore(libsize).reshape(-1, 1)
    X_full    = np.hstack([X_type, X_sample, X_libsize])

    FACTORS = {
        'cell_type':    X_type,
        'sample':       X_sample,
        'library_size': X_libsize,
    }

    print(f'\nVariance partitioning ({label}):')
    rows = []
    for i, col in enumerate(comp_cols):
        y = scores[:, i]
        r2_full = r2(X_full, y)
        partial = {}
        for name, X_factor in FACTORS.items():
            X_others = np.hstack([v for k, v in FACTORS.items() if k != name])
            r2_reduced = r2(X_others, y)
            partial[name] = max(r2_full - r2_reduced, 0.0)
        residual = max(1.0 - r2_full, 0.0)
        rows.append({'component': col, **partial, 'residual': residual})
        print(f"  {col}  R²_full={r2_full:.3f}  "
              f"type={partial['cell_type']:.3f}  "
              f"sample={partial['sample']:.3f}  "
              f"libsize={partial['library_size']:.3f}  "
              f"resid={residual:.3f}")

    return pd.DataFrame(rows).set_index('component')


def save_varpart_html(result_df, loadings_df, scores, comp_cols, title, out_path):
    """Stacked bar chart of variance partition + top-5 genes table."""
    comp_var = np.var(scores, axis=0)
    comp_var_frac = (comp_var / comp_var.sum() * 100).round(1)
    x_labels = [f'{col}<br>({comp_var_frac[i]:.1f}%)' for i, col in enumerate(comp_cols)]

    COLORS = {
        'cell_type':    '#4C72B0',
        'sample':       '#DD8452',
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
        title=title,
        xaxis_title='Component',
        yaxis_title='Variance explained (%)',
        yaxis=dict(range=[0, 100]),
        legend_title='Factor',
        width=750,
        height=500,
    )

    N_TOP = 5
    top_genes = {col: loadings_df[col].abs().nlargest(N_TOP).index.tolist() for col in comp_cols}
    header = '<tr><th>Rank</th>' + ''.join(f'<th>{col}</th>' for col in comp_cols) + '</tr>'
    rows_html = []
    for rank in range(N_TOP):
        cells = ''.join(
            f'<td>{top_genes[col][rank]}<br>'
            f'<small>({loadings_df.loc[top_genes[col][rank], col]:.3f})</small></td>'
            for col in comp_cols
        )
        rows_html.append(f'<tr><td>{rank+1}</td>{cells}</tr>')

    table_html = (
        f'<h3 style="font-family:sans-serif;margin-top:30px">'
        f'Top {N_TOP} genes by |loading| per component</h3>'
        f'<table border="1" cellpadding="4" cellspacing="0" '
        f'style="font-family:monospace;font-size:12px;border-collapse:collapse">'
        f'<thead style="background:#eee">{header}</thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table>'
    )

    plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
    with open(out_path, 'w') as f:
        f.write(f'<html><body>{plot_div}{table_html}</body></html>')
    print(f'Saved {out_path}')


print(f'{"=" * 70}\nL23 — mouse cheng22 {MOUSE_SUBCLASS}  (N_HVG = {N_HVG})\n{"=" * 70}')

# --- 1. Load and filter to L2/3 ---
print(f'Loading {IN_MOUSE}...')
m_adata = ad.read_h5ad(IN_MOUSE)
m_adata = m_adata[m_adata.obs['Subclass'] == MOUSE_SUBCLASS]
print(f'  {m_adata.n_obs} cells, {m_adata.n_vars} genes')

types   = m_adata.obs[CLUSTER_COL].values
samples = m_adata.obs[SAMPLE_COL].values
libsize = m_adata.obs[LIBSIZE_COL].values

# --- 2. Normalize: raw counts → log2(CP10k + 1) ---
print('Normalizing...')
X_raw = m_adata.raw[:, m_adata.var_names].X.toarray().astype(np.float32)
depths = X_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_norm = np.log2(X_raw / depths * MOUSE_CP + 1)

# --- 3. HVG selection ---
print(f'Selecting top {N_HVG} HVGs...')
gene_var = X_norm.var(axis=0)
hvg_idx  = np.argsort(gene_var)[::-1][:N_HVG]
gene_names = m_adata.var_names.values[hvg_idx]
X_hvg = X_norm[:, hvg_idx]

# --- 4. Scale + PCA ---
print('Scaling and PCA...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hvg)

pca = PCA(n_components=N_PCS, random_state=0)
scores = pca.fit_transform(X_scaled)   # (n_cells, N_PCS)
L      = pca.components_.T             # (N_HVG, N_PCS) gene loadings
print(f'  Explained variance ratio: {pca.explained_variance_ratio_.round(3)}')

pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
pc_load_df = pd.DataFrame(L, index=gene_names, columns=pc_cols)

# --- 5. Variance partitioning on PCs (pre-varimax) ---
pc_var_df = variance_partition(scores, pc_cols, types, samples, libsize, 'pre-varimax PCs')
pc_var_df.to_csv(OUT_PC_VAR_TSV, sep='\t')
print(f'Saved {OUT_PC_VAR_TSV}')
save_varpart_html(
    pc_var_df, pc_load_df, scores, pc_cols,
    f'Variance partitioning by PC — cheng22 mouse L2/3 IT (pre-varimax, N_HVG={N_HVG})',
    OUT_PC_VAR_HTML,
)

# --- 6. Varimax rotation ---
print('Running varimax...')
R            = varimax(L)
vx_scores    = scores @ R
vx_loadings  = L @ R

# --- 7. Reorder by descending variance ---
vx_var_order = np.argsort(vx_scores.var(axis=0))[::-1]
vx_scores    = vx_scores[:, vx_var_order]
vx_loadings  = vx_loadings[:, vx_var_order]
vx_cols      = [f'VX{i+1}' for i in range(N_PCS)]

vx_load_df = pd.DataFrame(vx_loadings, index=gene_names, columns=vx_cols)

# --- 8. Variance partitioning on VX components (post-varimax) ---
vx_var_df = variance_partition(vx_scores, vx_cols, types, samples, libsize, 'post-varimax VX')
vx_var_df.to_csv(OUT_VX_VAR_TSV, sep='\t')
print(f'Saved {OUT_VX_VAR_TSV}')
save_varpart_html(
    vx_var_df, vx_load_df, vx_scores, vx_cols,
    f'Variance partitioning by VX component — cheng22 mouse L2/3 IT '
    f'(post-varimax, N_HVG={N_HVG})',
    OUT_VX_VAR_HTML,
)

# --- 9. Save varimax coords ---
vx_df = pd.DataFrame(vx_scores, index=m_adata.obs_names, columns=vx_cols)
vx_df[CLUSTER_COL] = types
vx_df[SAMPLE_COL]  = samples
vx_df.to_csv(OUT_VX_COORDS, sep='\t')
print(f'Saved {OUT_VX_COORDS}')

# --- 10. Save varimax loadings ---
vx_load_df.to_csv(OUT_VX_LOADINGS, sep='\t')
print(f'Saved {OUT_VX_LOADINGS}')

print('\nDone.')
