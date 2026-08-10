"""Human Jorstad23 IT — per-subclass PCA + varimax rotation + variance partitioning.

Fuses l23_evo/01 (HVG -> StandardScaler -> PCA) and l23_evo/05 (Kaiser varimax, reorder
by descending score variance, variance partition) into one loop over the four IT
subclasses. UMAP, the PC-covariate correlation table and the one-vs-rest cluster markers
from l23_evo/01 are dropped — nothing downstream reads them.

Differences from the mouse template (it/21):
  - `.X` is already log-normalized (ln(CPM+1)); it is used directly, no re-normalization.
  - Gene identifiers are ENSG in `var_names`; symbols live in `var['feature_name']` and
    are what the loadings TSV is indexed by, because the ortholog table is symbol-based (G5).
  - `variance_partition` takes its factor dict as an argument, because the human side needs
    four factors (cell_type / donor / source / library_size) rather than the mouse three.
    `development_stage` is excluded: it is 1:1 with `donor_id` and would zero out the donor
    partial R² (l23_evo/05 does the same).

The coords TSV carries META_COLS alongside VX1..VX10 — script 03 reads `donor_id` and
script 04 reads `WithinArea_cluster` from it, and neither reopens the h5ad.

L2/3 control: this recipe is deterministic and identical to l23_evo/01 + 05, so
02.human_L23_varimax_coords.tsv should match l23_evo/05.varimax_coords.tsv to numerical
precision.

Reads:
  links/it_evo/jorstad23_human_WithinArea_{L23IT,L4IT,L5IT,L6IT}.h5ad
Outputs (per TOKEN in L23 / L4 / L5IT / L6IT):
  local_data/res/it_evo/02.human_<TOKEN>_varimax_coords.tsv
  local_data/res/it_evo/02.human_<TOKEN>_varimax_loadings.tsv
  local_data/res/it_evo/02.human_<TOKEN>_pc_variance_partition.tsv
  local_data/res/it_evo/02.human_<TOKEN>_vx_variance_partition.tsv
  local_data/fig/it_evo/02.human_<TOKEN>_pc_variance_partition.html
  local_data/fig/it_evo/02.human_<TOKEN>_vx_variance_partition.html
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

# --- file paths ---
LINK_DIR    = os.path.join(PROJECT_ROOT, 'links', 'it_evo')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')

SUBCLASSES = [
    {'token': 'L23',  'human_subclass': 'L2/3 IT', 'h5ad': 'jorstad23_human_WithinArea_L23IT.h5ad'},
    {'token': 'L4',   'human_subclass': 'L4 IT',   'h5ad': 'jorstad23_human_WithinArea_L4IT.h5ad'},
    {'token': 'L5IT', 'human_subclass': 'L5 IT',   'h5ad': 'jorstad23_human_WithinArea_L5IT.h5ad'},
    {'token': 'L6IT', 'human_subclass': 'L6 IT',   'h5ad': 'jorstad23_human_WithinArea_L6IT.h5ad'},
]

# --- parameters ---
N_HVG            = 2000
N_PCS            = 10
PCA_RANDOM_STATE = 0
CLUSTER_COL      = 'WithinArea_cluster'
DONOR_COL        = 'donor_id'
SOURCE_COL       = 'Source'
LIBSIZE_COL      = 'nCount_RNA'
GENE_NAME_COL    = 'feature_name'
META_COLS        = [CLUSTER_COL, DONOR_COL, SOURCE_COL, 'development_stage', LIBSIZE_COL]

FACTOR_COLORS = {
    'cell_type':    '#4C72B0',
    'donor':        '#DD8452',
    'source':       '#55A868',
    'library_size': '#C44E52',
    'residual':     '#CCCCCC',
}

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


def build_factors(obs):
    """Design matrices for the four human factors. Fails fast on all-zero dummies (G6)."""
    factors = {
        'cell_type': pd.get_dummies(obs[CLUSTER_COL]).values.astype(float),
        'donor':     pd.get_dummies(obs[DONOR_COL]).values.astype(float),
        'source':    pd.get_dummies(obs[SOURCE_COL]).values.astype(float),
        'library_size': zscore(obs[LIBSIZE_COL].values.astype(float)).reshape(-1, 1),
    }
    for name in ('cell_type', 'donor', 'source'):
        col_sums = factors[name].sum(axis=0)
        if (col_sums == 0).any():
            raise ValueError(
                f'{name} design matrix has {(col_sums == 0).sum()} all-zero dummy column(s) — '
                f'stale categorical levels survived subsetting (G6)'
            )
    return factors


def variance_partition(scores, comp_cols, factors, label):
    """Partial R² per component for each factor in `factors` (dict name -> design matrix)."""
    X_full = np.hstack(list(factors.values()))

    print(f'\nVariance partitioning ({label}):')
    rows = []
    for i, col in enumerate(comp_cols):
        y = scores[:, i]
        r2_full = r2(X_full, y)
        partial = {}
        for name in factors:
            X_others = np.hstack([v for k, v in factors.items() if k != name])
            partial[name] = max(r2_full - r2(X_others, y), 0.0)
        residual = max(1.0 - r2_full, 0.0)
        rows.append({'component': col, **partial, 'residual': residual})
        print(f"  {col}  R²_full={r2_full:.3f}  "
              + '  '.join(f'{k}={v:.3f}' for k, v in partial.items())
              + f'  resid={residual:.3f}')

    return pd.DataFrame(rows).set_index('component')


def save_varpart_html(result_df, loadings_df, scores, comp_cols, title, out_path):
    """Stacked bar chart of variance partition + top-5 genes table."""
    comp_var = np.var(scores, axis=0)
    comp_var_frac = (comp_var / comp_var.sum() * 100).round(1)
    x_labels = [f'{col}<br>({comp_var_frac[i]:.1f}%)' for i, col in enumerate(comp_cols)]

    fig = go.Figure()
    for factor, color in FACTOR_COLORS.items():
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


for cfg in SUBCLASSES:
    token    = cfg['token']
    subclass = cfg['human_subclass']
    in_h5ad  = os.path.join(LINK_DIR, cfg['h5ad'])

    out_vx_coords   = os.path.join(OUT_RES_DIR, f'02.human_{token}_varimax_coords.tsv')
    out_vx_loadings = os.path.join(OUT_RES_DIR, f'02.human_{token}_varimax_loadings.tsv')
    out_pc_var_tsv  = os.path.join(OUT_RES_DIR, f'02.human_{token}_pc_variance_partition.tsv')
    out_vx_var_tsv  = os.path.join(OUT_RES_DIR, f'02.human_{token}_vx_variance_partition.tsv')
    out_pc_var_html = os.path.join(OUT_FIG_DIR, f'02.human_{token}_pc_variance_partition.html')
    out_vx_var_html = os.path.join(OUT_FIG_DIR, f'02.human_{token}_vx_variance_partition.html')

    print(f'\n{"=" * 70}\n{token} — human {subclass}\n{"=" * 70}')

    # --- 1. Load; .X is already log-normalized ---
    print(f'Loading {in_h5ad}...')
    adata = ad.read_h5ad(in_h5ad)
    print(f'  {adata.n_obs} cells, {adata.n_vars} genes')
    X_norm = adata.X.toarray().astype(np.float32)
    gene_names = adata.var[GENE_NAME_COL].values

    factors = build_factors(adata.obs)

    # --- 2. HVG selection ---
    print(f'Selecting top {N_HVG} HVGs...')
    gene_var  = X_norm.var(axis=0)
    hvg_idx   = np.argsort(gene_var)[::-1][:N_HVG]
    hvg_names = gene_names[hvg_idx]
    X_hvg     = X_norm[:, hvg_idx]
    del X_norm
    gc.collect()

    # --- 3. Scale + PCA ---
    print('Scaling and PCA...')
    X_scaled = StandardScaler().fit_transform(X_hvg)
    del X_hvg
    gc.collect()

    pca = PCA(n_components=N_PCS, random_state=PCA_RANDOM_STATE)
    scores = pca.fit_transform(X_scaled)   # (n_cells, N_PCS)
    L      = pca.components_.T             # (N_HVG, N_PCS)
    del X_scaled
    gc.collect()
    print(f'  Explained variance ratio: {pca.explained_variance_ratio_.round(3)}')

    pc_cols    = [f'PC{i+1}' for i in range(N_PCS)]
    pc_load_df = pd.DataFrame(L, index=hvg_names, columns=pc_cols)

    # --- 4. Variance partitioning on PCs (pre-varimax) ---
    pc_var_df = variance_partition(scores, pc_cols, factors, f'{token} pre-varimax PCs')
    pc_var_df.to_csv(out_pc_var_tsv, sep='\t')
    print(f'Saved {out_pc_var_tsv}')
    save_varpart_html(
        pc_var_df, pc_load_df, scores, pc_cols,
        f'Variance partitioning by PC — Jorstad23 human {subclass} (pre-varimax)',
        out_pc_var_html,
    )

    # --- 5. Varimax rotation ---
    print('Running varimax...')
    R           = varimax(L)
    vx_scores   = scores @ R
    vx_loadings = L @ R

    # --- 6. Reorder by descending score variance ---
    vx_var_order = np.argsort(vx_scores.var(axis=0))[::-1]
    vx_scores    = vx_scores[:, vx_var_order]
    vx_loadings  = vx_loadings[:, vx_var_order]
    vx_cols      = [f'VX{i+1}' for i in range(N_PCS)]

    vx_load_df = pd.DataFrame(vx_loadings, index=hvg_names, columns=vx_cols)

    # --- 7. Variance partitioning on VX components (post-varimax) ---
    vx_var_df = variance_partition(vx_scores, vx_cols, factors, f'{token} post-varimax VX')
    vx_var_df.to_csv(out_vx_var_tsv, sep='\t')
    print(f'Saved {out_vx_var_tsv}')
    save_varpart_html(
        vx_var_df, vx_load_df, vx_scores, vx_cols,
        f'Variance partitioning by VX component — Jorstad23 human {subclass} (post-varimax)',
        out_vx_var_html,
    )

    # --- 8. Save varimax coords (with META_COLS for scripts 03/04) ---
    vx_df = pd.DataFrame(vx_scores, index=adata.obs_names, columns=vx_cols)
    for col in META_COLS:
        vx_df[col] = adata.obs[col].values
    vx_df.to_csv(out_vx_coords, sep='\t')
    print(f'Saved {out_vx_coords}')

    # --- 9. Save varimax loadings (index = HGNC symbols) ---
    vx_load_df.to_csv(out_vx_loadings, sep='\t')
    print(f'Saved {out_vx_loadings}')

    del adata, scores, L, vx_scores, vx_loadings, vx_df, vx_load_df, pc_load_df, factors
    gc.collect()

print('\nDone.')
