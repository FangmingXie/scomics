"""Archetype scores for jainlab26 wt + cux2cre L2/3 cells (combined).

PCA fit on wt (reference), varimax rotation derived from wt loadings, both
datasets projected into the same VX space. PCHA fitted on combined VX1 vs VX3.
Conservative one-vs-each Wilcoxon on combined cells for marker genes; archetype
scores computed for every cell.

Reads:
  local_data/res/cux2cre/04_v2.l23_jainlab26_wt_labeled.h5ad
  local_data/res/cux2cre/04_v2.l23_jainlab26_cux2cre_labeled.h5ad
  local_data/res/cux2cre/00_v2.jainlab26_wt_labeled.h5ad
  local_data/res/cux2cre/00_v2.jainlab26_cux2cre_labeled.h5ad
Outputs:
  local_data/res/cux2cre/07.combined_pcha_xp.tsv
  local_data/res/cux2cre/07.combined_pcha_aa.tsv
  local_data/res/cux2cre/07.combined_archetype_markers.tsv
  local_data/res/cux2cre/07.combined_archetype_scores.tsv
  local_data/fig/cux2cre/07.combined_archetype_scatter.html
  local_data/fig/cux2cre/07.combined_archetype_scores.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import select_hvg
from viz import scatter_categorical_html
from scomics.utils import pcha, norm

# --- file paths ---
OUT_RES_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre')
OUT_FIG_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre')
INPUT_WT_L23        = os.path.join(OUT_RES_DIR, '04_v2.l23_jainlab26_wt_labeled.h5ad')
INPUT_CUX2CRE_L23   = os.path.join(OUT_RES_DIR, '04_v2.l23_jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_00         = os.path.join(OUT_RES_DIR, '00_v2.jainlab26_wt_labeled.h5ad')
INPUT_CUX2CRE_00    = os.path.join(OUT_RES_DIR, '00_v2.jainlab26_cux2cre_labeled.h5ad')
OUT_PCHA_XP         = os.path.join(OUT_RES_DIR, '07.combined_pcha_xp.tsv')
OUT_PCHA_AA         = os.path.join(OUT_RES_DIR, '07.combined_pcha_aa.tsv')
OUT_MARKERS         = os.path.join(OUT_RES_DIR, '07.combined_archetype_markers.tsv')
OUT_SCORES          = os.path.join(OUT_RES_DIR, '07.combined_archetype_scores.tsv')
OUT_ARCH_HTML       = os.path.join(OUT_FIG_DIR, '07.combined_archetype_scatter.html')
OUT_SCORE_HTML      = os.path.join(OUT_FIG_DIR, '07.combined_archetype_scores.html')

# --- parameters ---
UMAP_X_MAX      = -4
UMAP_Y_MAX      = 10
N_HVG           = 3000
N_PCS           = 30
N_VX            = 5
VX_PCHA_COLS    = [0, 2]     # PCHA subspace: VX1 and VX3
NOC             = 3
N_TOP_CELLS     = 300
FRAC_IN_THRESH  = 0.25
FDR_THRESH      = 0.001
LOG2FC_THRESH   = np.log2(1.5)
ARCHETYPE_NAMES = ['A', 'B', 'C']
SCORE_PCTILE_LO = 2
SCORE_PCTILE_HI = 98
CLUSTER_COL     = 'Type_transferred'

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


def _umap_mask(adata_l23, adata_00):
    umap = pd.DataFrame(adata_00.obsm['X_umap'], index=adata_00.obs_names,
                        columns=['UMAP1', 'UMAP2'])
    u = umap.reindex(adata_l23.obs_names)
    return (u['UMAP1'] < UMAP_X_MAX) & (u['UMAP2'] < UMAP_Y_MAX)


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
                A, B = u.sum(), v.sum()
                C = (u ** 2 - v ** 2).sum()
                D = 2 * (u * v).sum()
                theta = 0.25 * np.arctan2(
                    D - gamma * 2 * A * B / n,
                    C - gamma * (A ** 2 - B ** 2) / n,
                )
                c, s = np.cos(theta), np.sin(theta)
                Rij = np.eye(p)
                Rij[i, i] = Rij[j, j] = c
                Rij[i, j] = -s; Rij[j, i] = s
                R = R @ Rij
        if np.max(np.abs(R - R_old)) < tol:
            break
    return R


# --- Load and filter both datasets ---
print('Loading wt L2/3...')
wt = ad.read_h5ad(INPUT_WT_L23)
wt = wt[_umap_mask(wt, ad.read_h5ad(INPUT_WT_00))].copy()
print(f'  {wt.n_obs} cells after UMAP filter')

print('Loading cux2cre L2/3...')
cux2cre = ad.read_h5ad(INPUT_CUX2CRE_L23)
cux2cre = cux2cre[_umap_mask(cux2cre, ad.read_h5ad(INPUT_CUX2CRE_00))].copy()
print(f'  {cux2cre.n_obs} cells after UMAP filter')

# --- Shared genes ---
shared_genes = np.intersect1d(wt.var_names, cux2cre.var_names)
print(f'Shared genes: {len(shared_genes)}')
wt      = wt[:,      shared_genes]
cux2cre = cux2cre[:, shared_genes]

# --- HVG selection from wt (reference), normalize both ---
x_wt      = _to_array(wt.X)
x_cux2cre = _to_array(cux2cre.X)
depths_wt      = x_wt.sum(axis=1)
depths_cux2cre = x_cux2cre.sum(axis=1)

hvg_mask = select_hvg(x_wt, depths_wt, N_HVG)
print(f'HVGs selected: {hvg_mask.sum()}')

xn_wt      = np.nan_to_num(norm(x_wt[:,      hvg_mask], depths_wt),      nan=0.0)
xn_cux2cre = np.nan_to_num(norm(x_cux2cre[:, hvg_mask], depths_cux2cre), nan=0.0)

hvg_names = wt.var_names.values[hvg_mask]
print(f'xn_wt: {xn_wt.shape}  |  xn_cux2cre: {xn_cux2cre.shape}')

# --- PCA: fit on wt, project both ---
print(f'Running PCA ({N_PCS} components, fit on wt)...')
pca_model = PCA(n_components=N_PCS, random_state=0)
pca_model.fit(xn_wt)
xp_wt      = pca_model.transform(xn_wt)
xp_cux2cre = pca_model.transform(xn_cux2cre)

# --- Varimax: rotation from wt loadings, reorder by wt variance ---
print(f'Running varimax on first {N_VX} PCs...')
R_vx     = varimax(pca_model.components_.T[:, :N_VX])
vx_wt      = xp_wt[:,      :N_VX] @ R_vx
vx_cux2cre = xp_cux2cre[:, :N_VX] @ R_vx
vx_order   = np.argsort(vx_wt.var(axis=0))[::-1]
vx_wt      = vx_wt[:,      vx_order]
vx_cux2cre = vx_cux2cre[:, vx_order]

# --- Combined VX1-VX3 subspace ---
vx_sub_wt      = vx_wt[:,      VX_PCHA_COLS]
vx_sub_cux2cre = vx_cux2cre[:, VX_PCHA_COLS]
vx_sub = np.vstack([vx_sub_wt, vx_sub_cux2cre])

n_wt      = wt.n_obs
n_cux2cre = cux2cre.n_obs
n_cells   = n_wt + n_cux2cre

xn            = np.vstack([xn_wt, xn_cux2cre])
cell_barcodes = np.concatenate([wt.obs_names.values, cux2cre.obs_names.values])
types         = np.concatenate([wt.obs[CLUSTER_COL].values, cux2cre.obs[CLUSTER_COL].values])
source        = np.array(['wt'] * n_wt + ['cux2cre'] * n_cux2cre)
n_hvg         = xn.shape[1]
print(f'Combined: {n_cells} cells ({n_wt} wt + {n_cux2cre} cux2cre)')

# --- Fit PCHA on combined VX1-VX3 ---
print(f'Fitting PCHA: NOC={NOC}, subspace=VX1 vs VX3 (combined)...')
aa, varexpl = pcha(vx_sub.T, noc=NOC)  # aa: (2, NOC)
xp_pcha = vx_sub
aa_vx   = aa.T                          # (NOC, 2)

pd.DataFrame(xp_pcha, index=cell_barcodes,
             columns=['VX1', 'VX3']).to_csv(OUT_PCHA_XP, sep='\t')
pd.DataFrame(aa_vx, columns=['VX1', 'VX3'],
             index=[f'archetype_{k+1}' for k in range(NOC)]).to_csv(OUT_PCHA_AA, sep='\t')
print(f'Saved {OUT_PCHA_XP} and {OUT_PCHA_AA}')

# --- Cell-to-archetype distances ---
dists     = np.stack([np.linalg.norm(vx_sub - aa_vx[k], axis=1) for k in range(NOC)], axis=1)
top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(NOC)]
print(f'Top {N_TOP_CELLS} cells selected per archetype.')

# --- Conservative one-vs-each Wilcoxon per archetype ---
print('Running conservative one-vs-each Wilcoxon per archetype...')
all_markers = []

for k in range(NOC):
    archetype_label = f'archetype_{k + 1}'
    print(f'  {archetype_label}...')
    in_idx  = top_cells[k]
    X_in    = xn[in_idx]
    frac_in = (X_in > 0).mean(axis=0)

    log2fc_mat    = []
    pval_mat      = []
    frac_out_list = []

    for j in range(NOC):
        if j == k:
            continue
        X_out    = xn[top_cells[j]]
        log2fc_j = (X_in.mean(axis=0) - X_out.mean(axis=0)) / np.log(2)
        pvals_j  = np.array([scipy.stats.ranksums(X_in[:, g], X_out[:, g])[1]
                              for g in range(n_hvg)])
        log2fc_mat.append(log2fc_j)
        pval_mat.append(pvals_j)
        frac_out_list.append((X_out > 0).mean(axis=0))

    log2fc   = np.stack(log2fc_mat).min(axis=0)
    pvals    = np.stack(pval_mat).max(axis=0)
    frac_out = np.stack(frac_out_list).mean(axis=0)

    _, fdr, _, _ = multipletests(pvals, method='fdr_bh')

    df = pd.DataFrame({
        'gene':      hvg_names,
        'archetype': archetype_label,
        'log2FC':    log2fc,
        'pval':      pvals,
        'fdr':       fdr,
        'frac_in':   frac_in,
        'frac_out':  frac_out,
    })
    df = df[(df['frac_in'] >= FRAC_IN_THRESH) & (df['fdr'] < FDR_THRESH) & (df['log2FC'] > LOG2FC_THRESH)]
    df = df.sort_values('log2FC', ascending=False)
    all_markers.append(df)
    print(f'    {len(df)} markers after filtering')

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(OUT_MARKERS, sep='\t', index=False)
print(f'Saved {OUT_MARKERS}  ({len(markers_df)} total markers)')

# --- Archetype score computation ---
print('Computing archetype scores...')
scores = np.zeros((n_cells, NOC), dtype=np.float32)

for k, name in enumerate(ARCHETYPE_NAMES):
    top_genes = markers_df[markers_df['archetype'] == f'archetype_{k+1}']['gene'].values
    cols = [np.where(hvg_names == g)[0][0] for g in top_genes if g in hvg_names]
    if not cols:
        print(f'  WARNING: no genes found for archetype {name}')
        continue
    mat          = xn[:, cols]
    lo           = np.percentile(mat, SCORE_PCTILE_LO, axis=0)
    hi           = np.percentile(mat, SCORE_PCTILE_HI, axis=0)
    rng          = np.where(hi > lo, hi - lo, 1.0)
    mat_norm     = np.clip((mat - lo) / rng, 0, 1)
    scores[:, k] = mat_norm.mean(axis=1)
    print(f'  Score {name}: {len(cols)} genes used')

scores_df = pd.DataFrame(scores, index=cell_barcodes,
                          columns=[f'score_{n}' for n in ARCHETYPE_NAMES])
scores_df.to_csv(OUT_SCORES, sep='\t')
print(f'Saved {OUT_SCORES}')

# --- PCHA scatter ---
print('\nGenerating PCHA scatter...')
panels = [(0, 1, 'VX1', 'VX3')]
scatter_categorical_html(
    xp_grid=[xp_pcha],
    cell_metadata={CLUSTER_COL: types, 'source': source},
    title=f'jainlab26 wt+cux2cre L2/3 — VX1 vs VX3 (NOC={NOC})',
    out_path=OUT_ARCH_HTML,
    panels=panels,
    arch_vis=aa,
    equal_aspect=True,
)
print(f'Saved {OUT_ARCH_HTML}')

# --- Archetype score scatter: 3×3 grid (Score A/B/C × Combined/WT/Cux2Cre) ---
print('Generating archetype score scatter...')
score_names = [f'Score {n}' for n in ARCHETYPE_NAMES]
score_vals  = [scores[:, k] for k in range(NOC)]
wt_mask     = source == 'wt'
cx_mask     = source == 'cux2cre'
groups = [
    ('Combined',     np.ones(n_cells, dtype=bool)),
    ('WT only',      wt_mask),
    ('Cux2Cre only', cx_mask),
]
n_groups = len(groups)
n_scores = len(ARCHETYPE_NAMES)

fig_score = make_subplots(
    rows=n_scores, cols=n_groups,
    subplot_titles=[f'{sn} — {gn}' for sn in score_names for gn, _ in groups],
    vertical_spacing=0.08,
    horizontal_spacing=0.05,
)

# colorbar y-centers for 3 equal rows with 0.08 vertical spacing
cb_y = [0.86, 0.50, 0.14]

for ri, (sname, sval) in enumerate(zip(score_names, score_vals)):
    cmin     = np.nanpercentile(sval, 5)
    cmax     = np.nanpercentile(sval, 95)
    caxis_id = 'coloraxis' if ri == 0 else f'coloraxis{ri + 1}'

    for gi, (gname, gmask) in enumerate(groups):
        fig_score.add_trace(go.Scatter(
            x=xp_pcha[gmask, 0], y=xp_pcha[gmask, 1], mode='markers',
            marker=dict(size=3, color=sval[gmask], coloraxis=caxis_id, opacity=0.6),
            showlegend=False,
        ), row=ri + 1, col=gi + 1)
        fig_score.add_trace(go.Scatter(
            x=aa[0, :], y=aa[1, :], mode='markers',
            marker=dict(size=8, color='black', symbol='diamond'),
            name='archetypes', showlegend=(ri == 0 and gi == 0), legendgroup='arch',
        ), row=ri + 1, col=gi + 1)
        fig_score.add_trace(go.Scatter(
            x=list(aa[0, :]) + [aa[0, 0]], y=list(aa[1, :]) + [aa[1, 0]], mode='lines',
            line=dict(color='black', width=1.5),
            showlegend=False, legendgroup='arch',
        ), row=ri + 1, col=gi + 1)

    fig_score.update_layout(**{
        caxis_id: dict(
            colorscale='RdBu_r', cmin=cmin, cmax=cmax,
            colorbar=dict(
                title=dict(text=sname, side='right'),
                y=cb_y[ri], len=0.24, yanchor='middle', thickness=12,
            ),
        )
    })

for ri in range(n_scores):
    for gi in range(n_groups):
        subplot_idx = ri * n_groups + gi + 1
        xref = 'x' if subplot_idx == 1 else f'x{subplot_idx}'
        fig_score.update_xaxes(title_text='VX1', row=ri + 1, col=gi + 1)
        fig_score.update_yaxes(title_text='VX3', scaleanchor=xref, scaleratio=1,
                                row=ri + 1, col=gi + 1)

fig_score.update_layout(
    title=f'jainlab26 wt+cux2cre L2/3 — archetype scores (NOC={NOC})',
    width=500 * n_groups,
    height=500 * n_scores,
)
fig_score.write_html(OUT_SCORE_HTML)
print(f'Saved {OUT_SCORE_HTML}')
print('Done.')
