"""Archetype markers for P56 gao25 astrocytes — fixed NOC.

Fits PCHA at analyst-chosen NOC on the VX subspace, assigns each archetype the
300 nearest cells (Euclidean in VX space), then runs Wilcoxon one-vs-rest
enrichment to find marker genes per archetype.
Saves inner PCA parameters needed for script 36 algebraic projection.

Reads:
  local_data/res/astro/33.varimax_coords.tsv
  local_data/res/astro/33.varimax_loadings.tsv  (HVG gene list)
  links/astro/gao25_scrna_astro.h5ad
Outputs:
  local_data/res/astro/35.pcha_xp.tsv
  local_data/res/astro/35.pcha_aa.tsv
  local_data/res/astro/35.inner_pca_components.tsv
  local_data/res/astro/35.inner_pca_mean.tsv
  local_data/res/astro/35.archetype_markers.tsv
  local_data/fig/astro/35.archetype_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html
from scomics.main import SCA

# --- file paths ---
INPUT_H5AD          = os.path.join(PROJECT_ROOT, 'links', 'astro', 'gao25_scrna_astro.h5ad')
OUT_RES_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_FIG_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
IN_VARIMAX_COORDS   = os.path.join(OUT_RES_DIR, '33.varimax_coords.tsv')
IN_VARIMAX_LOADINGS = os.path.join(OUT_RES_DIR, '33.varimax_loadings.tsv')
OUT_XP              = os.path.join(OUT_RES_DIR, '35.pcha_xp.tsv')
OUT_AA              = os.path.join(OUT_RES_DIR, '35.pcha_aa.tsv')
OUT_INNER_COMPS     = os.path.join(OUT_RES_DIR, '35.inner_pca_components.tsv')
OUT_INNER_MEAN      = os.path.join(OUT_RES_DIR, '35.inner_pca_mean.tsv')
OUT_MARKERS         = os.path.join(OUT_RES_DIR, '35.archetype_markers.tsv')
OUT_SCATTER_HTML    = os.path.join(OUT_FIG_DIR, '35.archetype_scatter.html')

# analyst-set after inspecting scripts 33 and 34
VX_COLS        = ['VX1', 'VX2', 'VX3', 'VX5', 'VX6']
NOC            = 4
NDIM           = 4
N_TOP_CELLS    = 300
FRAC_IN_THRESH = 0.25
FDR_THRESH     = 0.001
P56_AGE        = 'P56'

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load varimax coords (P56 only) ---
vx_df       = pd.read_csv(IN_VARIMAX_COORDS, sep='\t', index_col=0)
xn          = vx_df[VX_COLS].values
arch_labels = vx_df['archetype'].values
donors      = vx_df['donor_name'].values
print(f'Loaded VX coords: {xn.shape}')

# --- fit PCHA ---
print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(xn, arch_labels)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)

pd.DataFrame(sca.xp, index=vx_df.index,
             columns=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_XP, sep='\t')
pd.DataFrame(sca.aa.T, columns=[f'PC{i+1}' for i in range(NDIM)],
             index=[f'archetype_{k+1}' for k in range(NOC)]).to_csv(OUT_AA, sep='\t')
print(f'Saved {OUT_XP} and {OUT_AA}')

# --- save inner PCA parameters for script 36 ---
inner_comps = sca.pca_.components_[:NDIM]   # (NDIM, n_vx_cols)
inner_mean  = sca.pca_.mean_                  # (n_vx_cols,)

pd.DataFrame(inner_comps, columns=VX_COLS,
             index=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_INNER_COMPS, sep='\t')
pd.DataFrame(inner_mean.reshape(1, -1), columns=VX_COLS).to_csv(OUT_INNER_MEAN, sep='\t')
print(f'Saved {OUT_INNER_COMPS} and {OUT_INNER_MEAN}')

# --- back-project archetype vertices to VX space ---
aa_vx = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_   # (NOC, n_vx_cols)
print(f'Archetype VX coords shape: {aa_vx.shape}')

# --- nearest cells per archetype ---
dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(NOC)], axis=1)
top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(NOC)]
print(f'Top {N_TOP_CELLS} cells selected per archetype.')

# --- load raw counts (P56 only) and normalize ---
print('Loading h5ad for marker gene analysis...')
adata     = ad.read_h5ad(INPUT_H5AD)
adata_p56 = adata[adata.obs['Age'] == P56_AGE].copy()

hvg_genes  = pd.read_csv(IN_VARIMAX_LOADINGS, sep='\t', index_col=0).index.values
gene_names = adata_p56.var_names.values
hvg_idx    = np.where(np.isin(gene_names, hvg_genes))[0]
hvg_names  = gene_names[hvg_idx]

x_raw  = adata_p56.X[:, hvg_idx]
if sp.issparse(x_raw):
    x_raw = x_raw.toarray()
x_raw   = x_raw.astype(np.float32)
depths  = x_raw.sum(axis=1, keepdims=True)
depths  = np.where(depths == 0, 1, depths)
X_norm  = np.log2(x_raw / depths * 1e4 + 1)   # log2(CP10k + 1)
print(f'X_norm shape: {X_norm.shape}')

# --- Wilcoxon one-vs-rest per archetype ---
print('Running Wilcoxon enrichment per archetype...')
n_cells  = X_norm.shape[0]
n_hvg    = X_norm.shape[1]
all_markers = []

for k in range(NOC):
    archetype_label = f'archetype_{k + 1}'
    print(f'  {archetype_label}...')
    in_idx   = top_cells[k]
    out_mask = np.ones(n_cells, dtype=bool)
    out_mask[in_idx] = False

    X_in  = X_norm[in_idx]
    X_out = X_norm[out_mask]

    log2fc   = X_in.mean(axis=0) - X_out.mean(axis=0)   # already log2
    frac_in  = (X_in  > 0).mean(axis=0)
    frac_out = (X_out > 0).mean(axis=0)

    pvals = np.empty(n_hvg)
    for g in range(n_hvg):
        _, pvals[g] = scipy.stats.ranksums(X_in[:, g], X_out[:, g])

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
    df = df[(df['frac_in'] >= FRAC_IN_THRESH) & (df['fdr'] < FDR_THRESH)]
    df = df.sort_values('log2FC', ascending=False)
    all_markers.append(df)
    print(f'    {len(df)} markers after filtering')

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(OUT_MARKERS, sep='\t', index=False)
print(f'Saved {OUT_MARKERS}  ({len(markers_df)} total markers)')

# --- scatter visualization ---
xp_grid = [sca.xp]
aa      = sca.aa     # (NDIM, NOC)

scatter_categorical_html(
    xp_grid=xp_grid,
    cell_metadata={'archetype': arch_labels, 'donor_name': donors},
    title=f'P56 gao25 astrocytes — PCHA archetype space (NOC={NOC}, NDIM={NDIM})',
    out_path=OUT_SCATTER_HTML,
    noc_grid=np.array([NOC]),
    aa_grid=[aa],
)
print(f'Saved {OUT_SCATTER_HTML}')
print('Done.')
