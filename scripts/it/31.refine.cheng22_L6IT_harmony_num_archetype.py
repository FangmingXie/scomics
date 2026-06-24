"""Harmony archetype number selection — cheng22 mouse L6IT IT (embed + compute, refined).

Self-contained Harmony archetype-inference procedure (merged embed + sweep):
  1) z-score normalization (CP10k → log2(1+x) → z-score per gene) on top-2000 HVGs
  2) regress out library size (log depth) in expression space
  3) PCA (10 comps)
  4) Harmony batch correction across samples (Sample)
  5) PCHA NOC sweep directly on the top 5 Harmony-corrected PCs (H1–H5, NDIM=5,
     no PC dropped via drop_pcs=[]), with N_OUTER bootstrap repeats for mean ± std
     error bars.

Mirrors the normalization + regression + Harmony steps of
scripts/astro/41.embed_cheng22_nr_harmony_arch12.py. Plotting lives in
31.refine.viz.cheng22_L6IT_harmony_num_archetype.py.

Reads:
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
Outputs:
  local_data/res/it/31.refine.cheng22_L6IT_harmony_coords.tsv      (H1..H10 + Type + Sample)
  local_data/res/it/31.refine.cheng22_L6IT_harmony_pca_coords.tsv  (raw PCA, pre-Harmony)
  local_data/res/it/31.refine.cheng22_L6IT_harmony_loadings.tsv    (PCA gene loadings)
  local_data/res/it/31.refine.cheng22_L6IT_harmony_num_archetype_metrics.tsv   (metric grids w/ ARV mean/std)
  local_data/res/it/31.refine.cheng22_L6IT_harmony_num_archetype_plotdata.pkl  (proj + per-group archetypes)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import harmonypy as hm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg, run_noc_sweep
from scomics.main import SCA
from scomics.utils import norm, get_relative_variation

# --- file paths ---
OUT_RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
INPUT_MOUSE        = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
OUT_HARMONY_COORDS = os.path.join(OUT_RES_DIR, '31.refine.cheng22_L6IT_harmony_coords.tsv')
OUT_PCA_COORDS     = os.path.join(OUT_RES_DIR, '31.refine.cheng22_L6IT_harmony_pca_coords.tsv')
OUT_PCA_LOADINGS   = os.path.join(OUT_RES_DIR, '31.refine.cheng22_L6IT_harmony_loadings.tsv')
OUT_METRICS_TSV    = os.path.join(OUT_RES_DIR, '31.refine.cheng22_L6IT_harmony_num_archetype_metrics.tsv')
OUT_PLOTDATA       = os.path.join(OUT_RES_DIR, '31.refine.cheng22_L6IT_harmony_num_archetype_plotdata.pkl')

# --- parameters ---
MOUSE_SUBCLASS = 'L6IT'
CLUSTER_COL    = 'Type'
SAMPLE_COL     = 'Sample'
N_HVG          = 2000
N_PCS          = 10
# Top 5 Harmony-corrected PCs used directly as the PCHA space (no drop)
N_ARCH_PCS     = 5
NDIM           = N_ARCH_PCS   # 5
DROP_PCS       = []           # keep all NDIM dims (no PC dropped)
NOC_MIN        = 2
NOC_MAX        = 6
NREPEATS       = 10   # bootstrap resamples per ARV estimate
N_OUTER        = 20   # repeated ARV estimates → mean ± std

os.makedirs(OUT_RES_DIR, exist_ok=True)

# ============================ embed ============================

# --- 1. Load and filter to subclass ---
print('Loading mouse cheng22 data...')
adata = ad.read_h5ad(INPUT_MOUSE)
# remove mitochondrial genes (mirror astro/41)
mt_mask = np.array([g.lower().startswith('mt-') for g in adata.var_names])
adata = adata[:, ~mt_mask].copy()
print(f'  Removed {mt_mask.sum()} mito genes; {(~mt_mask).sum()} remaining')
adata = adata[adata.obs['Subclass'] == MOUSE_SUBCLASS].copy()
print(f'  {adata.n_obs} {MOUSE_SUBCLASS} cells, {adata.n_vars} genes')

types   = adata.obs[CLUSTER_COL].values
samples = adata.obs[SAMPLE_COL].values

# raw counts from .raw (cheng22 .X is log1p; .raw holds integer counts)
x = adata.raw[:, adata.var_names].X
x = x.toarray() if sp.issparse(x) else np.asarray(x, dtype=np.float64)
depths = x.sum(axis=1)

# --- 2. HVG selection ---
print(f'Selecting top {N_HVG} HVGs...')
hvg_mask = select_hvg(x, depths, N_HVG)
gene_names = adata.var_names.values[hvg_mask]

# --- 3. normalize (CP10k → log2(1+x) → z-score per gene) ---
xn = norm(x[:, hvg_mask], depths)

# --- 4. regress out library size (log depth) ---
print('Regressing out library size...')
log_depth = np.log(depths).reshape(-1, 1)
reg = LinearRegression().fit(log_depth, xn)
xn = xn - reg.predict(log_depth)

# --- 5. PCA ---
print(f'Fitting PCA (N_PCS={N_PCS})...')
pca = PCA(N_PCS, random_state=0)
pca_scores = pca.fit_transform(xn)   # (n, N_PCS)
L = pca.components_.T                 # (n_hvg, N_PCS)

# --- 6. Harmony batch correction across samples ---
print('Running Harmony batch correction (by Sample)...')
meta_df = pd.DataFrame({SAMPLE_COL: samples})
ho = hm.run_harmony(pca_scores, meta_df, SAMPLE_COL, random_state=0)
Z = np.asarray(ho.Z_corr)
# harmonypy returns Z_corr as (d, N); orient to (n_cells, N_PCS)
harmony_scores = Z.T if Z.shape[0] == N_PCS else Z
assert harmony_scores.shape == (adata.n_obs, N_PCS), \
    f'unexpected Harmony shape {harmony_scores.shape}, expected {(adata.n_obs, N_PCS)}'
print(f'  Harmony corrected embedding: {harmony_scores.shape}')

# --- 7. Save embedding artifacts ---
h_cols = [f'H{i+1}' for i in range(N_PCS)]
h_df = pd.DataFrame(harmony_scores, index=adata.obs_names, columns=h_cols)
h_df[CLUSTER_COL] = types
h_df[SAMPLE_COL]  = samples
h_df.to_csv(OUT_HARMONY_COORDS, sep='\t')
print(f'Saved {OUT_HARMONY_COORDS}')

pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
pc_df = pd.DataFrame(pca_scores, index=adata.obs_names, columns=pc_cols)
pc_df[CLUSTER_COL] = types
pc_df[SAMPLE_COL]  = samples
pc_df.to_csv(OUT_PCA_COORDS, sep='\t')
print(f'Saved {OUT_PCA_COORDS}')

pd.DataFrame(L, index=gene_names, columns=pc_cols).to_csv(OUT_PCA_LOADINGS, sep='\t')
print(f'Saved {OUT_PCA_LOADINGS}')

# ===================== archetype sweep =====================

# Feature matrix = top 5 Harmony-corrected PCs (used directly, no drop)
xf = harmony_scores[:, :N_ARCH_PCS]
sca = SCA(xf, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
print(f'Running NOC sweep {NOC_MIN}–{NOC_MAX}, NDIM={NDIM} (drop_pcs={DROP_PCS}), NREPEATS={NREPEATS}...')

ev_grid, _, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, samples, drop_pcs=DROP_PCS)

# --- repeated bootstrap ARV → mean ± std ---
print(f'Repeating bootstrap ARV N_OUTER={N_OUTER} times per NOC...')
arv_mean, arv_std, effev_mean, effev_std = [], [], [], []
for i, noc in enumerate(noc_grid):
    arv_reps = np.array([
        get_relative_variation(sca.bootstrap_proj_pcha(NDIM, noc, nrepeats=NREPEATS, drop_pcs=DROP_PCS))
        for _ in range(N_OUTER)
    ])
    effev_reps = ev_grid[i] * (1 - arv_reps)
    arv_mean.append(arv_reps.mean())
    arv_std.append(arv_reps.std())
    effev_mean.append(effev_reps.mean())
    effev_std.append(effev_reps.std())
    print(f"  NOC={noc}  ARV={arv_reps.mean():.4f}±{arv_reps.std():.4f}"
          f"  effEV={effev_reps.mean():.4f}±{effev_reps.std():.4f}")

arv_mean, arv_std = np.array(arv_mean), np.array(arv_std)
effev_mean, effev_std = np.array(effev_mean), np.array(effev_std)
effev_rep_grid = ev_grid * (1 - av_rep_grid)

# --- persist metric grids ---
metrics_df = pd.DataFrame({
    'NOC':        noc_grid,
    'EV':         ev_grid,
    'ARV_mean':   arv_mean,
    'ARV_std':    arv_std,
    'ARV_rep':    av_rep_grid,
    'effEV_mean': effev_mean,
    'effEV_std':  effev_std,
    'effEV_rep':  effev_rep_grid,
})
metrics_df.to_csv(OUT_METRICS_TSV, sep='\t', index=False)
print(f'  Saved {OUT_METRICS_TSV}')

# --- persist projection + per-group archetypes (drives the per-sample scatter) ---
plotdata = {
    'noc_grid':     noc_grid,
    'ndim':         NDIM,
    'samples':      samples,
    'xp':           xp_grid[0],
    'aa_reps_grid': aa_reps_grid,
}
with open(OUT_PLOTDATA, 'wb') as f:
    pickle.dump(plotdata, f)
print(f'  Saved {OUT_PLOTDATA}')

print('Done.')
