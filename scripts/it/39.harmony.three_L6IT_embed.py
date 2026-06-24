"""Harmony embedding — tri-dataset L6IT (load + merge + Harmony).

Integrates three mouse cortical IT datasets — cheng22 (P28NR), yoo25 (P21), and
gao25 (DevVIS P56) — into a single Harmony-corrected embedding (batch = sample
within dataset). This is the embedding half of the tri-dataset L6IT expansion of
28.harmony.cheng22_L4_num_archetype.py; the archetype/NOC sweep lives in
39.harmony.three_L6IT_num_archetype.py (which reads this script's coords output).

Procedure (mirrors template 28, on the merged object):
  1) per-dataset L6IT extraction on the shared gene set (inner join, 16,570 genes)
     - gao25 'L6 IT CTX Glut' -> L6IT; gao25 kept at a fixed-seed 25% subsample
     - sequencing depth comes from each dataset's validated total-counts column
       (cheng22 'n_counts', yoo25 'total_counts'); gao25 has no such column so
       depth is the full-.X row sum computed BEFORE gene subsetting
  2) z-score normalization (CP10k -> log2(1+x) -> z-score per gene) on top-2000 HVGs
  3) regress out library size (log depth) in expression space
  4) PCA (10 comps)
  5) Harmony batch correction across dataset:sample keys

Reads:
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  links/it/superdupermegaRNA_yoo25_IT_P21.h5ad
  links/it/DevVIS_scRNA_IT_CTX_Glut_P56.h5ad
Outputs:
  local_data/res/it/39.harmony.three_L6IT_coords.tsv      (H1..H10 + Type + Sample + Dataset)
  local_data/res/it/39.harmony.three_L6IT_pca_coords.tsv  (raw PCA, pre-Harmony)
  local_data/res/it/39.harmony.three_L6IT_loadings.tsv    (PCA gene loadings)
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import harmonypy as hm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg
from scomics.utils import norm

# --- file paths ---
OUT_RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
INPUT_CHENG22      = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
INPUT_YOO25        = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_yoo25_IT_P21.h5ad')
INPUT_GAO25        = os.path.join(PROJECT_ROOT, 'links', 'it', 'DevVIS_scRNA_IT_CTX_Glut_P56.h5ad')
OUT_HARMONY_COORDS = os.path.join(OUT_RES_DIR, '39.harmony.three_L6IT_coords.tsv')
OUT_PCA_COORDS     = os.path.join(OUT_RES_DIR, '39.harmony.three_L6IT_pca_coords.tsv')
OUT_PCA_LOADINGS   = os.path.join(OUT_RES_DIR, '39.harmony.three_L6IT_loadings.tsv')

# --- per-dataset config ---
# depth_col=None  -> compute depth from the full .X row sum (before gene subsetting)
# counts='raw'    -> pull integer counts from adata.raw; 'X' -> adata.X is already counts
DATASETS = [
    dict(tag='cheng22', path=INPUT_CHENG22, subclass_col='Subclass',
         subclass_val='L6IT', sample_col='Sample', depth_col='n_counts',
         type_col='Type', counts='raw'),
    dict(tag='yoo25', path=INPUT_YOO25, subclass_col='Subclass',
         subclass_val='L6IT', sample_col='Sample', depth_col='total_counts',
         type_col='Type', counts='raw'),
    dict(tag='gao25', path=INPUT_GAO25, subclass_col='subclass_label',
         subclass_val='L6 IT CTX Glut', sample_col='donor_name', depth_col=None,
         type_col='cluster_label', counts='X'),
]

# --- parameters ---
CLUSTER_COL    = 'Type'
SAMPLE_COL     = 'Sample'
DATASET_COL    = 'Dataset'
N_HVG          = 2000
N_PCS          = 10
GAO25_SUBSAMPLE_FRAC = 0.50   # keep this fraction of gao25 L6IT cells (1.0 = keep all)
RANDOM_SEED          = 0      # fixed seed for the gao25 subsample

os.makedirs(OUT_RES_DIR, exist_ok=True)

# ============================ load + merge ============================

# --- 1. Load each dataset and filter to L6IT ---
print('Loading datasets and filtering to L6IT...')
adatas = {}
for d in DATASETS:
    a = ad.read_h5ad(d['path'])
    a = a[a.obs[d['subclass_col']] == d['subclass_val']].copy()
    adatas[d['tag']] = a
    print(f"  {d['tag']:8s}: {a.n_obs} cells ({d['subclass_val']}), {a.n_vars} genes")

# --- 2. Shared gene set (inner join), ordered by cheng22, mito removed ---
common_set = set(adatas['cheng22'].var_names)
for tag in ('yoo25', 'gao25'):
    common_set &= set(adatas[tag].var_names)
common_genes = [g for g in adatas['cheng22'].var_names if g in common_set]
n_before = len(common_genes)
common_genes = [g for g in common_genes if not g.lower().startswith('mt-')]
print(f"  Shared genes: {n_before} (intersection); "
      f"{len(common_genes)} after removing {n_before - len(common_genes)} mito genes")

# --- 3. Extract counts + validated depth per dataset, then merge ---
rng = np.random.default_rng(RANDOM_SEED)
x_list, depth_list, samples_list, datasets_list, types_list, obs_list = [], [], [], [], [], []

for d in DATASETS:
    a = adatas[d['tag']]

    # gao25 fixed-seed subsample (applied to cells before depth/counts)
    if d['tag'] == 'gao25' and GAO25_SUBSAMPLE_FRAC < 1.0:
        n = a.n_obs
        k = int(round(GAO25_SUBSAMPLE_FRAC * n))
        idx = np.sort(rng.choice(n, size=k, replace=False))
        a = a[idx].copy()
        print(f"  gao25 subsampled to {k}/{n} cells (frac={GAO25_SUBSAMPLE_FRAC}, seed={RANDOM_SEED})")

    # depth (true library size) BEFORE gene subsetting
    if d['depth_col'] is not None:
        depth = a.obs[d['depth_col']].values.astype(np.float64)
        assert np.all(np.isfinite(depth)) and np.all(depth > 0), \
            f"{d['tag']}: invalid depth column '{d['depth_col']}' (NaN or <=0)"
    else:
        xfull = a.X
        depth = np.asarray(xfull.sum(axis=1)).ravel().astype(np.float64)
        assert np.all(np.isfinite(depth)) and np.all(depth > 0), \
            f"{d['tag']}: invalid computed depth (NaN or <=0)"

    # counts on the shared gene set (same order across datasets)
    xc = a.raw[:, common_genes].X if d['counts'] == 'raw' else a[:, common_genes].X
    xc = xc.toarray() if sp.issparse(xc) else np.asarray(xc, dtype=np.float64)

    x_list.append(xc)
    depth_list.append(depth)
    samples_list.append(np.array([f"{d['tag']}:{s}" for s in a.obs[d['sample_col']].astype(str).values]))
    datasets_list.append(np.repeat(d['tag'], a.n_obs))
    types_list.append(a.obs[d['type_col']].astype(str).values)
    obs_list.append(np.array([f"{d['tag']}:{n}" for n in a.obs_names]))
    print(f"  {d['tag']:8s}: depth median {np.median(depth):.0f}  ->  {xc.shape[0]} cells merged")

x        = np.vstack(x_list)
depths   = np.concatenate(depth_list)
samples  = np.concatenate(samples_list)
datasets = np.concatenate(datasets_list)
types    = np.concatenate(types_list)
obs_names = np.concatenate(obs_list)
n_obs = x.shape[0]
print(f"Merged object: {n_obs} cells x {len(common_genes)} genes")

# ============================ embed ============================

# --- 4. HVG selection (on merged counts) ---
print(f'Selecting top {N_HVG} HVGs...')
hvg_mask = select_hvg(x, depths, N_HVG)
gene_names = np.array(common_genes)[hvg_mask]

# --- 5. normalize (CP10k -> log2(1+x) -> z-score per gene) ---
xn = norm(x[:, hvg_mask], depths)

# --- 6. regress out library size (log depth) ---
print('Regressing out library size...')
log_depth = np.log(depths).reshape(-1, 1)
reg = LinearRegression().fit(log_depth, xn)
xn = xn - reg.predict(log_depth)

# --- 7. PCA ---
print(f'Fitting PCA (N_PCS={N_PCS})...')
pca = PCA(N_PCS, random_state=0)
pca_scores = pca.fit_transform(xn)   # (n, N_PCS)
L = pca.components_.T                 # (n_hvg, N_PCS)

# --- 8. Harmony batch correction across dataset:sample keys ---
print('Running Harmony batch correction (by dataset:sample)...')
meta_df = pd.DataFrame({SAMPLE_COL: samples})
ho = hm.run_harmony(pca_scores, meta_df, SAMPLE_COL, random_state=0)
Z = np.asarray(ho.Z_corr)
# harmonypy returns Z_corr as (d, N); orient to (n_cells, N_PCS)
harmony_scores = Z.T if Z.shape[0] == N_PCS else Z
assert harmony_scores.shape == (n_obs, N_PCS), \
    f'unexpected Harmony shape {harmony_scores.shape}, expected {(n_obs, N_PCS)}'
print(f'  Harmony corrected embedding: {harmony_scores.shape}')

# --- 9. Save embedding artifacts ---
h_cols = [f'H{i+1}' for i in range(N_PCS)]
h_df = pd.DataFrame(harmony_scores, index=obs_names, columns=h_cols)
h_df[CLUSTER_COL] = types
h_df[SAMPLE_COL]  = samples
h_df[DATASET_COL] = datasets
h_df.to_csv(OUT_HARMONY_COORDS, sep='\t')
print(f'Saved {OUT_HARMONY_COORDS}')

pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
pc_df = pd.DataFrame(pca_scores, index=obs_names, columns=pc_cols)
pc_df[CLUSTER_COL] = types
pc_df[SAMPLE_COL]  = samples
pc_df[DATASET_COL] = datasets
pc_df.to_csv(OUT_PCA_COORDS, sep='\t')
print(f'Saved {OUT_PCA_COORDS}')

pd.DataFrame(L, index=gene_names, columns=pc_cols).to_csv(OUT_PCA_LOADINGS, sep='\t')
print(f'Saved {OUT_PCA_LOADINGS}')

print('Done.')
