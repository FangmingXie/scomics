# Archetype scores from P56 marker genes across all postnatal ages.
# 1) Identifies top N_ARCHETYPE_GENES marker genes per archetype using P56 cells.
# 2) Scores each cell as mean z-scored (within age) log1p expression over marker genes.
# 3) Saves scores + PC coords to parquet. Visualization handled by script 20.

import json
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE            = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')
RES_DIR               = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
PARQUET_ALL_IN        = os.path.join(RES_DIR, '17.labels_all_ages.parquet')
PARQUET_SCORES        = os.path.join(RES_DIR, '19.archetype_scores.parquet')
JSON_MARKER_GENES     = os.path.join(RES_DIR, '19.archetype_marker_genes.json')

N_TOP_GENES       = 2000
N_ARCHETYPE_GENES = 10
NOC               = 4
P56_AGE_VAL       = 'P56'

os.makedirs(RES_DIR, exist_ok=True)

# Load expression data
adata = ad.read_h5ad(INPUT_FILE)
print(f'Total cells: {adata.shape[0]}')

x      = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
depths = x.sum(axis=1)
ages   = adata.obs['Age'].values
donors = adata.obs['donor_name'].values

# Subset to postnatal cells
keep     = np.array([a.startswith('P') for a in ages])
x, depths, ages, donors = x[keep], depths[keep], ages[keep], donors[keep]
print(f'Postnatal cells: {keep.sum()}')

# HVG selection from P56 only (same as script 17)
p56_mask = ages == P56_AGE_VAL
print(f'P56 cells: {p56_mask.sum()}')
hvg_mask   = select_hvg(x[p56_mask], depths[p56_mask], N_TOP_GENES)
gene_names = np.array(adata.var_names)[hvg_mask]

# CP10k normalize then log1p for all postnatal cells (no zscore yet — done per-gene below)
cp10k = x[:, hvg_mask] / depths[:, None] * 1e4
xl    = np.log1p(cp10k)   # (n_postnatal, n_hvg)
print(f'log1p matrix shape: {xl.shape}')

# Load P56 archetype labels from script 17 output
df_labels = pd.read_parquet(PARQUET_ALL_IN)
df_p56    = df_labels[df_labels['age'] == P56_AGE_VAL].reset_index(drop=True)
arch_p56  = df_p56['archetype'].values
xl_p56    = xl[p56_mask]

# Identify top marker genes per archetype (based on P56 mean difference)
arch_gene_idx = {}
for k in range(NOC):
    label     = f'Arch{k+1}'
    in_mask   = arch_p56 == label
    mean_in   = xl_p56[in_mask].mean(axis=0)
    mean_out  = xl_p56[~in_mask].mean(axis=0)
    top_idx   = np.argsort(mean_in - mean_out)[::-1][:N_ARCHETYPE_GENES]
    arch_gene_idx[label] = top_idx
    print(f'  {label} top genes: {list(gene_names[top_idx])}')

marker_genes = {label: list(gene_names[idx]) for label, idx in arch_gene_idx.items()}
with open(JSON_MARKER_GENES, 'w') as f:
    json.dump(marker_genes, f, indent=2)
print(f'Saved marker genes → {JSON_MARKER_GENES}')

# Z-score each gene within each age, then compute archetype scores
score_arrays = {f'Arch{k+1}_score': np.empty(len(xl)) for k in range(NOC)}
for age_val in np.unique(ages):
    age_mask = ages == age_val
    xl_age   = xl[age_mask]
    zl_age   = (xl_age - xl_age.mean(axis=0)) / (xl_age.std(axis=0) + 1e-8)
    for k in range(NOC):
        label = f'Arch{k+1}'
        score_arrays[f'{label}_score'][age_mask] = zl_age[:, arch_gene_idx[label]].mean(axis=1)

# Merge with PC coords, donor, age, archetype from script 17
pc_cols = [c for c in df_labels.columns if c.startswith('PC')]
df_scores = df_labels[pc_cols + ['donor_name', 'age', 'archetype']].copy()
for col, vals in score_arrays.items():
    df_scores[col] = vals
df_scores.to_parquet(PARQUET_SCORES, index=False)
print(f'Saved {PARQUET_SCORES}  ({len(df_scores)} cells, columns: {list(df_scores.columns)})')
