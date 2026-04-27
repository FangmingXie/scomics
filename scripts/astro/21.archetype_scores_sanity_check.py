# Sanity check: recompute archetype scores using only the top N_GENES_CHECK marker genes
# (subset of the top 10 saved in script 19 JSON). Scores will differ from parquet by design.

import json
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPTS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE        = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')
RES_DIR           = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
JSON_MARKER_GENES = os.path.join(RES_DIR, '19.archetype_marker_genes.json')
PARQUET_SCORES    = os.path.join(RES_DIR, '19.archetype_scores.parquet')

N_GENES_CHECK = 3

# Load saved marker genes
with open(JSON_MARKER_GENES) as f:
    marker_genes = json.load(f)
print('Marker genes loaded:')
for label, genes in marker_genes.items():
    print(f'  {label}: {genes}')

# Load expression data
adata = ad.read_h5ad(INPUT_FILE)
x      = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
depths = x.sum(axis=1)
ages   = adata.obs['Age'].values
all_genes = np.array(adata.var_names)

# Subset to postnatal cells
keep = np.array([a.startswith('P') for a in ages])
x, depths, ages = x[keep], depths[keep], ages[keep]
print(f'Postnatal cells: {keep.sum()}')

# CP10k → log1p
cp10k = x / depths[:, None] * 1e4
xl    = np.log1p(cp10k)

# Resolve gene name → column index in full gene space
gene_to_col = {g: i for i, g in enumerate(all_genes)}

score_arrays = {f'{label}_score': np.empty(len(xl)) for label in marker_genes}
for age_val in np.unique(ages):
    age_mask = ages == age_val
    xl_age   = xl[age_mask]
    zl_age   = (xl_age - xl_age.mean(axis=0)) / (xl_age.std(axis=0) + 1e-8)
    for label, genes in marker_genes.items():
        gene_cols = np.array([gene_to_col[g] for g in genes[:N_GENES_CHECK]])
        score_arrays[f'{label}_score'][age_mask] = zl_age[:, gene_cols].mean(axis=1)

# Compare against saved scores
df_saved = pd.read_parquet(PARQUET_SCORES)
score_cols = [c for c in df_saved.columns if c.endswith('_score')]

print(f'\nSanity check (top {N_GENES_CHECK} genes vs. top 10 — Pearson r per archetype score):')
for col in score_cols:
    r = np.corrcoef(score_arrays[col], df_saved[col].values)[0, 1]
    print(f'  {col}: r = {r:.4f}')
