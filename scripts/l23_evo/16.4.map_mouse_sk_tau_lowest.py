"""Map mouse L2/3 IT (Yoo25, P21) into human VX archetype space via Sinkhorn-Knopp.

Script 16.4: same as 16.3 but with τ=0.002.
16.3 (τ=0.005): p50 eff-N=20. Testing τ=0.002 to sharpen further.

Reads:
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
  links/l23_evo/yoo25_mouse_IT_P21.h5ad
  local_data/res/l23_evo/05.varimax_loadings.tsv
  data/human_mouse_orthologs.tsv
  local_data/res/l23_evo/09.pcha_xp.tsv
  local_data/res/l23_evo/09.pcha_aa.tsv
Output:
  local_data/res/l23_evo/16.4.mouse_sk_tau_lowest_coords.tsv
  local_data/fig/l23_evo/16.4.mouse_sk_tau_lowest_scatter.html
"""

import os
import sys
import io
import requests
import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from viz import scatter_categorical_html

# --- file paths ---
OUT_RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
INPUT_HUMAN      = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
INPUT_MOUSE      = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'yoo25_mouse_IT_P21.h5ad')
IN_LOADINGS      = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
IN_ORTHOLOGS     = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_XP            = os.path.join(OUT_RES_DIR, '09.pcha_xp.tsv')
IN_AA            = os.path.join(OUT_RES_DIR, '09.pcha_aa.tsv')
OUT_MOUSE_COORDS = os.path.join(OUT_RES_DIR, '16.4.mouse_sk_tau_lowest_coords.tsv')
OUT_HTML         = os.path.join(OUT_FIG_DIR, '16.4.mouse_sk_tau_lowest_scatter.html')

# --- parameters ---
VX_COLS            = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
N_TOP_GENES_PER_VX = 50
MOUSE_SUBCLASS     = 'L2/3'
N_DOWNSAMPLE_VIZ   = 5000
CLUSTER_COL        = 'WithinArea_cluster'
HUMAN_FRAC         = 0.2    # downsample human to this fraction before SK
SK_TAU             = 0.002  # temperature for exp(S / τ) — lower than 16.3 (0.005) to sharpen further
SK_MAX_ITER        = 100    # max SK iterations
SK_TOL             = 1e-6   # convergence tolerance
RANDOM_SEED        = 0

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- 1. Select high-loading genes from VX subspace ---
print('Selecting high-loading genes...')
load_df = pd.read_csv(IN_LOADINGS, sep='\t', index_col=0)
seen, vx_genes = set(), []
for col in VX_COLS:
    for gene in load_df[col].abs().nlargest(N_TOP_GENES_PER_VX).index:
        if gene not in seen:
            seen.add(gene)
            vx_genes.append(gene)
print(f'  {len(vx_genes)} unique genes from VX subspace')

# --- 2. Load ortholog table (cached) ---
if not os.path.exists(IN_ORTHOLOGS):
    print('Downloading human-mouse orthologs from BioMart...')
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Attribute name="external_gene_name"/>
    <Attribute name="mmusculus_homolog_associated_gene_name"/>
  </Dataset>
</Query>"""
    r = requests.get('https://www.ensembl.org/biomart/martservice', params={'query': xml})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep='\t')
    df.columns = ['human_symbol', 'mouse_symbol']
    df = df.dropna().query("human_symbol != '' and mouse_symbol != ''").drop_duplicates()
    df.to_csv(IN_ORTHOLOGS, sep='\t', index=False)
    print(f'  Saved {IN_ORTHOLOGS}  ({len(df)} orthologs)')
else:
    print(f'  Using cached {IN_ORTHOLOGS}')

ortho = pd.read_csv(IN_ORTHOLOGS, sep='\t')
ortho = ortho.drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
h2m = ortho.set_index('human_symbol')['mouse_symbol'].to_dict()
gene_pairs_pre = [(g, h2m[g]) for g in vx_genes if g in h2m]
print(f'  {len(gene_pairs_pre)} gene pairs with 1-to-1 orthologs')

# --- 3. Load human and mouse adata ---
print('Loading human expression...')
h_adata      = ad.read_h5ad(INPUT_HUMAN)
h_gene_names = (h_adata.var['feature_name'].values
                if 'feature_name' in h_adata.var.columns
                else h_adata.var_names.values)
h_gene_set   = set(h_gene_names)
types        = h_adata.obs[CLUSTER_COL].values

print('Loading mouse expression...')
m_adata      = ad.read_h5ad(INPUT_MOUSE)
m_adata      = m_adata[m_adata.obs['Subclass'] == MOUSE_SUBCLASS]
m_gene_names = m_adata.var_names.values
m_gene_set   = set(m_gene_names)

# Filter to gene pairs present in both datasets
gene_pairs  = [(hg, mg) for hg, mg in gene_pairs_pre if hg in h_gene_set and mg in m_gene_set]
human_genes = [p[0] for p in gene_pairs]
mouse_genes = [p[1] for p in gene_pairs]
print(f'  {len(gene_pairs)} gene pairs present in both datasets')

h_idx   = np.array([np.where(h_gene_names == g)[0][0] for g in human_genes])
X_human = h_adata.X[:, h_idx].toarray().astype(np.float32)
print(f'  X_human shape: {X_human.shape}')

# --- 4. Load and normalize mouse expression ---
m_idx        = np.array([np.where(m_gene_names == g)[0][0] for g in mouse_genes])
X_mouse_raw  = m_adata.X[:, m_idx].toarray().astype(np.float32)
depths       = X_mouse_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_mouse      = np.log2(X_mouse_raw / depths * 1e4 + 1)
print(f'  X_mouse shape: {X_mouse.shape}  ({m_adata.n_obs} cells)')

# --- 5. Downsample human cells ---
print('Downsampling human cells...')
rng         = np.random.default_rng(RANDOM_SEED)
n_human     = X_human.shape[0]
n_human_sub = int(n_human * HUMAN_FRAC)
h_sub_idx   = rng.choice(n_human, n_human_sub, replace=False)
X_human_sub = X_human[h_sub_idx]
print(f'  Downsampled human: {n_human} → {n_human_sub} cells')

# --- 6. Full cosine similarity matrix ---
print('Computing cosine similarity matrix...')

def l2norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

X_human_n = l2norm(X_human_sub)   # (n_human_sub, n_genes)
X_mouse_n = l2norm(X_mouse)       # (n_mouse, n_genes)
n_mouse   = X_mouse_n.shape[0]

S = X_mouse_n @ X_human_n.T       # (n_mouse, n_human_sub)
print(f'  Similarity matrix: {S.shape}, dtype={S.dtype}')

# --- 7. Sinkhorn-Knopp balanced assignment ---
print('Running Sinkhorn-Knopp...')

def sinkhorn(S, tau, max_iter, tol):
    """Entropy-regularized balanced assignment via Sinkhorn-Knopp.

    Row marginals = 1/n_mouse  (each mouse cell distributes unit mass)
    Col marginals = 1/n_human  (each human cell receives equal mass)

    Returns P (n_mouse × n_human_sub), rows sum to 1/n_mouse.
    """
    n_mouse, n_human = S.shape
    log_P = S / tau
    log_P -= log_P.max(axis=1, keepdims=True)  # numerical stability
    P = np.exp(log_P).astype(np.float64)

    r_target = np.full(n_mouse, 1.0 / n_mouse)
    c_target = np.full(n_human, 1.0 / n_human)

    for i in range(max_iter):
        # row normalize to r_target
        row_sums = P.sum(axis=1)
        P *= (r_target / row_sums)[:, None]

        # col normalize to c_target
        col_sums = P.sum(axis=0)
        P *= (c_target / col_sums)[None, :]

        # convergence check
        row_err = np.abs(P.sum(axis=1) - r_target).max()
        col_err = np.abs(P.sum(axis=0) - c_target).max()
        if max(row_err, col_err) < tol:
            print(f'  SK converged at iteration {i+1}  '
                  f'(row_err={row_err:.2e}, col_err={col_err:.2e})')
            break
    else:
        print(f'  SK did not converge in {max_iter} iters  '
              f'(row_err={row_err:.2e}, col_err={col_err:.2e})')
    return P

P = sinkhorn(S, SK_TAU, SK_MAX_ITER, SK_TOL)
del S

# --- 8. Hub diagnostic ---
print('Hub diagnostic...')
w_norm = P / P.sum(axis=1, keepdims=True)           # row-normalize to sum=1
eff_n  = 1.0 / (w_norm ** 2).sum(axis=1)            # effective N per mouse cell
print('Effective human cells per mouse cell:')
for pct in [5, 25, 50, 75, 95]:
    print(f'  p{pct}: {np.percentile(eff_n, pct):.1f}')

col_marginals = P.sum(axis=0)
print(f'Column marginal (human usage): '
      f'mean={col_marginals.mean():.4f}  '
      f'std={col_marginals.std():.4f}  '
      f'cv={col_marginals.std()/col_marginals.mean():.3f}')

# --- 9. Embed mouse cells in human PCHA space ---
print('Embedding mouse cells...')
xp_human     = pd.read_csv(IN_XP, sep='\t', index_col=0).values.astype(np.float32)
xp_human_sub = xp_human[h_sub_idx]

mouse_xp = w_norm.astype(np.float32) @ xp_human_sub  # (n_mouse, NDIM)

# --- 10. Save and visualize ---
ndim = xp_human.shape[1]
mouse_coords_df = pd.DataFrame(mouse_xp, index=m_adata.obs_names,
                                columns=[f'PC{i+1}' for i in range(ndim)])
mouse_coords_df['Subclass']    = m_adata.obs['Subclass'].values
mouse_coords_df['Type_leiden'] = m_adata.obs['Type_leiden'].values
mouse_coords_df.to_csv(OUT_MOUSE_COORDS, sep='\t')
print(f'Saved {OUT_MOUSE_COORDS}  ({len(mouse_coords_df)} rows)')

print('Generating visualization...')
aa       = pd.read_csv(IN_AA, sep='\t', index_col=0).values.T.astype(np.float32)
viz_hidx = rng.choice(n_human_sub, min(N_DOWNSAMPLE_VIZ, n_human_sub), replace=False)

xp_combined = np.vstack([xp_human_sub[viz_hidx], mouse_xp])
scatter_categorical_html(
    xp_grid=[xp_combined],
    cell_metadata={
        'species':     np.array(['human'] * len(viz_hidx) + ['mouse'] * n_mouse),
        'Type_leiden': np.concatenate([types[h_sub_idx][viz_hidx],
                                       m_adata.obs['Type_leiden'].values]),
    },
    title='Jorstad23 (human 20%) + Yoo25 L2/3 (mouse) — SK balanced assignment (τ=0.002)',
    out_path=OUT_HTML,
    panels=[(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')],
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    aa_grid=[aa],
)
print(f'Saved {OUT_HTML}')
print('Done.')
