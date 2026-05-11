"""Joint Harmony embedding of human + mouse L2/3 IT — map mouse cells to human PCHA space.

Builds a shared subtype-variation gene space via orthologs, aligns species with
Harmony, finds nearest human neighbors per mouse cell, then embeds mouse cells
as weighted averages of neighbor coordinates in the original human PCHA space.

Reads:
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
  links/l23_evo/yoo25_mouse_IT_P21.h5ad
  local_data/res/l23_evo/05.varimax_loadings.tsv
  local_data/res/l23_evo/14.mouse_varimax_loadings.tsv
  data/human_mouse_orthologs.tsv
  local_data/res/l23_evo/09.pcha_xp.tsv
  local_data/res/l23_evo/09.pcha_aa.tsv
Outputs:
  local_data/res/l23_evo/15.mouse_harmony_coords.tsv
  local_data/fig/l23_evo/15.mouse_harmony_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import harmonypy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from viz import scatter_categorical_html

# --- file paths ---
OUT_RES_DIR          = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR          = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
INPUT_HUMAN          = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
INPUT_MOUSE          = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'yoo25_mouse_IT_P21.h5ad')
IN_HUMAN_VX_LOADINGS = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
IN_MOUSE_VX_LOADINGS = os.path.join(OUT_RES_DIR, '14.mouse_varimax_loadings.tsv')
IN_ORTHOLOGS         = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_HUMAN_XP          = os.path.join(OUT_RES_DIR, '09.pcha_xp.tsv')
IN_HUMAN_AA          = os.path.join(OUT_RES_DIR, '09.pcha_aa.tsv')
OUT_MOUSE_COORDS     = os.path.join(OUT_RES_DIR, '15.mouse_harmony_coords.tsv')
OUT_HTML             = os.path.join(OUT_FIG_DIR, '15.mouse_harmony_scatter.html')

# --- parameters ---
HUMAN_VX_COLS      = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
MOUSE_VX_COLS      = ['VX3', 'VX4', 'VX5', 'VX6']   # components with high Type_leiden R² (0.52, 0.37, 0.38, 0.37)
N_TOP_GENES_PER_VX = 50
N_PCS_JOINT        = 20
K_NEIGHBORS        = 10
CHUNK_SIZE         = 500
MOUSE_SUBCLASS     = 'L2/3'
N_DOWNSAMPLE_VIZ   = 5000
HUMAN_CLUSTER_COL  = 'WithinArea_cluster'
MOUSE_CLUSTER_COL  = 'Type_leiden'

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def l2norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


# --- 1. Shared gene space ---
print('Building shared gene space...')
human_load = pd.read_csv(IN_HUMAN_VX_LOADINGS, sep='\t', index_col=0)
mouse_load = pd.read_csv(IN_MOUSE_VX_LOADINGS, sep='\t', index_col=0)

seen_h, human_vx_genes = set(), []
for col in HUMAN_VX_COLS:
    for g in human_load[col].abs().nlargest(N_TOP_GENES_PER_VX).index:
        if g not in seen_h:
            seen_h.add(g)
            human_vx_genes.append(g)

seen_m, mouse_vx_genes = set(), []
for col in MOUSE_VX_COLS:
    for g in mouse_load[col].abs().nlargest(N_TOP_GENES_PER_VX).index:
        if g not in seen_m:
            seen_m.add(g)
            mouse_vx_genes.append(g)

print(f'  {len(human_vx_genes)} human VX genes, {len(mouse_vx_genes)} mouse VX genes')

# Ortholog dicts (1-to-1 only)
ortho = pd.read_csv(IN_ORTHOLOGS, sep='\t')
ortho = ortho.drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
h2m = ortho.set_index('human_symbol')['mouse_symbol'].to_dict()
m2h = ortho.set_index('mouse_symbol')['human_symbol'].to_dict()

# --- 2. Load datasets ---
print('Loading human h5ad...')
h_adata = ad.read_h5ad(INPUT_HUMAN)
h_gene_names = (h_adata.var['feature_name'].values
                if 'feature_name' in h_adata.var.columns
                else h_adata.var_names.values)
h_gene_set = set(h_gene_names)
n_human = h_adata.n_obs

print('Loading mouse h5ad...')
m_adata = ad.read_h5ad(INPUT_MOUSE)
m_adata = m_adata[m_adata.obs['Subclass'] == MOUSE_SUBCLASS]
m_gene_names = m_adata.var_names.values
m_gene_set = set(m_gene_names)
n_mouse = m_adata.n_obs

# Gene pairs: union of (a) human VX gene → mouse ortholog, (b) mouse VX gene → human ortholog
pair_set = set()
for hg in human_vx_genes:
    if hg in h2m and h2m[hg] in m_gene_set and hg in h_gene_set:
        pair_set.add((hg, h2m[hg]))
for mg in mouse_vx_genes:
    if mg in m2h and m2h[mg] in h_gene_set and mg in m_gene_set:
        pair_set.add((m2h[mg], mg))

gene_pairs = sorted(pair_set)
human_genes = [p[0] for p in gene_pairs]
mouse_genes = [p[1] for p in gene_pairs]
print(f'  {len(gene_pairs)} shared gene pairs')

# --- 3. Joint expression matrix ---
print('Building joint expression matrix...')
h_idx = np.array([np.where(h_gene_names == g)[0][0] for g in human_genes])
X_human = h_adata.X[:, h_idx].toarray().astype(np.float32)

m_idx = np.array([np.where(m_gene_names == g)[0][0] for g in mouse_genes])
X_mouse_raw = m_adata.X[:, m_idx].toarray().astype(np.float32)
depths = X_mouse_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_mouse = np.log2(X_mouse_raw / depths * 1e4 + 1)

X_joint = np.vstack([X_human, X_mouse])
print(f'  X_joint shape: {X_joint.shape}')

scaler = StandardScaler()
X_joint_scaled = scaler.fit_transform(X_joint)

# --- 4. Joint PCA ---
print(f'Joint PCA (n={N_PCS_JOINT})...')
pca = PCA(n_components=N_PCS_JOINT, random_state=0)
Z_pca = pca.fit_transform(X_joint_scaled)

# --- 5. Harmony ---
print('Running Harmony...')
meta = pd.DataFrame({'species': ['human'] * n_human + ['mouse'] * n_mouse})
ho = harmonypy.run_harmony(Z_pca, meta, vars_use='species', random_state=0)
Z_corr = ho.Z_corr   # harmonypy returns (n_cells, n_pcs)
Z_human = Z_corr[:n_human]
Z_mouse = Z_corr[n_human:]
print(f'  Z_human: {Z_human.shape}, Z_mouse: {Z_mouse.shape}')

# --- 6. Nearest human neighbors (chunked cosine) ---
print('Computing nearest human neighbors (chunked cosine)...')
Z_human_n = l2norm(Z_human.astype(np.float32))
Z_mouse_n = l2norm(Z_mouse.astype(np.float32))

topk_indices = np.empty((n_mouse, K_NEIGHBORS), dtype=np.int32)
topk_scores  = np.empty((n_mouse, K_NEIGHBORS), dtype=np.float32)

for start in range(0, n_mouse, CHUNK_SIZE):
    chunk = Z_mouse_n[start:start + CHUNK_SIZE]
    sim_chunk = chunk @ Z_human_n.T
    idx = np.argpartition(sim_chunk, -K_NEIGHBORS, axis=1)[:, -K_NEIGHBORS:]
    scores = np.take_along_axis(sim_chunk, idx, axis=1)
    topk_indices[start:start + len(chunk)] = idx
    topk_scores[start:start + len(chunk)]  = scores
    print(f'  chunk {start}–{start + len(chunk)} / {n_mouse}')

# --- 7. Hub diagnostic ---
unique, counts = np.unique(topk_indices, return_counts=True)
print(f'\nHub diagnostic:')
print(f'  Unique human cells used: {len(unique)} / {n_human}')
sorted_counts = np.sort(counts)[::-1]
cumsum = np.cumsum(sorted_counts)
total_slots = n_mouse * K_NEIGHBORS
for pct in [25, 50, 75]:
    n_needed = int(np.searchsorted(cumsum, total_slots * pct / 100)) + 1
    print(f'  Top {n_needed} cells cover {pct}% of neighbor slots')

# --- 8. Embed mouse cells in human PCHA space ---
print('\nEmbedding mouse cells in human PCHA space...')
xp_human = pd.read_csv(IN_HUMAN_XP, sep='\t', index_col=0).values.astype(np.float32)
aa = pd.read_csv(IN_HUMAN_AA, sep='\t', index_col=0).values.T.astype(np.float32)  # (NDIM, NOC)
ndim = xp_human.shape[1]

weights = topk_scores / topk_scores.sum(axis=1, keepdims=True)
mouse_xp = np.stack([weights[i] @ xp_human[topk_indices[i]] for i in range(n_mouse)])

mouse_coords_df = pd.DataFrame(mouse_xp, index=m_adata.obs_names,
                                columns=[f'PC{i+1}' for i in range(ndim)])
mouse_coords_df[MOUSE_CLUSTER_COL] = m_adata.obs[MOUSE_CLUSTER_COL].values
mouse_coords_df.to_csv(OUT_MOUSE_COORDS, sep='\t')
print(f'Saved {OUT_MOUSE_COORDS}  ({len(mouse_coords_df)} rows)')

# --- 9. Visualize human + mouse overlaid ---
print('\nGenerating visualization...')
rng = np.random.default_rng(0)
hidx = rng.choice(n_human, min(N_DOWNSAMPLE_VIZ, n_human), replace=False)
h_types = h_adata.obs[HUMAN_CLUSTER_COL].values

xp_combined = np.vstack([xp_human[hidx], mouse_xp])
species_col = np.array(['human'] * len(hidx) + ['mouse'] * n_mouse)

scatter_categorical_html(
    xp_grid=[xp_combined],
    cell_metadata={
        'species': species_col,
        MOUSE_CLUSTER_COL: np.concatenate([h_types[hidx], m_adata.obs[MOUSE_CLUSTER_COL].values]),
    },
    title='Jorstad23 (human) + Yoo25 L2/3 (mouse) — Harmony joint embedding',
    out_path=OUT_HTML,
    panels=[(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')],
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    aa_grid=[aa],
)
print(f'Saved {OUT_HTML}')
print('Done.')
