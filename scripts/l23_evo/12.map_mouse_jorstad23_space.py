"""Map mouse L2/3 IT (Yoo25, P21) into human VX archetype space.

Identifies high-loading genes per VX component, finds mouse orthologs,
computes cosine similarity between mouse and human cells, then embeds
each mouse cell as a weighted average of its top-k most similar human
cells' coordinates in the PCHA space.

Reads:
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
  links/l23_evo/yoo25_mouse_IT_P21.h5ad
  local_data/res/l23_evo/05.varimax_loadings.tsv
  data/human_mouse_orthologs.tsv           (downloaded on first run)
  local_data/res/l23_evo/09.pcha_xp.tsv
  local_data/res/l23_evo/09.pcha_aa.tsv
Output:
  local_data/res/l23_evo/12.mouse_embedded_coords.tsv
  local_data/fig/l23_evo/12.mouse_embedded_scatter.html
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
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
INPUT_HUMAN    = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
INPUT_MOUSE    = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'yoo25_mouse_IT_P21.h5ad')
IN_LOADINGS    = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
IN_ORTHOLOGS   = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_XP          = os.path.join(OUT_RES_DIR, '09.pcha_xp.tsv')
IN_AA          = os.path.join(OUT_RES_DIR, '09.pcha_aa.tsv')
OUT_MOUSE_COORDS = os.path.join(OUT_RES_DIR, '12.mouse_embedded_coords.tsv')
OUT_HTML       = os.path.join(OUT_FIG_DIR, '12.mouse_embedded_scatter.html')

# --- parameters ---
VX_COLS            = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
N_TOP_GENES_PER_VX = 50
K_NEIGHBORS        = 10
MOUSE_SUBCLASS     = 'L2/3'
N_DOWNSAMPLE_VIZ   = 5000
CLUSTER_COL        = 'WithinArea_cluster'
CHUNK_SIZE         = 500

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

# --- 2. Download ortholog table (cached) ---
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

# --- 3. Load human and mouse adata to get gene name sets ---
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

h_idx      = np.array([np.where(h_gene_names == g)[0][0] for g in human_genes])
X_human    = h_adata.X[:, h_idx].toarray().astype(np.float32)
print(f'  X_human shape: {X_human.shape}')

# --- 4. Load and normalize mouse expression ---
m_idx      = np.array([np.where(m_gene_names == g)[0][0] for g in mouse_genes])
X_mouse_raw = m_adata.X[:, m_idx].toarray().astype(np.float32)
depths     = X_mouse_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_mouse    = np.log2(X_mouse_raw / depths * 1e4 + 1)
print(f'  X_mouse shape: {X_mouse.shape}  ({m_adata.n_obs} cells)')

# --- 5. Cosine similarity (chunked) ---
print('Computing cosine similarity (chunked)...')

def l2norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

X_human_n = l2norm(X_human)
X_mouse_n = l2norm(X_mouse)
n_mouse   = X_mouse_n.shape[0]
n_human   = X_human_n.shape[0]

topk_indices = np.empty((n_mouse, K_NEIGHBORS), dtype=np.int32)
topk_scores  = np.empty((n_mouse, K_NEIGHBORS), dtype=np.float32)

for start in range(0, n_mouse, CHUNK_SIZE):
    chunk      = X_mouse_n[start:start + CHUNK_SIZE]
    sim_chunk  = chunk @ X_human_n.T
    idx        = np.argpartition(sim_chunk, -K_NEIGHBORS, axis=1)[:, -K_NEIGHBORS:]
    scores     = np.take_along_axis(sim_chunk, idx, axis=1)
    topk_indices[start:start + len(chunk)] = idx
    topk_scores[start:start + len(chunk)]  = scores
    print(f'  chunk {start}–{start + len(chunk)} / {n_mouse}')

# --- 6. Load PCHA results from script 09 ---
print('Loading PCHA results...')
xp_human = pd.read_csv(IN_XP, sep='\t', index_col=0).values.astype(np.float32)
aa       = pd.read_csv(IN_AA, sep='\t', index_col=0).values.T.astype(np.float32)  # (NDIM, NOC)
print(f'  xp_human shape: {xp_human.shape},  aa shape: {aa.shape}')

# --- 7. Embed mouse cells ---
print('Embedding mouse cells...')
weights   = topk_scores / topk_scores.sum(axis=1, keepdims=True)
mouse_xp  = np.stack([
    weights[i] @ xp_human[topk_indices[i]]
    for i in range(n_mouse)
])
ndim = xp_human.shape[1]
mouse_coords_df = pd.DataFrame(mouse_xp, index=m_adata.obs_names,
                                columns=[f'PC{i+1}' for i in range(ndim)])
mouse_coords_df['Subclass'] = m_adata.obs['Subclass'].values
if 'Type_leiden' in m_adata.obs.columns:
    mouse_coords_df['Type_leiden'] = m_adata.obs['Type_leiden'].values
mouse_coords_df.to_csv(OUT_MOUSE_COORDS, sep='\t')
print(f'Saved {OUT_MOUSE_COORDS}  ({len(mouse_coords_df)} rows)')

# --- 8. Visualize human + mouse overlaid ---
print('Generating visualization...')
rng   = np.random.default_rng(0)
hidx  = rng.choice(n_human, min(N_DOWNSAMPLE_VIZ, n_human), replace=False)

xp_combined = np.vstack([xp_human[hidx], mouse_xp])
species_col = np.array(['human'] * len(hidx) + ['mouse'] * n_mouse)

cell_metadata = {'species': species_col}
if 'Type_leiden' in m_adata.obs.columns:
    cell_metadata['Type_leiden'] = np.concatenate([
        types[hidx],
        m_adata.obs['Type_leiden'].values,
    ])

scatter_categorical_html(
    xp_grid=[xp_combined],
    cell_metadata=cell_metadata,
    title='Jorstad23 (human) + Yoo25 L2/3 (mouse) — reprojected VX subspace',
    out_path=OUT_HTML,
    panels=[(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')],
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
    aa_grid=[aa],
)
print(f'Saved {OUT_HTML}')
print('Done.')
