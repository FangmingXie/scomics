"""Cross-species VX subspace matching via CCA of loading matrices.

Applies CCA between human (Jorstad23) and mouse (Cheng22) VX loading matrices
in shared ortholog gene space. Genes are observations; VX components are features.
CCA finds linear combinations of VX components maximally correlated across species.

Reads:
  local_data/res/l23_evo/05.varimax_loadings.tsv
  local_data/res/l23_evo/05.variance_partition.tsv
  local_data/res/l23_evo/05.varimax_coords.tsv
  local_data/res/l23_evo/18.mouse_varimax_loadings.tsv
  local_data/res/l23_evo/18.mouse_vx_variance_partition.tsv
  local_data/res/l23_evo/18.mouse_varimax_coords.tsv
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/res/l23_evo/24.orthoaxis_cca_correlations.tsv
  local_data/res/l23_evo/24.orthoaxis_cca_weights_human.tsv
  local_data/res/l23_evo/24.orthoaxis_cca_weights_mouse.tsv
  local_data/res/l23_evo/24.orthoaxis_cca_top_genes.tsv
  local_data/fig/l23_evo/24.orthoaxis_cca_cell_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cross_decomposition import CCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

# --- file paths ---
OUT_RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_HUMAN_LOADINGS = os.path.join(OUT_RES_DIR, '05.varimax_loadings.tsv')
IN_HUMAN_VARPART  = os.path.join(OUT_RES_DIR, '05.variance_partition.tsv')
IN_HUMAN_COORDS   = os.path.join(OUT_RES_DIR, '05.varimax_coords.tsv')
IN_MOUSE_LOADINGS = os.path.join(OUT_RES_DIR, '18.mouse_varimax_loadings.tsv')
IN_MOUSE_VARPART  = os.path.join(OUT_RES_DIR, '18.mouse_vx_variance_partition.tsv')
IN_MOUSE_COORDS   = os.path.join(OUT_RES_DIR, '18.mouse_varimax_coords.tsv')
IN_ORTHOLOGS      = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_CCA_CORR_TSV  = os.path.join(OUT_RES_DIR, '24.orthoaxis_cca_correlations.tsv')
OUT_CCA_WHUMAN    = os.path.join(OUT_RES_DIR, '24.orthoaxis_cca_weights_human.tsv')
OUT_CCA_WMOUSE    = os.path.join(OUT_RES_DIR, '24.orthoaxis_cca_weights_mouse.tsv')
OUT_CCA_GENES_TSV = os.path.join(OUT_RES_DIR, '24.orthoaxis_cca_top_genes.tsv')
OUT_SCATTER_HTML  = os.path.join(OUT_FIG_DIR, '24.orthoaxis_cca_cell_scatter.html')

# --- parameters ---
CELL_TYPE_R2_THRESHOLD = 0.1
HUMAN_VX_COLS          = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
MOUSE_VX_COLS          = ['VX1', 'VX2', 'VX6', 'VX7', 'VX8', 'VX10']
HUMAN_CLUSTER_COL      = 'WithinArea_cluster'
MOUSE_CLUSTER_COL      = 'Type'
N_TOP_GENES            = 20
N_DOWNSAMPLE_SCATTER   = 3000

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load loadings ---
human_loadings = pd.read_csv(IN_HUMAN_LOADINGS, sep='\t', index_col=0)
mouse_loadings = pd.read_csv(IN_MOUSE_LOADINGS, sep='\t', index_col=0)

# --- load cell coords ---
human_coords = pd.read_csv(IN_HUMAN_COORDS, sep='\t', index_col=0)
mouse_coords = pd.read_csv(IN_MOUSE_COORDS, sep='\t', index_col=0)

# --- ortholog matching: keep 1-to-1 pairs present in both loadings ---
orthologs = pd.read_csv(IN_ORTHOLOGS, sep='\t')
orthologs = orthologs.drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
shared = orthologs[
    orthologs['human_symbol'].isin(human_loadings.index) &
    orthologs['mouse_symbol'].isin(mouse_loadings.index)
].reset_index(drop=True)

print(f'Shared ortholog genes: {len(shared)}')

# --- build z-scored loading matrices (genes × VX) ---
human_shared = human_loadings.loc[shared['human_symbol'].values, HUMAN_VX_COLS].reset_index(drop=True)
mouse_shared = mouse_loadings.loc[shared['mouse_symbol'].values, MOUSE_VX_COLS].reset_index(drop=True)

X = zscore(human_shared.values, axis=0)   # n_genes × n_human_vx
Y = zscore(mouse_shared.values, axis=0)   # n_genes × n_mouse_vx

n_components = min(len(HUMAN_VX_COLS), len(MOUSE_VX_COLS))

# --- fit CCA ---
print(f'Fitting CCA with {n_components} components on {len(shared)} shared genes...')
cca = CCA(n_components=n_components, max_iter=1000)
cca.fit(X, Y)

X_c, Y_c = cca.transform(X, Y)   # each: n_genes × n_components

# --- canonical correlations ---
cca_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
cca_dim_names = [f'CCA{i+1}' for i in range(n_components)]

cca_corr_df = pd.DataFrame({
    'cca_dim': cca_dim_names,
    'canonical_correlation': cca_corrs,
})
cca_corr_df.to_csv(OUT_CCA_CORR_TSV, sep='\t', index=False)
print(f'Saved CCA correlations: {OUT_CCA_CORR_TSV}')
for dim, r in zip(cca_dim_names, cca_corrs):
    print(f'  {dim}: r = {r:.4f}')

# --- CCA weights ---
human_weights_df = pd.DataFrame(
    cca.x_weights_,
    index=HUMAN_VX_COLS,
    columns=cca_dim_names,
)
mouse_weights_df = pd.DataFrame(
    cca.y_weights_,
    index=MOUSE_VX_COLS,
    columns=cca_dim_names,
)
human_weights_df.to_csv(OUT_CCA_WHUMAN, sep='\t')
mouse_weights_df.to_csv(OUT_CCA_WMOUSE, sep='\t')
print(f'Saved human weights: {OUT_CCA_WHUMAN}')
print(f'Saved mouse weights: {OUT_CCA_WMOUSE}')

# --- top genes per CCA axis (ranked by mean |score| across both species) ---
# gene CCA scores: back-project gene loadings through CCA weights
human_gene_cca = X @ cca.x_weights_   # n_genes × n_components
mouse_gene_cca = Y @ cca.y_weights_   # n_genes × n_components

top_genes_records = []
for i, dim in enumerate(cca_dim_names):
    mean_abs = (np.abs(human_gene_cca[:, i]) + np.abs(mouse_gene_cca[:, i])) / 2
    order = np.argsort(mean_abs)[::-1][:N_TOP_GENES]
    gene_df = pd.DataFrame({
        'cca_dim': dim,
        'canonical_correlation': cca_corrs[i],
        'human_gene': shared.loc[order, 'human_symbol'].values,
        'mouse_gene': shared.loc[order, 'mouse_symbol'].values,
        'human_gene_score': human_gene_cca[order, i],
        'mouse_gene_score': mouse_gene_cca[order, i],
        'mean_abs_score': mean_abs[order],
    })
    top_genes_records.append(gene_df)

top_genes_df = pd.concat(top_genes_records, ignore_index=True)
top_genes_df.to_csv(OUT_CCA_GENES_TSV, sep='\t', index=False)
print(f'Saved top CCA genes: {OUT_CCA_GENES_TSV}')

# --- project cells onto CCA axes ---
# human cells: VX scores @ CCA x_weights
human_vx_mat = human_coords[HUMAN_VX_COLS].values
mouse_vx_mat = mouse_coords[MOUSE_VX_COLS].values

# z-score VX scores per column (same normalization as gene matrix)
human_vx_z = zscore(human_vx_mat, axis=0)
mouse_vx_z = zscore(mouse_vx_mat, axis=0)

human_cca_cells = human_vx_z @ cca.x_weights_   # n_human_cells × n_components
mouse_cca_cells = mouse_vx_z @ cca.y_weights_    # n_mouse_cells × n_components

# --- joint scatter: one subplot per consecutive CCA pair (CCA1v2, CCA2v3, CCA3v4, ...) ---
human_types = human_coords[HUMAN_CLUSTER_COL].values
mouse_types = mouse_coords[MOUSE_CLUSTER_COL].values

def _color_map(labels):
    import matplotlib.pyplot as plt
    unique = sorted(set(str(l) for l in labels))
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {v: cycle[i % len(cycle)] for i, v in enumerate(unique)}

human_cmap = _color_map(human_types)
mouse_cmap = _color_map(mouse_types)

rng = np.random.default_rng(42)
h_idx = rng.choice(len(human_coords), min(N_DOWNSAMPLE_SCATTER, len(human_coords)), replace=False)
m_idx = rng.choice(len(mouse_coords), min(N_DOWNSAMPLE_SCATTER, len(mouse_coords)), replace=False)

# consecutive pairs: (CCA1,CCA2), (CCA2,CCA3), ..., (CCA{n-1},CCA{n})
cca_pairs = [(i, i + 1) for i in range(n_components - 1)]
n_panels = len(cca_pairs)

fig = make_subplots(rows=1, cols=n_panels,
                    subplot_titles=[f'CCA{a+1} vs CCA{b+1}' for a, b in cca_pairs])

for col, (ci, cj) in enumerate(cca_pairs, start=1):
    show_legend = (col == 1)
    for label in sorted(set(str(l) for l in human_types[h_idx])):
        mask = np.array([str(l) == label for l in human_types[h_idx]])
        fig.add_trace(go.Scatter(
            x=human_cca_cells[h_idx[mask], ci],
            y=human_cca_cells[h_idx[mask], cj],
            mode='markers',
            marker=dict(color=human_cmap[label], size=4, opacity=0.6, symbol='circle'),
            name=f'H: {label}',
            legendgroup=f'human_{label}',
            showlegend=show_legend,
            text=[label] * mask.sum(),
        ), row=1, col=col)
    for label in sorted(set(str(l) for l in mouse_types[m_idx])):
        mask = np.array([str(l) == label for l in mouse_types[m_idx]])
        fig.add_trace(go.Scatter(
            x=mouse_cca_cells[m_idx[mask], ci],
            y=mouse_cca_cells[m_idx[mask], cj],
            mode='markers',
            marker=dict(color=mouse_cmap[label], size=4, opacity=0.6, symbol='diamond'),
            name=f'M: {label}',
            legendgroup=f'mouse_{label}',
            showlegend=show_legend,
            text=[label] * mask.sum(),
        ), row=1, col=col)
    ri = cca_corrs[ci]
    rj = cca_corrs[cj]
    fig.update_xaxes(title_text=f'CCA{ci+1} (r={ri:.3f})', row=1, col=col)
    fig.update_yaxes(title_text=f'CCA{cj+1} (r={rj:.3f})', row=1, col=col)

fig.update_layout(
    title='CCA joint cell scatter: Human (circle) vs Mouse (diamond)',
    width=500 * n_panels,
    height=600,
)
fig.write_html(OUT_SCATTER_HTML)
print(f'Saved CCA cell scatter: {OUT_SCATTER_HTML}')

# --- sanity check ---
if cca_corrs[0] > 0:
    print(f'\nCHECK PASSED: CCA1 canonical correlation = {cca_corrs[0]:.4f} > 0')
else:
    print(f'\nWARNING: CCA1 canonical correlation = {cca_corrs[0]:.4f} — unexpected')
