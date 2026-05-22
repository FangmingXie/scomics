"""Cross-species VX subspace matching via Pearson correlation of loading vectors.

Compares human (Jorstad23) and mouse (Cheng22) varimax loading vectors in
shared ortholog gene space to identify corresponding VX axes across species.

Reads:
  local_data/res/l23_evo/05.varimax_loadings.tsv
  local_data/res/l23_evo/05.variance_partition.tsv
  local_data/res/l23_evo/05.varimax_coords.tsv
  local_data/res/l23_evo/18.mouse_varimax_loadings.tsv
  local_data/res/l23_evo/18.mouse_vx_variance_partition.tsv
  local_data/res/l23_evo/18.mouse_varimax_coords.tsv
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/res/l23_evo/23.orthoaxis_corr_matrix.tsv
  local_data/fig/l23_evo/23.orthoaxis_corr_heatmap.html
  local_data/res/l23_evo/23.orthoaxis_top_genes.tsv
  local_data/fig/l23_evo/23.orthoaxis_cell_scatter.html
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore
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
OUT_CORR_TSV      = os.path.join(OUT_RES_DIR, '23.orthoaxis_corr_matrix.tsv')
OUT_CORR_HTML     = os.path.join(OUT_FIG_DIR, '23.orthoaxis_corr_heatmap.html')
OUT_GENES_TSV     = os.path.join(OUT_RES_DIR, '23.orthoaxis_top_genes.tsv')
OUT_SCATTER_HTML  = os.path.join(OUT_FIG_DIR, '23.orthoaxis_cell_scatter.html')

# --- parameters ---
CELL_TYPE_R2_THRESHOLD = 0.1
HUMAN_VX_COLS          = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
MOUSE_VX_COLS          = ['VX1', 'VX2', 'VX6', 'VX7', 'VX8', 'VX10']
HUMAN_CLUSTER_COL      = 'WithinArea_cluster'
MOUSE_CLUSTER_COL      = 'Type'
N_TOP_GENES            = 20
TOP_K_PAIRS            = 3
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

# --- align loading matrices to shared gene set ---
human_shared = human_loadings.loc[shared['human_symbol'].values, HUMAN_VX_COLS]
mouse_shared = mouse_loadings.loc[shared['mouse_symbol'].values, MOUSE_VX_COLS]

# reset index so arrays align by position
human_shared = human_shared.reset_index(drop=True)
mouse_shared = mouse_shared.reset_index(drop=True)

# --- z-score each VX loading vector over shared gene set ---
human_z = pd.DataFrame(
    zscore(human_shared.values, axis=0),
    columns=HUMAN_VX_COLS,
)
mouse_z = pd.DataFrame(
    zscore(mouse_shared.values, axis=0),
    columns=MOUSE_VX_COLS,
)

# --- Pearson correlation matrix ---
corr_matrix = pd.DataFrame(index=HUMAN_VX_COLS, columns=MOUSE_VX_COLS, dtype=float)
for hvx in HUMAN_VX_COLS:
    for mvx in MOUSE_VX_COLS:
        r = np.corrcoef(human_z[hvx].values, mouse_z[mvx].values)[0, 1]
        corr_matrix.loc[hvx, mvx] = r

corr_matrix.to_csv(OUT_CORR_TSV, sep='\t')
print(f'Saved correlation matrix: {OUT_CORR_TSV}')
print(corr_matrix.to_string())

# --- PC correlation: PCA on each species' VX loading matrix in shared gene space ---
# SVD of z-scored loading matrices; left singular vectors are gene-space PCs
n_pcs = min(len(HUMAN_VX_COLS), len(MOUSE_VX_COLS))
Uh, _, _ = np.linalg.svd(human_z.values, full_matrices=False)   # n_genes × n_human_vx
Um, _, _ = np.linalg.svd(mouse_z.values, full_matrices=False)   # n_genes × n_mouse_vx
Uh = Uh[:, :n_pcs]
Um = Um[:, :n_pcs]

pc_names = [f'PC{i+1}' for i in range(n_pcs)]
pc_corr_matrix = pd.DataFrame(index=pc_names, columns=pc_names, dtype=float)
for i, hpc in enumerate(pc_names):
    for j, mpc in enumerate(pc_names):
        r = np.corrcoef(Uh[:, i], Um[:, j])[0, 1]
        pc_corr_matrix.loc[hpc, mpc] = r

# --- two-panel heatmap: VX correlation | PC correlation ---
abs_vals = corr_matrix.abs().values.astype(float)
vx_text  = [[f'{v:.2f}' for v in row] for row in abs_vals]

pc_abs_vals = pc_corr_matrix.abs().values.astype(float)
pc_text     = [[f'{v:.2f}' for v in row] for row in pc_abs_vals]

fig_heatmap = make_subplots(
    rows=1, cols=2,
    subplot_titles=['VX loading correlation', 'PC of VX subspace correlation'],
    horizontal_spacing=0.15,
)

fig_heatmap.add_trace(go.Heatmap(
    z=abs_vals,
    x=MOUSE_VX_COLS,
    y=HUMAN_VX_COLS,
    text=vx_text,
    texttemplate='%{text}',
    colorscale='Blues',
    zmin=abs_vals.min(),
    zmax=abs_vals.max(),
    colorbar=dict(title='|Pearson r|', x=0.42),
    showscale=True,
), row=1, col=1)

fig_heatmap.add_trace(go.Heatmap(
    z=pc_abs_vals,
    x=[f'M-{p}' for p in pc_names],
    y=[f'H-{p}' for p in pc_names],
    text=pc_text,
    texttemplate='%{text}',
    colorscale='Blues',
    zmin=pc_abs_vals.min(),
    zmax=pc_abs_vals.max(),
    colorbar=dict(title='|Pearson r|', x=1.0),
    showscale=True,
), row=1, col=2)

fig_heatmap.update_xaxes(title_text='Mouse VX (Cheng22)', row=1, col=1)
fig_heatmap.update_yaxes(title_text='Human VX (Jorstad23)', row=1, col=1)
fig_heatmap.update_xaxes(title_text='Mouse PC (Cheng22)', row=1, col=2)
fig_heatmap.update_yaxes(title_text='Human PC (Jorstad23)', row=1, col=2)
fig_heatmap.update_layout(
    title='Human vs Mouse VX loading correlation (shared orthologs, |r|)',
    width=1100,
    height=500,
)
fig_heatmap.write_html(OUT_CORR_HTML)
print(f'Saved heatmap: {OUT_CORR_HTML}')

# --- top genes per matched pair ---
# for each human VX, find best-matched mouse VX
best_matches = {}
for hvx in HUMAN_VX_COLS:
    best_mvx = corr_matrix.loc[hvx].abs().idxmax()
    r = corr_matrix.loc[hvx, best_mvx]
    best_matches[hvx] = (best_mvx, r)

top_genes_records = []
for hvx, (mvx, r) in best_matches.items():
    product = human_shared[hvx].abs().values * mouse_shared[mvx].abs().values
    order = np.argsort(product)[::-1][:N_TOP_GENES]
    gene_df = pd.DataFrame({
        'human_vx': hvx,
        'mouse_vx': mvx,
        'corr': r,
        'human_gene': shared.loc[order, 'human_symbol'].values,
        'mouse_gene': shared.loc[order, 'mouse_symbol'].values,
        'human_loading': human_shared[hvx].values[order],
        'mouse_loading': mouse_shared[mvx].values[order],
        'loading_product': product[order],
    })
    top_genes_records.append(gene_df)

top_genes_df = pd.concat(top_genes_records, ignore_index=True)
top_genes_df.to_csv(OUT_GENES_TSV, sep='\t', index=False)
print(f'Saved top genes: {OUT_GENES_TSV}')

# --- cell scatter: top-k matched pairs ---
# For each matched pair ranked by |r|, show human and mouse cell VX score distributions
# Both z-score normalized, colored by cell type, plotted on same x-axis range

# pick top-k pairs by |r| (deduplicated on mouse VX side)
pair_scores = [(hvx, mvx, abs(r)) for hvx, (mvx, r) in best_matches.items()]
pair_scores.sort(key=lambda x: -x[2])
seen_mvx = set()
top_pairs = []
for hvx, mvx, absr in pair_scores:
    if mvx not in seen_mvx:
        top_pairs.append((hvx, mvx, best_matches[hvx][1]))
        seen_mvx.add(mvx)
    if len(top_pairs) == TOP_K_PAIRS:
        break

# build color maps per species
def _color_map(labels):
    import matplotlib.pyplot as plt
    unique = sorted(set(str(l) for l in labels))
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {v: cycle[i % len(cycle)] for i, v in enumerate(unique)}

human_types = human_coords[HUMAN_CLUSTER_COL].values
mouse_types = mouse_coords[MOUSE_CLUSTER_COL].values
human_cmap = _color_map(human_types)
mouse_cmap = _color_map(mouse_types)

# downsample for plotting
rng = np.random.default_rng(42)
h_idx = rng.choice(len(human_coords), min(N_DOWNSAMPLE_SCATTER, len(human_coords)), replace=False)
m_idx = rng.choice(len(mouse_coords), min(N_DOWNSAMPLE_SCATTER, len(mouse_coords)), replace=False)

n_pairs = len(top_pairs)
fig_scatter = make_subplots(
    rows=n_pairs, cols=2,
    subplot_titles=[title for p in top_pairs for title in
                    [f'Human {p[0]} (r={p[2]:.2f})', f'Mouse {p[1]} (r={p[2]:.2f})']],
    shared_xaxes='columns',
)

for row, (hvx, mvx, r) in enumerate(top_pairs, start=1):
    # z-score VX scores across all cells per species
    h_scores = zscore(human_coords[hvx].values)[h_idx]
    m_scores = zscore(mouse_coords[mvx].values)[m_idx]
    h_labels = human_types[h_idx]
    m_labels = mouse_types[m_idx]
    h_jitter = rng.uniform(-0.4, 0.4, len(h_idx))
    m_jitter = rng.uniform(-0.4, 0.4, len(m_idx))

    for label in sorted(set(str(l) for l in h_labels)):
        mask = np.array([str(l) == label for l in h_labels])
        fig_scatter.add_trace(go.Scatter(
            x=h_scores[mask],
            y=h_jitter[mask],
            mode='markers',
            marker=dict(color=human_cmap[label], size=3, opacity=0.5),
            name=label,
            legendgroup=f'human_{label}',
            showlegend=(row == 1),
            text=[label] * mask.sum(),
        ), row=row, col=1)

    for label in sorted(set(str(l) for l in m_labels)):
        mask = np.array([str(l) == label for l in m_labels])
        fig_scatter.add_trace(go.Scatter(
            x=m_scores[mask],
            y=m_jitter[mask],
            mode='markers',
            marker=dict(color=mouse_cmap[label], size=3, opacity=0.5),
            name=label,
            legendgroup=f'mouse_{label}',
            showlegend=(row == 1),
            text=[label] * mask.sum(),
        ), row=row, col=2)

fig_scatter.update_yaxes(visible=False)
fig_scatter.update_layout(
    title='Cell VX scores: top matched human–mouse VX pairs (z-scored)',
    height=300 * n_pairs,
    width=900,
)
fig_scatter.write_html(OUT_SCATTER_HTML)
print(f'Saved cell scatter: {OUT_SCATTER_HTML}')

# --- sanity check ---
max_r = corr_matrix.abs().values.max()
print(f'\nMax |r| in correlation matrix: {max_r:.3f}')
if max_r < 0.3:
    print('WARNING: max |r| < 0.3 — shared signal between species may be weak')
else:
    print('CHECK PASSED: at least one cross-species VX pair with |r| > 0.3')
