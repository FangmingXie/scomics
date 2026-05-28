"""Cross-species archetype conservation: mouse ABC vs human ABCD.

For each (mouse archetype, human archetype) pair, computes:
  1. Gene-overlap statistics via 1-to-1 orthologs: Jaccard + hypergeometric p-value
  2. Spearman correlation of log2FC profiles across all shared HVG orthologs

Produces a 3x4 similarity heatmap (three metrics) and per-pair conservation scatter.

Reads:
  local_data/res/l23_evo/21.mouse_archetype_markers.tsv
  local_data/res/l23_evo/25.human_archetype_markers.tsv
  data/human_mouse_orthologs.tsv
  local_data/res/l23_evo/18.mouse_varimax_loadings.tsv   (mouse HVG list)
  local_data/res/l23_evo/01.pca_loadings.tsv             (human HVG list)
  local_data/res/l23_evo/21.mouse_pcha_xp.tsv
  local_data/res/l23_evo/21.mouse_pcha_aa.tsv
  local_data/res/l23_evo/25.human_pcha_xp.tsv
  local_data/res/l23_evo/25.human_pcha_aa.tsv
  links/l23_evo/cheng22_mouse_IT_P28.h5ad
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  local_data/res/l23_evo/26.archetype_jaccard.tsv
  local_data/res/l23_evo/26.archetype_pvals.tsv
  local_data/res/l23_evo/26.archetype_spearman.tsv
  local_data/res/l23_evo/26.archetype_gene_conservation.tsv
  local_data/fig/l23_evo/26.archetype_conservation_heatmap.html
  local_data/fig/l23_evo/26.archetype_conservation_scatter.html
"""

import os
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import spearmanr
import anndata as ad
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_MOUSE_MARKERS   = os.path.join(OUT_RES_DIR, '21.mouse_archetype_markers.tsv')
IN_HUMAN_MARKERS   = os.path.join(OUT_RES_DIR, '25.human_archetype_markers.tsv')
IN_ORTHOLOGS       = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_MOUSE_HVG       = os.path.join(OUT_RES_DIR, '18.mouse_varimax_loadings.tsv')
IN_HUMAN_HVG       = os.path.join(OUT_RES_DIR, '01.pca_loadings.tsv')
IN_MOUSE_PCHA_XP   = os.path.join(OUT_RES_DIR, '21.mouse_pcha_xp.tsv')
IN_MOUSE_PCHA_AA   = os.path.join(OUT_RES_DIR, '21.mouse_pcha_aa.tsv')
IN_HUMAN_PCHA_XP   = os.path.join(OUT_RES_DIR, '25.human_pcha_xp.tsv')
IN_HUMAN_PCHA_AA   = os.path.join(OUT_RES_DIR, '25.human_pcha_aa.tsv')
IN_MOUSE_H5AD      = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'cheng22_mouse_IT_P28.h5ad')
IN_HUMAN_H5AD      = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_JACCARD        = os.path.join(OUT_RES_DIR, '26.archetype_jaccard.tsv')
OUT_PVALS          = os.path.join(OUT_RES_DIR, '26.archetype_pvals.tsv')
OUT_CORR           = os.path.join(OUT_RES_DIR, '26.archetype_spearman.tsv')
OUT_GENES          = os.path.join(OUT_RES_DIR, '26.archetype_gene_conservation.tsv')
OUT_HEATMAP_HTML   = os.path.join(OUT_FIG_DIR, '26.archetype_conservation_heatmap.html')
OUT_SCATTER_HTML   = os.path.join(OUT_FIG_DIR, '26.archetype_conservation_scatter.html')

# --- parameters ---
MOUSE_ARCHETYPE_NAMES = ['A', 'B', 'C']
HUMAN_ARCHETYPE_NAMES = ['A', 'B', 'C', 'D']
MOUSE_SUBCLASS        = 'L2/3'
N_TOP_CELLS           = 300

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

n_mouse = len(MOUSE_ARCHETYPE_NAMES)
n_human = len(HUMAN_ARCHETYPE_NAMES)
mouse_labels = [f'mouse_{n}' for n in MOUSE_ARCHETYPE_NAMES]
human_labels = [f'human_{n}' for n in HUMAN_ARCHETYPE_NAMES]

# --- load markers and orthologs ---
mouse_df = pd.read_csv(IN_MOUSE_MARKERS, sep='\t')
human_df = pd.read_csv(IN_HUMAN_MARKERS, sep='\t')

ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol')
         .drop_duplicates('mouse_symbol'))
mouse_to_human = dict(zip(ortho['mouse_symbol'], ortho['human_symbol']))
human_to_mouse = dict(zip(ortho['human_symbol'], ortho['mouse_symbol']))
N_ORTHOLOGS = len(ortho)
print(f'Orthologs: {N_ORTHOLOGS} 1-to-1 pairs')

mouse_sets   = {}
mouse_log2fc = {}
for k, name in enumerate(MOUSE_ARCHETYPE_NAMES):
    sub = mouse_df[mouse_df['archetype'] == f'archetype_{k+1}']
    mouse_sets[k]   = set(sub['gene'])
    mouse_log2fc[k] = dict(zip(sub['gene'], sub['log2FC']))

human_sets   = {}
human_log2fc = {}
for j, name in enumerate(HUMAN_ARCHETYPE_NAMES):
    sub = human_df[human_df['archetype'] == f'archetype_{j+1}']
    human_sets[j]   = set(sub['gene'])
    human_log2fc[j] = dict(zip(sub['gene'], sub['log2FC']))

print('Mouse archetype sizes:', {MOUSE_ARCHETYPE_NAMES[k]: len(v) for k, v in mouse_sets.items()})
print('Human archetype sizes:', {HUMAN_ARCHETYPE_NAMES[j]: len(v) for j, v in human_sets.items()})

# --- 3x4 overlap statistics (Jaccard + hypergeometric) ---
jaccard_mat  = np.zeros((n_mouse, n_human))
pval_mat     = np.zeros((n_mouse, n_human))
n_shared_mat = np.zeros((n_mouse, n_human), dtype=int)

for k in range(n_mouse):
    m_orth = {mouse_to_human[g] for g in mouse_sets[k] if g in mouse_to_human}
    K = len(m_orth)
    for j in range(n_human):
        h_set    = human_sets[j]
        n_draw   = len(h_set)
        shared   = m_orth & h_set
        n_shared = len(shared)
        union    = m_orth | h_set
        jaccard_mat[k, j]  = n_shared / len(union) if union else 0.0
        pval_mat[k, j]     = scipy.stats.hypergeom.sf(n_shared - 1, N_ORTHOLOGS, K, n_draw)
        n_shared_mat[k, j] = n_shared
        print(f'  mouse {MOUSE_ARCHETYPE_NAMES[k]} vs human {HUMAN_ARCHETYPE_NAMES[j]}: '
              f'shared={n_shared}, J={jaccard_mat[k,j]:.3f}, p={pval_mat[k,j]:.2e}')

pd.DataFrame(jaccard_mat, index=mouse_labels, columns=human_labels).to_csv(OUT_JACCARD, sep='\t')
pd.DataFrame(pval_mat,    index=mouse_labels, columns=human_labels).to_csv(OUT_PVALS,   sep='\t')
print(f'Saved {OUT_JACCARD} and {OUT_PVALS}')

# --- per-pair gene conservation table ---
rows = []
for k in range(n_mouse):
    for j in range(n_human):
        for mg in mouse_sets[k]:
            hg = mouse_to_human.get(mg)
            if hg is None:
                continue
            status = 'conserved' if hg in human_sets[j] else 'mouse_only'
            rows.append({'mouse_arch': MOUSE_ARCHETYPE_NAMES[k], 'human_arch': HUMAN_ARCHETYPE_NAMES[j],
                         'mouse_gene': mg, 'human_gene': hg, 'status': status,
                         'mouse_log2FC': mouse_log2fc[k].get(mg, np.nan),
                         'human_log2FC': human_log2fc[j].get(hg, np.nan)})
        for hg in human_sets[j]:
            mg = human_to_mouse.get(hg)
            if mg is None or mg in mouse_sets[k]:
                continue
            rows.append({'mouse_arch': MOUSE_ARCHETYPE_NAMES[k], 'human_arch': HUMAN_ARCHETYPE_NAMES[j],
                         'mouse_gene': mg, 'human_gene': hg, 'status': 'human_only',
                         'mouse_log2FC': np.nan,
                         'human_log2FC': human_log2fc[j].get(hg, np.nan)})

genes_df = pd.DataFrame(rows)
genes_df.to_csv(OUT_GENES, sep='\t', index=False)
print(f'Saved {OUT_GENES}  ({len(genes_df)} rows)')

# --- Spearman correlation of full log2FC profiles across all shared HVG orthologs ---
print('\nComputing Spearman correlation of full log2FC profiles...')

# find shared HVG orthologs
mouse_hvg = set(pd.read_csv(IN_MOUSE_HVG, sep='\t', index_col=0).index)
human_hvg = set(pd.read_csv(IN_HUMAN_HVG, sep='\t', index_col=0).index)
shared_ortho = ortho[ortho['mouse_symbol'].isin(mouse_hvg) & ortho['human_symbol'].isin(human_hvg)]
print(f'Shared HVG orthologs: {len(shared_ortho)}')

# reconstruct top_cells from saved PCHA results (distances in XP space)
mouse_xp = pd.read_csv(IN_MOUSE_PCHA_XP, sep='\t', index_col=0)
mouse_aa = pd.read_csv(IN_MOUSE_PCHA_AA, sep='\t', index_col=0).values   # (3, 5)
dists_m  = np.stack([np.linalg.norm(mouse_xp.values - mouse_aa[k], axis=1)
                     for k in range(n_mouse)], axis=1)
top_cells_mouse = [np.argsort(dists_m[:, k])[:N_TOP_CELLS] for k in range(n_mouse)]

human_xp = pd.read_csv(IN_HUMAN_PCHA_XP, sep='\t', index_col=0)
human_aa = pd.read_csv(IN_HUMAN_PCHA_AA, sep='\t', index_col=0).values   # (4, 5)
dists_h  = np.stack([np.linalg.norm(human_xp.values - human_aa[k], axis=1)
                     for k in range(n_human)], axis=1)
top_cells_human = [np.argsort(dists_h[:, k])[:N_TOP_CELLS] for k in range(n_human)]

# load mouse expression, aligned to XP order, filtered to shared ortholog genes
print('Loading mouse h5ad...')
m_adata = ad.read_h5ad(IN_MOUSE_H5AD)
m_adata = m_adata[m_adata.obs['Subclass'] == MOUSE_SUBCLASS]
m_adata = m_adata[mouse_xp.index]
m_gene_names = m_adata.var_names.values
m_idx = np.where(np.isin(m_gene_names, shared_ortho['mouse_symbol'].values))[0]
X_mouse_raw = m_adata.X[:, m_idx].toarray().astype(np.float32)
depths = X_mouse_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_mouse = np.log2(X_mouse_raw / depths * 1e4 + 1)
m_genes_used = m_gene_names[m_idx]

# load human expression, aligned to XP order, filtered to shared ortholog genes
print('Loading human h5ad...')
h_adata = ad.read_h5ad(IN_HUMAN_H5AD)
h_adata = h_adata[human_xp.index]
h_gene_names = (h_adata.var['feature_name'].values
                if 'feature_name' in h_adata.var.columns
                else h_adata.var_names.values)
h_idx = np.where(np.isin(h_gene_names, shared_ortho['human_symbol'].values))[0]
X_human = h_adata.X[:, h_idx].toarray().astype(np.float32)
h_genes_used = h_gene_names[h_idx]

# align mouse and human columns via ortholog pairs
m_gene_to_col = {g: i for i, g in enumerate(m_genes_used)}
h_gene_to_col = {g: i for i, g in enumerate(h_genes_used)}
m_to_h = dict(zip(shared_ortho['mouse_symbol'], shared_ortho['human_symbol']))

aligned_mouse = [g for g in m_genes_used if g in m_to_h and m_to_h[g] in h_gene_to_col]
aligned_human = [m_to_h[g] for g in aligned_mouse]
m_cols = [m_gene_to_col[g] for g in aligned_mouse]
h_cols = [h_gene_to_col[g] for g in aligned_human]
print(f'Aligned ortholog gene pairs: {len(aligned_mouse)}')

X_m = X_mouse[:, m_cols]   # (n_mouse_cells, n_genes)
X_h = X_human[:, h_cols]   # (n_human_cells, n_genes)

# compute mean log2FC per archetype: top_cells vs. all other cells
def archetype_log2fc(X, top_cells_list, noc):
    n_cells = X.shape[0]
    fc = np.zeros((noc, X.shape[1]))
    for k in range(noc):
        in_idx   = top_cells_list[k]
        out_mask = np.ones(n_cells, dtype=bool)
        out_mask[in_idx] = False
        fc[k] = (X[in_idx].mean(axis=0) - X[out_mask].mean(axis=0)) / np.log(2)
    return fc

mouse_fc = archetype_log2fc(X_m, top_cells_mouse, n_mouse)  # (3, n_genes)
human_fc = archetype_log2fc(X_h, top_cells_human, n_human)  # (4, n_genes)

corr_mat = np.zeros((n_mouse, n_human))
for k in range(n_mouse):
    for j in range(n_human):
        corr_mat[k, j], _ = spearmanr(mouse_fc[k], human_fc[j])

pd.DataFrame(corr_mat, index=mouse_labels, columns=human_labels).to_csv(OUT_CORR, sep='\t')
print(f'Saved {OUT_CORR}')
print('Spearman correlations:')
print(pd.DataFrame(corr_mat, index=MOUSE_ARCHETYPE_NAMES, columns=HUMAN_ARCHETYPE_NAMES).round(3))

# --- heatmap HTML: Jaccard | -log10(p) | Spearman ---
annot_text = [[
    f"J={jaccard_mat[k,j]:.2f}<br>p={pval_mat[k,j]:.1e}<br>n={n_shared_mat[k,j]}"
    for j in range(n_human)] for k in range(n_mouse)]
corr_annot = [[f"r={corr_mat[k,j]:.3f}" for j in range(n_human)] for k in range(n_mouse)]

panels = [
    (jaccard_mat,                    annot_text,  'Jaccard similarity',   'Jaccard'),
    (-np.log10(pval_mat + 1e-300),   annot_text,  '−log₁₀(p-value)',      '−log₁₀(p)'),
    (corr_mat,                       corr_annot,  'Spearman correlation',  'Spearman r'),
]
colorbar_xs = [0.27, 0.63, 1.0]

fig_hm = make_subplots(rows=1, cols=3,
                       subplot_titles=[p[2] for p in panels],
                       horizontal_spacing=0.12)

for col_idx, (zmat, ann, title, cblabel) in enumerate(panels, start=1):
    fig_hm.add_trace(go.Heatmap(
        z=zmat, x=HUMAN_ARCHETYPE_NAMES, y=MOUSE_ARCHETYPE_NAMES,
        colorscale='Blues' if col_idx < 3 else 'RdBu',
        zmid=0 if col_idx == 3 else None,
        colorbar=dict(title=cblabel, x=colorbar_xs[col_idx - 1], len=0.7),
        text=ann, texttemplate='%{text}', textfont=dict(size=10),
        showscale=True,
    ), row=1, col=col_idx)
    fig_hm.update_xaxes(title_text='Human archetype', row=1, col=col_idx)

fig_hm.update_yaxes(title_text='Mouse archetype', row=1, col=1)
fig_hm.update_layout(
    title='Mouse (ABC) vs Human (ABCD) archetype conservation',
    height=380, width=1200,
)
fig_hm.write_html(OUT_HEATMAP_HTML)
print(f'Saved {OUT_HEATMAP_HTML}')

# --- conservation scatter: one subplot per mouse archetype (best match by Spearman) ---
best_human = [int(np.argmax(corr_mat[k])) for k in range(n_mouse)]

STATUS_COLORS = {'conserved': '#e41a1c', 'mouse_only': '#aaaaaa'}

fig_sc = make_subplots(
    rows=1, cols=n_mouse,
    subplot_titles=[
        f'Mouse {MOUSE_ARCHETYPE_NAMES[k]} → Human {HUMAN_ARCHETYPE_NAMES[best_human[k]]}'
        f'  (r={corr_mat[k, best_human[k]]:.3f})'
        for k in range(n_mouse)
    ],
    horizontal_spacing=0.08,
)

for k in range(n_mouse):
    j = best_human[k]
    sub = genes_df[(genes_df['mouse_arch'] == MOUSE_ARCHETYPE_NAMES[k]) &
                   (genes_df['human_arch']  == HUMAN_ARCHETYPE_NAMES[j])  &
                   (genes_df['status'].isin(['conserved', 'mouse_only']))]

    for status, color in STATUS_COLORS.items():
        pts = sub[sub['status'] == status]
        fig_sc.add_trace(go.Scatter(
            x=pts['mouse_log2FC'], y=pts['human_log2FC'],
            mode='markers',
            marker=dict(color=color, size=7, opacity=0.8),
            name=status, showlegend=(k == 0),
            text=pts['mouse_gene'],
            hovertemplate='%{text}<br>mouse log2FC=%{x:.2f}<br>human log2FC=%{y:.2f}<extra></extra>',
        ), row=1, col=k + 1)

    fig_sc.update_xaxes(title_text='Mouse log2FC (marker set)', row=1, col=k + 1)
    fig_sc.update_yaxes(title_text='Human log2FC (marker set)', row=1, col=k + 1)

fig_sc.update_layout(
    title='Archetype marker gene conservation (best match by Spearman)',
    height=450, width=300 * n_mouse + 100,
)
fig_sc.write_html(OUT_SCATTER_HTML)
print(f'Saved {OUT_SCATTER_HTML}')
print('Done.')
