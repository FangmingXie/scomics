"""Archetype markers for P56 gao25 gray-matter astrocytes — fixed NOC.

Based on script 35; uses 33.2 varimax results (no mt genes, Arch4 excluded).
Fits PCHA at analyst-chosen NOC on the VX subspace, assigns each archetype the
300 nearest cells (Euclidean in VX space), then runs Wilcoxon one-vs-rest
enrichment to find marker genes per archetype.
Saves inner PCA parameters needed for algebraic projection.

Reads:
  local_data/res/astro/33.2.varimax_coords.tsv
  local_data/res/astro/33.2.varimax_loadings.tsv  (HVG gene list)
  links/astro/gao25_scrna_astro.h5ad
Outputs:
  local_data/res/astro/35.2.pcha_xp.tsv
  local_data/res/astro/35.2.pcha_aa.tsv
  local_data/res/astro/35.2.inner_pca_components.tsv
  local_data/res/astro/35.2.inner_pca_mean.tsv
  local_data/res/astro/35.2.archetype_markers.tsv
  local_data/fig/astro/35.2.archetype_scatter.html
  local_data/fig/astro/35.2.vx_panels.html
  local_data/fig/astro/35.2.gene_expr_vx_panels.html
  local_data/fig/astro/35.2.pc_panels.html
  local_data/fig/astro/35.2.gene_expr_pc_panels.html
  local_data/fig/astro/35.2.umap.html
  local_data/fig/astro/35.2.gene_expr_umap.html
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html, gene_expr_scatter_html, scatter_2d_categorical_html
from scomics.main import SCA

# --- file paths ---
INPUT_H5AD          = os.path.join(PROJECT_ROOT, 'links', 'astro', 'gao25_scrna_astro.h5ad')
OUT_RES_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'astro')
OUT_FIG_DIR         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
IN_VARIMAX_COORDS   = os.path.join(OUT_RES_DIR, '33.2.varimax_coords.tsv')
IN_VARIMAX_LOADINGS = os.path.join(OUT_RES_DIR, '33.2.varimax_loadings.tsv')
OUT_XP              = os.path.join(OUT_RES_DIR, '35.2.pcha_xp.tsv')
OUT_AA              = os.path.join(OUT_RES_DIR, '35.2.pcha_aa.tsv')
OUT_INNER_COMPS     = os.path.join(OUT_RES_DIR, '35.2.inner_pca_components.tsv')
OUT_INNER_MEAN      = os.path.join(OUT_RES_DIR, '35.2.inner_pca_mean.tsv')
OUT_MARKERS         = os.path.join(OUT_RES_DIR, '35.2.archetype_markers.tsv')
OUT_SCATTER_HTML    = os.path.join(OUT_FIG_DIR, '35.2.archetype_scatter.html')
OUT_VX_PANELS_HTML  = os.path.join(OUT_FIG_DIR, '35.2.vx_panels.html')
OUT_GENE_EXPR_HTML  = os.path.join(OUT_FIG_DIR, '35.2.gene_expr_vx_panels.html')
OUT_PC_PANELS_HTML  = os.path.join(OUT_FIG_DIR, '35.2.pc_panels.html')
OUT_GENE_EXPR_PC_HTML = os.path.join(OUT_FIG_DIR, '35.2.gene_expr_pc_panels.html')
OUT_UMAP_HTML         = os.path.join(OUT_FIG_DIR, '35.2.umap.html')
OUT_GENE_EXPR_UMAP_HTML = os.path.join(OUT_FIG_DIR, '35.2.gene_expr_umap.html')

# analyst-set after inspecting scripts 33.2 and 34.2
VX_COLS        = ['VX3', 'VX4', 'VX6', 'VX7', 'VX9']
NOC            = 4
NDIM           = 4
N_TOP_CELLS    = 300
FRAC_IN_THRESH = 0.25
FDR_THRESH     = 0.001
P56_AGE        = 'P56'

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- load varimax coords (P56 GM only) ---
vx_df       = pd.read_csv(IN_VARIMAX_COORDS, sep='\t', index_col=0)
xn          = vx_df[VX_COLS].values
arch_labels = vx_df['archetype'].values
donors      = vx_df['donor_name'].values
print(f'Loaded VX coords: {xn.shape}')

# --- fit PCHA ---
print(f'Fitting PCHA: NOC={NOC}, NDIM={NDIM}...')
sca = SCA(xn, arch_labels)
sca.setup_feature_matrix(method='data')
sca.proj_and_pcha(NDIM, NOC)

pd.DataFrame(sca.xp, index=vx_df.index,
             columns=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_XP, sep='\t')
pd.DataFrame(sca.aa.T, columns=[f'PC{i+1}' for i in range(NDIM)],
             index=[f'archetype_{k+1}' for k in range(NOC)]).to_csv(OUT_AA, sep='\t')
print(f'Saved {OUT_XP} and {OUT_AA}')

# --- save inner PCA parameters ---
inner_comps = sca.pca_.components_[:NDIM]   # (NDIM, n_vx_cols)
inner_mean  = sca.pca_.mean_                  # (n_vx_cols,)

pd.DataFrame(inner_comps, columns=VX_COLS,
             index=[f'PC{i+1}' for i in range(NDIM)]).to_csv(OUT_INNER_COMPS, sep='\t')
pd.DataFrame(inner_mean.reshape(1, -1), columns=VX_COLS).to_csv(OUT_INNER_MEAN, sep='\t')
print(f'Saved {OUT_INNER_COMPS} and {OUT_INNER_MEAN}')

# --- back-project archetype vertices to VX space ---
aa_vx = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_   # (NOC, n_vx_cols)
print(f'Archetype VX coords shape: {aa_vx.shape}')

# --- nearest cells per archetype ---
dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(NOC)], axis=1)
top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(NOC)]
print(f'Top {N_TOP_CELLS} cells selected per archetype.')

# --- load raw counts and normalize (GM cells only, aligned to vx_df) ---
print('Loading h5ad for marker gene analysis...')
adata      = ad.read_h5ad(INPUT_H5AD)
adata_p56  = adata[adata.obs['Age'] == P56_AGE].copy()
# align to GM cells in vx_df (excludes Arch4)
adata_p56  = adata_p56[vx_df.index].copy()
print(f'  GM P56 cells for markers: {adata_p56.shape[0]}')

hvg_genes  = pd.read_csv(IN_VARIMAX_LOADINGS, sep='\t', index_col=0).index.values
gene_names = adata_p56.var_names.values
hvg_idx    = np.where(np.isin(gene_names, hvg_genes))[0]
hvg_names  = gene_names[hvg_idx]

x_raw  = adata_p56.X[:, hvg_idx]
if sp.issparse(x_raw):
    x_raw = x_raw.toarray()
x_raw   = x_raw.astype(np.float32)
depths  = x_raw.sum(axis=1, keepdims=True)
depths  = np.where(depths == 0, 1, depths)
X_norm  = np.log2(x_raw / depths * 1e4 + 1)   # log2(CP10k + 1)
print(f'X_norm shape: {X_norm.shape}')

# --- Wilcoxon one-vs-rest per archetype ---
print('Running Wilcoxon enrichment per archetype...')
n_cells  = X_norm.shape[0]
n_hvg    = X_norm.shape[1]
all_markers = []

for k in range(NOC):
    archetype_label = f'archetype_{k + 1}'
    print(f'  {archetype_label}...')
    in_idx   = top_cells[k]
    out_mask = np.ones(n_cells, dtype=bool)
    out_mask[in_idx] = False

    X_in  = X_norm[in_idx]
    X_out = X_norm[out_mask]

    log2fc   = X_in.mean(axis=0) - X_out.mean(axis=0)
    frac_in  = (X_in  > 0).mean(axis=0)
    frac_out = (X_out > 0).mean(axis=0)

    pvals = np.empty(n_hvg)
    for g in range(n_hvg):
        _, pvals[g] = scipy.stats.ranksums(X_in[:, g], X_out[:, g])

    _, fdr, _, _ = multipletests(pvals, method='fdr_bh')

    df = pd.DataFrame({
        'gene':      hvg_names,
        'archetype': archetype_label,
        'log2FC':    log2fc,
        'pval':      pvals,
        'fdr':       fdr,
        'frac_in':   frac_in,
        'frac_out':  frac_out,
    })
    df = df[(df['frac_in'] >= FRAC_IN_THRESH) & (df['fdr'] < FDR_THRESH)]
    df = df.sort_values('log2FC', ascending=False)
    all_markers.append(df)
    print(f'    {len(df)} markers after filtering')

markers_df = pd.concat(all_markers, ignore_index=True)
markers_df.to_csv(OUT_MARKERS, sep='\t', index=False)
print(f'Saved {OUT_MARKERS}  ({len(markers_df)} total markers)')

# --- scatter visualization ---
xp_grid = [sca.xp]
aa      = sca.aa     # (NDIM, NOC)

scatter_categorical_html(
    xp_grid=xp_grid,
    cell_metadata={'archetype': arch_labels, 'donor_name': donors},
    title=f'P56 gao25 gray-matter astrocytes — PCHA archetype space (NOC={NOC}, NDIM={NDIM})',
    out_path=OUT_SCATTER_HTML,
    noc_grid=np.array([NOC]),
    aa_grid=[aa],
)
print(f'Saved {OUT_SCATTER_HTML}')

# --- multi-panel VX scatter: VX3 vs VX4/6/7/9 ---
PANEL_PAIRS = [('VX3', 'VX4'), ('VX3', 'VX6'), ('VX3', 'VX7'), ('VX3', 'VX9')]
import matplotlib.pyplot as plt
cat_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
unique_archetypes = sorted(set(arch_labels))
arch_color = {a: cat_palette[i % len(cat_palette)] for i, a in enumerate(unique_archetypes)}

fig_vx = make_subplots(
    rows=2, cols=4,
    specs=[
        [{'type': 'xy'}] * 4,
        [{'type': 'scene', 'colspan': 4}, None, None, None],
    ],
    subplot_titles=[f'{x} vs {y}' for x, y in PANEL_PAIRS] + ['3D  VX3–VX4–VX6'],
    row_heights=[0.4, 0.6],
)

for panel_idx, (xcol, ycol) in enumerate(PANEL_PAIRS):
    xvals = vx_df[xcol].values
    yvals = vx_df[ycol].values
    for arch in unique_archetypes:
        mask = arch_labels == arch
        fig_vx.add_trace(go.Scatter(
            x=xvals[mask], y=yvals[mask], mode='markers',
            marker=dict(size=3, color=arch_color[arch], opacity=0.5),
            name=arch, legendgroup=arch, showlegend=(panel_idx == 0),
        ), row=1, col=panel_idx + 1)
    fig_vx.update_xaxes(title_text=xcol, row=1, col=panel_idx + 1)
    fig_vx.update_yaxes(title_text=ycol, row=1, col=panel_idx + 1)

for arch in unique_archetypes:
    mask = arch_labels == arch
    fig_vx.add_trace(go.Scatter3d(
        x=vx_df['VX3'].values[mask],
        y=vx_df['VX4'].values[mask],
        z=vx_df['VX6'].values[mask],
        mode='markers',
        marker=dict(size=2, color=arch_color[arch], opacity=0.5),
        name=arch, legendgroup=arch, showlegend=False,
    ), row=2, col=1)

fig_vx.update_layout(
    title='P56 GM astrocytes — VX pairwise scatter (Arch1-3)',
    scene=dict(xaxis_title='VX3', yaxis_title='VX4', zaxis_title='VX6',
               dragmode='orbit'),
    width=1800, height=1000,
    legend=dict(itemsizing='constant'),
)
fig_vx.write_html(OUT_VX_PANELS_HTML)
print(f'Saved {OUT_VX_PANELS_HTML}')

# --- gene expression scatter: top 5 |loading| genes per VX component ---
vx_load_df = pd.read_csv(IN_VARIMAX_LOADINGS, sep='\t', index_col=0)
top_genes_ordered = list(dict.fromkeys(
    gene
    for vx_col in VX_COLS
    for gene in vx_load_df[vx_col].abs().nlargest(5).index.tolist()
))

gene_to_hidx = {g: i for i, g in enumerate(hvg_names)}
from scipy.stats import zscore as _zscore
gene_vals = {}
for gene in top_genes_ordered:
    if gene in gene_to_hidx:
        expr = X_norm[:, gene_to_hidx[gene]]
        gene_vals[gene] = _zscore(expr)

# top 5 marker genes per PCHA archetype (Arch1–3)
for arch_name in ['archetype_1', 'archetype_2', 'archetype_3']:
    for gene in markers_df[markers_df['archetype'] == arch_name]['gene'].head(5):
        if gene not in gene_vals and gene in gene_to_hidx:
            gene_vals[gene] = _zscore(X_norm[:, gene_to_hidx[gene]])

# VX_COLS indices: VX3=0, VX4=1, VX6=2, VX7=3, VX9=4
panels_vx = [
    (0, 1, 'VX3', 'VX4'),
    (0, 2, 'VX3', 'VX6'),
    (0, 3, 'VX3', 'VX7'),
    (0, 4, 'VX3', 'VX9'),
]

gene_expr_scatter_html(
    x=xn[:, 0], y=xn[:, 1],
    gene_vals=gene_vals,
    title='Top |loading| genes per VX — P56 GM astrocytes',
    out_path=OUT_GENE_EXPR_HTML,
    xp=xn,
    panels=panels_vx,
    panel_3d=(0, 1, 2, 'VX3', 'VX4', 'VX6'),
)

# --- PCA on VX components (5 PCs) ---
from sklearn.decomposition import PCA as _PCA
pca_vx = _PCA(n_components=5, random_state=0).fit(xn)
xp5 = pca_vx.transform(xn)   # (n_cells, 5)
pc_labels = [f'PC{i+1}' for i in range(5)]
pc_var_frac = pca_vx.explained_variance_ratio_ * 100

# --- PC panels (categorical coloring, 2×2 grid + 3D at bottom) ---
PC_PANEL_PAIRS = [(0, 1), (0, 2), (0, 3), (0, 4)]   # PC1 vs PC2/3/4/5

fig_pc = make_subplots(
    rows=3, cols=2,
    specs=[
        [{'type': 'xy'}, {'type': 'xy'}],
        [{'type': 'xy'}, {'type': 'xy'}],
        [{'type': 'scene', 'colspan': 2}, None],
    ],
    subplot_titles=[
        f'{pc_labels[cx]} vs {pc_labels[cy]}'
        f'  ({pc_var_frac[cx]:.1f}% / {pc_var_frac[cy]:.1f}%)'
        for cx, cy in PC_PANEL_PAIRS
    ] + ['3D  PC1–PC2–PC3'],
    row_heights=[0.33, 0.33, 0.44],
)

for panel_idx, (cx, cy) in enumerate(PC_PANEL_PAIRS):
    row, col = divmod(panel_idx, 2)
    row += 1; col += 1
    for arch in unique_archetypes:
        mask = arch_labels == arch
        fig_pc.add_trace(go.Scatter(
            x=xp5[mask, cx], y=xp5[mask, cy], mode='markers',
            marker=dict(size=3, color=arch_color[arch], opacity=0.5),
            name=arch, legendgroup=arch, showlegend=(panel_idx == 0),
        ), row=row, col=col)
    fig_pc.update_xaxes(title_text=pc_labels[cx], row=row, col=col)
    fig_pc.update_yaxes(title_text=pc_labels[cy], row=row, col=col)

for arch in unique_archetypes:
    mask = arch_labels == arch
    fig_pc.add_trace(go.Scatter3d(
        x=xp5[mask, 0], y=xp5[mask, 1], z=xp5[mask, 2], mode='markers',
        marker=dict(size=2, color=arch_color[arch], opacity=0.5),
        name=arch, legendgroup=arch, showlegend=False,
    ), row=3, col=1)

fig_pc.update_layout(
    title='P56 GM astrocytes — PCs from VX subspace (Arch1-3)',
    scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3',
               dragmode='orbit'),
    width=900, height=1300,
    legend=dict(itemsizing='constant'),
)
fig_pc.write_html(OUT_PC_PANELS_HTML)
print(f'Saved {OUT_PC_PANELS_HTML}')

# --- gene expression on PC panels (with 3D at bottom) ---
panels_pc = [(cx, cy, pc_labels[cx], pc_labels[cy]) for cx, cy in PC_PANEL_PAIRS]

gene_expr_scatter_html(
    x=xp5[:, 0], y=xp5[:, 1],
    gene_vals=gene_vals,
    title='Top |loading| genes per VX — PC subspace — P56 GM astrocytes',
    out_path=OUT_GENE_EXPR_PC_HTML,
    xp=xp5,
    panels=panels_pc,
    panel_3d=(0, 1, 2, 'PC1', 'PC2', 'PC3'),
)

# --- UMAP (2D) on 5 PCs ---
print('Computing UMAP on 5 PCs...')
import umap as _umap
umap_coords = _umap.UMAP(n_components=2, random_state=0).fit_transform(xp5)
print(f'  UMAP shape: {umap_coords.shape}')

scatter_2d_categorical_html(
    xp_grid=[umap_coords],
    cell_metadata={'archetype': arch_labels, 'donor_name': donors},
    title='UMAP — P56 GM astrocytes (5 PCs from VX subspace)',
    out_path=OUT_UMAP_HTML,
    xlabel='UMAP1', ylabel='UMAP2',
)

gene_expr_scatter_html(
    x=umap_coords[:, 0], y=umap_coords[:, 1],
    gene_vals=gene_vals,
    title='Top |loading| genes per VX — UMAP — P56 GM astrocytes',
    out_path=OUT_GENE_EXPR_UMAP_HTML,
    xlabel='UMAP1', ylabel='UMAP2',
)
print('Done.')
