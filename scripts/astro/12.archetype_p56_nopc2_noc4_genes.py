# Archetype analysis — P56 astrocyte cells (all clusters), PC2 dropped, NOC=4 (Gao et al. 2025)
# Finds archetype-specific genes and saves a gene expression scatter.
# Groups/replicates: donor_name
# Outputs:
#   - metrics PNG
#   - interactive HTML (per-donor archetype overlay)
#   - interactive HTML (2D + 3D PCA scatter colored by metadata, including archetype assignment)
#   - interactive HTML (PC1 vs PC3 scatter colored by archetype-specific gene expression)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg, run_noc_sweep
from viz import save_metrics_plot, scatter_per_group_html, scatter_categorical_html, gene_expr_scatter_html

from scomics.main import SCA
from scomics.utils import norm

SCRIPTS_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT           = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE             = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')
FIG_DIR                = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
FIG_NOC_METRICS        = os.path.join(FIG_DIR, '12.archetype_p56_nopc2_noc4_metrics.png')
FIG_NOC_ARCHETYPES_REP = os.path.join(FIG_DIR, '12.archetype_p56_nopc2_noc4_interactive_rep.html')
FIG_NOC_PCA_CAT        = os.path.join(FIG_DIR, '12.archetype_p56_nopc2_noc4_pca_metadata.html')
FIG_GENE_SCATTER       = os.path.join(FIG_DIR, '12.archetype_p56_nopc2_noc4_gene_scatter.html')

P56_AGE_VAL       = 'P56'
NDIM              = 5
NOC               = 4
NREPEATS          = 10
N_TOP_GENES       = 2000
DROP_PCS          = [1]
N_ARCHETYPE_CELLS = 300
N_TOP_ARCHETYPE   = 5

# load and filter to P56
adata = ad.read_h5ad(INPUT_FILE)
adata = adata[adata.obs['Age'] == P56_AGE_VAL].copy()
print(f'P56 cells: {adata.shape[0]}')

x      = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
depths = x.sum(axis=1)
types  = adata.obs['cluster_label'].values
donors = adata.obs['donor_name'].values

hvg_mask = select_hvg(x, depths, N_TOP_GENES)
gene_names = np.array(adata.var_names)[hvg_mask]
xn  = norm(x[:, hvg_mask], depths)
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

os.makedirs(FIG_DIR, exist_ok=True)
noc_grid = np.array([NOC])

ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, donors, drop_pcs=DROP_PCS)

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
donor_to_color = {d: cycle[i % len(cycle)] for i, d in enumerate(np.unique(donors))}

save_metrics_plot(noc_grid, ev_grid, av_grid, av_rep_grid,
                  NDIM, f'Archetype selection P56 (no PC2) NOC={NOC} (NDIM={NDIM})', FIG_NOC_METRICS)

scatter_per_group_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                       donors, donor_to_color,
                       f'Per-donor archetype overlay P56 (no PC2) NOC={NOC} (NDIM={NDIM})',
                       FIG_NOC_ARCHETYPES_REP)

# --- nearest-archetype assignment per cell ---
# With DROP_PCS=[1], xp[:, 0]=PC1, xp[:, 1]=PC3, xp[:, 2]=PC4
xp = xp_grid[0]
aa = aa_grid[0]  # shape: (ndim, noc)

dists_all = np.stack([np.linalg.norm(xp - aa[:, k], axis=1) for k in range(NOC)], axis=1)
arch_assign = np.argmin(dists_all, axis=1)
arch_labels = np.array([f'Arch{k+1}' for k in arch_assign])

scatter_categorical_html(
    xp_grid=xp_grid,
    cell_metadata={'cluster_label': types, 'donor_name': donors, 'archetype': arch_labels},
    title=f'P56 astrocytes (no PC2) NOC={NOC} — PCA scatter colored by metadata (NDIM={NDIM})',
    out_path=FIG_NOC_PCA_CAT,
    noc_grid=noc_grid, ev_grid=ev_grid, av_grid=av_grid, aa_grid=aa_grid,
)

# --- archetype-specific genes ---
arch_specific_idx = []
for k in range(NOC):
    closest  = np.argsort(dists_all[:, k])[:N_ARCHETYPE_CELLS]
    rest     = np.argsort(dists_all[:, k])[N_ARCHETYPE_CELLS:]
    mean_diff = xn[closest].mean(axis=0) - xn[rest].mean(axis=0)
    top_idx  = np.argsort(mean_diff)[::-1][:N_TOP_ARCHETYPE]
    arch_specific_idx.append(top_idx)
    print(f'  Archetype {k+1} top genes: {list(gene_names[top_idx])}')

seen = set()
ordered_idx = []
for top_idx in arch_specific_idx:
    for idx in top_idx:
        if idx not in seen:
            seen.add(idx)
            ordered_idx.append(idx)

gene_vals = {gene_names[i]: xn[:, i] for i in ordered_idx}

gene_expr_scatter_html(
    x=xp[:, 0], y=xp[:, 1], z=xp[:, 2],
    gene_vals=gene_vals,
    aa=aa,
    title=f'P56 (no PC2) NOC={NOC} — archetype-specific gene expression',
    out_path=FIG_GENE_SCATTER,
    xlabel='PC1', ylabel='PC3', zlabel='PC4',
)
