# Archetype analysis — P56 astrocyte cells, cluster 5225_Astro-TE NN_3, PC2 dropped, NOC=2 (Gao et al. 2025)
# Finds top genes contributing to PC1 and PC3, and saves a gene expression scatter.
# Groups/replicates: donor_name
# Outputs:
#   - metrics PNG
#   - interactive HTML (per-donor archetype overlay)
#   - interactive HTML (2D + 3D PCA scatter colored by metadata)
#   - interactive HTML (PC1 vs PC3 scatter colored by top gene expression)

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
FIG_NOC_METRICS        = os.path.join(FIG_DIR, '10.archetype_p56_nn3_nopc2_noc2_metrics.png')
FIG_NOC_ARCHETYPES_REP = os.path.join(FIG_DIR, '10.archetype_p56_nn3_nopc2_noc2_interactive_rep.html')
FIG_NOC_PCA_CAT        = os.path.join(FIG_DIR, '10.archetype_p56_nn3_nopc2_noc2_pca_metadata.html')
FIG_GENE_SCATTER       = os.path.join(FIG_DIR, '10.archetype_p56_nn3_nopc2_noc2_gene_scatter.html')

P56_AGE_VAL   = 'P56'
CLUSTER_VAL   = '5225_Astro-TE NN_3'
NDIM          = 5
NOC           = 2
NREPEATS      = 10
N_TOP_GENES   = 2000
DROP_PCS      = [1]
N_TOP_LOADING = 20

# load and filter to P56, cluster NN_3
adata = ad.read_h5ad(INPUT_FILE)
adata = adata[(adata.obs['Age'] == P56_AGE_VAL) &
              (adata.obs['cluster_label'] == CLUSTER_VAL)].copy()
print(f'P56 {CLUSTER_VAL} cells: {adata.shape[0]}')

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
                  NDIM, f'Archetype selection P56 NN_3 (no PC2) NOC={NOC} (NDIM={NDIM})', FIG_NOC_METRICS)

scatter_per_group_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                       donors, donor_to_color,
                       f'Per-donor archetype overlay P56 NN_3 (no PC2) NOC={NOC} (NDIM={NDIM})',
                       FIG_NOC_ARCHETYPES_REP)

scatter_categorical_html(
    xp_grid=xp_grid,
    cell_metadata={'cluster_label': types, 'donor_name': donors},
    title=f'P56 astrocytes NN_3 (no PC2) NOC={NOC} — PCA scatter colored by metadata (NDIM={NDIM})',
    out_path=FIG_NOC_PCA_CAT,
    noc_grid=noc_grid, ev_grid=ev_grid, av_grid=av_grid, aa_grid=aa_grid,
)

# --- gene expression scatter ---
# With DROP_PCS=[1], pca fits components [0..5]; dropping index 1 leaves [0,2,3,4,5].
# xp[:, 0] = PC1 (pca component 0), xp[:, 1] = PC3 (pca component 2).
xp = xp_grid[0]
pc1_loadings = sca.pca_.components_[0]
pc3_loadings = sca.pca_.components_[2]

top_pc1_idx = np.argsort(np.abs(pc1_loadings))[::-1][:N_TOP_LOADING]
top_pc3_idx = np.argsort(np.abs(pc3_loadings))[::-1][:N_TOP_LOADING]

# union: PC1-top genes first, then PC3-top genes not already included
seen = set()
ordered_idx = []
for idx in list(top_pc1_idx) + list(top_pc3_idx):
    if idx not in seen:
        seen.add(idx)
        ordered_idx.append(idx)

gene_vals = {gene_names[i]: xn[:, i] for i in ordered_idx}

gene_expr_scatter_html(
    x=xp[:, 0], y=xp[:, 1],
    gene_vals=gene_vals,
    title=f'P56 NN_3 (no PC2) NOC={NOC} — gene expression',
    out_path=FIG_GENE_SCATTER,
    xlabel='PC1', ylabel='PC3',
)
