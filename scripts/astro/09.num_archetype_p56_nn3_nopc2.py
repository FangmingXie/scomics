# Archetype number selection — P56 astrocyte cells, cluster 5225_Astro-TE NN_3, PC2 dropped (Gao et al. 2025)
# Groups/replicates: donor_name
# Outputs:
#   - metrics PNG with EV, ARV, ARV_rep, effective EV, effective EV_rep
#   - interactive HTML (per-donor archetype overlay)
#   - interactive HTML (2D + 3D PCA scatter colored by metadata)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import select_hvg, run_noc_sweep
from viz import save_metrics_plot, scatter_per_group_html, scatter_categorical_html

from scomics.main import SCA
from scomics.utils import norm

SCRIPTS_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT           = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE             = os.path.join(PROJECT_ROOT, 'local_data', 'source', 'gao25', 'DevVIS_scRNA_AstroTE.h5ad')
FIG_DIR                = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'astro')
FIG_NOC_METRICS        = os.path.join(FIG_DIR, '09.num_archetype_p56_nn3_nopc2_metrics.png')
FIG_NOC_ARCHETYPES_REP = os.path.join(FIG_DIR, '09.num_archetype_p56_nn3_nopc2_interactive_rep.html')
FIG_NOC_PCA_CAT        = os.path.join(FIG_DIR, '09.num_archetype_p56_nn3_nopc2_pca_metadata.html')

P56_AGE_VAL    = 'P56'
CLUSTER_VAL    = '5225_Astro-TE NN_3'
NDIM           = 5
NOC_MIN        = 2
NOC_MAX        = 5
NREPEATS       = 10
N_TOP_GENES    = 2000
DROP_PCS       = [1]

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
xn  = norm(x[:, hvg_mask], depths)
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

os.makedirs(FIG_DIR, exist_ok=True)
noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)

ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, donors, drop_pcs=DROP_PCS)

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
donor_to_color = {d: cycle[i % len(cycle)] for i, d in enumerate(np.unique(donors))}

save_metrics_plot(noc_grid, ev_grid, av_grid, av_rep_grid,
                  NDIM, f'Archetype selection P56 NN_3 (no PC2) (NDIM={NDIM})', FIG_NOC_METRICS)

scatter_per_group_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                       donors, donor_to_color,
                       f'Per-donor archetype overlay P56 NN_3 (no PC2) (NDIM={NDIM})',
                       FIG_NOC_ARCHETYPES_REP)

scatter_categorical_html(
    xp_grid=xp_grid,
    cell_metadata={'cluster_label': types, 'donor_name': donors},
    title=f'P56 astrocytes NN_3 (no PC2) — PCA scatter colored by metadata (NDIM={NDIM})',
    out_path=FIG_NOC_PCA_CAT,
    noc_grid=noc_grid, ev_grid=ev_grid, av_grid=av_grid, aa_grid=aa_grid,
)
