# Number of archetypes selection — Cheng 2022 dataset
# Outputs:
#   - metrics PNG with EV, ARV, ARV_rep, effective EV, effective EV_rep
#   - interactive HTML (3D per-sample overlay)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run_noc_sweep
from viz import save_metrics_plot, scatter_per_group_html

from SingleCellArchetype.main import SCA
from SingleCellArchetype.utils import norm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'local_data', 'processed', 'cheng22', 'cheng22.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cheng22')
FIG_NOC_METRICS = os.path.join(FIG_DIR, '04.num_archetype_cheng22_metrics.png')
FIG_NOC_ARCHETYPES_REP = os.path.join(FIG_DIR, '04.num_archetype_cheng22_interactive_rep.html')

NDIM = 10
NOC_MIN = 2
NOC_MAX = 6
NREPEATS = 10

# load data
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

x = np.array(adata.X.todense())
depths = adata.obs['n_counts'].values
types = adata.obs['Type'].values
samples = adata.obs['sample'].values

xn = norm(x, depths)
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

os.makedirs(FIG_DIR, exist_ok=True)
noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)

ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
    sca, noc_grid, NDIM, NREPEATS, samples)

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sample_to_color = {s: cycle[i % len(cycle)] for i, s in enumerate(np.unique(samples))}

save_metrics_plot(noc_grid, ev_grid, av_grid, av_rep_grid,
                  NDIM, f'Archetype selection (NDIM={NDIM})', FIG_NOC_METRICS)

scatter_per_group_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                        samples, sample_to_color,
                        f'Per-sample archetype overlay (NDIM={NDIM})',
                        FIG_NOC_ARCHETYPES_REP)
