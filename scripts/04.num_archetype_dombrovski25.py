# Number of archetypes selection — Dombrowski 2025 fly dataset
# Analyzes each (cell type × age) combination separately.
# Cell types: LC4, LPLC1, LPLC2 (cell type X excluded)
# Ages: APF_48h, APF_72h, APF_96h (from orig.ident)
# Replicates: genotype
# Outputs per (celltype, age):
#   - metrics PNG with EV, ARV, ARV_rep, effective EV, effective EV_rep
#   - interactive HTML (2D + 3D, cells colored by genotype)
#   - interactive HTML (3D per-genotype overlay)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

sys.path.insert(0, os.path.dirname(__file__))
from common import (select_hvg, run_noc_sweep, save_metrics_plot,
                    save_interactive_html_pro, save_group_overlay_html)

from SingleCellArchetype.main import SCA
from SingleCellArchetype.utils import norm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'local_data', 'processed', 'dombrowski25_fly', 'dombrowski25_fly.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'dombrowski25_fly')

CELLTYPES = ['LC4', 'LPLC1', 'LPLC2']  # exclude 'X'
N_TOP_GENES = 2000
NDIM = 10
NOC_MIN = 2
NOC_MAX = 6
NREPEATS = 10


def run_one(celltype, age, x_all, depths_all, obs_df_all, types_all, ages_all,
            noc_grid, ndim, nrepeats, fig_dir):
    """Run full archetype-number sweep for one (celltype, age) and save outputs."""
    mask = (types_all == celltype) & (ages_all == age)
    n_cells = mask.sum()
    print(f"\n=== celltype={celltype}  age={age}  n_cells={n_cells} ===")
    if n_cells == 0:
        print("  Skipping: no cells.")
        return

    x = x_all[mask]
    depths = depths_all[mask]
    obs_df = obs_df_all[mask]
    genotypes = obs_df['genotype'].values
    genotype_ids = np.unique(genotypes)

    # drop constant genes to avoid NaN in zscore
    x = x[:, x.var(axis=0) > 0]

    xn = norm(x, depths)
    sca = SCA(xn, genotypes)
    sca.setup_feature_matrix(method='data')

    ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid = run_noc_sweep(
        sca, noc_grid, ndim, nrepeats, genotypes)

    tag = f"{celltype}_{age}"
    fig_metrics = os.path.join(fig_dir, f"04.{tag}_metrics.png")
    fig_interactive = os.path.join(fig_dir, f"04.{tag}_interactive.html")
    fig_rep = os.path.join(fig_dir, f"04.{tag}_interactive_rep.html")

    save_metrics_plot(noc_grid, ev_grid, av_grid, av_rep_grid,
                      ndim, f'{celltype}  {age}  (NDIM={ndim})', fig_metrics)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    genotype_to_color = {g: cycle[i % len(cycle)] for i, g in enumerate(genotype_ids)}
    cell_metadata = {col: obs_df[col].values for col in obs_df.columns}

    save_interactive_html_pro(noc_grid, ev_grid, av_grid, xp_grid, aa_grid,
                              cell_metadata, ndim,
                              f'{celltype}  {age} — 2D & 3D view (NDIM={ndim})',
                              fig_interactive)

    save_group_overlay_html(noc_grid, ev_grid, av_rep_grid, xp_grid, aa_reps_grid,
                            genotypes, genotype_to_color, ndim,
                            f'{celltype}  {age} — per-genotype overlay (NDIM={ndim})',
                            fig_rep)


# --- main ---
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

x_raw = np.array(adata.X)
depths_raw = adata.obs['nCount_RNA'].values
hvg_mask = select_hvg(x_raw, depths_raw, N_TOP_GENES)

x_all = x_raw[:, hvg_mask]
depths_all = depths_raw
obs_df_all = adata.obs
types_all = adata.obs['type1'].values
ages_all = adata.obs['orig.ident'].values

os.makedirs(FIG_DIR, exist_ok=True)
noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)

for celltype in CELLTYPES:
    for age in np.unique(ages_all):
        run_one(celltype, age, x_all, depths_all, obs_df_all, types_all, ages_all,
                noc_grid, NDIM, NREPEATS, FIG_DIR)
