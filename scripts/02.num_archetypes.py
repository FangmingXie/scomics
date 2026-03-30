# Number of archetypes selection tutorial
# Runs PCHA with NDIM=10 and a range of NOC (2-6), computing explained variance
# and archetype relative variation (bootstrap stability) to identify optimal NOC.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import anndata as ad

from SingleCellArchetype.main import SCA
from SingleCellArchetype.utils import norm, plot_archetype, get_relative_variation

# infer project root from this script's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'data_snrna_v1.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig')
FIG_NOC_METRICS = os.path.join(FIG_DIR, '02.num_archetypes_metrics.png')
FIG_NOC_ARCHETYPES = os.path.join(FIG_DIR, '02.num_archetypes_triangles.png')

NDIM = 8
NOC_MIN = 2
NOC_MAX = 6
NREPEATS = 10

# load data
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

x = np.array(adata.X.todense())
depths = adata.obs['n_counts'].values
types = adata.obs['Type'].values

xn = norm(x, depths)
sca = SCA(xn, types)
sca.setup_feature_matrix(method='data')

noc_grid = np.arange(NOC_MIN, NOC_MAX + 1)
ev_grid = []
av_grid = []
xp_grid = []
aa_grid = []

for noc in noc_grid:
    xp, aa, ev = sca.proj_and_pcha(NDIM, noc)
    aa_boots = sca.bootstrap_proj_pcha(NDIM, noc, nrepeats=NREPEATS)
    av = get_relative_variation(aa_boots)
    print(f"NOC={noc}  EV={ev:.4f}  ARV={av:.4f}  effectiveEV={ev*(1-av):.4f}")
    ev_grid.append(ev)
    av_grid.append(av)
    xp_grid.append(xp)
    aa_grid.append(aa)

ev_grid = np.array(ev_grid)
av_grid = np.array(av_grid)

# --- plot: EV, ARV, effective EV vs NOC ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(noc_grid, ev_grid, '-o', label='explained variance (EV)')
ax.plot(noc_grid, av_grid, '-o', color='gray', label='archetype relative variation (ARV)')
ax.plot(noc_grid, ev_grid * (1 - av_grid), '-o', label='effective EV')
ax.set_xlabel('Number of archetypes (NOC)')
ax.set_ylabel('Score')
ax.set_ylim([0, 1])
ax.set_xticks(noc_grid)
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=8)
ax.set_title(f'Archetype selection (NDIM={NDIM})')
sns.despine(ax=ax)
ax.grid(False)

os.makedirs(FIG_DIR, exist_ok=True)
fig.tight_layout()
fig.savefig(FIG_NOC_METRICS, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {FIG_NOC_METRICS}")

# --- plot: PCA scatter + archetype triangles for each NOC (multi-panel) ---
ncols = len(noc_grid)
types_colorvec = np.char.add('C', sca.types_idx.astype(str))

fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
for i, noc in enumerate(noc_grid):
    ax = axes[0, i]
    ax.scatter(xp_grid[i][:, 0], xp_grid[i][:, 1], c=types_colorvec, s=1, zorder=0)
    plot_archetype(ax, aa_grid[i], fmt='-o', color='k', zorder=1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_aspect('equal')
    ax.set_title(f'NOC={noc}\nEV={ev_grid[i]:.3f}  ARV={av_grid[i]:.3f}')
    sns.despine(ax=ax)
    ax.grid(False)

fig.tight_layout()
fig.savefig(FIG_NOC_ARCHETYPES, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {FIG_NOC_ARCHETYPES}")
