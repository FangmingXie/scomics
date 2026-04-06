# The minimum example code to run `SingleCellArchetype`

# import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')

import anndata as ad

from SingleCellArchetype.main import SCA
from SingleCellArchetype.utils import norm, plot_archetype

# infer project root from this script's location
# __file__                                     -> .../SingleCellArchetype/scripts/tutorial_minimum.py
# os.path.abspath(__file__)                    -> absolute path of this script
# os.path.dirname(...)                         -> .../SingleCellArchetype/scripts/
# os.path.dirname(os.path.dirname(...))        -> .../SingleCellArchetype/   (project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'data_snrna_v1.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig')

# load sample data
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

# set up the SingleCellArchetype object
x = np.array(adata.X.todense())
depths = adata.obs['n_counts'].values
types = adata.obs['Type'].values

xn = norm(x, depths)
sca = SCA(xn, types)

# Infer archetypes
ndim = 2
noc = 3
xp, aa, varexpl = sca.proj_and_pcha(ndim, noc)

# plot results
types_colorvec = np.char.add('C', sca.types_idx.astype(str))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(xp[:, 0], xp[:, 1], c=types_colorvec, s=2)
plot_archetype(ax, aa, fmt='-o', color='k',
               label='Inferred archetypes and Pareto front')
ax.set_title('Single cells colored by type')
ax.legend(bbox_to_anchor=(1, 1))

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_aspect('equal')
sns.despine(ax=ax)
ax.grid(False)

os.makedirs(FIG_DIR, exist_ok=True)
fig.savefig(os.path.join(FIG_DIR, 'tutorial_minimum.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
