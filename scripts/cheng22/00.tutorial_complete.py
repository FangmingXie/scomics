# The full tutorial on `scomics`

# import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
sns.set_context('poster')

import anndata as ad

from scomics.main import SCA
from scomics.utils import norm, plot_archetype

# infer project root from this script's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'data_snrna_v1.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig')

# load sample data
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

# set up SCA object using the sample data
x = np.array(adata.X.todense())
depths = adata.obs['n_counts'].values
types = adata.obs['Type'].values

xn = norm(x, depths)
sca = SCA(xn, types)

# Run scomics SCA with different controls
ndim = 2
noc = 3
nrepeats = 3
p = 0.8

sca.setup_feature_matrix(method='data')
xp, aa, varexpl = sca.proj_and_pcha(ndim, noc)
aa_dsamps = sca.bootstrap_proj_pcha(ndim, noc, nrepeats=nrepeats, is_bootstrap=False, downsamp_p=p)

# gshuff
sca.setup_feature_matrix(method='gshuff')
xp_gshuff, aa_gshuff, varexpl_gshuff = sca.proj_and_pcha(ndim, noc)
aa_gshuff_dsamps = sca.bootstrap_proj_pcha(ndim, noc, nrepeats=nrepeats, is_bootstrap=False, downsamp_p=p)

# tshuff
sca.setup_feature_matrix(method='tshuff')
xp_tshuff, aa_tshuff, varexpl_tshuff = sca.proj_and_pcha(ndim, noc)
aa_tshuff_dsamps = sca.bootstrap_proj_pcha(ndim, noc, nrepeats=nrepeats, is_bootstrap=False, downsamp_p=p)

# plot
types_colorvec = np.char.add('C', sca.types_idx.astype(str))

fig, axs = plt.subplots(1, 3, figsize=(8*3, 6))
ax = axs[0]
ax.scatter(xp[:, 0], xp[:, 1], c=types_colorvec, s=2)
plot_archetype(ax, aa, fmt='-o', color='k', zorder=2)
for i in range(nrepeats):
    plot_archetype(ax, aa_dsamps[i], fmt='--', color='gray', zorder=0)
ax.set_title('Data\n(single cells colored by type)')

ax = axs[1]
ax.set_title('Shuffled')
ax.scatter(xp_gshuff[:, 0], xp_gshuff[:, 1], c=types_colorvec, s=2)
plot_archetype(ax, aa_gshuff, fmt='-o', color='k', zorder=2)
for i in range(nrepeats):
    plot_archetype(ax, aa_gshuff_dsamps[i], fmt='--', color='gray', zorder=0)

ax = axs[2]
ax.set_title('Shuffled within each type')
ax.scatter(xp_tshuff[:, 0], xp_tshuff[:, 1], c=types_colorvec, s=2)
plot_archetype(ax, aa_tshuff, fmt='-o', color='k', zorder=2)
for i in range(nrepeats):
    plot_archetype(ax, aa_tshuff_dsamps[i], fmt='--', color='gray', zorder=0)

for i in range(3):
    ax = axs[i]
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_aspect('equal')
    sns.despine(ax=ax)
    ax.grid(False)

os.makedirs(FIG_DIR, exist_ok=True)
fig.savefig(os.path.join(FIG_DIR, 'tutorial_complete_controls.png'), bbox_inches='tight', dpi=150)
plt.close(fig)

# quantify t-ratio (2D only)
t_ratio, t_ratio_shuffs, pval = sca.t_ratio_test(ndim, noc, nrepeats=20)
print(f'p-value: {pval}')

fig, ax = plt.subplots()
ax.axvline(t_ratio, color='r', label='data')
sns.histplot(t_ratio_shuffs, ax=ax, stat='density', label='shuffled')
ax.set_xlabel('t-ratio')
ax.legend()
sns.despine(ax=ax)
fig.savefig(os.path.join(FIG_DIR, 'tutorial_complete_tratio.png'), bbox_inches='tight', dpi=150)
plt.close(fig)

# quantify stability
def get_relative_variation(aa_list):
    aa_avg = np.mean(aa_list, axis=0).T
    ref = np.mean(pairwise_distances(aa_avg))
    aa_std = np.std(aa_list, axis=0).T
    epsilon = np.mean(np.sqrt(np.sum(np.power(aa_std, 2), axis=1)))
    return epsilon/ref

var_data   = get_relative_variation(aa_dsamps)
var_gshuff = get_relative_variation(aa_gshuff_dsamps)

fig, ax = plt.subplots(figsize=(3, 6))
ax.bar(np.arange(2), [var_data, var_gshuff])
ax.set_xticks(np.arange(2))
ax.set_xticklabels(['data', 'shuffled'], rotation=90)
ax.set_ylabel('Archetype variation\n(higher being less stable)')
sns.despine(ax=ax)
fig.savefig(os.path.join(FIG_DIR, 'tutorial_complete_stability.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
