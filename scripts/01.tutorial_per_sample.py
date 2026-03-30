# Per-sample archetypal analysis tutorial
# Runs PCHA on each sample's cells using the global PCA space, then
# overlays per-sample archetype triangles and reports cross-sample variance.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances

import anndata as ad

from SingleCellArchetype.main import SCA
from SingleCellArchetype.utils import norm, plot_archetype, get_relative_variation, mean_archetype_dist

# infer project root from this script's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'data_snrna_v1.h5ad')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig')
FIG_PER_SAMPLE = os.path.join(FIG_DIR, 'tutorial_per_sample_archetypes.png')
FIG_VARIATION = os.path.join(FIG_DIR, 'tutorial_per_sample_variation.png')

NDIM = 2
NOC = 3

# load data
adata = ad.read_h5ad(INPUT_FILE)
print(adata)

x = np.array(adata.X.todense())
depths = adata.obs['n_counts'].values
types = adata.obs['Type'].values
samples = adata.obs['sample'].values

xn = norm(x, depths)
sca = SCA(xn, types)

# global analysis — also fits and stores sca.pca_
sca.setup_feature_matrix(method='data')
xp, aa, varexpl = sca.proj_and_pcha(NDIM, NOC)

# per-sample PCHA in global PCA space
unique_samples = np.unique(samples)
print(f"Samples: {unique_samples}")

aa_per_sample = []
for samp in unique_samples:
    mask = (samples == samp)
    _, aa_samp, _ = sca.pcha_on_subset(mask, NOC)
    aa_per_sample.append(aa_samp)

# cross-sample archetype variation
rv = get_relative_variation(aa_per_sample)
print(f"Cross-sample archetype variation: {rv:.4f}")

# --- plot: per-sample archetype triangles on global PCA scatter ---
types_colorvec = np.char.add('C', sca.types_idx.astype(str))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(xp[:, 0], xp[:, 1], c=types_colorvec, s=2, zorder=0)

for i, samp in enumerate(unique_samples):
    plot_archetype(ax, aa_per_sample[i], fmt='--',
                   color=f'C{i}', alpha=0.6, zorder=1, label=samp)

plot_archetype(ax, aa, fmt='-o', color='k', zorder=2, label='global')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_aspect('equal')
ax.set_title(f'Per-sample archetypes\nCross-sample variation: {rv:.3f}')
ax.legend(fontsize=8, markerscale=1.5, bbox_to_anchor=(1.01, 1), loc='upper left')
sns.despine(ax=ax)
ax.grid(False)

os.makedirs(FIG_DIR, exist_ok=True)
fig.tight_layout()
fig.savefig(FIG_PER_SAMPLE, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {FIG_PER_SAMPLE}")

# --- plot: distance of each sample's archetypes from the global archetypes ---
# For each sample, compute the mean distance between its archetypes and the global archetypes,
# normalized by the mean pairwise distance between global archetypes.

# normalize by mean pairwise distance between global archetypes
global_ref = np.mean(pairwise_distances(aa.T)[np.triu_indices(aa.shape[1], k=1)])
dists = [mean_archetype_dist(aa_per_sample[i], aa) / global_ref
         for i in range(len(unique_samples))]

fig, ax = plt.subplots(figsize=(max(4, len(unique_samples)*0.6 + 2), 5))
ax.bar(np.arange(len(unique_samples)), dists,
       color=[f'C{i}' for i in range(len(unique_samples))])
ax.set_xticks(np.arange(len(unique_samples)))
ax.set_xticklabels(unique_samples, rotation=45, ha='right')
ax.set_ylabel('Mean archetype distance from global\n(relative to global inter-archetype distance)')
ax.set_title(f'Cross-sample variation: {rv:.3f}')
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(FIG_VARIATION, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {FIG_VARIATION}")
