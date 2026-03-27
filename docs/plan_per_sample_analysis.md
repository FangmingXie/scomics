# Plan: Per-Sample Archetypal Analysis
*Date: 2026-03-27*

## Context
The existing tutorial (`scripts/tutorial_complete.py`) runs archetypal analysis on all cells pooled together. The goal is a new script that additionally runs PCHA independently on cells from each sample, overlays per-sample archetypes in the shared global PCA space, and reports cross-sample variance using the existing `get_relative_variation` utility.

The sample identity column is **`sample`** in `adata.obs` (e.g. `'P28_1a'`).

---

## Changes Required

### 1. `src/SingleCellArchetype/utils.py`

**Modify `proj`** to return the fitted PCA model alongside the projection:
```python
def proj(x_norm, ndim, method='PCA', skip_pc1=False):
    ...
    return x_proj, pca_model   # was: return x_proj
```

**Add `proj_transform`** — projects data using an already-fitted PCA model (no re-fitting):
```python
def proj_transform(x_norm, pca_model, skip_pc1=False):
    x_proj = pca_model.transform(x_norm)
    if skip_pc1:
        x_proj = x_proj[:, 1:]
    else:
        x_proj = x_proj[:, :-1]
    return x_proj
```

### 2. `src/SingleCellArchetype/main.py`

**Update all callers of `proj`** to unpack `(xp, pca_model)`:
- `proj_and_pcha`: store `self.pca_` after fitting
- `bootstrap_proj_pcha`: update unpacking (discards the returned model)

**Add method `pcha_on_subset`**:
```python
def pcha_on_subset(self, mask, noc, skip_pc1=False, **kwargs):
    """Run PCHA on a cell subset using the globally-fitted PCA (call proj_and_pcha first)."""
    if not hasattr(self, 'pca_'):
        raise ValueError("Call proj_and_pcha first to fit the global PCA.")
    xp_sub = proj_transform(self.xf[mask], self.pca_, skip_pc1=skip_pc1)
    aa, varexpl = pcha(xp_sub.T, noc=noc, **kwargs)
    return xp_sub, aa, varexpl
```

### 3. New script: `scripts/tutorial_per_sample.py`

Copy of `tutorial_complete.py` with the per-sample analysis section added after the global analysis.

**Structure:**
```python
# FILE PATHS at top
PROJECT_ROOT = ...
INPUT_FILE = ...
FIG_PER_SAMPLE = os.path.join(FIG_DIR, 'tutorial_per_sample_archetypes.png')

# load, normalize, create SCA
sca.setup_feature_matrix('data')
xp, aa, varexpl = sca.proj_and_pcha(ndim, noc)   # fits global PCA, stores sca.pca_

samples = adata.obs['sample'].values
unique_samples = np.unique(samples)

aa_per_sample = []
for samp in unique_samples:
    mask = (samples == samp)
    xp_sub, aa_samp, _ = sca.pcha_on_subset(mask, noc)
    aa_per_sample.append(aa_samp)

# Plot: scatter of all cells (global PCA coords) + per-sample archetype triangles
fig, ax = plt.subplots(...)
ax.scatter(xp[:,0], xp[:,1], c=types_colorvec, s=2)
for i, samp in enumerate(unique_samples):
    plot_archetype(ax, aa_per_sample[i], fmt='--', color=f'C{i}', label=samp)
plot_archetype(ax, aa, fmt='-o', color='k', zorder=2, label='global')

# Variance across samples
rv = get_relative_variation(aa_per_sample)
ax.set_title(f'Per-sample archetypes\nCross-sample variation: {rv:.3f}')
fig.savefig(FIG_PER_SAMPLE)
```

---

## Critical Files

| File | Change |
|------|--------|
| `src/SingleCellArchetype/utils.py` | modify `proj`, add `proj_transform` |
| `src/SingleCellArchetype/main.py` | update `proj_and_pcha`, `bootstrap_proj_pcha`; add `pcha_on_subset` |
| `scripts/tutorial_per_sample.py` | new script (copy + extend from `tutorial_complete.py`) |

## Reused Utilities

- `get_relative_variation` (`utils.py:188`) — cross-sample archetype variance
- `plot_archetype` (`utils.py:212`) — archetype triangle plotting
- `pcha` (`utils.py:67`) — PCHA wrapper (unchanged)
- `norm`, `SCA` — unchanged

---

## Verification

Run in `archetype` conda env:
```bash
conda run -n archetype python scripts/tutorial_per_sample.py
```
Expected output in `local_data/fig/`:
- `tutorial_per_sample_archetypes.png` — cells scatter with per-sample archetype triangles overlaid
