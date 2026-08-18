"""Permutation significance of the whole-subspace overlap Σcos²θ, per IT subclass (script-21 style).

Script 21 tests individual canonical axes (CCA1, CCA2, ...). This asks the axis-free question
instead: is there ANY overall conservation between the mouse and human Gate-A loading subspaces
of a subclass, without picking a direction? The statistic is the subspace overlap

    Σcos²θ = Σ_{i=1..4} cos²θ_i = Σ_{i=1..4} r_i²,

the sum over the top 4 canonical components of the squared principal-angle cosines (equivalently
the squared canonical correlations), where Qx, Qy are orthonormal bases of the two centered
ortholog loading blocks. Fixed at k=4 for every subclass (all have >= 4 components), so the totals
are directly comparable. It is 0 when those directions are orthogonal and 4 when they coincide —
a single scalar summarizing how aligned the two subspaces are, defining no axis.

One panel per subclass, in a row (L2/3, L4, L5IT, L6IT). Each is a script-21-style permutation
null: the grey histogram is Σcos²θ under N_PERM shuffles of the mouse gene labels (which destroys
the ortholog pairing while leaving each species' subspace intact), the coloured line is the
observed Σcos²θ, and the box quotes observed / z / empirical p / null 99th pct / n. Observed
sitting far in the right tail in every panel = significant overall conservation in every subclass.

Permuting rows of Y commutes with orthonormalization (Qy_perm = P·Qy), so each replicate is one
k×k SVD of Qxᵀ(P·Qy) (the top-4 squared singular values). The observed value is checked against
Σ of the top-4 squared canonical correlations in 16's persisted spectrum. Caveats mirror 21: the
null is bounded and right-skewed, so z is a standardized effect size (not a Gaussian deviate) and
the empirical p floors at 1/(N_PERM+1); the gene-exchangeability null ignores co-expression.

Reads (per TOKEN):
  local_data/res/it_evo/02.human_<TOKEN>_varimax_loadings.tsv
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_loadings.tsv
  local_data/res/it_evo/16.<TOKEN>_axis_cca_spectrum.tsv   (observed Σcos²θ cross-check)
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/fig/it_evo/34.it_subspace_overlap_significance.pdf
  local_data/res/it_evo/34.it_subspace_overlap_significance.tsv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_ORTHOLOGS = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_PDF      = os.path.join(FIG_DIR, '34.it_subspace_overlap_significance.pdf')
OUT_TSV      = os.path.join(RES_DIR, '34.it_subspace_overlap_significance.tsv')

# --- Gate-A VX sets, mirror script 16/21 ---
SUBCLASSES = [
    {'token': 'L23',  'label': 'L2/3', 'color': 'C0',
     'human_vx': ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9'],
     'mouse_loadings': '19.cheng22_L23_varimax_loadings.tsv'},
    {'token': 'L4',   'label': 'L4', 'color': 'C1',
     'human_vx': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX4', 'VX5', 'VX6'],
     'mouse_loadings': '21.cheng22_L4_varimax_loadings.tsv'},
    {'token': 'L5IT', 'label': 'L5IT', 'color': 'C2',
     'human_vx': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'],
     'mouse_loadings': '23.cheng22_L5IT_varimax_loadings.tsv'},
    {'token': 'L6IT', 'label': 'L6IT', 'color': 'C3',
     'human_vx': ['VX6', 'VX7', 'VX8', 'VX9'],
     'mouse_vx': ['VX1', 'VX2', 'VX9', 'VX10'],
     'mouse_loadings': '25.cheng22_L6IT_varimax_loadings.tsv'},
]

# --- config ---
TOP_K    = 4           # Σcos²θ over the top 4 canonical components (every subclass has >= 4)
N_PERM   = 20000       # large so the null tail (hence empirical p) is well resolved (mirror 21)
SEED     = 0
N_BINS   = 60
OBS_TOL  = 1e-6        # recomputed Σcos²θ(top 4) vs Σ of 16's top-4 squared canonical r

os.makedirs(FIG_DIR, exist_ok=True)

ORTHO = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))


def orthonormal(Mtx):
    """Orthonormal basis (QR) for the column span of centered M."""
    return np.linalg.qr(Mtx - Mtx.mean(axis=0))[0]


def analyze(cfg):
    """Observed Σcos²θ, its permutation null, k, and n for one subclass."""
    token = cfg['token']
    H = pd.read_csv(os.path.join(RES_DIR, f'02.human_{token}_varimax_loadings.tsv'),
                    sep='\t', index_col=0)
    M = pd.read_csv(os.path.join(IT_RES_DIR, cfg['mouse_loadings']), sep='\t', index_col=0)
    shared = ORTHO[ORTHO['human_symbol'].isin(H.index)
                   & ORTHO['mouse_symbol'].isin(M.index)].reset_index(drop=True)
    n = len(shared)

    Qx = orthonormal(H.loc[shared['human_symbol'].values, cfg['human_vx']].values)
    Qy = orthonormal(M.loc[shared['mouse_symbol'].values, cfg['mouse_vx']].values)

    def sumcos2_top(A):
        sv = np.clip(np.linalg.svd(A, compute_uv=False), 0, 1)   # principal-angle cosines
        return float(np.sum(sv[:TOP_K] ** 2))

    obs = sumcos2_top(Qx.T @ Qy)                     # Σcos²θ over the top TOP_K components

    # cross-check against Σ of 16's top-4 squared canonical correlations (default spectrum)
    spec = pd.read_csv(os.path.join(RES_DIR, f'16.{token}_axis_cca_spectrum.tsv'),
                       sep='\t').set_index('component')
    sc2_16 = float(sum(float(spec.loc[f'CCA{i}', 'r']) ** 2 for i in range(1, TOP_K + 1)))
    if not abs(obs - sc2_16) < OBS_TOL:
        raise ValueError(f'{cfg["label"]}: recomputed Σcos²θ(top{TOP_K}) {obs:.6f} != '
                         f'spectrum sum {sc2_16:.6f}')

    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = sumcos2_top(Qx.T @ Qy[rng.permutation(n)])
    print(f'  {cfg["label"]:5s}: n={n}, Σcos²θ(top{TOP_K})={obs:.3f}, '
          f'null mean {null.mean():.3f}, z={(obs - null.mean()) / null.std():.1f}')
    return {**cfg, 'n': n, 'obs': obs, 'null': null}


print('--- subspace overlap Σcos²θ permutation significance, per IT subclass ---')
results = [analyze(cfg) for cfg in SUBCLASSES]

# --- tidy stats table ---
rows = []
for res in results:
    null, obs = res['null'], res['obs']
    mu, sd = float(null.mean()), float(null.std())
    n_ge = int((null >= obs).sum())
    rows.append({'subclass': res['label'], 'n': res['n'], 'k': TOP_K, 'sumcos2': obs,
                 'null_mean': mu, 'null_sd': sd, 'z': (obs - mu) / sd,
                 'null_p99': float(np.percentile(null, 99)),
                 'n_ge': n_ge, 'p_emp': (1 + n_ge) / (N_PERM + 1)})
pd.DataFrame(rows).to_csv(OUT_TSV, sep='\t', index=False)
print(f'  Saved {OUT_TSV}')

# --- figure: one script-21-style null panel per subclass, in a row ---
plt.rcParams['pdf.fonttype'] = 42
fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.0), squeeze=False)
for ax, res in zip(axes[0], results):
    null, obs = res['null'], res['obs']
    z = (obs - null.mean()) / null.std()
    n_ge = int((null >= obs).sum())
    p99 = np.percentile(null, 99)
    p_str = (f'p < {1 / (N_PERM + 1):.1e}  (0 / {N_PERM})' if n_ge == 0
             else f'p = {(1 + n_ge) / (N_PERM + 1):.2e}')

    ax.hist(null, bins=N_BINS, density=True, color='#bdbdbd', edgecolor='none',
            label=f'permutation null (n={N_PERM})')
    ax.axvline(p99, color='0.4', lw=1.0, ls='--', label=f'null 99th pct = {p99:.3f}')
    ax.axvline(obs, color=res['color'], lw=2.4, label=f'observed = {obs:.3f}')

    stats = (f'Σcos²θ (top {TOP_K}) = {obs:.3f}\n'
             f'z = {z:.1f}\n'
             f'{p_str}\n'
             f'null 99th pct = {p99:.3f}\n'
             f'{res["n"]} shared orthologs')
    ax.text(0.96, 0.97, stats, transform=ax.transAxes, ha='right', va='top', fontsize=8,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))
    ax.set_xlim(0, max(obs, p99) * 1.15)
    ax.set_xlabel(f'subspace overlap  Σcos²θ (top {TOP_K})')
    ax.set_ylabel('null density')
    ax.set_title(f'{res["label"]}: subspace-overlap permutation test')
    ax.legend(loc='center right', fontsize=7, framealpha=0.9)
    sns.despine(ax=ax)

fig.suptitle('Overall mouse–human subspace conservation per IT subclass  (universe: hvg_intersect)',
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
