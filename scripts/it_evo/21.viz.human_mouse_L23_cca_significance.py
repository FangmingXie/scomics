"""Permutation significance of the L2/3 CCA1 and CCA2 canonical correlations (L2/3 only, plots).

Two panels, one per conserved canonical component. Each shows the permutation null distribution
of the canonical correlation — mouse gene labels shuffled N_PERM times, so the pairing is
destroyed while each species' loading subspace is untouched — with the observed correlation
marked. Permuting rows of Y commutes with orthonormalization (Qy_perm = P·Qy), so each replicate
is a single k×k SVD of Qxᵀ(P·Qy); the full spectrum is retained per replicate and column i is
this component's null.

Caveats worth reading off the figure: the null is bounded on [0,1] and right-skewed, so z is a
standardized effect size, NOT a Gaussian deviate — converting it to a normal tail is meaningless
(the largest of 20000 null draws sits near z≈4.4). The empirical p is quoted separately and
floors at 1/(N_PERM+1). The null also assumes gene exchangeability, which co-expression violates,
so it is anti-conservative; the block-cross-validated r (from 16, shown for context) is the
honest generalization number and is lower.

Reads:
  local_data/res/it_evo/02.human_L23_varimax_loadings.tsv
  local_data/res/it/19.cheng22_L23_varimax_loadings.tsv
  local_data/res/it_evo/16.L23_axis_cca_spectrum.tsv   (blocked-CV r, for context)
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/fig/it_evo/21.human_mouse_L23_cca_significance.pdf
"""

import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_H_LOADINGS = os.path.join(RES_DIR, '02.human_L23_varimax_loadings.tsv')
IN_M_LOADINGS = os.path.join(IT_RES_DIR, '19.cheng22_L23_varimax_loadings.tsv')
IN_SPECTRUM   = os.path.join(RES_DIR, '16.L23_axis_cca_spectrum.tsv')
IN_ORTHOLOGS  = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_PDF       = os.path.join(FIG_DIR, '21.human_mouse_L23_cca_significance.pdf')

# --- config (Gate-A L2/3 VX sets, mirror script 16) ---
HUMAN_VX  = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
MOUSE_VX  = ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9']
COMPONENTS = [0, 1]                 # CCA1, CCA2
N_PERM    = 20000                   # large so the null tail (hence empirical p) is well resolved
SEED      = 0
N_BINS    = 70

os.makedirs(FIG_DIR, exist_ok=True)


def orthonormal(Mtx):
    """Orthonormal basis (QR) for the column span of centered M."""
    return np.linalg.qr(Mtx - Mtx.mean(axis=0))[0]


# --- shared orthologs and centered loading blocks ---
ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
H = pd.read_csv(IN_H_LOADINGS, sep='\t', index_col=0)
M = pd.read_csv(IN_M_LOADINGS, sep='\t', index_col=0)
shared = ortho[ortho['human_symbol'].isin(H.index)
               & ortho['mouse_symbol'].isin(M.index)].reset_index(drop=True)
n_genes = len(shared)
X = H.loc[shared['human_symbol'].values, HUMAN_VX].values
Y = M.loc[shared['mouse_symbol'].values, MOUSE_VX].values

Qx, Qy = orthonormal(X), orthonormal(Y)
obs_spec = np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), 0.0, 1.0)

# --- permutation null: retain the full spectrum per replicate ---
print(f'Permutation null: {N_PERM} shuffles on {n_genes} shared orthologs...')
rng = np.random.default_rng(SEED)
null_spec = np.empty((N_PERM, Qy.shape[1]))
for i in range(N_PERM):
    null_spec[i] = np.clip(np.linalg.svd(Qx.T @ Qy[rng.permutation(n_genes)],
                                         compute_uv=False), 0.0, 1.0)

# --- blocked-CV r (context) from script 16 ---
spec_tsv = pd.read_csv(IN_SPECTRUM, sep='\t').set_index('component')
cv_blocked = {0: spec_tsv.loc['CCA1', 'r_cv_blocked'], 1: spec_tsv.loc['CCA2', 'r_cv_blocked']}

# --- plot ---
plt.rcParams['pdf.fonttype'] = 42
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

for ax, comp in zip(axes, COMPONENTS):
    obs = obs_spec[comp]
    null = null_spec[:, comp]
    mu, sd = null.mean(), null.std()
    z = (obs - mu) / sd
    n_ge = int((null >= obs).sum())
    p_emp = (1 + n_ge) / (N_PERM + 1)
    p95, p99 = np.percentile(null, 95), np.percentile(null, 99)
    null_max = null.max()
    skew = scipy.stats.skew(null)

    ax.hist(null, bins=N_BINS, density=True, color='#bdbdbd', edgecolor='none',
            label=f'permutation null (n={N_PERM})')
    ax.axvline(mu, color='0.4', lw=1.0, ls=':', label=f'null mean = {mu:.3f}')
    ax.axvline(p99, color='0.4', lw=1.0, ls='--', label=f'null 99th pct = {p99:.3f}')
    ax.axvline(obs, color='#d62728', lw=2.2, label=f'observed = {obs:.3f}')
    ax.axvline(cv_blocked[comp], color='#1f77b4', lw=1.8, ls='-.',
               label=f'blocked-CV r = {cv_blocked[comp]:.3f}')

    p_str = f'p < {1/(N_PERM+1):.1e}  (0 / {N_PERM})' if n_ge == 0 else f'p = {p_emp:.2e}'
    stats = (f'observed r = {obs:.3f}\n'
             f'z = {z:.1f}   (effect size, non-Gaussian)\n'
             f'{p_str}\n'
             f'null: mean {mu:.3f}, sd {sd:.3f}\n'
             f'null 95th/99th/max = {p95:.3f} / {p99:.3f} / {null_max:.3f}\n'
             f'null skew = +{skew:.2f}  (right-skewed)\n'
             f'blocked-CV r = {cv_blocked[comp]:.3f}\n'
             f'{n_genes} shared orthologs')
    ax.text(0.97, 0.97, stats, transform=ax.transAxes, ha='right', va='top', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))

    ax.set_xlim(0, max(obs, cv_blocked[comp]) * 1.12)
    ax.set_xlabel('canonical correlation')
    ax.set_ylabel('null density')
    ax.set_title(f'L2/3 CCA{comp + 1}: permutation significance')
    ax.legend(loc='center right', fontsize=8, framealpha=0.9)
    sns.despine(ax=ax)
    print(f'  CCA{comp+1}: obs {obs:.3f}, null {mu:.3f}±{sd:.3f}, z {z:.1f}, '
          f'{n_ge}/{N_PERM} ≥ obs, blocked-CV {cv_blocked[comp]:.3f}')

fig.suptitle('L2/3 conserved canonical correlations vs the gene-label permutation null')
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
