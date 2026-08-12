"""Permutation significance of the cross-species canonical correlations, per IT subclass (plots).

Two figures from the same permutation nulls:
  A. a single detailed panel for L2/3 CCA1 — the headline conserved axis — with the full stats
     block (observed r, z, empirical p, null mean/sd/percentiles/max/skew, blocked-CV r, n);
  B. a grid of null distributions: rows are subclasses (L2/3, L4, L5IT, L6IT), columns are the
     top CCAs (CCA1..CCA5), each cell a compact null + observed + blocked-CV + z/p annotation.

The null shuffles mouse gene labels N_PERM times, destroying the pairing while each species'
loading subspace is untouched. Permuting rows of Y commutes with orthonormalization
(Qy_perm = P·Qy), so each replicate is a single k×k SVD of Qxᵀ(P·Qy) and the full spectrum is
retained per replicate. A subclass has only k = min(kx, ky) components (L23 6, L5IT 5,
L4/L6IT 4), so grid cells past a subclass's component count are left blank.

Caveats read off the figures: the null is bounded on [0,1] and right-skewed, so z is a
standardized effect size, NOT a Gaussian deviate (the largest of N_PERM null draws sits near
z≈4.4); the empirical p is quoted separately and floors at 1/(N_PERM+1). The null assumes gene
exchangeability, which co-expression violates, so it is anti-conservative — the blocked-CV r
(shown) is the honest generalization number.

Reads (per TOKEN):
  local_data/res/it_evo/02.human_<TOKEN>_varimax_loadings.tsv
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_loadings.tsv
  local_data/res/it_evo/16.<TOKEN>_axis_cca_spectrum.tsv   (blocked-CV r, for context)
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/fig/it_evo/21.human_mouse_L23_cca1_significance.pdf   (detailed single panel)
  local_data/fig/it_evo/21.human_mouse_cca_significance.pdf        (4×5 grid)
"""

import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_ORTHOLOGS = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_DETAIL   = os.path.join(FIG_DIR, '21.human_mouse_L23_cca1_significance.pdf')
OUT_GRID     = os.path.join(FIG_DIR, '21.human_mouse_cca_significance.pdf')

# --- Gate-A VX sets, mirror script 16 ---
SUBCLASSES = [
    {'token': 'L23',  'human_vx': ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9'], 'mouse_loadings': '19.cheng22_L23_varimax_loadings.tsv'},
    {'token': 'L4',   'human_vx': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX4', 'VX5', 'VX6'], 'mouse_loadings': '21.cheng22_L4_varimax_loadings.tsv'},
    {'token': 'L5IT', 'human_vx': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'], 'mouse_loadings': '23.cheng22_L5IT_varimax_loadings.tsv'},
    {'token': 'L6IT', 'human_vx': ['VX6', 'VX7', 'VX8', 'VX9'],
     'mouse_vx': ['VX1', 'VX2', 'VX9', 'VX10'], 'mouse_loadings': '25.cheng22_L6IT_varimax_loadings.tsv'},
]

# --- config ---
N_TOP_CCA = 5          # columns: CCA1..CCA5
N_PERM    = 20000      # large so the null tail (hence empirical p) is well resolved
SEED      = 0
N_BINS    = 60

os.makedirs(FIG_DIR, exist_ok=True)


def orthonormal(Mtx):
    """Orthonormal basis (QR) for the column span of centered M."""
    return np.linalg.qr(Mtx - Mtx.mean(axis=0))[0]


def analyze(cfg):
    """Observed spectrum, full permutation null, blocked-CV r, and n for one subclass."""
    ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
             .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
    H = pd.read_csv(os.path.join(RES_DIR, f'02.human_{cfg["token"]}_varimax_loadings.tsv'),
                    sep='\t', index_col=0)
    M = pd.read_csv(os.path.join(IT_RES_DIR, cfg['mouse_loadings']), sep='\t', index_col=0)
    shared = ortho[ortho['human_symbol'].isin(H.index)
                   & ortho['mouse_symbol'].isin(M.index)].reset_index(drop=True)
    n = len(shared)
    X = H.loc[shared['human_symbol'].values, cfg['human_vx']].values
    Y = M.loc[shared['mouse_symbol'].values, cfg['mouse_vx']].values
    Qx, Qy = orthonormal(X), orthonormal(Y)
    obs = np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), 0.0, 1.0)

    rng = np.random.default_rng(SEED)
    null = np.empty((N_PERM, Qy.shape[1]))
    for i in range(N_PERM):
        null[i] = np.clip(np.linalg.svd(Qx.T @ Qy[rng.permutation(n)], compute_uv=False), 0.0, 1.0)

    spec = pd.read_csv(os.path.join(RES_DIR, f'16.{cfg["token"]}_axis_cca_spectrum.tsv'),
                       sep='\t').set_index('component')
    k = obs.shape[0]
    cv = {i: float(spec.loc[f'CCA{i+1}', 'r_cv_blocked']) for i in range(k)}

    # cell-level variance carried by each canonical direction, as a fraction of the total
    # variance in the Gate-A VX subspace: EV_i = âᵀ Σ â / trace(Σ) with â the unit canonical
    # weight vector (16) and Σ the cell VX covariance, computed per species.
    def ev_fraction(coords_path, vx_cols, weights_path):
        C = pd.read_csv(coords_path, sep='\t', index_col=0)[vx_cols].values
        cov = np.cov(C, rowvar=False)
        total = np.trace(cov)
        W = pd.read_csv(weights_path, sep='\t', index_col=0)
        out = {}
        for i in range(k):
            a = W.loc[f'CCA{i+1}', vx_cols].to_numpy(dtype=float)   # object row -> float
            a = a / np.linalg.norm(a)
            out[i] = float(a @ cov @ a / total)
        return out

    ev_h = ev_fraction(os.path.join(RES_DIR, f'02.human_{cfg["token"]}_varimax_coords.tsv'),
                       cfg['human_vx'],
                       os.path.join(RES_DIR, f'16.{cfg["token"]}_axis_cca_weights_human.tsv'))
    ev_m = ev_fraction(os.path.join(IT_RES_DIR, cfg['mouse_loadings'].replace('loadings', 'coords')),
                       cfg['mouse_vx'],
                       os.path.join(RES_DIR, f'16.{cfg["token"]}_axis_cca_weights_mouse.tsv'))
    return {'token': cfg['token'], 'n': n, 'obs': obs, 'null': null, 'cv': cv, 'k': k,
            'ev_h': ev_h, 'ev_m': ev_m}


results = [analyze(cfg) for cfg in SUBCLASSES]
by_token = {res['token']: res for res in results}
plt.rcParams['pdf.fonttype'] = 42


# --- figure A: detailed single panel for L2/3 CCA1 ---
def detail_panel(res, comp, out_path):
    obs = res['obs'][comp]
    null = res['null'][:, comp]
    mu, sd = null.mean(), null.std()
    z = (obs - mu) / sd
    n_ge = int((null >= obs).sum())
    p95, p99, null_max = np.percentile(null, 95), np.percentile(null, 99), null.max()
    skew = scipy.stats.skew(null)
    p_str = (f'p < {1/(N_PERM+1):.1e}  (0 / {N_PERM})' if n_ge == 0
             else f'p = {(1 + n_ge) / (N_PERM + 1):.2e}')

    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.hist(null, bins=90, density=True, color='#bdbdbd', edgecolor='none',
            label=f'permutation null (n={N_PERM})')
    ax.axvline(mu, color='0.4', lw=1.0, ls=':', label=f'null mean = {mu:.3f}')
    ax.axvline(p99, color='0.4', lw=1.0, ls='--', label=f'null 99th pct = {p99:.3f}')
    ax.axvline(res['cv'][comp], color='#1f77b4', lw=1.8, ls='-.',
               label=f'blocked-CV r = {res["cv"][comp]:.3f}')
    ax.axvline(obs, color='#d62728', lw=2.2, label=f'observed = {obs:.3f}')

    stats = (f'observed r = {obs:.3f}\n'
             f'z = {z:.1f}   (effect size, non-Gaussian)\n'
             f'{p_str}\n'
             f'null: mean {mu:.3f}, sd {sd:.3f}\n'
             f'null 95th/99th/max = {p95:.3f} / {p99:.3f} / {null_max:.3f}\n'
             f'null skew = +{skew:.2f}  (right-skewed)\n'
             f'blocked-CV r = {res["cv"][comp]:.3f}\n'
             f'explained var of VX subspace: '
             f'human {res["ev_h"][comp]:.0%}, mouse {res["ev_m"][comp]:.0%}\n'
             f'{res["n"]} shared orthologs')
    ax.text(0.97, 0.97, stats, transform=ax.transAxes, ha='right', va='top', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))
    ax.set_xlim(0, max(obs, res['cv'][comp]) * 1.12)
    ax.set_xlabel('canonical correlation')
    ax.set_ylabel('null density')
    ax.set_title(f'{res["token"]} CCA{comp + 1}: permutation significance')
    ax.legend(loc='center right', fontsize=8, framealpha=0.9)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  Saved {out_path}')


# --- figure B: 4×5 grid (rows = subclasses, cols = CCA1..CCA5) ---
def grid_figure(results, out_path):
    nrows, ncols = len(results), N_TOP_CCA
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows), squeeze=False)
    for r, res in enumerate(results):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= res['k']:                                # component doesn't exist
                ax.axis('off')
                continue
            obs = res['obs'][c]
            null = res['null'][:, c]
            mu, sd = null.mean(), null.std()
            z = (obs - mu) / sd
            n_ge = int((null >= obs).sum())
            p_str = f'p<{1/(N_PERM+1):.0e}' if n_ge == 0 else f'p={(1 + n_ge) / (N_PERM + 1):.1e}'

            ax.hist(null, bins=N_BINS, density=True, color='#bdbdbd', edgecolor='none')
            ax.axvline(np.percentile(null, 99), color='0.45', lw=0.8, ls='--')
            ax.axvline(res['cv'][c], color='#1f77b4', lw=1.4, ls='-.')
            ax.axvline(obs, color='#d62728', lw=2.0)

            ax.text(0.96, 0.96,
                    f'r={obs:.3f}\nz={z:.1f}\n{p_str}\nCV={res["cv"][c]:.2f}\n'
                    f'EV h/m={res["ev_h"][c]:.0%}/{res["ev_m"][c]:.0%}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.75', alpha=0.9))
            ax.set_xlim(0, max(obs, res['cv'][c], np.percentile(null, 99)) * 1.15)
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'CCA{c + 1}', fontsize=11)
            if c == 0:
                ax.set_ylabel(f'{res["token"]}\n(n={res["n"]})', fontsize=10)
            if r == nrows - 1:
                ax.set_xlabel('canonical corr', fontsize=8)
            sns.despine(ax=ax, left=True)

    handles = [
        plt.Line2D([], [], color='#bdbdbd', lw=6, label=f'permutation null (n={N_PERM})'),
        plt.Line2D([], [], color='0.45', lw=1.0, ls='--', label='null 99th pct'),
        plt.Line2D([], [], color='#d62728', lw=2.0, label='observed r'),
        plt.Line2D([], [], color='#1f77b4', lw=1.4, ls='-.', label='blocked-CV r'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle('Cross-species canonical correlations vs the gene-label permutation null '
                 '(z is a bounded-support effect size, not a Gaussian deviate)', y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  Saved {out_path}')


for res in results:
    for c in range(min(N_TOP_CCA, res['k'])):
        null = res['null'][:, c]
        z = (res['obs'][c] - null.mean()) / null.std()
        print(f'  {res["token"]:5s} CCA{c+1}: obs {res["obs"][c]:.3f}, z {z:.1f}, '
              f'{int((null >= res["obs"][c]).sum())}/{N_PERM}≥obs, CV {res["cv"][c]:.3f}')

detail_panel(by_token['L23'], 0, OUT_DETAIL)
grid_figure(results, OUT_GRID)
print('\nDone.')
