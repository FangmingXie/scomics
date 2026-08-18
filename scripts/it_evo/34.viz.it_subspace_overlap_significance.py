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

Two stacked panels per subclass, in a row (L2/3, L4, L5IT, L6IT). These live on DIFFERENT footings
(different ambient gene space, hence different chance floors), so they are drawn separately rather
than overlaid — overlaying invites two invalid reads (cv_null vs the permutation null; a CV line vs
the histogram) that the split removes.

TOP panel — the only like-for-like significance test (full-gene set, in-sample; footing 1). A
script-21-style permutation null: the grey histogram is Σcos²θ under N_PERM shuffles of the mouse
gene labels (which destroys the ortholog pairing while leaving each species' subspace intact), the
dashed line the null 99th pct, and the coloured line the observed Σcos²θ. Its box quotes observed /
z / empirical p / null 99th pct / n. Observed far in the right tail in every panel = significant
overall conservation. Chance floor here ≈ p_x·p_y/n (~0.05–0.10; ambient N = n ≈ 300–480 genes).

BOTTOM track — the held-out CV estimator, on its OWN footing (small held-out block; ambient N =
n_test ≈ n/5 ≈ 65–95 genes, so a much larger chance floor ≈ k²/n_test ~ 0.2). Top row: a dumbbell
from an open marker at cv_null (chance floor) to a filled marker at cv_raw (achieved held-out
overlap), whose span is the floor-corrected gap. Below it, obs and that gap (cv_raw − cv_null) are
drawn as two bars BOTH anchored at 0, so the one meaningful cross-panel comparison — obs vs the
corrected gap — reads off their right edges directly (a dashed guide marks obs). It addresses the permutation
null's anti-conservatism (gene exchangeability ignores co-expression): per random K-fold split it
fits CCA on train genes, keeps the top-4 canonical directions, projects the HELD-OUT gene loadings
onto them, and measures Σcos²θ of the held-out variates (averaged over CV_R seeds). The two
dumbbell ends differ only in the held-out pairing:
  - cv_null (random-gene, 5-fold): held-out Σcos²θ with the ortholog pairing PERMUTED on the
    held-out genes — the finite-sample floor (two 4-D subspaces of a ~70-gene block overlap even
    with no signal), the "random genes" baseline, 16's headline random-CV quantity. NOT comparable
    to the top-panel null: it is ~5× higher purely because n_test ≈ n/5 shrinks the ambient space.
  - cv_raw (true pairing): the achieved held-out overlap, carrying that same floor.
  - gap cv_raw − cv_null: the conserved held-out overlap above chance — the honest cross-validated
    counterpart of obs (obs is the optimistic in-sample plug-in; the gap sits a little lower).
Random (not module-blocked) folds, as requested; raw, baseline, and corrected values all go to the
TSV.

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
from scipy.linalg import cholesky, solve_triangular

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
CV_K     = 5           # random cross-validation folds over genes (mirror 16)
CV_R     = 10          # CV seeds averaged (mirror 16)
CV_SEED  = 0

os.makedirs(FIG_DIR, exist_ok=True)

ORTHO = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))


def orthonormal(Mtx):
    """Orthonormal basis (QR) for the column span of centered M."""
    return np.linalg.qr(Mtx - Mtx.mean(axis=0))[0]


def cca_fit(X, Y):
    """Full closed-form CCA (mirrors 16). Returns (corrs, A, B); variates are Xc @ A, Yc @ B."""
    n = X.shape[0]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Lx = cholesky(Xc.T @ Xc / (n - 1), lower=True)
    Ly = cholesky(Yc.T @ Yc / (n - 1), lower=True)
    K = solve_triangular(Lx, Xc.T @ Yc / (n - 1), lower=True)
    K = solve_triangular(Ly, K.T, lower=True).T
    U, D, Vt = np.linalg.svd(K)
    return np.clip(D, 0.0, 1.0), solve_triangular(Lx.T, U, lower=False), solve_triangular(Ly.T, Vt.T, lower=False)


def random_folds(n, K, seed):
    rng = np.random.default_rng(seed)
    f = np.empty(n, dtype=int)
    for i, g in enumerate(rng.permutation(n)):
        f[g] = i % K
    return f


def random_cv_sumcos2(X, Y, k):
    """Random-fold CV of Σcos²θ over the top-k TRAIN canonical directions (fully top-k compatible).

    Per fold: fit CCA on the train genes, keep the top-k canonical directions, project the HELD-OUT
    gene loadings onto them, and measure Σcos²θ of the held-out variates (plus a permuted-pairing
    baseline). Averaged over folds and CV_R seeds. Random (not module-blocked) folds, as requested."""
    n = X.shape[0]
    rng = np.random.default_rng(CV_SEED)
    raw_seeds, null_seeds = [], []
    for s in range(CV_R):
        fold = random_folds(n, CV_K, CV_SEED + s)
        raw, null = [], []
        for f in range(CV_K):
            te = fold == f
            tr = ~te
            if te.sum() < TOP_K or tr.sum() <= X.shape[1]:
                continue
            _, A, B = cca_fit(X[tr], Y[tr])
            Xt, Yt = X[te] @ A[:, :k], Y[te] @ B[:, :k]       # top-k train directions, held-out genes
            Qxt = np.linalg.qr(Xt - Xt.mean(0))[0]
            Qyt = np.linalg.qr(Yt - Yt.mean(0))[0]
            raw.append(float(np.sum(np.clip(np.linalg.svd(Qxt.T @ Qyt, compute_uv=False), 0, 1) ** 2)))
            Qyp = Qyt[rng.permutation(Qyt.shape[0])]           # destroy pairing on held-out genes
            null.append(float(np.sum(np.clip(np.linalg.svd(Qxt.T @ Qyp, compute_uv=False), 0, 1) ** 2)))
        if raw:
            raw_seeds.append(np.mean(raw))
            null_seeds.append(np.mean(null))
    return float(np.mean(raw_seeds)), float(np.mean(null_seeds))


def analyze(cfg):
    """Observed Σcos²θ, its permutation null, k, and n for one subclass."""
    token = cfg['token']
    H = pd.read_csv(os.path.join(RES_DIR, f'02.human_{token}_varimax_loadings.tsv'),
                    sep='\t', index_col=0)
    M = pd.read_csv(os.path.join(IT_RES_DIR, cfg['mouse_loadings']), sep='\t', index_col=0)
    shared = ORTHO[ORTHO['human_symbol'].isin(H.index)
                   & ORTHO['mouse_symbol'].isin(M.index)].reset_index(drop=True)
    n = len(shared)

    Xraw = H.loc[shared['human_symbol'].values, cfg['human_vx']].values
    Yraw = M.loc[shared['mouse_symbol'].values, cfg['mouse_vx']].values
    Qx, Qy = orthonormal(Xraw), orthonormal(Yraw)

    def sumcos2_top(A):
        sv = np.clip(np.linalg.svd(A, compute_uv=False), 0, 1)   # principal-angle cosines
        return float(np.sum(sv[:TOP_K] ** 2))

    obs = sumcos2_top(Qx.T @ Qy)                     # Σcos²θ over the top TOP_K components
    cv_raw, cv_null = random_cv_sumcos2(Xraw, Yraw, TOP_K)   # held-out top-4 Σcos²θ + baseline

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
          f'perm-null mean {null.mean():.3f}, z={(obs - null.mean()) / null.std():.1f}; '
          f'random-CV held-out {cv_raw - cv_null:.3f} (raw {cv_raw:.3f} - baseline {cv_null:.3f})')
    return {**cfg, 'n': n, 'obs': obs, 'null': null, 'cv_raw': cv_raw, 'cv_null': cv_null}


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
                 'n_ge': n_ge, 'p_emp': (1 + n_ge) / (N_PERM + 1),
                 'cv_random_raw': res['cv_raw'], 'cv_random_baseline': res['cv_null'],
                 'cv_random_heldout': res['cv_raw'] - res['cv_null']})
pd.DataFrame(rows).to_csv(OUT_TSV, sep='\t', index=False)
print(f'  Saved {OUT_TSV}')

# --- figure: per subclass, a footing-1 permutation-test panel over a footing-2/3 CV track ---
# The histogram panel is the ONLY like-for-like test (obs vs the full-set permutation null). The
# held-out CV estimator lives on a different footing (small held-out block, floor ~k²/n_test), so it
# gets its own slim track below rather than being overlaid on the null — cv_null is the chance floor,
# cv_raw the achieved held-out overlap, their gap the floor-corrected signal comparable to obs.
plt.rcParams['pdf.fonttype'] = 42
fig = plt.figure(figsize=(4.2 * len(results), 5.1), layout='constrained')
gs = fig.add_gridspec(2, len(results), height_ratios=[3.3, 1.5], hspace=0.08)
XMAX = max(max(r['obs'], np.percentile(r['null'], 99), r['cv_raw']) for r in results) * 1.12

for j, res in enumerate(results):
    ax = fig.add_subplot(gs[0, j])
    axcv = fig.add_subplot(gs[1, j], sharex=ax)
    null, obs = res['null'], res['obs']
    z = (obs - null.mean()) / null.std()
    n_ge = int((null >= obs).sum())
    p99 = np.percentile(null, 99)
    p_str = (f'p < {1 / (N_PERM + 1):.1e}  (0 / {N_PERM})' if n_ge == 0
             else f'p = {(1 + n_ge) / (N_PERM + 1):.2e}')
    cv_null_v = res['cv_null']                        # footing-2 chance floor (random-gene held-out CV)
    cv_raw_v = res['cv_raw']                          # footing-2 achieved held-out overlap (true pairing)
    cv_corr = cv_raw_v - cv_null_v                    # footing-3 floor-corrected signal (~ comparable to obs)

    # --- footing-1 panel: pure permutation significance test (obs vs full-set null) ---
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
    ax.set_xlim(0, XMAX)
    ax.set_ylabel('null density')
    ax.set_title(f'{res["label"]}: subspace-overlap permutation test')
    ax.legend(loc='center right', fontsize=7, framealpha=0.9)
    ax.tick_params(labelbottom=True)
    sns.despine(ax=ax)

    # --- footing-2/3 track: held-out CV estimator, a DIFFERENT footing ---
    # top row: raw decomposition dumbbell (chance floor cv_null → achieved cv_raw); its span = the gap.
    yd = 1.15
    axcv.plot([cv_null_v, cv_raw_v], [yd, yd], color='0.5', lw=1.4, zorder=1)
    axcv.plot(cv_null_v, yd, 'o', mfc='white', mec='0.4', mew=1.2, ms=6, zorder=2)
    axcv.plot(cv_raw_v, yd, 'o', color='0.4', ms=6, zorder=2)
    axcv.text(cv_null_v, yd + 0.30, f'{cv_null_v:.3f}', ha='center', va='bottom', fontsize=6, color='0.4')
    axcv.text(cv_raw_v, yd + 0.30, f'{cv_raw_v:.3f}', ha='center', va='bottom', fontsize=6, color='0.4')
    # bottom rows: obs and the corrected gap, BOTH anchored at 0 so their right edges compare directly.
    axcv.axvline(obs, ymin=0.10, ymax=0.52, color=res['color'], lw=1.0, ls='--', alpha=0.6, zorder=1)
    axcv.barh(0.2, obs, height=0.5, color=res['color'], alpha=0.85, zorder=2)
    axcv.barh(-0.6, cv_corr, height=0.5, color='0.3', zorder=2)
    axcv.text(obs, 0.2, f' {obs:.3f}', va='center', ha='left', fontsize=6, color=res['color'])
    axcv.text(cv_corr, -0.6, f' {cv_corr:.3f}', va='center', ha='left', fontsize=6, color='0.3')
    axcv.set_ylim(-1.15, 1.85)
    axcv.set_yticks([yd, 0.2, -0.6])
    axcv.set_yticklabels(['cv_null→cv_raw', 'obs', 'cv_raw−cv_null'] if j == 0 else [], fontsize=6.5)
    axcv.tick_params(axis='y', length=0)
    axcv.set_xlabel(f'subspace overlap  Σcos²θ (top {TOP_K})')
    sns.despine(ax=axcv, left=True)

fig.suptitle('Overall mouse–human subspace conservation per IT subclass  (universe: hvg_intersect)\n'
             'top: full-set permutation test (obs vs null);  bottom: held-out CV estimator '
             '(different footing — floor cv_null → achieved cv_raw; obs vs corrected gap as '
             '0-anchored bars)', fontsize=10)
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
