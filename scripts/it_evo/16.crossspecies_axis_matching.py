"""Which mouse IT axis best matches which human IT axis — closed-form CCA on ortholog loadings.

An "axis" is a linear mixture of a species' varimax (VX) components; "matching" is high
correlation of the orthologous-gene loadings. With genes as observations and VX components
as features (rows paired by 1-to-1 orthology), maximizing corr(Xa, Yb) is exactly CCA, and
CCA here is the cosines of the principal angles between the two gene-loading subspaces:
orthonormalize each centered block (QR), and the singular values of `Qxᵀ Qy` are the
canonical correlations (== `cos(scipy.linalg.subspace_angles)`). No NIPALS, no
x_weights_/x_rotations_ ambiguity.

CCA is invariant to any invertible linear transform of either block, so (i) column z-scoring
is a no-op for the canonical correlations — it only affects the pairwise-Pearson baseline
(step 6); (ii) the varimax rotation is irrelevant to the spectrum, only the *weights* carry
VX identity; (iii) enlarging the VX set enlarges the span, so canonical correlations rise
monotonically (step 5 / gate sensitivity).

Primary inputs are the varimax loadings restricted to the Gate-A VX sets (the subspace the
archetypes live in), all 2000 × 10 with genes in the index. The Gate-A VX sets mirror
04/05's SUBCLASSES config (kept in sync here, as 09 mirrors 04/05's noc); `*_pcha_gene_loadings`
is deliberately NOT used — by the invariance above it spans the same thing and its asymmetric
column counts would cap n_components for non-biological reasons.

Per subclass pair the script computes:
  1. Closed-form CCA spectrum, Σcos²θ (subspace overlap), and per-block condition number.
  2. Cross-validation over genes — component-wise (signed) and subspace-level (Σcos²θ) —
     with folds BLOCKED BY CO-EXPRESSION MODULE (random folds leak through co-expressed
     genes and inflate CV; blocking recovers ≈0 under no signal). Both blocked and random
     reported, over R module-assignment seeds.
  3. Bootstrap weight stability (median |cos|, 5th pct, frac<0.9); weights flagged stable
     only when the bootstrap median ≥ 0.9.
  4. Permutation null (mouse gene labels), full spectrum per replicate, z per component and
     for Σcos²θ. Reported as z (a bounded-support standardized effect size, NOT a normal
     deviate) plus an empirical bound; the null assumes gene exchangeability which
     co-expression violates, so it is anti-conservative and subordinate to CV.
  5. Gate-A sensitivity: all-10-VX and leave-one-VX-out, each against its own perm null (z).
  6. Pairwise Pearson baseline on column-z-scored loadings (the gain from allowing mixtures).
  7. Top genes per stable canonical pair by |x_loading · y_loading|.
Steps 1, 2, 4 also run for all 16 human × mouse pairs → five aligned specificity grids.

Reads (per diagonal TOKEN):
  local_data/res/it_evo/02.human_<TOKEN>_varimax_loadings.tsv
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_loadings.tsv
  data/human_mouse_orthologs.tsv
Reads (--compat, L2/3 control against l23_evo/23,24):
  local_data/res/l23_evo/05.varimax_loadings.tsv
  local_data/res/l23_evo/18.mouse_varimax_loadings.tsv
Outputs (local_data/res/it_evo/, per diagonal TOKEN):
  16.<TOKEN>_axis_cca_spectrum.tsv
  16.<TOKEN>_axis_cca_weights_{human,mouse}.tsv
  16.<TOKEN>_axis_pairwise_corr.tsv
  16.<TOKEN>_axis_gate_sensitivity.tsv
  16.<TOKEN>_axis_cca_top_genes.tsv
  16.<TOKEN>_cv_modules.tsv
Outputs (grid, once):
  16.crossspecies_axis_specificity_{cos2,cos2z,cos2cv,cca1z,cca1,cos2q}.tsv
"""

import os
import argparse
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, solve_triangular
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
IT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
ITEVO_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
L23_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IN_ORTHOLOGS  = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')

# Gate-A VX sets — mirror it_evo/04 (human) and it_evo/05 (mouse) SUBCLASSES configs; kept
# in sync here as 09 mirrors their noc. `mouse_loadings` is it/{19,21,23,25}.
SUBCLASSES = [
    {'token': 'L23',
     'human_vx': ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9'],
     'mouse_loadings': '19.cheng22_L23_varimax_loadings.tsv'},
    {'token': 'L4',
     'human_vx': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX4', 'VX5', 'VX6'],
     'mouse_loadings': '21.cheng22_L4_varimax_loadings.tsv'},
    {'token': 'L5IT',
     'human_vx': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'],
     'mouse_loadings': '23.cheng22_L5IT_varimax_loadings.tsv'},
    {'token': 'L6IT',
     'human_vx': ['VX6', 'VX7', 'VX8', 'VX9'],
     'mouse_vx': ['VX1', 'VX2', 'VX9', 'VX10'],
     'mouse_loadings': '25.cheng22_L6IT_varimax_loadings.tsv'},
]

# --- parameters ---
ALL_VX          = [f'VX{i}' for i in range(1, 11)]
N_PERM          = 4000     # permutation null replicates (z quoted to ±0.2)
N_BOOT          = 500      # gene bootstraps for weight stability
CV_K            = 5        # cross-validation folds over genes
CV_R            = 10       # module-assignment seeds (spread of blocked/random CV)
MODULE_DIVISOR  = 8        # n_modules ≈ n_genes / MODULE_DIVISOR
BOOT_NCOMP      = 2        # bootstrap stability computed for components 1..2 (Pilot §6)
STABLE_THRESH   = 0.9      # bootstrap median |cos| gate for interpreting weights
N_TOP_GENES     = 20
SEED            = 0

# --- compat (L2/3 control on old l23_evo inputs, matching 23/24's hardcoded VX sets) ---
COMPAT_HUMAN_VX = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
COMPAT_MOUSE_VX = ['VX1', 'VX2', 'VX6', 'VX7', 'VX8', 'VX10']
COMPAT_SPECTRUM = np.array([0.5981, 0.4111, 0.3412, 0.2212, 0.1680, 0.0390])


# ======================================================================================
# Core CCA (closed form)
# ======================================================================================

def orthonormal(M):
    """Orthonormal basis (QR) for the column span of centered M — genes × k."""
    return np.linalg.qr(M - M.mean(axis=0))[0]


def cca_spectrum(X, Y):
    """Canonical correlations = cos(principal angles) between spans of centered X, Y."""
    s = np.linalg.svd(orthonormal(X).T @ orthonormal(Y), compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def cca_fit(X, Y):
    """Full closed-form CCA. Returns (corrs, A, B): canonical correlations and the weight
    matrices mapping centered X, Y to canonical variates (Xc @ A, Yc @ B)."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Sxx = Xc.T @ Xc / (n - 1)
    Syy = Yc.T @ Yc / (n - 1)
    Sxy = Xc.T @ Yc / (n - 1)
    Lx = cholesky(Sxx, lower=True)                        # Sxx = Lx Lxᵀ
    Ly = cholesky(Syy, lower=True)
    K = solve_triangular(Lx, Sxy, lower=True)             # Lx⁻¹ Sxy
    K = solve_triangular(Ly, K.T, lower=True).T           # Lx⁻¹ Sxy Ly⁻ᵀ
    U, D, Vt = np.linalg.svd(K)
    A = solve_triangular(Lx.T, U, lower=False)            # Lx⁻ᵀ U
    B = solve_triangular(Ly.T, Vt.T, lower=False)         # Ly⁻ᵀ V
    return np.clip(D, 0.0, 1.0), A, B


def sumcos2(s):
    """Subspace overlap Σcos²θ = Σ sᵢ² = ‖Qxᵀ Qy‖²_F."""
    return float(np.dot(s, s))


def cond_zscored(M):
    """Condition number of the column-standardized block (Pilot §7 near-orthogonality)."""
    Mz = (M - M.mean(axis=0)) / M.std(axis=0)
    return float(np.linalg.cond(Mz))


# ======================================================================================
# Permutation null — permuting rows of Y commutes with orthonormalization (Qy_perm = P Qy),
# so each replicate is one k×k SVD of Qxᵀ (P Qy). Retain the full spectrum per replicate.
# ======================================================================================

def permutation_null(X, Y, n_perm, seed):
    Qx, Qy = orthonormal(X), orthonormal(Y)
    n, k = Qy.shape[0], min(X.shape[1], Y.shape[1])
    rng = np.random.default_rng(seed)
    null_spec = np.empty((n_perm, k))
    for i in range(n_perm):
        Qyp = Qy[rng.permutation(n)]
        s = np.clip(np.linalg.svd(Qx.T @ Qyp, compute_uv=False), 0.0, 1.0)
        null_spec[i] = s
    null_sumcos2 = (null_spec ** 2).sum(axis=1)
    return null_spec, null_sumcos2


# ======================================================================================
# Co-expression modules and fold assignment for cross-validation
# ======================================================================================

def coexpression_modules(X):
    """Ward-cluster the human loading matrix (genes × kx) into ≈ n/MODULE_DIVISOR modules."""
    n = X.shape[0]
    n_modules = max(2, round(n / MODULE_DIVISOR))
    Z = linkage(X, method='ward')
    return fcluster(Z, t=n_modules, criterion='maxclust')


def module_folds(modules, K, seed):
    """Assign whole modules to K folds (genes in a module always share a fold)."""
    rng = np.random.default_rng(seed)
    mods = np.unique(modules)
    perm = rng.permutation(len(mods))
    fold_of_mod = {mods[perm[i]]: i % K for i in range(len(mods))}
    return np.array([fold_of_mod[m] for m in modules])


def random_folds(n, K, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    f = np.empty(n, dtype=int)
    for i, g in enumerate(idx):
        f[g] = i % K
    return f


def cv_one_assignment(X, Y, fold_of_gene, r, rng):
    """One K-fold pass. Returns (comp_folds K×r signed, sub_folds K, sub_null_folds K)."""
    K = int(fold_of_gene.max()) + 1
    comp_folds, sub_folds, sub_null = [], [], []
    for f in range(K):
        te = fold_of_gene == f
        tr = ~te
        if te.sum() < 3 or tr.sum() <= X.shape[1]:
            continue
        _, A, B = cca_fit(X[tr], Y[tr])
        Xt, Yt = X[te] @ A, Y[te] @ B                     # Pearson centers internally
        cc = np.array([np.corrcoef(Xt[:, i], Yt[:, i])[0, 1] for i in range(r)])
        comp_folds.append(cc)
        Qxt = np.linalg.qr(Xt - Xt.mean(axis=0))[0]
        Qyt = np.linalg.qr(Yt - Yt.mean(axis=0))[0]
        s = np.clip(np.linalg.svd(Qxt.T @ Qyt, compute_uv=False), 0.0, 1.0)
        sub_folds.append(sumcos2(s))
        Qyp = Qyt[rng.permutation(Qyt.shape[0])]          # destroy pairing on held-out genes
        sp = np.clip(np.linalg.svd(Qxt.T @ Qyp, compute_uv=False), 0.0, 1.0)
        sub_null.append(sumcos2(sp))
    return np.array(comp_folds), np.array(sub_folds), np.array(sub_null)


def cross_validate(X, Y, modules, r, seed):
    """Blocked (by module) and random K-fold CV over R seeds.

    Returns dict with component-wise signed means (blocked/random), subspace Σcos²θ
    means (blocked/random) and its blocked null, plus seed-0 blocked per-fold comp values.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    blk_comp, rnd_comp = [], []
    blk_sub, rnd_sub, blk_sub_null, rnd_sub_null = [], [], [], []
    comp_folds0 = None
    for s in range(CV_R):
        cf_b, sf_b, sn_b = cv_one_assignment(X, Y, module_folds(modules, CV_K, seed + s), r, rng)
        cf_r, sf_r, sn_r = cv_one_assignment(X, Y, random_folds(n, CV_K, seed + 1000 + s), r, rng)
        if cf_b.size:
            blk_comp.append(cf_b.mean(axis=0))
            blk_sub.append(sf_b.mean())
            blk_sub_null.append(sn_b.mean())
        if cf_r.size:
            rnd_comp.append(cf_r.mean(axis=0))
            rnd_sub.append(sf_r.mean())
            rnd_sub_null.append(sn_r.mean())
        if s == 0:
            comp_folds0 = cf_b
    sub_blk = float(np.mean(blk_sub))
    sub_blk_null = float(np.mean(blk_sub_null))
    sub_rnd = float(np.mean(rnd_sub))
    sub_rnd_null = float(np.mean(rnd_sub_null))
    return {
        'comp_blocked': np.mean(blk_comp, axis=0),
        'comp_random':  np.mean(rnd_comp, axis=0),
        # raw held-out Σcos²θ carries a random-subspace baseline (≈ r²/n_test); the
        # generalization signal is the excess over the permuted-pairing null, which is the
        # quantity that returns ≈0 under a destroyed pairing (Verification §2).
        'sub_blocked_raw': sub_blk, 'sub_blocked_null': sub_blk_null,
        'sub_random_raw':  sub_rnd, 'sub_random_null':  sub_rnd_null,
        'sub_blocked': sub_blk - sub_blk_null,   # excess (headline)
        'sub_random':  sub_rnd - sub_rnd_null,
        'comp_folds0':  comp_folds0,
    }


# ======================================================================================
# Bootstrap weight stability
# ======================================================================================

def _abscos(a, b):
    return abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def bootstrap_weights(X, Y, A_full, B_full, n_boot, seed):
    """|cos| between resampled and full-data canonical weight vectors, components 1..BOOT_NCOMP."""
    n = X.shape[0]
    ncomp = min(BOOT_NCOMP, A_full.shape[1])
    rng = np.random.default_rng(seed)
    cos_h = {c: [] for c in range(ncomp)}
    cos_m = {c: [] for c in range(ncomp)}
    n_skip = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            _, A, B = cca_fit(X[idx], Y[idx])
        except LinAlgError:
            n_skip += 1
            continue
        for c in range(ncomp):
            cos_h[c].append(_abscos(A[:, c], A_full[:, c]))
            cos_m[c].append(_abscos(B[:, c], B_full[:, c]))

    def summarize(store):
        out = {}
        for c, vals in store.items():
            v = np.array(vals)
            out[c] = (float(np.median(v)), float(np.percentile(v, 5)), float((v < STABLE_THRESH).mean()))
        return out

    return summarize(cos_h), summarize(cos_m), n_skip


# ======================================================================================
# Per-pair analysis (steps 1, 2, 4) — used by both the grid and the per-token detail
# ======================================================================================

def load_loadings(human_token, mouse_cfg):
    """Return (X, Y, shared_df, human_vx, mouse_vx) for a (human, mouse) subclass pair."""
    human = pd.read_csv(
        os.path.join(ITEVO_RES_DIR, f'02.human_{human_token}_varimax_loadings.tsv'),
        sep='\t', index_col=0)
    mouse = pd.read_csv(
        os.path.join(IT_RES_DIR, mouse_cfg['mouse_loadings']), sep='\t', index_col=0)
    ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
             .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
    shared = ortho[ortho['human_symbol'].isin(human.index)
                   & ortho['mouse_symbol'].isin(mouse.index)].reset_index(drop=True)
    return human, mouse, shared


def analyze_pair(X, Y, seed):
    """Steps 1, 2, 4 for one (X, Y) pair. Modules from the human block X."""
    kx, ky = X.shape[1], Y.shape[1]
    r = min(kx, ky)
    spec = cca_spectrum(X, Y)
    obs_sc2 = sumcos2(spec)

    null_spec, null_sc2 = permutation_null(X, Y, N_PERM, seed)
    null_mean = null_spec.mean(axis=0)
    null_sd = null_spec.std(axis=0)
    z = (spec - null_mean) / null_sd
    sc2_mean, sc2_sd = float(null_sc2.mean()), float(null_sc2.std())
    sc2_z = (obs_sc2 - sc2_mean) / sc2_sd
    sc2_p_emp = (1 + int((null_sc2 >= obs_sc2).sum())) / (N_PERM + 1)

    modules = coexpression_modules(X)
    cv = cross_validate(X, Y, modules, r, seed)

    return {
        'n': X.shape[0], 'kx': kx, 'ky': ky, 'r': r,
        'spec': spec, 'sumcos2': obs_sc2, 'sumcos2_frac': obs_sc2 / r,
        'null_mean': null_mean, 'null_sd': null_sd, 'z': z,
        'sc2_null_mean': sc2_mean, 'sc2_null_sd': sc2_sd, 'sc2_z': float(sc2_z),
        'sc2_p_emp': sc2_p_emp,
        'cond_x': cond_zscored(X), 'cond_y': cond_zscored(Y),
        'cv': cv, 'modules': modules,
    }


# ======================================================================================
# Per-token detail (steps 3, 5, 6, 7) and file writing
# ======================================================================================

def sign_fix(a, b):
    """Flip both weight vectors by the sign of a's largest-|·| entry (keeps corr sign)."""
    s = np.sign(a[np.argmax(np.abs(a))])
    if s == 0:
        s = 1.0
    return a * s, b * s


def write_spectrum(token, res, suffix):
    cv = res['cv']
    r = res['r']
    rows = []
    for i in range(len(res['spec'])):
        folds0 = res['cv']['comp_folds0']
        fold_str = ';'.join(f'{v:.4f}' for v in folds0[:, i]) if (folds0 is not None and i < r) else ''
        rows.append({
            'component': f'CCA{i+1}', 'r': res['spec'][i], 'frac': np.nan,
            'null_mean': res['null_mean'][i], 'null_sd': res['null_sd'][i], 'z': res['z'][i],
            'r_cv_blocked': cv['comp_blocked'][i] if i < r else np.nan,
            'r_cv_random': cv['comp_random'][i] if i < r else np.nan,
            'r_cv_folds': fold_str,
            'cond_x': res['cond_x'], 'cond_y': res['cond_y'],
        })
    rows.append({
        'component': 'subspace', 'r': res['sumcos2'], 'frac': res['sumcos2_frac'],
        'null_mean': res['sc2_null_mean'], 'null_sd': res['sc2_null_sd'], 'z': res['sc2_z'],
        # r_cv_blocked/random are the excess-over-permuted-null generalization signal;
        # raw held-out Σcos²θ and its baseline null are recorded in r_cv_folds.
        'r_cv_blocked': cv['sub_blocked'], 'r_cv_random': cv['sub_random'],
        'r_cv_folds': (f"blk_raw={cv['sub_blocked_raw']:.4f};blk_null={cv['sub_blocked_null']:.4f};"
                       f"rnd_raw={cv['sub_random_raw']:.4f};rnd_null={cv['sub_random_null']:.4f};"
                       f"p_emp={res['sc2_p_emp']:.2e}"),
        'cond_x': res['cond_x'], 'cond_y': res['cond_y'],
    })
    out = os.path.join(OUT_RES_DIR, f'16.{token}_axis_cca_spectrum{suffix}.tsv')
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    print(f'  saved {out}')


def write_weights(token, res, human_vx, mouse_vx, A, B, boot_h, boot_m, suffix):
    r = res['r']
    for side, vx, W, boot, fname in [
            ('human', human_vx, A, boot_h, 'human'),
            ('mouse', mouse_vx, B, boot_m, 'mouse')]:
        rows = []
        for i in range(r):
            a, b = sign_fix(A[:, i].copy(), B[:, i].copy())
            w = (a if side == 'human' else b)
            w = w / np.linalg.norm(w)
            row = {'component': f'CCA{i+1}', 'canonical_r': res['spec'][i]}
            row.update({v: w[j] for j, v in enumerate(vx)})
            if i in boot:
                med, p5, frac = boot[i]
                row.update({'boot_cos_median': med, 'boot_cos_p5': p5, 'boot_frac_below': frac})
            else:
                row.update({'boot_cos_median': np.nan, 'boot_cos_p5': np.nan, 'boot_frac_below': np.nan})
            stable = (i in boot_h and i in boot_m
                      and boot_h[i][0] >= STABLE_THRESH and boot_m[i][0] >= STABLE_THRESH)
            row['stable'] = stable
            rows.append(row)
        out = os.path.join(OUT_RES_DIR, f'16.{token}_axis_cca_weights_{fname}{suffix}.tsv')
        pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
        print(f'  saved {out}')


def write_pairwise(token, X, Y, human_vx, mouse_vx, suffix):
    """Step 6: component × component Pearson on column-z-scored loadings."""
    Xz = (X - X.mean(0)) / X.std(0)
    Yz = (Y - Y.mean(0)) / Y.std(0)
    mat = np.array([[np.corrcoef(Xz[:, i], Yz[:, j])[0, 1]
                     for j in range(Y.shape[1])] for i in range(X.shape[1])])
    df = pd.DataFrame(mat, index=human_vx, columns=mouse_vx)
    out = os.path.join(OUT_RES_DIR, f'16.{token}_axis_pairwise_corr{suffix}.tsv')
    df.to_csv(out, sep='\t')
    print(f'  saved {out}  (max |r| = {np.abs(mat).max():.3f})')


def write_gate_sensitivity(token, human_token, mouse_cfg, human_vx, mouse_vx, seed, suffix):
    """Step 5: all-10-VX and leave-one-VX-out, each against its own perm null (z)."""
    human, mouse, shared = load_loadings(human_token, mouse_cfg)
    Xh = human.loc[shared['human_symbol'].values]
    Ym = mouse.loc[shared['mouse_symbol'].values]

    def evaluate(hvx, mvx):
        X, Y = Xh[hvx].values, Ym[mvx].values
        spec = cca_spectrum(X, Y)
        _, null_sc2 = permutation_null(X, Y, N_PERM, seed)
        null_spec, _ = permutation_null(X, Y, N_PERM, seed + 7)
        cca1_z = (spec[0] - null_spec[:, 0].mean()) / null_spec[:, 0].std()
        sc2 = sumcos2(spec)
        sc2_z = (sc2 - null_sc2.mean()) / null_sc2.std()
        return spec[0], float(cca1_z), sc2, sc2 / min(len(hvx), len(mvx)), float(sc2_z)

    rows = []
    for tag, side, dropped, hvx, mvx in (
            [('gateA', '', '', human_vx, mouse_vx),
             ('all10', '', '', ALL_VX, ALL_VX)]
            + [('drop_human', 'human', v, [x for x in human_vx if x != v], mouse_vx)
               for v in human_vx]
            + [('drop_mouse', 'mouse', v, human_vx, [x for x in mouse_vx if x != v])
               for v in mouse_vx]):
        cca1, cca1_z, sc2, sc2_frac, sc2_z = evaluate(hvx, mvx)
        rows.append({'setting': tag, 'side': side, 'vx_dropped': dropped,
                     'kx': len(hvx), 'ky': len(mvx),
                     'cca1': cca1, 'cca1_z': cca1_z,
                     'sumcos2': sc2, 'sumcos2_frac': sc2_frac, 'sumcos2_z': sc2_z})
    out = os.path.join(OUT_RES_DIR, f'16.{token}_axis_gate_sensitivity{suffix}.tsv')
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    print(f'  saved {out}')


def write_top_genes(token, res, X, Y, shared, A, B, stable_flags, suffix):
    """Step 7: top genes per stable canonical pair by |x_loading · y_loading|."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    records = []
    for i in range(res['r']):
        if not stable_flags[i]:
            continue
        hs = Xc @ A[:, i]
        ms = Yc @ B[:, i]
        prod = np.abs(hs * ms)
        order = np.argsort(prod)[::-1][:N_TOP_GENES]
        records.append(pd.DataFrame({
            'component': f'CCA{i+1}', 'canonical_r': res['spec'][i],
            'human_gene': shared.loc[order, 'human_symbol'].values,
            'mouse_gene': shared.loc[order, 'mouse_symbol'].values,
            'human_score': hs[order], 'mouse_score': ms[order],
            'abs_product': prod[order],
        }))
    out = os.path.join(OUT_RES_DIR, f'16.{token}_axis_cca_top_genes{suffix}.tsv')
    if records:
        pd.concat(records, ignore_index=True).to_csv(out, sep='\t', index=False)
        print(f'  saved {out}  ({sum(stable_flags)} stable pair(s))')
    else:
        pd.DataFrame(columns=['component', 'canonical_r', 'human_gene', 'mouse_gene',
                              'human_score', 'mouse_score', 'abs_product']).to_csv(
            out, sep='\t', index=False)
        print(f'  saved {out}  (no stable pairs — empty)')


def write_modules(token, res, shared, suffix):
    df = pd.DataFrame({'human_gene': shared['human_symbol'].values,
                       'mouse_gene': shared['mouse_symbol'].values,
                       'module': res['modules']})
    out = os.path.join(OUT_RES_DIR, f'16.{token}_cv_modules{suffix}.tsv')
    df.to_csv(out, sep='\t', index=False)
    print(f'  saved {out}  ({df["module"].nunique()} modules)')


def run_detail(token, human_token, mouse_cfg, human_vx, mouse_vx, res, X, Y, shared, suffix):
    """Full per-token detail (steps 3, 5, 6, 7) given a precomputed analyze_pair `res`."""
    print(f'\n{"="*70}\n{token} — full detail  (n={res["n"]}, kx={res["kx"]}, ky={res["ky"]})')
    print(f'  spectrum : {np.round(res["spec"], 4)}')
    print(f'  Σcos²θ   : {res["sumcos2"]:.3f}  (frac {res["sumcos2_frac"]:.3f}, '
          f'z {res["sc2_z"]:.1f}, null {res["sc2_null_mean"]:.3f})')
    print(f'  z/comp   : {np.round(res["z"], 1)}')
    print(f'  CV comp blocked: {np.round(res["cv"]["comp_blocked"], 3)}  '
          f'random: {np.round(res["cv"]["comp_random"], 3)}')
    print(f'  CV Σcos²θ excess blocked {res["cv"]["sub_blocked"]:.3f} '
          f'(raw {res["cv"]["sub_blocked_raw"]:.3f} − null {res["cv"]["sub_blocked_null"]:.3f}) | '
          f'random {res["cv"]["sub_random"]:.3f}')

    _, A, B = cca_fit(X, Y)
    boot_h, boot_m, n_skip = bootstrap_weights(X, Y, A, B, N_BOOT, SEED)
    print(f'  bootstrap ({N_BOOT}, {n_skip} skipped):')
    for c in boot_h:
        print(f'    comp{c+1}: human med {boot_h[c][0]:.3f} p5 {boot_h[c][1]:.3f} '
              f'frac<0.9 {boot_h[c][2]:.2f} | mouse med {boot_m[c][0]:.3f}')
    stable_flags = [
        (i in boot_h and i in boot_m
         and boot_h[i][0] >= STABLE_THRESH and boot_m[i][0] >= STABLE_THRESH)
        for i in range(res['r'])]

    write_spectrum(token, res, suffix)
    write_weights(token, res, human_vx, mouse_vx, A, B, boot_h, boot_m, suffix)
    write_pairwise(token, X, Y, human_vx, mouse_vx, suffix)
    write_gate_sensitivity(token, human_token, mouse_cfg, human_vx, mouse_vx, SEED, suffix)
    write_top_genes(token, res, X, Y, shared, A, B, stable_flags, suffix)
    write_modules(token, res, shared, suffix)


# ======================================================================================
# Main
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='*', default=None,
                        help='subset of diagonal tokens for full per-token detail (default: all four)')
    parser.add_argument('--no-grid', action='store_true', help='skip the 16-cell specificity grid')
    parser.add_argument('--compat', action='store_true',
                        help='L2/3 control on old l23_evo/05+18 inputs; validates plumbing')
    args = parser.parse_args()
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    if args.compat:
        print('*** COMPAT: L2/3 control on l23_evo/05 + 18 loadings ***')
        human = pd.read_csv(os.path.join(L23_RES_DIR, '05.varimax_loadings.tsv'), sep='\t', index_col=0)
        mouse = pd.read_csv(os.path.join(L23_RES_DIR, '18.mouse_varimax_loadings.tsv'), sep='\t', index_col=0)
        ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
                 .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
        shared = ortho[ortho['human_symbol'].isin(human.index)
                       & ortho['mouse_symbol'].isin(mouse.index)].reset_index(drop=True)
        X = human.loc[shared['human_symbol'].values, COMPAT_HUMAN_VX].values
        Y = mouse.loc[shared['mouse_symbol'].values, COMPAT_MOUSE_VX].values
        spec = cca_spectrum(X, Y)
        print(f'  n_shared = {len(shared)} (expect 521)')
        print(f'  spectrum = {np.round(spec, 4)}')
        print(f'  expected = {COMPAT_SPECTRUM}')
        ok = np.allclose(spec, COMPAT_SPECTRUM, atol=1e-3)
        print(f'  {"CHECK PASSED" if ok else "CHECK FAILED"}: spectrum matches 23/24 to 1e-3')
        res = analyze_pair(X, Y, SEED)
        write_spectrum('compat', res, suffix='')  # spectrum for the record; plumbing only
        return

    tokens = args.tokens or [c['token'] for c in SUBCLASSES]
    cfg_by_token = {c['token']: c for c in SUBCLASSES}
    grid_tokens = [c['token'] for c in SUBCLASSES]

    # --- grid: steps 1, 2, 4 for all 16 human × mouse pairs; reuse diagonal for detail ---
    grid = {}
    if not args.no_grid or tokens:
        print('Analyzing pairs (grid + diagonal detail)...')
        pairs = ([(ht, mt) for ht in grid_tokens for mt in grid_tokens]
                 if not args.no_grid else [(t, t) for t in tokens])
        for ht, mt in pairs:
            mcfg = cfg_by_token[mt]
            human, mouse, shared = load_loadings(ht, mcfg)
            hvx = cfg_by_token[ht]['human_vx']
            mvx = mcfg['mouse_vx']
            X = human.loc[shared['human_symbol'].values, hvx].values
            Y = mouse.loc[shared['mouse_symbol'].values, mvx].values
            res = analyze_pair(X, Y, SEED)
            grid[(ht, mt)] = res
            print(f'  H:{ht:5s} M:{mt:5s} n={res["n"]:3d} CCA1={res["spec"][0]:.3f} '
                  f'(z{res["z"][0]:.1f}) Σcos²θ={res["sumcos2"]:.3f} '
                  f'(frac{res["sumcos2_frac"]:.2f} z{res["sc2_z"]:.1f} CVblk{res["cv"]["sub_blocked"]:.3f})')

    # --- per-token detail (steps 3, 5, 6, 7) on diagonal cells ---
    for token in tokens:
        mcfg = cfg_by_token[token]
        human, mouse, shared = load_loadings(token, mcfg)
        hvx = cfg_by_token[token]['human_vx']
        mvx = mcfg['mouse_vx']
        X = human.loc[shared['human_symbol'].values, hvx].values
        Y = mouse.loc[shared['mouse_symbol'].values, mvx].values
        res = grid.get((token, token)) or analyze_pair(X, Y, SEED)
        run_detail(token, token, mcfg, hvx, mvx, res, X, Y, shared, suffix='')

    # --- five specificity grids + BH q ---
    if not args.no_grid:
        write_grids(grid, grid_tokens)

    print('\nDone.')


def write_grids(grid, tokens):
    print(f'\n{"="*70}\nSpecificity grids (human rows × mouse cols)')
    idx, cols = tokens, tokens

    def mat(fn):
        return pd.DataFrame([[fn(grid[(ht, mt)]) for mt in cols] for ht in idx],
                            index=[f'human_{t}' for t in idx],
                            columns=[f'mouse_{t}' for t in cols])

    grids = {
        'cos2':   mat(lambda r: r['sumcos2_frac']),
        'cos2z':  mat(lambda r: r['sc2_z']),
        'cos2cv': mat(lambda r: r['cv']['sub_blocked']),
        'cca1z':  mat(lambda r: r['z'][0]),
        'cca1':   mat(lambda r: r['spec'][0]),
    }
    # BH q across the 16 Σcos²θ empirical permutation p-values (one independent test / cell)
    pvals = np.array([[grid[(ht, mt)]['sc2_p_emp'] for mt in cols] for ht in idx])
    q = multipletests(pvals.ravel(), method='fdr_bh')[1].reshape(pvals.shape)
    grids['cos2q'] = pd.DataFrame(q, index=[f'human_{t}' for t in idx],
                                  columns=[f'mouse_{t}' for t in cols])

    for name, df in grids.items():
        out = os.path.join(OUT_RES_DIR, f'16.crossspecies_axis_specificity_{name}.tsv')
        df.to_csv(out, sep='\t')
        print(f'  saved {out}')
    print('\nΣcos²θ (frac of min(kx,ky)):')
    print(grids['cos2'].round(3).to_string())
    print('\nblocked-CV Σcos²θ:')
    print(grids['cos2cv'].round(3).to_string())


if __name__ == '__main__':
    main()
