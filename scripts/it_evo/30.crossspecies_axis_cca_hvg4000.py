"""L2/3 cross-species axis matching at N_HVG = 4000 — closed-form CCA on ortholog loadings.

Standalone L2/3-only copy of `16.crossspecies_axis_matching.py` reading the N=4000 record
(27 human, 28 mouse, 29 extension) instead of the N=2000 one (02, it/19, 26). The method is
unchanged; see 16's docstring for the derivation. In brief: with genes as observations and
VX components as features (rows paired by 1-to-1 orthology), maximizing corr(Xa, Yb) is
exactly CCA, and the canonical correlations are the cosines of the principal angles between
the two gene-loading subspaces.

Trimmed relative to 16 (plan/it_evo/10): the four-subclass loop, the 16-cell specificity grid
and `--compat` all drop out; `--universe` and `--ladder` are kept.

Per the plan the computed quantities are:
  1. Closed-form CCA spectrum, Σcos²θ (subspace overlap), per-block condition number.
  2. Cross-validation over genes, folds BLOCKED BY CO-EXPRESSION MODULE (random folds leak
     through co-expressed genes and inflate CV). Both blocked and random, over R seeds.
  3. Bootstrap weight stability; weights interpretable only when the median |cos| >= 0.9.
  4. Permutation null (mouse gene labels); z is a standardized effect size on bounded
     support, NOT a normal deviate, and is subordinate to CV because gene exchangeability
     is violated by co-expression.
  5. Gate-A sensitivity: all-10-VX and leave-one-VX-out, each against its own perm null.
  6. Pairwise Pearson baseline on column-z-scored loadings.
  7. Top genes per stable canonical pair.

GATE A IS RE-DERIVED, NOT INHERITED. Varimax orders components by score variance and that
order changes with the gene set, so 16's `human_vx=[VX2,VX6,VX7,VX8,VX9,VX10]` /
`mouse_vx=[VX1,VX2,VX5,VX7,VX8,VX9]` are meaningless in this basis. HUMAN_VX/MOUSE_VX below
were re-selected from 27/28's variance-partition tables by the original criterion; the
derivation is recorded in plan/it_evo/10 §2. Note kx=5 != ky=6 — nothing here requires them
to be equal, and r = min(kx, ky) = 5.

`r` FALLS AS THE UNIVERSE GROWS and is NOT comparable across universes or across N records —
always quote it with n. What compares across records is bootstrap stability, blocked-CV r
against its own permuted null, and the per-gene canonical projections.

Reads:
  local_data/res/it_evo/27.human_L23_varimax_loadings_hvg4000.tsv       (HVG membership)
  local_data/res/it_evo/28.mouse_L23_varimax_loadings_hvg4000.tsv       (HVG membership)
  local_data/res/it_evo/29.{human,mouse}_L23_varimax_loadings_full_hvg4000.tsv (expanded)
  data/human_mouse_orthologs.tsv
Outputs (local_data/res/it_evo/, TAG='_hvg4000' + universe suffix):
  30.L23_axis_cca_spectrum<TAG><SUFFIX>.tsv
  30.L23_axis_cca_weights_{human,mouse}<TAG><SUFFIX>.tsv
  30.L23_axis_pairwise_corr<TAG><SUFFIX>.tsv
  30.L23_axis_gate_sensitivity<TAG><SUFFIX>.tsv
  30.L23_axis_cca_top_genes<TAG><SUFFIX>.tsv
  30.L23_cv_modules<TAG><SUFFIX>.tsv
Outputs (--ladder):
  30.L23_universe_ladder<TAG>.tsv
"""

import os
import argparse
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, solve_triangular
from scipy.cluster.hierarchy import linkage, fcluster

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_RES_DIR   = RES_DIR
IN_ORTHOLOGS  = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_H_HVG      = os.path.join(RES_DIR, '27.human_L23_varimax_loadings_hvg4000.tsv')
IN_M_HVG      = os.path.join(RES_DIR, '28.mouse_L23_varimax_loadings_hvg4000.tsv')
IN_H_FULL     = os.path.join(RES_DIR, '29.human_L23_varimax_loadings_full_hvg4000.tsv')
IN_M_FULL     = os.path.join(RES_DIR, '29.mouse_L23_varimax_loadings_full_hvg4000.tsv')

TOKEN = 'L23'
TAG   = '_hvg4000'

# --- Gate A at N_HVG=4000 (re-derived; see plan/it_evo/10 §2 and the module docstring) ---
# human: cell_type partial R² is the largest factor on VX3 (.725), VX6 (.366), VX8 (.246),
#        VX9 (.287), VX10 (.196); dropped VX1 (library_size .713), VX2/VX4/VX5 (donor),
#        VX7 (source .215).
# mouse: cell_type largest on VX1 (.611), VX3 (.576), VX4 (.180), VX6 (.170), VX9 (.323),
#        VX10 (.156); dropped VX2/VX5/VX8 (sample) and VX7 (library_size .304).
HUMAN_VX = ['VX3', 'VX6', 'VX8', 'VX9', 'VX10']
MOUSE_VX = ['VX1', 'VX3', 'VX4', 'VX6', 'VX9', 'VX10']

# --- parameters (identical to 16) ---
ALL_VX          = [f'VX{i}' for i in range(1, 11)]
N_PERM          = 4000     # permutation null replicates (z quoted to ±0.2)
N_BOOT          = 500      # gene bootstraps for weight stability
CV_K            = 5        # cross-validation folds over genes
CV_R            = 10       # module-assignment seeds (spread of blocked/random CV)
MODULE_DIVISOR  = 8        # n_modules ≈ n_genes / MODULE_DIVISOR
BOOT_NCOMP      = 2        # bootstrap stability computed for components 1..2
STABLE_THRESH   = 0.9      # bootstrap median |cos| gate for interpreting weights
N_TOP_GENES     = 20
SEED            = 0

# --- gene universe (see 16's docstring) ---
#   hvg_intersect  HVG in both species (n=1640 here) — the historical kind of set
#   human_hvg      HVG in human (n=3450), whatever the mouse side does
#   mouse_hvg      HVG in mouse (n=3675), whatever the human side does
#   hvg_union      HVG in either species (n=5485) — primary for this record
UNIVERSES        = ['hvg_intersect', 'human_hvg', 'mouse_hvg', 'hvg_union']
UNIVERSE_SUFFIX  = {'hvg_intersect': '', 'human_hvg': '_humanhvg',
                    'mouse_hvg': '_mousehvg', 'hvg_union': '_union'}
DEFAULT_UNIVERSE = 'hvg_union'     # the plan's primary; 16 defaulted to hvg_intersect
LADDER_REF       = 'hvg_intersect' # weight-alignment reference in the ladder


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
    """Condition number of the column-standardized block."""
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
        # quantity that returns ≈0 under a destroyed pairing.
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
# Gene universe and per-pair analysis
# ======================================================================================

def load_loadings(universe):
    """Return (human_loadings, mouse_loadings, shared_df) for the given gene universe.

    The 4000-HVG TSVs are always read — under the expanded universes they no longer supply
    the loadings, but they define HVG membership.
    """
    if universe not in UNIVERSE_SUFFIX:
        raise ValueError(f'unknown universe {universe!r}; expected one of {UNIVERSES}')
    hvg_h = pd.read_csv(IN_H_HVG, sep='\t', index_col=0)
    hvg_m = pd.read_csv(IN_M_HVG, sep='\t', index_col=0)

    if universe == 'hvg_intersect':
        human, mouse = hvg_h, hvg_m
    else:
        human = pd.read_csv(IN_H_FULL, sep='\t', index_col=0)
        mouse = pd.read_csv(IN_M_FULL, sep='\t', index_col=0)

    ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
             .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
    shared = ortho[ortho['human_symbol'].isin(human.index)
                   & ortho['mouse_symbol'].isin(mouse.index)]
    if universe == 'hvg_union':
        shared = shared[shared['human_symbol'].isin(hvg_h.index)
                        | shared['mouse_symbol'].isin(hvg_m.index)]
    elif universe == 'human_hvg':
        shared = shared[shared['human_symbol'].isin(hvg_h.index)]
    elif universe == 'mouse_hvg':
        shared = shared[shared['mouse_symbol'].isin(hvg_m.index)]
    return human, mouse, shared.reset_index(drop=True)


def blocks(universe):
    """(X, Y, shared) — the Gate-A restricted, ortholog-paired loading blocks."""
    human, mouse, shared = load_loadings(universe)
    X = human.loc[shared['human_symbol'].values, HUMAN_VX].values
    Y = mouse.loc[shared['mouse_symbol'].values, MOUSE_VX].values
    return X, Y, shared


def analyze_pair(X, Y, seed):
    """Steps 1, 2, 4. Modules from the human block X."""
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
# File writing (steps 1, 3, 5, 6, 7)
# ======================================================================================

def sign_fix(a, b):
    """Flip both weight vectors by the sign of a's largest-|·| entry (keeps corr sign)."""
    s = np.sign(a[np.argmax(np.abs(a))])
    if s == 0:
        s = 1.0
    return a * s, b * s


def write_spectrum(res, suffix):
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
    out = os.path.join(OUT_RES_DIR, f'30.{TOKEN}_axis_cca_spectrum{TAG}{suffix}.tsv')
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    print(f'  saved {out}')


def write_weights(res, A, B, boot_h, boot_m, suffix):
    r = res['r']
    for side, vx in [('human', HUMAN_VX), ('mouse', MOUSE_VX)]:
        boot = boot_h if side == 'human' else boot_m
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
            row['stable'] = (i in boot_h and i in boot_m
                             and boot_h[i][0] >= STABLE_THRESH and boot_m[i][0] >= STABLE_THRESH)
            rows.append(row)
        out = os.path.join(OUT_RES_DIR, f'30.{TOKEN}_axis_cca_weights_{side}{TAG}{suffix}.tsv')
        pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
        print(f'  saved {out}')


def write_pairwise(X, Y, suffix):
    """Step 6: component × component Pearson on column-z-scored loadings."""
    Xz = (X - X.mean(0)) / X.std(0)
    Yz = (Y - Y.mean(0)) / Y.std(0)
    mat = np.array([[np.corrcoef(Xz[:, i], Yz[:, j])[0, 1]
                     for j in range(Y.shape[1])] for i in range(X.shape[1])])
    df = pd.DataFrame(mat, index=HUMAN_VX, columns=MOUSE_VX)
    out = os.path.join(OUT_RES_DIR, f'30.{TOKEN}_axis_pairwise_corr{TAG}{suffix}.tsv')
    df.to_csv(out, sep='\t')
    print(f'  saved {out}  (max |r| = {np.abs(mat).max():.3f})')


def write_gate_sensitivity(universe, seed, suffix):
    """Step 5: all-10-VX and leave-one-VX-out, each against its own perm null (z).

    This is where the one borderline Gate-A call is quantified: human VX10 keeps cell_type
    as its dominant factor (.196 vs donor .024 / source .007 / libsize .036) but carries 9
    ribosomal-protein genes in its top-50 loadings, so its drop_human row is the sensitivity
    of the result to that judgment.
    """
    human, mouse, shared = load_loadings(universe)
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
            [('gateA', '', '', HUMAN_VX, MOUSE_VX),
             ('all10', '', '', ALL_VX, ALL_VX)]
            + [('drop_human', 'human', v, [x for x in HUMAN_VX if x != v], MOUSE_VX)
               for v in HUMAN_VX]
            + [('drop_mouse', 'mouse', v, HUMAN_VX, [x for x in MOUSE_VX if x != v])
               for v in MOUSE_VX]):
        cca1, cca1_z, sc2, sc2_frac, sc2_z = evaluate(hvx, mvx)
        rows.append({'setting': tag, 'side': side, 'vx_dropped': dropped,
                     'kx': len(hvx), 'ky': len(mvx),
                     'cca1': cca1, 'cca1_z': cca1_z,
                     'sumcos2': sc2, 'sumcos2_frac': sc2_frac, 'sumcos2_z': sc2_z})
    out = os.path.join(OUT_RES_DIR, f'30.{TOKEN}_axis_gate_sensitivity{TAG}{suffix}.tsv')
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    print(f'  saved {out}')


def write_top_genes(res, X, Y, shared, A, B, stable_flags, suffix):
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
    out = os.path.join(OUT_RES_DIR, f'30.{TOKEN}_axis_cca_top_genes{TAG}{suffix}.tsv')
    if records:
        pd.concat(records, ignore_index=True).to_csv(out, sep='\t', index=False)
        print(f'  saved {out}  ({sum(stable_flags)} stable pair(s))')
    else:
        pd.DataFrame(columns=['component', 'canonical_r', 'human_gene', 'mouse_gene',
                              'human_score', 'mouse_score', 'abs_product']).to_csv(
            out, sep='\t', index=False)
        print(f'  saved {out}  (no stable pairs — empty)')


def write_modules(res, shared, suffix):
    df = pd.DataFrame({'human_gene': shared['human_symbol'].values,
                       'mouse_gene': shared['mouse_symbol'].values,
                       'module': res['modules']})
    out = os.path.join(OUT_RES_DIR, f'30.{TOKEN}_cv_modules{TAG}{suffix}.tsv')
    df.to_csv(out, sep='\t', index=False)
    print(f'  saved {out}  ({df["module"].nunique()} modules)')


def run_detail(universe, suffix):
    """Full detail (steps 1-7) for one gene universe."""
    X, Y, shared = blocks(universe)
    res = analyze_pair(X, Y, SEED)

    print(f'\n{"="*70}\n{TOKEN} N_HVG=4000 — full detail  (universe={universe}, '
          f'n={res["n"]}, kx={res["kx"]}, ky={res["ky"]})')
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

    write_spectrum(res, suffix)
    write_weights(res, A, B, boot_h, boot_m, suffix)
    write_pairwise(X, Y, suffix)
    write_gate_sensitivity(universe, SEED, suffix)
    write_top_genes(res, X, Y, shared, A, B, stable_flags, suffix)
    write_modules(res, shared, suffix)


# ======================================================================================
# Gene-universe ladder
# ======================================================================================

def run_ladder():
    """Sweep the gene universe; one row per universe x component.

    `cos_to_hvg_intersect` is the alignment of each fitted weight vector against the
    LADDER_REF solution — the check that widening the universe re-estimates the same axis
    rather than finding a different one.
    """
    print(f'\n{"="*70}\n{TOKEN} N_HVG=4000 — gene-universe ladder\n{"="*70}')
    ref = {}
    rows = []
    for universe in UNIVERSES:
        X, Y, _ = blocks(universe)
        res = analyze_pair(X, Y, SEED)
        _, A, B = cca_fit(X, Y)
        boot_h, boot_m, _ = bootstrap_weights(X, Y, A, B, N_BOOT, SEED)
        if universe == LADDER_REF:
            ref = {c: (A[:, c].copy(), B[:, c].copy()) for c in range(res['r'])}
        for i in range(res['r']):
            stable = (i in boot_h and i in boot_m
                      and boot_h[i][0] >= STABLE_THRESH and boot_m[i][0] >= STABLE_THRESH)
            rows.append({
                'universe': universe, 'n': res['n'], 'component': f'CCA{i+1}',
                'r': res['spec'][i], 'z': res['z'][i],
                'r_cv_blocked': res['cv']['comp_blocked'][i],
                'r_cv_random': res['cv']['comp_random'][i],
                'boot_cos_median_human': boot_h[i][0] if i in boot_h else np.nan,
                'boot_cos_median_mouse': boot_m[i][0] if i in boot_m else np.nan,
                'stable': stable,
                'cos_to_hvg_intersect_human': _abscos(A[:, i], ref[i][0]) if i in ref else np.nan,
                'cos_to_hvg_intersect_mouse': _abscos(B[:, i], ref[i][1]) if i in ref else np.nan,
                'sumcos2_frac': res['sumcos2_frac'], 'sumcos2_z': res['sc2_z'],
                'sumcos2_cv_blocked': res['cv']['sub_blocked'],
            })
        print(f'  {universe:14s} n={res["n"]:6d}  '
              f'CCA1 r={res["spec"][0]:.3f} z={res["z"][0]:5.1f} '
              f'cv={res["cv"]["comp_blocked"][0]:.3f} boot={boot_h[0][0]:.2f}  |  '
              f'CCA2 r={res["spec"][1]:.3f} z={res["z"][1]:5.1f} '
              f'cv={res["cv"]["comp_blocked"][1]:.3f} boot={boot_h[1][0]:.2f}')

    out = os.path.join(OUT_RES_DIR, f'30.{TOKEN}_universe_ladder{TAG}.tsv')
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    print(f'  saved {out}')
    print('\n  r is a property of the gene universe — not comparable across rows.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--universe', choices=UNIVERSES, default=DEFAULT_UNIVERSE,
                        help=f'gene universe (default: {DEFAULT_UNIVERSE})')
    parser.add_argument('--ladder', action='store_true',
                        help='sweep all universes and write the ladder TSV instead of the '
                             'per-universe detail')
    args = parser.parse_args()
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    print(f'Gate A (N_HVG=4000): human {HUMAN_VX} (kx={len(HUMAN_VX)}), '
          f'mouse {MOUSE_VX} (ky={len(MOUSE_VX)})')

    if args.ladder:
        run_ladder()
    else:
        print(f'Gene universe: {args.universe}  '
              f'(output suffix {TAG + UNIVERSE_SUFFIX[args.universe]!r})')
        run_detail(args.universe, UNIVERSE_SUFFIX[args.universe])

    print('\nDone.')


if __name__ == '__main__':
    main()
