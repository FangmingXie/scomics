"""Extend the N_HVG=4000 varimax gene loadings to the whole transcriptome — no refit.

Standalone L2/3-only copy of `26.extend_varimax_loadings.py` reading scripts 27 (human) and
28 (mouse) instead of 02 and it/19. The mathematics is unchanged; see 26's docstring for the
derivation. In brief: a varimax loading is a rescaled correlation between a gene and the
FIXED cell scores, so with Z the per-gene z-scored expression and S the saved varimax scores,

    G = Z' S / (n-1)  =  L @ (S' S / (n-1))        ->        L = G @ inv(S' S / (n-1))

is exact on the 4000 fitted HVGs and defined for every other expressed gene. The
reconstruction assertion against the stored 4000-gene TSVs (RECON_TOL) is the self-check
that the new basis is internally consistent (plan/it_evo/10 §3).

Normalization MUST match 27/28 exactly or the extended loadings are not comparable to the
stored ones: human `.X` is already log-normalized and is used as-is (27...py:220), mouse is
log2(raw/depth*1e4 + 1) rebuilt from `.raw` (28...py:216-220) — re-normalizing the mouse `.X`
is the G2 double-normalization bug.

GENE-SET GATE (plan §1, Verification 2). The two N=4000 HVG sets must intersect through
1-to-1 orthologs to exactly EXPECT_INTERSECT and union to exactly EXPECT_UNION. Those two
numbers come from the independent HVG-overlap sweep in the plan and depend on nothing
downstream, so a mismatch means the variance ranking has drifted — the script stops rather
than writing a record built on a different gene set.

Reads:
  links/it_evo/jorstad23_human_WithinArea_L23IT.h5ad
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  local_data/res/it_evo/27.human_L23_varimax_{coords,loadings}_hvg4000.tsv
  local_data/res/it_evo/28.mouse_L23_varimax_{coords,loadings}_hvg4000.tsv
  data/human_mouse_orthologs.tsv
Outputs (local_data/res/it_evo/):
  29.human_L23_varimax_loadings_full_hvg4000.tsv   all expressed human genes x VX1..VX10
  29.mouse_L23_varimax_loadings_full_hvg4000.tsv   all expressed mouse genes x VX1..VX10
  29.L23_gene_universe_hvg4000.tsv                 ortholog pairs x membership/detection/tier
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_RES_DIR    = RES_DIR
IN_ORTHOLOGS   = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_HUMAN_H5AD  = os.path.join(PROJECT_ROOT, 'links', 'it_evo',
                              'jorstad23_human_WithinArea_L23IT.h5ad')
IN_MOUSE_H5AD  = os.path.join(PROJECT_ROOT, 'links', 'it',
                              'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
IN_H_COORDS    = os.path.join(RES_DIR, '27.human_L23_varimax_coords_hvg4000.tsv')
IN_H_LOADINGS  = os.path.join(RES_DIR, '27.human_L23_varimax_loadings_hvg4000.tsv')
IN_M_COORDS    = os.path.join(RES_DIR, '28.mouse_L23_varimax_coords_hvg4000.tsv')
IN_M_LOADINGS  = os.path.join(RES_DIR, '28.mouse_L23_varimax_loadings_hvg4000.tsv')
OUT_H_FULL     = os.path.join(OUT_RES_DIR, '29.human_L23_varimax_loadings_full_hvg4000.tsv')
OUT_M_FULL     = os.path.join(OUT_RES_DIR, '29.mouse_L23_varimax_loadings_full_hvg4000.tsv')
OUT_UNIVERSE   = os.path.join(OUT_RES_DIR, '29.L23_gene_universe_hvg4000.tsv')

# --- parameters ---
N_PCS         = 10
VX_COLS       = [f'VX{i+1}' for i in range(N_PCS)]
GENE_NAME_COL = 'feature_name'     # human var column carrying HGNC symbols (27...py:65)
RECON_TOL     = 1e-4               # max |extended - stored| over the original HVGs
MOUSE_CP      = 1e4                # counts-per-10k target, matching 28...py:219

# Gene-set gate (plan/it_evo/10 §1) — independent of everything downstream.
#
# The plan quotes 1641 / 5486 (and human_hvg 3452). Those came from a sweep that accumulated
# gene variance in float64; 27/28 follow 02 and it/19 exactly, which cast to float32 first,
# and the two rankings disagree on 3 genes at the N=4000 cut (relative variance gap ~1e-4,
# below float32 accumulation error over 47k cells). Restricted to the ortholog universe the
# difference is {ANKLE2, OPRD1, ZNF697} float64-only vs {PTDSS2} float32-only, i.e. exactly
# 3450 + 3 - 1 = 3452, which reconciles both figures. The plan already recorded the same
# mechanism at N=2000 ("human 1999/2000, one float32 tie at the cut boundary" — QSER1 vs
# CNTNAP1). Matching 02's recipe outranks matching the sweep, so the gate holds the pipeline
# values; what it still enforces is that the ranking has not DRIFTED, which is what the plan
# asked of it. Independently confirmed: 27's top-4000 reproduces the float32 ranking 4000/4000
# and contains 02's stored top-2000 in full, so the nesting the plan's marker-saturation
# argument relies on is intact.
EXPECT_INTERSECT = 1640            # plan §1 sweep (float64): 1641
EXPECT_UNION     = 5485            # plan §1 sweep (float64): 5486

os.makedirs(OUT_RES_DIR, exist_ok=True)


# ======================================================================================
# Out-of-sample loading extension
# ======================================================================================

def extend_loadings(X, genes, S):
    """Varimax loadings for every gene in X, given the fixed varimax scores S.

    X is cells x genes (dense ndarray or scipy sparse) already normalized exactly as the
    source script normalized it; S is cells x N_PCS from the saved coords TSV. Genes with
    zero variance have no defined loading and are dropped (reported, never imputed).

    Returns (loadings_df, detection_rate_series) over the retained genes.
    """
    n = S.shape[0]
    if S.shape[0] != X.shape[0]:
        raise ValueError(f'cell count mismatch: X has {X.shape[0]}, scores have {S.shape[0]}')

    if sp.issparse(X):
        Xc = X.tocsc()
        mu = np.asarray(Xc.mean(axis=0)).ravel()
        sq = np.asarray(Xc.multiply(Xc).mean(axis=0)).ravel()
        var = np.maximum(sq - mu ** 2, 0.0) * n / (n - 1)
        XtS = Xc.T @ S
        detection = np.asarray((Xc > 0).mean(axis=0)).ravel()
    else:
        mu = X.mean(axis=0)
        var = X.var(axis=0, ddof=1)
        XtS = X.T @ S
        detection = (X > 0).mean(axis=0)

    sd = np.sqrt(var)
    keep = sd > 0
    n_drop = int((~keep).sum())
    if n_drop:
        print(f'    dropping {n_drop} zero-variance genes (no defined loading)')

    # Z = (X - mu) / sd  =>  Z'S = (X'S - mu (1'S)) / sd; S is centered so 1'S ~ 0, but the
    # term is kept exact rather than assumed.
    ZtS = (XtS[keep] - np.outer(mu[keep], S.sum(axis=0))) / sd[keep][:, None]
    G = ZtS / (n - 1)
    Scov = S.T @ S / (n - 1)
    L = G @ np.linalg.inv(Scov)

    kept_genes = np.asarray(genes)[keep]
    return (pd.DataFrame(L, index=kept_genes, columns=VX_COLS),
            pd.Series(detection[keep], index=kept_genes))


def check_reconstruction(L_ext, stored, label):
    """Fail-fast: the extension must reproduce the stored HVG loadings."""
    missing = [g for g in stored.index if g not in L_ext.index]
    if missing:
        raise ValueError(
            f'{label}: {len(missing)} stored HVG genes absent from the extended loadings '
            f'(e.g. {missing[:5]}) — expression matrix and loadings TSV disagree')
    err = float(np.abs(L_ext.loc[stored.index, VX_COLS].values - stored[VX_COLS].values).max())
    if not err < RECON_TOL:
        raise ValueError(
            f'{label}: extended loadings deviate from the stored HVG loadings by {err:.3e} '
            f'(tolerance {RECON_TOL:.0e}) — normalization or score matrix does not match '
            f'the source script')
    print(f'    reconstruction check OK: max |extended - stored| = {err:.2e} over '
          f'{len(stored)} HVGs')
    return err


# ======================================================================================
# Per-species loading of expression + scores
# ======================================================================================

def load_human():
    """Human: .X is ALREADY log-normalized (27...py:220) — do not renormalize."""
    coords = pd.read_csv(IN_H_COORDS, sep='\t', index_col=0)
    stored = pd.read_csv(IN_H_LOADINGS, sep='\t', index_col=0)
    adata = ad.read_h5ad(IN_HUMAN_H5AD)
    print(f'    h5ad {adata.n_obs} cells x {adata.n_vars} genes; coords {len(coords)} cells')
    adata = adata[coords.index.values]          # align cells to the saved scores; KeyError if absent
    genes = adata.var[GENE_NAME_COL].astype(str).values
    return adata.X.astype(np.float64), genes, coords[VX_COLS].values, stored


def load_mouse():
    """Mouse: rebuild log2(CP10k + 1) from .raw counts (28...py:216-220). The stored .X is
    already normalized, so normalizing it again is the G2 double-normalization bug."""
    coords = pd.read_csv(IN_M_COORDS, sep='\t', index_col=0)
    stored = pd.read_csv(IN_M_LOADINGS, sep='\t', index_col=0)
    adata = ad.read_h5ad(IN_MOUSE_H5AD)
    print(f'    h5ad {adata.n_obs} cells x {adata.n_vars} genes; coords {len(coords)} cells')
    adata = adata[coords.index.values]
    X_raw = adata.raw[:, adata.var_names].X.toarray().astype(np.float32)
    depths = X_raw.sum(axis=1, keepdims=True)
    depths[depths == 0] = 1
    X_norm = np.log2(X_raw / depths * MOUSE_CP + 1)
    return X_norm, np.asarray(adata.var_names), coords[VX_COLS].values, stored


# ======================================================================================
# Gene universe
# ======================================================================================

def build_universe(H, M, h_det, m_det, h_hvg, m_hvg):
    """Ortholog pairs expressed in both datasets, tagged with HVG membership and tier.

    tier describes HVG membership per gene, NOT a universe name — 30 composes its universes
    from in_human_hvg/in_mouse_hvg itself, so these labels stay stable if the universe list
    changes: 'both' | 'human_only' | 'mouse_only' | 'neither'.
    """
    ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
             .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
    u = ortho[ortho['human_symbol'].isin(H.index)
              & ortho['mouse_symbol'].isin(M.index)].reset_index(drop=True)
    u['human_detection'] = u['human_symbol'].map(h_det).values
    u['mouse_detection'] = u['mouse_symbol'].map(m_det).values
    u['in_human_hvg'] = u['human_symbol'].isin(h_hvg).values
    u['in_mouse_hvg'] = u['mouse_symbol'].isin(m_hvg).values
    u['tier'] = np.where(u['in_human_hvg'] & u['in_mouse_hvg'], 'both',
                np.where(u['in_human_hvg'], 'human_only',
                np.where(u['in_mouse_hvg'], 'mouse_only', 'neither')))
    return u


def report_universe(u, n_h_hvg, n_m_hvg):
    """Print the universe sizes and enforce the gene-set gate."""
    n_h = int(u['in_human_hvg'].sum())
    n_m = int(u['in_mouse_hvg'].sum())
    n_int = int((u['in_human_hvg'] & u['in_mouse_hvg']).sum())
    n_uni = int((u['in_human_hvg'] | u['in_mouse_hvg']).sum())
    n_all = len(u)
    expected = n_h * n_m / n_all
    print(f'\n  gene universe (30 --universe ...):')
    print(f'    [all expressed]{n_all:6d}  1-to-1 orthologs detected in both datasets')
    print(f'    hvg_union      {n_uni:6d}  HVG in either species  <- primary')
    print(f'    mouse_hvg      {n_m:6d}  HVG in mouse')
    print(f'    human_hvg      {n_h:6d}  HVG in human')
    print(f'    hvg_intersect  {n_int:6d}  HVG in both')
    print(f'    {n_h} of {n_h_hvg} human HVGs and {n_m} of {n_m_hvg} mouse HVGs have a '
          f'1-to-1 ortholog here;')
    print(f'    chance overlap {expected:.0f} vs observed {n_int} = {n_int / expected:.2f}x '
          f'enrichment.')

    if (n_int, n_uni) != (EXPECT_INTERSECT, EXPECT_UNION):
        raise ValueError(
            f'GENE-SET GATE FAILED: hvg_intersect={n_int} (expected {EXPECT_INTERSECT}), '
            f'hvg_union={n_uni} (expected {EXPECT_UNION}). The N=4000 variance ranking does '
            f'not reproduce the sweep in plan/it_evo/10 §1 — stop and diagnose before '
            f'building a record on this gene set.')
    print(f'    GATE OK: hvg_intersect={n_int}, hvg_union={n_uni} match plan §1.')


# ======================================================================================
# Main
# ======================================================================================

print(f'{"=" * 70}\nL23 — extending N_HVG=4000 varimax loadings to all expressed genes\n{"=" * 70}')

print('  human:')
Xh, hgenes, Sh, stored_h = load_human()
H, h_det = extend_loadings(Xh, hgenes, Sh)
check_reconstruction(H, stored_h, 'L23 human')
del Xh

print('  mouse:')
Xm, mgenes, Sm, stored_m = load_mouse()
M, m_det = extend_loadings(Xm, mgenes, Sm)
check_reconstruction(M, stored_m, 'L23 mouse')
del Xm

u = build_universe(H, M, h_det, m_det, set(stored_h.index), set(stored_m.index))
report_universe(u, len(stored_h), len(stored_m))

H.to_csv(OUT_H_FULL, sep='\t')
M.to_csv(OUT_M_FULL, sep='\t')
u.to_csv(OUT_UNIVERSE, sep='\t', index=False)
print(f'\n  saved {OUT_H_FULL}  ({H.shape[0]} genes)')
print(f'  saved {OUT_M_FULL}  ({M.shape[0]} genes)')
print(f'  saved {OUT_UNIVERSE}  ({len(u)} ortholog pairs)')

print('\nDone.')
