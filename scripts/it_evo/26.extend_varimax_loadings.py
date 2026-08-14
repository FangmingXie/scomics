"""Extend varimax gene loadings to the whole transcriptome — out-of-sample, no refit.

Scripts 02 (human) and it/{19,21,23,25} (mouse) fit PCA+varimax on the top-2000 HVGs, so
their loadings TSVs have only 2000 rows. Script 16 then intersects the two species' HVG
sets through 1-to-1 orthologs, which collapses to ~357 genes for L2/3. That intersection is
close to arbitrary: 1820 human-HVG and 1871 mouse-HVG orthologs out of 12,812 orthologs
expressed in both datasets give 266 by chance vs 357 observed — a 1.34x enrichment. Two
HVG lists chosen independently per species (different datasets, cell counts, normalizations)
simply do not agree, and the gene set that survives is a near-random sample.

A varimax loading is a rescaled correlation between a gene and the FIXED cell scores, so it
can be computed for any gene without touching the PCA basis, the varimax rotation, Gate A,
or the archetypes. With Z the per-gene z-scored expression (cells x genes) and S the saved
varimax scores (cells x 10), the in-sample loadings L satisfy S = Z @ L with L having
orthonormal columns (L = pca.components_.T @ R, R orthogonal). Hence

    G = Z' S / (n-1)  =  L @ (S' S / (n-1))        ->        L = G @ inv(S' S / (n-1))

which is exact for the original HVGs and defined for every other gene. The script asserts
the reconstruction against the stored 2000-gene TSVs (RECON_TOL) before writing anything;
that assertion is the primary correctness test.

This is the same universe expansion l23_evo/{29,30} applied to regulon enrichment (2000 ->
20,415 genes); it was never applied to the loadings side.

Normalization MUST match the source scripts exactly or the extended loadings are not
comparable to the stored ones: human .X is already log-normalized and is used as-is
(02...py:229), mouse is log2(raw/depth*1e4 + 1) rebuilt from .raw (it/19...py:200-205) —
re-normalizing the mouse .X is the G2 double-normalization bug.

Cells are selected by the barcodes in the coords TSV rather than by re-applying the
subclass filter, so the score matrix and the expression matrix are aligned by construction.

Reads (per TOKEN):
  links/it_evo/jorstad23_human_WithinArea_<...>.h5ad
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  local_data/res/it_evo/02.human_<TOKEN>_varimax_{coords,loadings}.tsv
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_{coords,loadings}.tsv
  data/human_mouse_orthologs.tsv
Outputs (local_data/res/it_evo/):
  26.human_<TOKEN>_varimax_loadings_full.tsv    all expressed human genes x VX1..VX10
  26.mouse_<TOKEN>_varimax_loadings_full.tsv    all expressed mouse genes x VX1..VX10
  26.<TOKEN>_gene_universe.tsv                  ortholog pairs x membership/detection/tier
"""

import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
LINK_ITEVO_DIR = os.path.join(PROJECT_ROOT, 'links', 'it_evo')
LINK_IT_DIR    = os.path.join(PROJECT_ROOT, 'links', 'it')
IT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
ITEVO_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IN_ORTHOLOGS   = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_MOUSE_H5AD  = os.path.join(LINK_IT_DIR, 'superdupermegaRNA_cheng22_IT_P28NR.h5ad')

# Mirrors 02's SUBCLASSES (human h5ad) and 16's SUBCLASSES (mouse loadings file numbering).
SUBCLASSES = [
    {'token': 'L23',  'human_h5ad': 'jorstad23_human_WithinArea_L23IT.h5ad', 'mouse_prefix': '19.cheng22_L23'},
    {'token': 'L4',   'human_h5ad': 'jorstad23_human_WithinArea_L4IT.h5ad',  'mouse_prefix': '21.cheng22_L4'},
    {'token': 'L5IT', 'human_h5ad': 'jorstad23_human_WithinArea_L5IT.h5ad',  'mouse_prefix': '23.cheng22_L5IT'},
    {'token': 'L6IT', 'human_h5ad': 'jorstad23_human_WithinArea_L6IT.h5ad',  'mouse_prefix': '25.cheng22_L6IT'},
]

# --- parameters ---
N_PCS         = 10
VX_COLS       = [f'VX{i+1}' for i in range(N_PCS)]
GENE_NAME_COL = 'feature_name'     # human var column carrying HGNC symbols (02...py:70)
RECON_TOL     = 1e-4               # max |extended - stored| over the original HVGs
MOUSE_CP      = 1e4                # counts-per-10k target, matching it/19...py:204

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

def load_human(cfg):
    """Human: .X is ALREADY log-normalized (02...py:229) — do not renormalize."""
    coords = pd.read_csv(
        os.path.join(ITEVO_RES_DIR, f"02.human_{cfg['token']}_varimax_coords.tsv"),
        sep='\t', index_col=0)
    stored = pd.read_csv(
        os.path.join(ITEVO_RES_DIR, f"02.human_{cfg['token']}_varimax_loadings.tsv"),
        sep='\t', index_col=0)
    adata = ad.read_h5ad(os.path.join(LINK_ITEVO_DIR, cfg['human_h5ad']))
    print(f'    h5ad {adata.n_obs} cells x {adata.n_vars} genes; coords {len(coords)} cells')
    adata = adata[coords.index.values]          # align cells to the saved scores; KeyError if absent
    genes = adata.var[GENE_NAME_COL].astype(str).values
    return adata.X.astype(np.float64), genes, coords[VX_COLS].values, stored


def load_mouse(cfg):
    """Mouse: rebuild log2(CP10k + 1) from .raw counts (it/19...py:200-205). The stored .X is
    already normalized, so normalizing it again is the G2 double-normalization bug."""
    coords = pd.read_csv(
        os.path.join(IT_RES_DIR, f"{cfg['mouse_prefix']}_varimax_coords.tsv"),
        sep='\t', index_col=0)
    stored = pd.read_csv(
        os.path.join(IT_RES_DIR, f"{cfg['mouse_prefix']}_varimax_loadings.tsv"),
        sep='\t', index_col=0)
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

    tier describes HVG membership per gene, NOT a universe name — 16 composes its universes
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
    n_h = int(u['in_human_hvg'].sum())
    n_m = int(u['in_mouse_hvg'].sum())
    n_int = int((u['in_human_hvg'] & u['in_mouse_hvg']).sum())
    n_uni = int((u['in_human_hvg'] | u['in_mouse_hvg']).sum())
    n_all = len(u)
    expected = n_h * n_m / n_all
    print(f'\n  gene universe (16 --universe ...):')
    print(f'    [all expressed]{n_all:6d}  1-to-1 orthologs detected in both datasets')
    print(f'    hvg_union      {n_uni:6d}  HVG in either species')
    print(f'    mouse_hvg      {n_m:6d}  HVG in mouse')
    print(f'    human_hvg      {n_h:6d}  HVG in human')
    print(f'    hvg_intersect  {n_int:6d}  HVG in both  <- script 16 historical gene set')
    print(f'    {n_h} of {n_h_hvg} human HVGs and {n_m} of {n_m_hvg} mouse HVGs have a '
          f'1-to-1 ortholog here;')
    print(f'    chance overlap {expected:.0f} vs observed {n_int} = {n_int / expected:.2f}x '
          f'enrichment — the two HVG lists barely agree.')


# ======================================================================================
# Main
# ======================================================================================

def run_token(cfg):
    token = cfg['token']
    print(f'\n{"=" * 70}\n{token} — extending varimax loadings to all expressed genes\n{"=" * 70}')

    print('  human:')
    Xh, hgenes, Sh, stored_h = load_human(cfg)
    H, h_det = extend_loadings(Xh, hgenes, Sh)
    check_reconstruction(H, stored_h, f'{token} human')
    del Xh

    print('  mouse:')
    Xm, mgenes, Sm, stored_m = load_mouse(cfg)
    M, m_det = extend_loadings(Xm, mgenes, Sm)
    check_reconstruction(M, stored_m, f'{token} mouse')
    del Xm

    u = build_universe(H, M, h_det, m_det, set(stored_h.index), set(stored_m.index))
    report_universe(u, len(stored_h), len(stored_m))

    out_h = os.path.join(OUT_RES_DIR, f'26.human_{token}_varimax_loadings_full.tsv')
    out_m = os.path.join(OUT_RES_DIR, f'26.mouse_{token}_varimax_loadings_full.tsv')
    out_u = os.path.join(OUT_RES_DIR, f'26.{token}_gene_universe.tsv')
    H.to_csv(out_h, sep='\t')
    M.to_csv(out_m, sep='\t')
    u.to_csv(out_u, sep='\t', index=False)
    print(f'\n  saved {out_h}  ({H.shape[0]} genes)')
    print(f'  saved {out_m}  ({M.shape[0]} genes)')
    print(f'  saved {out_u}  ({len(u)} ortholog pairs)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='*', default=['L23'],
                        help='subclass tokens to extend (default: L23)')
    args = parser.parse_args()
    cfg_by_token = {c['token']: c for c in SUBCLASSES}
    for token in args.tokens:
        if token not in cfg_by_token:
            raise ValueError(f'unknown token {token}; expected one of {list(cfg_by_token)}')
        run_token(cfg_by_token[token])
    print('\nDone.')


if __name__ == '__main__':
    main()
