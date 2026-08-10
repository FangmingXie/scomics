"""Archetype scores + loadings for Cheng22 mouse L2/3 / L4 / L5 IT / L6 IT — seeded PCHA.

The l23_evo/58 procedure applied to the four IT subclasses, re-using the already-computed
varimax record in local_data/res/it/{19,21,23,25} — nothing upstream is recomputed. Per
subclass: seeded PCHA on the selected VX subspace, top cells per archetype by distance in
VX space, conservative one-vs-each Wilcoxon markers, per-cell [0,1] archetype scores.

Three deliberate departures from l23_evo/58, all forced by re-using the it/ varimax record:

  G1  The reused it/*_varimax_coords.tsv name their sample column `Sample`, not `sample`.

  G2  cheng22 `.X` is already log1p(CP10k) (max 6.40, non-integer) while `.raw.X` holds the
      integer counts (max 1031). l23_evo/18 and 58 treat `.X` as raw counts and normalize it
      a second time; it/{19,21,23,25} correctly normalize from `.raw`, and those are the
      scripts that produced the loadings re-used here. This script therefore reads
      `.raw[:, var_names].X`, so markers and scores are on the same scale the HVG loadings
      were fit on. l23_evo is left alone as a frozen record.

  G11 The shared marker-script line `log2fc = (mean_in - mean_out) / np.log(2)` converts a
      natural-log difference to log2. That is right for the human `.X` (ln(CPM+1)) but wrong
      here: `X_norm` below is already log2(CP10k+1), so the division inflates every mouse
      log2FC by 1/ln2 = 1.443x — with LOG2FC_THRESH = log2(1.5) the l23_evo mouse filter
      effectively admits genes at 1.32x, not 1.5x. The division is dropped here. Expect
      fewer markers than 58 produced; that is the intended direction.

Reads:
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_{coords,loadings}.tsv
  links/it_evo/superdupermegaRNA_cheng22_IT_P28NR.h5ad
Outputs (per TOKEN in L23 / L4 / L5IT / L6IT):
  local_data/res/it_evo/05.mouse_<TOKEN>_pcha_xp.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_pcha_aa.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_pcha_inner_components.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_pcha_inner_mean.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_pcha_gene_loadings.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_top_cells.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_archetype_markers.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_archetype_scores.tsv
  local_data/fig/it_evo/05.mouse_<TOKEN>_archetype_scatter.html
  local_data/fig/it_evo/05.mouse_<TOKEN>_archetype_scores.html
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import anndata as ad
import scipy.stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html, gene_expr_scatter_html
from scomics.main import SCA

# --- file paths ---
IT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
INPUT_MOUSE = os.path.join(PROJECT_ROOT, 'links', 'it_evo', 'superdupermegaRNA_cheng22_IT_P28NR.h5ad')

# `vx_cols` are the fixed selections made in it/{20,22,24,26}; `noc` is the largest NOC
# before ARV_mean jumps in local_data/res/it/{20,22,24,26}.refine.*_metrics.tsv
# (L23 0.027->0.470 at 4; L4 0.025->0.474 at 4; L5IT 0.025->0.544 at 3; L6IT 0.029->0.146 at 4).
# `n_top_cells` is reduced for the two small subclasses so noc*n_top_cells stays near the
# ~22% of cells that the L2/3 precedent draws (G3).
SUBCLASSES = [
    {'token': 'L23',  'mouse_subclass': 'L2/3',
     'coords': '19.cheng22_L23_varimax_coords.tsv',
     'loadings': '19.cheng22_L23_varimax_loadings.tsv',
     'vx_cols': ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9'], 'noc': 3, 'n_top_cells': 300},
    {'token': 'L4',   'mouse_subclass': 'L4',
     'coords': '21.cheng22_L4_varimax_coords.tsv',
     'loadings': '21.cheng22_L4_varimax_loadings.tsv',
     'vx_cols': ['VX1', 'VX4', 'VX5', 'VX6'], 'noc': 3, 'n_top_cells': 300},
    {'token': 'L5IT', 'mouse_subclass': 'L5IT',
     'coords': '23.cheng22_L5IT_varimax_coords.tsv',
     'loadings': '23.cheng22_L5IT_varimax_loadings.tsv',
     'vx_cols': ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'], 'noc': 2, 'n_top_cells': 100},
    {'token': 'L6IT', 'mouse_subclass': 'L6IT',
     'coords': '25.cheng22_L6IT_varimax_coords.tsv',
     'loadings': '25.cheng22_L6IT_varimax_loadings.tsv',
     'vx_cols': ['VX1', 'VX2', 'VX9', 'VX10'], 'noc': 3, 'n_top_cells': 100},
]

# --- parameters ---
SUBCLASS_COL     = 'Subclass'
CLUSTER_COL      = 'Type'
SAMPLE_COL       = 'Sample'          # G1 — the it/ record spells it capitalized
PCHA_SEED        = 0
FRAC_IN_THRESH   = 0.25
FDR_THRESH       = 0.001
LOG2FC_THRESH    = np.log2(1.5)
SCORE_PCTILE_LO  = 2
SCORE_PCTILE_HI  = 98
TOP_CELL_FRAC_MAX = 0.25             # G3 — noc * n_top_cells must stay under this
ALPHABET         = ['A', 'B', 'C', 'D', 'E', 'F']

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

print(f'Loading {INPUT_MOUSE}...')
m_adata_all = ad.read_h5ad(INPUT_MOUSE)
print(f'  {m_adata_all.n_obs} cells x {m_adata_all.n_vars} genes')

for cfg in SUBCLASSES:
    token       = cfg['token']
    subclass    = cfg['mouse_subclass']
    vx_cols     = cfg['vx_cols']
    noc         = cfg['noc']
    n_top_cells = cfg['n_top_cells']
    ndim        = len(vx_cols) - 1
    archetype_names = ALPHABET[:noc]

    in_vx_coords   = os.path.join(IT_RES_DIR, cfg['coords'])
    in_vx_loadings = os.path.join(IT_RES_DIR, cfg['loadings'])
    out_pcha_xp    = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_xp.tsv')
    out_pcha_aa    = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_aa.tsv')
    out_inner_comps = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_inner_components.tsv')
    out_inner_mean  = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_inner_mean.tsv')
    out_gene_loadings = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_gene_loadings.tsv')
    out_top_cells  = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_top_cells.tsv')
    out_markers    = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_archetype_markers.tsv')
    out_scores     = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_archetype_scores.tsv')
    out_arch_html  = os.path.join(OUT_FIG_DIR, f'05.mouse_{token}_archetype_scatter.html')
    out_score_html = os.path.join(OUT_FIG_DIR, f'05.mouse_{token}_archetype_scores.html')

    print(f'\n{"=" * 70}\n{token} — cheng22 mouse {subclass}  '
          f'(NOC={noc}, NDIM={ndim}, VX={vx_cols})\n{"=" * 70}')

    # --- load varimax coords ---
    vx_df   = pd.read_csv(in_vx_coords, sep='\t', index_col=0)
    xn      = vx_df[vx_cols].values
    types   = vx_df[CLUSTER_COL].values
    samples = vx_df[SAMPLE_COL].values
    n_cells = len(vx_df)

    if noc * n_top_cells > TOP_CELL_FRAC_MAX * n_cells:
        raise ValueError(
            f'{token}: noc*n_top_cells = {noc * n_top_cells} exceeds '
            f'{TOP_CELL_FRAC_MAX:.0%} of {n_cells} cells — the "top" sets stop being '
            f'archetype-pure (G3)'
        )

    # --- fit PCHA (seeded inside the loop, G8) ---
    print(f'Fitting PCHA: NOC={noc}, NDIM={ndim}...')
    sca = SCA(xn, types)
    sca.setup_feature_matrix(method='data')
    # py_pcha draws its furthest-sum start index and S init from the global NumPy RNG and
    # takes no seed argument; re-seeding here (not once at import) keeps each subclass
    # independent of how many draws the previous one consumed.
    np.random.seed(PCHA_SEED)
    sca.proj_and_pcha(ndim, noc)

    pc_names = [f'PC{i+1}' for i in range(ndim)]
    pd.DataFrame(sca.xp, index=vx_df.index, columns=pc_names).to_csv(out_pcha_xp, sep='\t')
    pd.DataFrame(sca.aa.T, columns=pc_names,
                 index=[f'archetype_{k+1}' for k in range(noc)]).to_csv(out_pcha_aa, sep='\t')
    print(f'Saved {out_pcha_xp} and {out_pcha_aa}')

    # --- inner PCA parameters (VX axes -> PCHA space) ---
    inner_comps = sca.pca_.components_[:ndim]   # (ndim, len(vx_cols))
    inner_mean  = sca.pca_.mean_                # (len(vx_cols),)
    pd.DataFrame(inner_comps, columns=vx_cols,
                 index=pc_names).to_csv(out_inner_comps, sep='\t')
    pd.DataFrame(inner_mean.reshape(1, -1), columns=vx_cols).to_csv(out_inner_mean, sep='\t')
    print(f'Saved {out_inner_comps} and {out_inner_mean}')

    # gene-level loadings: compose genes->VX (it/ varimax) with VX->PC (inner PCA).
    # Units inherited from the it/ script — weights on StandardScaler-z-scored
    # log2(CP10k+1) HVG expression, not the un-z-scored X_norm built below.
    vx_load = pd.read_csv(in_vx_loadings, sep='\t', index_col=0)   # genes x VX1..VX10
    gene_ld = vx_load[vx_cols].values @ inner_comps.T
    pd.DataFrame(gene_ld, index=vx_load.index,
                 columns=pc_names).to_csv(out_gene_loadings, sep='\t')
    print(f'Saved {out_gene_loadings}  ({gene_ld.shape[0]} genes x {gene_ld.shape[1]} PCs)')

    # --- top cells per archetype, by distance in VX space ---
    aa_vx     = sca.aa.T @ inner_comps + inner_mean            # (noc, len(vx_cols))
    dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(noc)], axis=1)
    top_cells = [np.argsort(dists[:, k])[:n_top_cells] for k in range(noc)]

    for k in range(noc):
        for j in range(k + 1, noc):
            shared = np.intersect1d(top_cells[k], top_cells[j])
            if len(shared):
                raise ValueError(
                    f'{token}: top-cell sets for archetype_{k+1} and archetype_{j+1} share '
                    f'{len(shared)} cells — n_top_cells={n_top_cells} is too large (G3)'
                )
    print(f'Top {n_top_cells} cells per archetype (pairwise disjoint).')

    cell_index = vx_df.index.values
    pd.DataFrame([
        {'archetype': f'archetype_{k+1}', 'rank': r, 'cell': cell_index[i]}
        for k in range(noc) for r, i in enumerate(top_cells[k])
    ]).to_csv(out_top_cells, sep='\t', index=False)
    print(f'Saved {out_top_cells}')

    # --- normalized expression, HVGs only (G2: normalize from .raw integer counts) ---
    print('Loading expression and filtering to HVGs...')
    m_adata = m_adata_all[m_adata_all.obs[SUBCLASS_COL] == subclass]
    m_adata = m_adata[vx_df.index]        # align to the varimax coord order
    hvg_genes = vx_load.index.values

    X_raw  = m_adata.raw[:, m_adata.var_names].X.toarray().astype(np.float32)
    depths = X_raw.sum(axis=1, keepdims=True)
    depths[depths == 0] = 1
    X_norm = np.log2(X_raw / depths * 1e4 + 1)   # already base 2 — see G11 above
    del X_raw
    gc.collect()

    gene_names = m_adata.var_names.values
    hvg_idx    = np.where(np.isin(gene_names, hvg_genes))[0]
    X_norm     = X_norm[:, hvg_idx]
    hvg_names  = gene_names[hvg_idx]
    cell_barcodes = m_adata.obs_names.values
    n_hvg = X_norm.shape[1]
    print(f'X_norm shape: {X_norm.shape}')

    # --- conservative one-vs-each Wilcoxon per archetype ---
    print('Running conservative one-vs-each Wilcoxon per archetype...')
    all_markers = []
    for k in range(noc):
        archetype_label = f'archetype_{k + 1}'
        print(f'  {archetype_label}...')
        X_in    = X_norm[top_cells[k]]
        frac_in = (X_in > 0).mean(axis=0)

        log2fc_mat, pval_mat, frac_out_list = [], [], []
        for j in range(noc):
            if j == k:
                continue
            X_out = X_norm[top_cells[j]]
            # no / np.log(2): X_norm is already log2 (G11)
            log2fc_j = X_in.mean(axis=0) - X_out.mean(axis=0)
            pvals_j  = np.array([scipy.stats.ranksums(X_in[:, g], X_out[:, g])[1]
                                 for g in range(n_hvg)])
            log2fc_mat.append(log2fc_j)
            pval_mat.append(pvals_j)
            frac_out_list.append((X_out > 0).mean(axis=0))

        # worst-case across pairwise comparisons
        log2fc   = np.stack(log2fc_mat).min(axis=0)
        pvals    = np.stack(pval_mat).max(axis=0)
        frac_out = np.stack(frac_out_list).mean(axis=0)

        _, fdr, _, _ = multipletests(pvals, method='fdr_bh')

        df = pd.DataFrame({
            'gene':      hvg_names,
            'archetype': archetype_label,
            'log2FC':    log2fc,
            'pval':      pvals,
            'fdr':       fdr,
            'frac_in':   frac_in,
            'frac_out':  frac_out,
        })
        df = df[(df['frac_in'] >= FRAC_IN_THRESH) & (df['fdr'] < FDR_THRESH)
                & (df['log2FC'] > LOG2FC_THRESH)]
        df = df.sort_values('log2FC', ascending=False)
        all_markers.append(df)
        print(f'    {len(df)} markers after filtering')
        if len(df) == 0:
            print(f'    WARNING: {archetype_label} has 0 markers — NOC may be too high '
                  f'(return to Gate C)')

    markers_df = pd.concat(all_markers, ignore_index=True)
    markers_df.to_csv(out_markers, sep='\t', index=False)
    print(f'Saved {out_markers}  ({len(markers_df)} total markers)')

    # --- archetype score computation ---
    print('Computing archetype scores...')
    scores = np.zeros((n_cells, noc), dtype=np.float32)
    for k, name in enumerate(archetype_names):
        top_genes = markers_df[markers_df['archetype'] == f'archetype_{k+1}']['gene'].values
        cols = [np.where(hvg_names == g)[0][0] for g in top_genes if g in hvg_names]
        if not cols:
            print(f'  WARNING: no genes found for archetype {name}')
            continue
        mat = X_norm[:, cols]
        lo  = np.percentile(mat, SCORE_PCTILE_LO, axis=0)
        hi  = np.percentile(mat, SCORE_PCTILE_HI, axis=0)
        rng = np.where(hi > lo, hi - lo, 1.0)
        scores[:, k] = np.clip((mat - lo) / rng, 0, 1).mean(axis=1)
        print(f'  Score {name}: {len(cols)} genes used')

    pd.DataFrame(scores, index=cell_barcodes,
                 columns=[f'score_{n}' for n in archetype_names]).to_csv(out_scores, sep='\t')
    print(f'Saved {out_scores}')

    # --- PCHA scatter ---
    print('Generating PCHA scatter...')
    panels = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')][:max(ndim - 1, 1)]
    panel_3d = (0, 1, 2, 'PC1', 'PC2', 'PC3') if ndim >= 3 else None
    scatter_categorical_html(
        xp_grid=[sca.xp],
        cell_metadata={CLUSTER_COL: types, SAMPLE_COL: samples},
        title=f'Cheng22 mouse {subclass} — varimax PCHA space (NOC={noc})',
        out_path=out_arch_html,
        panels=panels,
        panel_3d=panel_3d,
        arch_vis=sca.aa,
    )

    # --- archetype score scatter ---
    print('Generating archetype score scatter...')
    gene_vals = {f'Score {n}': scores[:, k] for k, n in enumerate(archetype_names)}
    gene_expr_scatter_html(
        gene_vals=gene_vals,
        x=sca.xp[:, 0], y=sca.xp[:, 1],
        title=f'Cheng22 mouse {subclass} — archetype scores (NOC={noc})',
        out_path=out_score_html,
        xp=sca.xp,
        panels=panels,
        panel_3d=panel_3d,
        aa=sca.aa,
        pctile_low=5, pctile_high=95,
        colorbar_title='archetype score [0–1]',
    )

    del m_adata, X_norm, sca, vx_df, vx_load
    gc.collect()

print('\nDone.')
