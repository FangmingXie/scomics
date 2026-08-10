"""Archetype scores + loadings for Jorstad23 human L2/3 / L4 / L5 IT / L6 IT — seeded PCHA.

The l23_evo/56 procedure applied to the four IT subclasses. Per subclass: seeded PCHA on
the Gate-A VX subspace, top cells per archetype by distance in VX space, conservative
one-vs-each Wilcoxon markers, per-cell [0,1] archetype scores, plus the inner-PCA
parameters and composed gene-level loadings.

Human gene identifiers are ENSG in `var_names`; everything here is keyed on
`var['feature_name']` symbols so it lines up with the symbol-based ortholog table (G5).
`.X` is ln(CPM+1), so the `/ np.log(2)` in the log2FC line is correct on this side —
unlike the mouse side, see script 05's G11 note.

`*_top_cells.tsv` is new relative to l23_evo/56: script 09 consumes it instead of
re-deriving top cells by distance in PCHA space, which would give a different cell set
than the VX-space back-projection used here.

L2/3 control: 04.human_L23_pcha_{xp,aa}.tsv should match l23_evo/56.human_pcha_{xp,aa}.tsv
— both seeded with PCHA_SEED=0 on a numerically identical varimax record.

Reads:
  local_data/res/it_evo/02.human_<TOKEN>_varimax_{coords,loadings}.tsv
  links/it_evo/jorstad23_human_WithinArea_<HTOKEN>.h5ad
Outputs (per TOKEN in L23 / L4 / L5IT / L6IT):
  local_data/res/it_evo/04.human_<TOKEN>_pcha_xp.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_aa.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_inner_components.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_inner_mean.tsv
  local_data/res/it_evo/04.human_<TOKEN>_pcha_gene_loadings.tsv
  local_data/res/it_evo/04.human_<TOKEN>_top_cells.tsv
  local_data/res/it_evo/04.human_<TOKEN>_archetype_markers.tsv
  local_data/res/it_evo/04.human_<TOKEN>_archetype_scores.tsv
  local_data/fig/it_evo/04.human_<TOKEN>_archetype_scatter.html
  local_data/fig/it_evo/04.human_<TOKEN>_archetype_scores.html
"""

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html, gene_expr_scatter_html
from scomics.main import SCA

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokens', nargs='*', default=None,
                    help='subset of tokens to run (default: all four)')
args = parser.parse_args()

# --- file paths ---
LINK_DIR    = os.path.join(PROJECT_ROOT, 'links', 'it_evo')
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')

# `vx_cols` are the Gate-A selections (see script 03 for the per-component reasoning);
# `noc` is the Gate-B read of 03.refine.human_<TOKEN>_num_archetype_metrics.tsv — the
# largest NOC before ARV_mean jumps.
SUBCLASSES = [
    {'token': 'L23',  'human_subclass': 'L2/3 IT', 'h5ad': 'jorstad23_human_WithinArea_L23IT.h5ad',
     'vx_cols': ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10'], 'noc': 4},
    # ARV_mean 0.029 at 3 -> 0.405 at 4. ARV_rep is elevated across the whole sweep
    # (0.227 at NOC=2, 0.203 at 3), i.e. L4 donors disagree more than L2/3's do at every
    # NOC — a property of the data, not of this choice, so it does not move the elbow.
    {'token': 'L4',   'human_subclass': 'L4 IT',   'h5ad': 'jorstad23_human_WithinArea_L4IT.h5ad',
     'vx_cols': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'], 'noc': 3},
    # ARV_mean 0.024 / 0.101 / 0.596 at NOC 3 / 4 / 5 — the decisive break is 4->5, and
    # ARV_rep agrees sharply (0.068 / 0.076 / 0.573).
    {'token': 'L5IT', 'human_subclass': 'L5 IT',   'h5ad': 'jorstad23_human_WithinArea_L5IT.h5ad',
     'vx_cols': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10'], 'noc': 4},
    # ARV_mean 0.022 at 2 -> 0.579 at 3 (ARV_rep 0.139 -> 0.326). Verified this is not an
    # artifact of keeping the borderline VX9: re-running the sweep on ['VX6','VX7','VX8']
    # also stops at 2 (ARV_mean 0.025 -> 0.396, ARV_rep 0.160 -> 0.626). Human L6 IT
    # supports a 2-vertex structure only.
    {'token': 'L6IT', 'human_subclass': 'L6 IT',   'h5ad': 'jorstad23_human_WithinArea_L6IT.h5ad',
     'vx_cols': ['VX6', 'VX7', 'VX8', 'VX9'], 'noc': 2},
]
if args.tokens:
    SUBCLASSES = [c for c in SUBCLASSES if c['token'] in args.tokens]

# --- parameters ---
CLUSTER_COL      = 'WithinArea_cluster'
GENE_NAME_COL    = 'feature_name'
PCHA_SEED        = 0
N_TOP_CELLS      = 300
FRAC_IN_THRESH   = 0.25
FDR_THRESH       = 0.001
LOG2FC_THRESH    = np.log2(1.5)
SCORE_PCTILE_LO  = 2
SCORE_PCTILE_HI  = 98
TOP_CELL_FRAC_MAX = 0.25             # G3 — noc * N_TOP_CELLS must stay under this
ALPHABET         = ['A', 'B', 'C', 'D', 'E', 'F']

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

for cfg in SUBCLASSES:
    token    = cfg['token']
    subclass = cfg['human_subclass']
    vx_cols  = cfg['vx_cols']
    noc      = cfg['noc']
    if noc is None:
        raise ValueError(
            f'{token}: noc is unset — read '
            f'03.refine.human_{token}_num_archetype_metrics.tsv and fill it in (Gate B)')
    ndim = len(vx_cols) - 1
    archetype_names = ALPHABET[:noc]

    in_h5ad        = os.path.join(LINK_DIR, cfg['h5ad'])
    in_vx_coords   = os.path.join(OUT_RES_DIR, f'02.human_{token}_varimax_coords.tsv')
    in_vx_loadings = os.path.join(OUT_RES_DIR, f'02.human_{token}_varimax_loadings.tsv')
    out_pcha_xp    = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_xp.tsv')
    out_pcha_aa    = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_aa.tsv')
    out_inner_comps = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_inner_components.tsv')
    out_inner_mean  = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_inner_mean.tsv')
    out_gene_loadings = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_gene_loadings.tsv')
    out_top_cells  = os.path.join(OUT_RES_DIR, f'04.human_{token}_top_cells.tsv')
    out_markers    = os.path.join(OUT_RES_DIR, f'04.human_{token}_archetype_markers.tsv')
    out_scores     = os.path.join(OUT_RES_DIR, f'04.human_{token}_archetype_scores.tsv')
    out_arch_html  = os.path.join(OUT_FIG_DIR, f'04.human_{token}_archetype_scatter.html')
    out_score_html = os.path.join(OUT_FIG_DIR, f'04.human_{token}_archetype_scores.html')

    print(f'\n{"=" * 70}\n{token} — human {subclass}  '
          f'(NOC={noc}, NDIM={ndim}, VX={vx_cols})\n{"=" * 70}')

    # --- load varimax coords ---
    vx_df   = pd.read_csv(in_vx_coords, sep='\t', index_col=0)
    xn      = vx_df[vx_cols].values
    types   = vx_df[CLUSTER_COL].values
    n_cells = len(vx_df)

    if noc * N_TOP_CELLS > TOP_CELL_FRAC_MAX * n_cells:
        raise ValueError(
            f'{token}: noc*N_TOP_CELLS = {noc * N_TOP_CELLS} exceeds '
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

    # gene-level loadings: compose genes->VX (script 02) with VX->PC (inner PCA).
    # Units inherited from script 02 — weights on StandardScaler-z-scored HVG expression,
    # not the raw ln(CPM+1) matrix built below for markers/scores.
    vx_load = pd.read_csv(in_vx_loadings, sep='\t', index_col=0)   # genes x VX1..VX10
    gene_ld = vx_load[vx_cols].values @ inner_comps.T
    pd.DataFrame(gene_ld, index=vx_load.index,
                 columns=pc_names).to_csv(out_gene_loadings, sep='\t')
    print(f'Saved {out_gene_loadings}  ({gene_ld.shape[0]} genes x {gene_ld.shape[1]} PCs)')

    # --- top cells per archetype, by distance in VX space ---
    aa_vx     = sca.aa.T @ inner_comps + inner_mean            # (noc, len(vx_cols))
    dists     = np.stack([np.linalg.norm(xn - aa_vx[k], axis=1) for k in range(noc)], axis=1)
    top_cells = [np.argsort(dists[:, k])[:N_TOP_CELLS] for k in range(noc)]

    for k in range(noc):
        for j in range(k + 1, noc):
            shared = np.intersect1d(top_cells[k], top_cells[j])
            if len(shared):
                raise ValueError(
                    f'{token}: top-cell sets for archetype_{k+1} and archetype_{j+1} share '
                    f'{len(shared)} cells — N_TOP_CELLS={N_TOP_CELLS} is too large (G3)'
                )
    print(f'Top {N_TOP_CELLS} cells per archetype (pairwise disjoint).')

    cell_index = vx_df.index.values
    pd.DataFrame([
        {'archetype': f'archetype_{k+1}', 'rank': r, 'cell': cell_index[i]}
        for k in range(noc) for r, i in enumerate(top_cells[k])
    ]).to_csv(out_top_cells, sep='\t', index=False)
    print(f'Saved {out_top_cells}')

    # --- load gene expression (HVGs only; .X is already log-normalized) ---
    print('Loading h5ad and filtering to HVGs...')
    adata      = ad.read_h5ad(in_h5ad)
    hvg_genes  = vx_load.index.values
    gene_names = adata.var[GENE_NAME_COL].values
    hvg_idx    = np.where(np.isin(gene_names, hvg_genes))[0]
    # slice the sparse matrix to the HVG columns before densifying (G7)
    X_norm     = adata.X[:, hvg_idx].toarray().astype(np.float32)
    hvg_names  = gene_names[hvg_idx]
    cell_barcodes = adata.obs_names.values
    n_hvg = X_norm.shape[1]
    print(f'X_norm shape: {X_norm.shape}')

    if not np.array_equal(cell_barcodes, cell_index):
        raise ValueError(
            f'{token}: h5ad cell order differs from {in_vx_coords} — top-cell indices '
            f'would point at the wrong cells'
        )

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
            # human .X is ln(CPM+1) — the /ln(2) converts the difference to log2
            log2fc_j = (X_in.mean(axis=0) - X_out.mean(axis=0)) / np.log(2)
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
                  f'(return to Gate B)')

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
        cell_metadata={CLUSTER_COL: types},
        title=f'Jorstad23 human {subclass} — varimax PCHA space (NOC={noc})',
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
        title=f'Jorstad23 human {subclass} — archetype scores (NOC={noc})',
        out_path=out_score_html,
        xp=sca.xp,
        panels=panels,
        panel_3d=panel_3d,
        aa=sca.aa,
        pctile_low=5, pctile_high=95,
        colorbar_title='archetype score [0–1]',
    )

    del adata, X_norm, sca, vx_df, vx_load
    gc.collect()

print('\nDone.')
