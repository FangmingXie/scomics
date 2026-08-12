"""Top-CCA2 gene expression on the human L2/3 CCA1×CCA2 embedding (L2/3 only, plots only).

Companion to script 18. Same human cell embedding — each Jorstad23 L2/3 cell positioned by
its projection onto the conserved human CCA1/CCA2 canonical directions (script 16), with the
PCHA archetype vertices (04) overlaid — but here the cells are colored by the ln(CPM+1)
expression of the genes that contribute most to CCA2 (SORCS3, KCNH7, SGCZ, …).

"Contribute most to CCA2" is the conserved ranking used in 16's report: genes ordered by
|human_CCA2_loading · mouse_CCA2_loading| over shared orthologs, using the 16 canonical
weights. Script 16 gates its `*_top_genes.tsv` to bootstrap-stable pairs and so withholds
CCA2; this script recomputes the ranking from the persisted weights + loadings (no CCA refit).

Reads:
  local_data/res/it_evo/02.human_L23_varimax_{coords,loadings}.tsv
  local_data/res/it/19.cheng22_L23_varimax_loadings.tsv
  local_data/res/it_evo/16.L23_axis_cca_weights_{human,mouse}.tsv
  local_data/res/it_evo/04.human_L23_pcha_{aa,inner_components,inner_mean}.tsv
  data/human_mouse_orthologs.tsv
  links/it_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  local_data/fig/it_evo/19.human_L23_cca2_gene_expr_cca.pdf
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_VX_COORDS  = os.path.join(RES_DIR, '02.human_L23_varimax_coords.tsv')
IN_H_LOADINGS = os.path.join(RES_DIR, '02.human_L23_varimax_loadings.tsv')
IN_M_LOADINGS = os.path.join(IT_RES_DIR, '19.cheng22_L23_varimax_loadings.tsv')
IN_W_HUMAN    = os.path.join(RES_DIR, '16.L23_axis_cca_weights_human.tsv')
IN_W_MOUSE    = os.path.join(RES_DIR, '16.L23_axis_cca_weights_mouse.tsv')
IN_AA         = os.path.join(RES_DIR, '04.human_L23_pcha_aa.tsv')
IN_INNER_CMP  = os.path.join(RES_DIR, '04.human_L23_pcha_inner_components.tsv')
IN_INNER_MEAN = os.path.join(RES_DIR, '04.human_L23_pcha_inner_mean.tsv')
IN_ORTHOLOGS  = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_H5AD       = os.path.join(PROJECT_ROOT, 'links', 'it_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_PDF       = os.path.join(FIG_DIR, '19.human_L23_cca2_gene_expr_cca.pdf')

# --- config ---
HUMAN_VX    = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']
MOUSE_VX    = ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9']
CCA_AXES    = ['CCA1', 'CCA2']
AXIS_CCA    = 'CCA2'                 # axis whose top genes to plot
TOP_N_GENES = 9
NCOLS       = 3
GENE_NAME_COL = 'feature_name'
AA_RENAME   = {'archetype_1': "D'", 'archetype_2': "C'", 'archetype_3': "B'", 'archetype_4': "A'"}
EXPR_CMAP   = 'viridis'
EXPR_PCTILE_HI = 99                  # clip color scale at this expression percentile (vmin=0)
POINT_SIZE  = 3

os.makedirs(FIG_DIR, exist_ok=True)


def cca_embedding():
    """Human cells and PCHA archetype vertices projected onto CCA1×CCA2 (as script 18)."""
    vx_df = pd.read_csv(IN_VX_COORDS, sep='\t', index_col=0)
    weights = pd.read_csv(IN_W_HUMAN, sep='\t', index_col=0)
    aa_df = pd.read_csv(IN_AA, sep='\t', index_col=0)
    inner_cmp = pd.read_csv(IN_INNER_CMP, sep='\t', index_col=0)
    inner_mean = pd.read_csv(IN_INNER_MEAN, sep='\t', index_col=0)[HUMAN_VX].values.ravel()

    W = weights.loc[CCA_AXES, HUMAN_VX].values.T            # 6 × 2
    r_cca = weights.loc[CCA_AXES, 'canonical_r'].values
    C = vx_df[HUMAN_VX].values
    mean_vx = C.mean(axis=0)
    cca_cells = (C - mean_vx) @ W

    aa_vx = aa_df.values @ inner_cmp.loc[list(aa_df.columns), HUMAN_VX].values + inner_mean
    cca_aa = (aa_vx - mean_vx) @ W
    aa_labels = np.array([AA_RENAME[i] for i in aa_df.index])
    ang = np.arctan2(cca_aa[:, 1] - cca_aa[:, 1].mean(), cca_aa[:, 0] - cca_aa[:, 0].mean())
    order = np.argsort(ang)
    return vx_df.index, cca_cells, cca_aa[order], aa_labels[order], r_cca


def top_cca_genes():
    """Genes ranked by |human · mouse| CCA-axis loading over shared orthologs (16 weights)."""
    ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
             .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
    H = pd.read_csv(IN_H_LOADINGS, sep='\t', index_col=0)
    M = pd.read_csv(IN_M_LOADINGS, sep='\t', index_col=0)
    shared = ortho[ortho['human_symbol'].isin(H.index)
                   & ortho['mouse_symbol'].isin(M.index)].reset_index(drop=True)
    wh = pd.read_csv(IN_W_HUMAN, sep='\t', index_col=0).loc[AXIS_CCA, HUMAN_VX].values
    wm = pd.read_csv(IN_W_MOUSE, sep='\t', index_col=0).loc[AXIS_CCA, MOUSE_VX].values
    X = H.loc[shared['human_symbol'].values, HUMAN_VX].values
    Y = M.loc[shared['mouse_symbol'].values, MOUSE_VX].values
    hs = (X - X.mean(0)) @ wh
    ms = (Y - Y.mean(0)) @ wm
    order = np.argsort(np.abs(hs * ms))[::-1][:TOP_N_GENES]
    return list(shared.loc[order, 'human_symbol'].values)


def grid_gene_scatter(cca_cells, expr, genes, cca_aa, aa_labels, r_cca, out_path):
    """One CCA1×CCA2 panel per gene, colored by expression; archetype diamonds + polygon."""
    plt.rcParams['pdf.fonttype'] = 42
    n = len(genes)
    nrows = int(np.ceil(n / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(4.2 * NCOLS, 4 * nrows), squeeze=False)
    for k, gene in enumerate(genes):
        ax = axes[k // NCOLS][k % NCOLS]
        vals = expr[:, k]
        vmax = np.percentile(vals, EXPR_PCTILE_HI)
        vmax = vmax if vmax > 0 else vals.max()
        srt = np.argsort(vals)                              # high expression drawn last
        sc = ax.scatter(cca_cells[srt, 0], cca_cells[srt, 1], c=vals[srt], cmap=EXPR_CMAP,
                        vmin=0, vmax=vmax, s=POINT_SIZE, linewidths=0, rasterized=True)
        ax.plot(list(cca_aa[:, 0]) + [cca_aa[0, 0]], list(cca_aa[:, 1]) + [cca_aa[0, 1]],
                '-', color='black', linewidth=1.0)
        ax.scatter(cca_aa[:, 0], cca_aa[:, 1], marker='D', color='black', s=30, zorder=3)
        for (vx_, vy_), label in zip(cca_aa, aa_labels):
            ax.annotate(label, (vx_, vy_), textcoords='offset points', xytext=(5, 5),
                        fontsize=8, fontweight='bold', color='black', zorder=4)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'CCA1 (r={r_cca[0]:.3f})')
        ax.set_ylabel(f'CCA2 (r={r_cca[1]:.3f})')
        ax.set_title(gene, fontstyle='italic')
        fig.colorbar(sc, ax=ax, label='ln(CPM+1)', shrink=0.8)
        sns.despine(ax=ax)
    for k in range(n, nrows * NCOLS):                       # blank any unused axes
        axes[k // NCOLS][k % NCOLS].axis('off')
    fig.suptitle('Jorstad23 human L2/3 IT — top-CCA2 gene expression on the conserved '
                 'CCA1×CCA2 embedding')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  Saved {out_path}')


# --- run ---
cell_index, cca_cells, cca_aa, aa_labels, r_cca = cca_embedding()
genes = top_cca_genes()
print(f'Top {TOP_N_GENES} {AXIS_CCA} genes: {", ".join(genes)}')

print('Loading human expression...')
adata = ad.read_h5ad(IN_H5AD)
adata = adata[cell_index]                                   # align to embedding order
feat = adata.var[GENE_NAME_COL].values
gene_idx = [int(np.where(feat == g)[0][0]) for g in genes]
expr = adata.X[:, gene_idx].toarray().astype(np.float32)    # slice before densify (G7)

print(f'--- L2/3: {AXIS_CCA} top-gene expression on human CCA1×CCA2 '
      f'(r = {r_cca[0]:.3f}, {r_cca[1]:.3f}) ---')
grid_gene_scatter(cca_cells, expr, genes, cca_aa, aa_labels, r_cca, OUT_PDF)
print('\nDone.')
