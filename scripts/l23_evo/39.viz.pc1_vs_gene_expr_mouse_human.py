"""PC1-vs-gene-expression scatter PDF — mouse Cheng22 & human Jorstad23 L2/3 IT.

For each requested gene, plot per-cell PCHA PC1 (x) against that gene's normalized
expression (y), mouse and human side by side, one gene per page in a single multi-page
PDF. Points are drawn rasterized (rasterized=True) so each dense cloud is a single
embedded bitmap, while axes/text stay vector. PC1 is read from the cached PCHA embeddings
used by scripts 21.viz / 25.viz (no recompute); the same display-only PC1 sign flip is
applied so orientation matches those archetype figures.

Gene expression:
  - Mouse h5ad holds raw counts -> normalized here as log2(CP10k + 1) (as in script 21).
  - Human h5ad holds already-log-normalized X -> used directly.
  - Human gene symbol = mouse symbol upper-cased (Robo1 -> ROBO1).

Reads:
  local_data/res/l23_evo/21.mouse_pcha_xp.tsv
  local_data/res/l23_evo/25.human_pcha_xp.tsv
  links/l23_evo/cheng22_mouse_IT_P28.h5ad
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  local_data/fig/l23_evo/39.pc1_vs_gene_expr_mouse_human.pdf
  local_data/fig/l23_evo/39.pc1_pc2_gene_expr_mouse_human.pdf

Usage:
  python 39.viz.pc1_vs_gene_expr_mouse_human.py [--genes Robo1 Rorb ...]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_MOUSE_XP = os.path.join(OUT_RES_DIR, '21.mouse_pcha_xp.tsv')
IN_HUMAN_XP = os.path.join(OUT_RES_DIR, '25.human_pcha_xp.tsv')
INPUT_MOUSE = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'cheng22_mouse_IT_P28.h5ad')
INPUT_HUMAN = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_PDF     = os.path.join(OUT_FIG_DIR, '39.pc1_vs_gene_expr_mouse_human.pdf')
OUT_PDF_EMB = os.path.join(OUT_FIG_DIR, '39.pc1_pc2_gene_expr_mouse_human.pdf')

# --- parameters ---
MOUSE_SUBCLASS = 'L2/3'
GENES          = ['Robo1', 'Kirrel3', 'Ntng1', 'Tox', 'Rorb', 'Cntn5', 'Cdh12',   # mouse symbols
                  'Grm1', 'Dscaml1', 'Nfia', 'Epha6', 'Meis2', 'Rfx3', 'Kcnh5', 'Sema6d']
# Display-only PC1/PC2 sign flips, matching archetype figures 21.viz / 25.viz.
MOUSE_PC1_SIGN = 1.0    # mouse FLIP = [1, -1] -> PC1 unchanged, PC2 flipped
MOUSE_PC2_SIGN = -1.0
HUMAN_PC1_SIGN = -1.0   # human FLIP = [-1, 1] -> PC1 flipped, PC2 unchanged
HUMAN_PC2_SIGN = 1.0
POINT_SIZE     = 4
DPI            = 300
N_PC1_BINS     = 10          # PC1 bins for the mean-expression overlay line
EMB_CMAP       = 'RdBu_r'    # PC1-vs-PC2 expression colormap
EMB_PCTILE     = (2, 98)     # per-panel color-scale clip percentiles

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--genes', nargs='+', default=GENES,
                    help='Gene symbols in mouse casing (human = upper-cased). Default: %(default)s')
args = parser.parse_args()
genes = list(args.genes)

os.makedirs(OUT_FIG_DIR, exist_ok=True)


def binned_mean(x, y, n_bins):
    """Mean of y within n_bins equal-width bins over x; returns (bin_centers, means) for non-empty bins."""
    edges   = np.linspace(np.min(x), np.max(x), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx     = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    means   = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
    keep    = ~np.isnan(means)
    return centers[keep], means[keep]


def load_mouse_expr(genes_mouse, cell_index):
    """Return DataFrame (cells x genes) of log2(CP10k+1) mouse expression, reindexed to cell_index."""
    print('Loading mouse h5ad (raw counts)...')
    adata = ad.read_h5ad(INPUT_MOUSE)
    adata = adata[adata.obs['Subclass'] == MOUSE_SUBCLASS]
    print(f'  mouse L2/3 cells: {adata.n_obs}')

    var_names = adata.var_names.values
    missing = [g for g in genes_mouse if g not in set(var_names)]
    if missing:
        raise ValueError(f'Genes not found in mouse var_names: {missing}')

    X_raw  = adata.X.toarray().astype(np.float32)
    depths = X_raw.sum(axis=1, keepdims=True)
    depths[depths == 0] = 1
    data = {}
    for g in genes_mouse:
        idx = int(np.where(var_names == g)[0][0])
        data[g] = np.log2(X_raw[:, idx] / depths[:, 0] * 1e4 + 1)
        print(f'  mouse {g}: var index {idx}')
    df = pd.DataFrame(data, index=adata.obs_names.values)
    df = df.reindex(cell_index)
    if df.isna().any().any():
        raise ValueError('Mouse expression has NaN after reindexing to PC1 barcodes (barcode mismatch).')
    return df


def load_human_expr(genes_human, cell_index):
    """Return DataFrame (cells x genes) of already-log-normalized human expression, reindexed to cell_index."""
    print('Loading human h5ad (already log-normalized)...')
    adata = ad.read_h5ad(INPUT_HUMAN)
    print(f'  human cells: {adata.n_obs}')

    gene_names = (adata.var['feature_name'].values
                  if 'feature_name' in adata.var.columns
                  else adata.var_names.values)
    name_set = set(gene_names)
    missing = [g for g in genes_human if g not in name_set]
    if missing:
        raise ValueError(f'Genes not found in human feature_name: {missing}')

    data = {}
    for g in genes_human:
        idx = int(np.where(gene_names == g)[0][0])
        data[g] = adata.X[:, idx].toarray().astype(np.float32).ravel()
        print(f'  human {g}: var index {idx}')
    df = pd.DataFrame(data, index=adata.obs_names.values)
    df = df.reindex(cell_index)
    if df.isna().any().any():
        raise ValueError('Human expression has NaN after reindexing to PC1 barcodes (barcode mismatch).')
    return df


# --- load cached PC1 (display-flipped to match archetype figures) ---
mouse_xp = pd.read_csv(IN_MOUSE_XP, sep='\t', index_col=0)
human_xp = pd.read_csv(IN_HUMAN_XP, sep='\t', index_col=0)
mouse_pc1 = mouse_xp['PC1'].values * MOUSE_PC1_SIGN
mouse_pc2 = mouse_xp['PC2'].values * MOUSE_PC2_SIGN
human_pc1 = human_xp['PC1'].values * HUMAN_PC1_SIGN
human_pc2 = human_xp['PC2'].values * HUMAN_PC2_SIGN

# --- gene symbols per species ---
genes_human = [g.upper() for g in genes]

# --- expression matrices (aligned to PC1 barcode order) ---
mouse_expr = load_mouse_expr(genes, mouse_xp.index)
human_expr = load_human_expr(genes_human, human_xp.index)

# --- multi-page PDF: one gene per page, mouse (left) + human (right) ---
plt.rcParams['pdf.fonttype'] = 42   # editable vector text
print(f'Writing {OUT_PDF} ({len(genes)} pages)...')
with PdfPages(OUT_PDF) as pdf:
    for g_mouse, g_human in zip(genes, genes_human):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

        for ax, pc1, vals, sym, sp_title in [
            (axes[0], mouse_pc1, mouse_expr[g_mouse].values, g_mouse, 'Cheng22 mouse L2/3 IT'),
            (axes[1], human_pc1, human_expr[g_human].values, g_human, 'Jorstad23 human L2/3 IT'),
        ]:
            ax.scatter(pc1, vals, s=POINT_SIZE, linewidths=0, color='#bbbbbb', rasterized=True)
            bc, bm = binned_mean(pc1, vals, N_PC1_BINS)   # bin-mean overlay (vector)
            ax.plot(bc, bm, '-o', color='#c0392b', linewidth=1.5, markersize=4,
                    zorder=3, label=f'mean over {N_PC1_BINS} PC1 bins')
            ax.set_xlabel('PC1')
            ax.set_ylabel(f'{sym} (log-norm expr)')
            ax.set_title(sp_title)
            ax.legend(frameon=False, fontsize=8, loc='upper right')
            sns.despine(ax=ax)

        fig.suptitle(f'{g_mouse} / {g_human} — PC1 vs expression')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)

print(f'Saved {OUT_PDF}')

# --- multi-page PDF: one gene per page, PC1-vs-PC2 colored by expression ---
print(f'Writing {OUT_PDF_EMB} ({len(genes)} pages)...')
with PdfPages(OUT_PDF_EMB) as pdf:
    for g_mouse, g_human in zip(genes, genes_human):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

        for ax, pc1, pc2, vals, sym, sp_title in [
            (axes[0], mouse_pc1, mouse_pc2, mouse_expr[g_mouse].values, g_mouse, 'Cheng22 mouse L2/3 IT'),
            (axes[1], human_pc1, human_pc2, human_expr[g_human].values, g_human, 'Jorstad23 human L2/3 IT'),
        ]:
            vmin, vmax = np.nanpercentile(vals, EMB_PCTILE)
            sc = ax.scatter(pc1, pc2, c=vals, cmap=EMB_CMAP, vmin=vmin, vmax=vmax,
                            s=POINT_SIZE, linewidths=0, rasterized=True)
            ax.set_aspect('equal', adjustable='box')   # true PC1/PC2 geometry
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(sp_title)
            fig.colorbar(sc, ax=ax, label=f'{sym} (log-norm expr)', shrink=0.8)
            sns.despine(ax=ax)

        fig.suptitle(f'{g_mouse} / {g_human} — PC1 vs PC2, colored by expression')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)

print(f'Saved {OUT_PDF_EMB}')
print('Done.')
