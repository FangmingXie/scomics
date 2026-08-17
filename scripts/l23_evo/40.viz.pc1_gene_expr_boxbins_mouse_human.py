"""PC1-vs-gene-expression views (incl. per-bin boxplots) — mouse Cheng22 & human Jorstad23 L2/3 IT.

Short-gene-list variant of script 39. Produces the same three PDFs as 39, plus a fourth PDF
that shows, per gene, a boxplot of expression across 10 PC1 bins (mouse and human side by
side) — a distributional alternative to the scatter + mean-expression view — plus a fifth
PDF that collapses ALL selected genes into one figure: a gene x PC1-bin heatmap per species
(mouse left, human right). PC1 is read from the cached PCHA embeddings used by scripts
21.viz / 25.viz (no recompute); the same display-only PC1 sign flip is applied so
orientation matches those archetype figures.

Heatmap panel details:
  - Cell value = that gene's mean log2(CP10k+1) over cells in one of N_PC1_BINS equal-width
    bins spanning the species' own PC1 min–max (same binning as the mean±std overlay), so
    mouse and human columns are relative-PC1 comparable.
  - Each gene row is z-scored WITHIN its species across that species' bins (mean 0, sd 1), so
    every gene's PC1 shape reads at full contrast and the diverging colormap is centered on
    each gene's own species-specific mean. Mouse-vs-human amplitude is therefore NOT
    comparable across the two panels — only the shape is. Color limits are symmetric and
    shared by both panels (+/- the largest |z| over all genes and both species).
  - Rows are ordered ONCE by hierarchical clustering (correlation distance, average linkage,
    optimal leaf ordering) of the concatenated mouse+human z-scored profile, and that single
    order is applied to both panels so rows can be read across species.

Gene expression (both species normalized identically for comparability):
  - Mouse h5ad holds raw counts in X          -> log2(CP10k + 1) (as in script 21).
  - Human h5ad holds raw counts in .raw.X     -> log2(CP10k + 1), same formula.
    (The pre-normalized human X has an unrecorded log base, so it is not used.)
  - Human gene symbol = mouse symbol upper-cased (Grm1 -> GRM1).

Reads:
  local_data/res/l23_evo/21.mouse_pcha_xp.tsv
  local_data/res/l23_evo/25.human_pcha_xp.tsv
  links/l23_evo/cheng22_mouse_IT_P28.h5ad
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  local_data/fig/l23_evo/40.pc1_vs_gene_expr_mouse_human.pdf
  local_data/fig/l23_evo/40.pc1_pc2_gene_expr_mouse_human.pdf
  local_data/fig/l23_evo/40.pc1_binmean_overlay_mouse_human.pdf
  local_data/fig/l23_evo/40.pc1_gene_expr_boxbins_mouse_human.pdf
  local_data/fig/l23_evo/40.pc1_gene_expr_heatmap_mouse_human.pdf

Usage:
  python 40.viz.pc1_gene_expr_boxbins_mouse_human.py [--genes Grm1 Kcnh5 ...]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage, leaves_list
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_MOUSE_XP = os.path.join(OUT_RES_DIR, '21.mouse_pcha_xp.tsv')
IN_HUMAN_XP = os.path.join(OUT_RES_DIR, '25.human_pcha_xp.tsv')
INPUT_MOUSE = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'cheng22_mouse_IT_P28.h5ad')
INPUT_HUMAN = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_PDF     = os.path.join(OUT_FIG_DIR, '40.pc1_vs_gene_expr_mouse_human.pdf')
OUT_PDF_EMB = os.path.join(OUT_FIG_DIR, '40.pc1_pc2_gene_expr_mouse_human.pdf')
OUT_PDF_OVL = os.path.join(OUT_FIG_DIR, '40.pc1_binmean_overlay_mouse_human.pdf')
OUT_PDF_BOX = os.path.join(OUT_FIG_DIR, '40.pc1_gene_expr_boxbins_mouse_human.pdf')
OUT_PDF_HEAT = os.path.join(OUT_FIG_DIR, '40.pc1_gene_expr_heatmap_mouse_human.pdf')

# --- parameters ---
MOUSE_SUBCLASS = 'L2/3'
GENES          = ['Grm1', 'Dscaml1', 'Kcnh5', 'Ntng1', 'Cdh13', 'Nfia', 'Rorb',    # mouse symbols
                  'Epha6', 'Pde1a', 'Cntnap2', 'Tox', 'Astn2', 'Gpc6']
# Display-only PC1/PC2 sign flips, matching archetype figures 21.viz / 25.viz.
MOUSE_PC1_SIGN = 1.0    # mouse FLIP = [1, -1] -> PC1 unchanged, PC2 flipped
MOUSE_PC2_SIGN = -1.0
HUMAN_PC1_SIGN = -1.0   # human FLIP = [-1, 1] -> PC1 flipped, PC2 unchanged
HUMAN_PC2_SIGN = 1.0
POINT_SIZE     = 4
DPI            = 300
N_PC1_BINS     = 10          # PC1 bins for the mean-expression overlay line and boxplots
EMB_CMAP       = 'RdBu_r'    # PC1-vs-PC2 expression colormap
EMB_PCTILE     = (2, 98)     # per-panel color-scale clip percentiles
MOUSE_COLOR    = '#2166ac'   # mouse curve/boxes
HUMAN_COLOR    = '#b2182b'   # human curve/boxes
FILL_ALPHA     = 0.3         # alpha for the +/- std fill bands and box faces
HEAT_CMAP      = 'RdBu_r'    # gene x PC1-bin heatmap color (per-gene z, symmetric about 0)
CLUSTER_METRIC = 'correlation'   # row distance for hierarchical clustering (PC1 shape)
CLUSTER_METHOD = 'average'       # linkage method

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


def binned_stats_relative(x, y, n_bins):
    """Bin y into n_bins equal-width bins over x's own [min, max] range.

    Returns (rel_centers, means, stds) for all n_bins. rel_centers are the bin centers
    mapped to [0, 1] ((b+0.5)/n_bins), so different-range inputs (mouse vs human PC1)
    overlay on exactly the same n_bins x-positions. Empty bins yield NaN mean/std.
    """
    edges   = np.linspace(np.min(x), np.max(x), n_bins + 1)
    idx     = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    means   = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
    stds    = np.array([y[idx == b].std()  if np.any(idx == b) else np.nan for b in range(n_bins)])
    centers = (np.arange(n_bins) + 0.5) / n_bins
    return centers, means, stds


def binned_groups(x, y, n_bins):
    """Split y into n_bins lists by equal-width bins over x's own [min, max] range.

    Returns (positions, groups) for non-empty bins only: positions are 1-based bin indices
    (1..n_bins) and groups[i] is the array of y-values in that bin (for ax.boxplot).
    """
    edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    idx   = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    positions, groups = [], []
    for b in range(n_bins):
        vals = y[idx == b]
        if vals.size:
            positions.append(b + 1)
            groups.append(vals)
    return positions, groups


def binmean_matrix(pc1, expr_df, gene_list, species):
    """Rows = genes, cols = N_PC1_BINS relative PC1 bins; value = mean log2(CP10k+1) in bin.

    Uses the same relative binning as binned_stats_relative, so mouse and human matrices
    have column-comparable (relative-PC1) bins. Fails fast if any bin is empty, since an
    empty bin would leave a NaN that hierarchical clustering cannot consume.
    """
    rows = []
    for g in gene_list:
        _, means, _ = binned_stats_relative(pc1, expr_df[g].values, N_PC1_BINS)
        empty = np.where(np.isnan(means))[0]
        if empty.size:
            raise ValueError(f'{species} {g}: PC1 bins {(empty + 1).tolist()} are empty at '
                             f'N_PC1_BINS={N_PC1_BINS}; cannot build the heatmap.')
        rows.append(means)
    return np.vstack(rows)


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
    """Return DataFrame (cells x genes) of log2(CP10k+1) human expression, reindexed to cell_index.

    Normalized from raw counts (adata.raw.X) with the same log2(CP10k+1) formula as mouse,
    so both species are on an identical log2-CP10k scale. The pre-normalized adata.X is not
    used (its log base is unrecorded).
    """
    print('Loading human h5ad (raw counts from .raw)...')
    adata = ad.read_h5ad(INPUT_HUMAN)
    print(f'  human cells: {adata.n_obs}')
    if adata.raw is None:
        raise ValueError('Human h5ad has no .raw; cannot recompute log2(CP10k+1) from raw counts.')

    gene_names = (adata.raw.var['feature_name'].values
                  if 'feature_name' in adata.raw.var.columns
                  else adata.raw.var_names.values)
    name_set = set(gene_names)
    missing = [g for g in genes_human if g not in name_set]
    if missing:
        raise ValueError(f'Genes not found in human raw.var feature_name: {missing}')

    X_raw  = adata.raw.X.toarray().astype(np.float32)
    depths = X_raw.sum(axis=1, keepdims=True)
    depths[depths == 0] = 1
    data = {}
    for g in genes_human:
        idx = int(np.where(gene_names == g)[0][0])
        data[g] = np.log2(X_raw[:, idx] / depths[:, 0] * 1e4 + 1)
        print(f'  human {g}: raw.var index {idx}')
    df = pd.DataFrame(data, index=adata.obs_names.values)
    df = df.reindex(cell_index)
    if df.isna().any().any():
        raise ValueError('Human expression has NaN after reindexing to PC1 barcodes (barcode mismatch).')
    return df


def draw_boxbins(ax, positions, groups, color, sym, sp_title):
    """Draw per-bin boxplots (fliers hidden) colored by species onto ax."""
    bp = ax.boxplot(groups, positions=positions, widths=0.6, showfliers=False,
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(FILL_ALPHA)
        patch.set_edgecolor(color)
    for part in ('whiskers', 'caps'):
        for line in bp[part]:
            line.set_color(color)
    for median in bp['medians']:
        median.set_color('black')
    ax.set_xticks(range(1, N_PC1_BINS + 1))
    ax.set_xticklabels(range(1, N_PC1_BINS + 1))
    ax.set_xlabel(f'PC1 bin (1..{N_PC1_BINS}, low→high)')
    ax.set_ylabel(f'{sym} log2(CP10k+1) expr')
    ax.set_title(sp_title)
    sns.despine(ax=ax)


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

# --- multi-page PDF: one gene per page, mouse+human mean +/- std over 10 relative PC1 bins ---
print(f'Writing {OUT_PDF_OVL} ({len(genes)} pages)...')
with PdfPages(OUT_PDF_OVL) as pdf:
    for g_mouse, g_human in zip(genes, genes_human):
        fig, ax = plt.subplots(figsize=(5.4, 4.2))

        for pc1, vals, color, lbl in [
            (mouse_pc1, mouse_expr[g_mouse].values, MOUSE_COLOR, f'mouse {g_mouse}'),
            (human_pc1, human_expr[g_human].values, HUMAN_COLOR, f'human {g_human}'),
        ]:
            bc, bm, bs = binned_stats_relative(pc1, vals, N_PC1_BINS)
            ax.fill_between(bc, bm - bs, bm + bs, color=color, alpha=FILL_ALPHA, linewidth=0)
            ax.plot(bc, bm, '-o', color=color, linewidth=1.5, markersize=4, label=lbl)

        ax.set_xlabel(f'PC1 (relative, per-species min–max; {N_PC1_BINS} bins)')
        ax.set_ylabel('log2(CP10k+1) expr')
        ax.set_title(f'{g_mouse} / {g_human} — mean ± std over PC1 bins')
        ax.legend(frameon=False, fontsize=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)

print(f'Saved {OUT_PDF_OVL}')

# --- multi-page PDF: one gene per page, per-bin boxplots (mouse left, human right) ---
print(f'Writing {OUT_PDF_BOX} ({len(genes)} pages)...')
with PdfPages(OUT_PDF_BOX) as pdf:
    for g_mouse, g_human in zip(genes, genes_human):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

        for ax, pc1, vals, color, sym, sp_title in [
            (axes[0], mouse_pc1, mouse_expr[g_mouse].values, MOUSE_COLOR, g_mouse, 'Cheng22 mouse L2/3 IT'),
            (axes[1], human_pc1, human_expr[g_human].values, HUMAN_COLOR, g_human, 'Jorstad23 human L2/3 IT'),
        ]:
            positions, groups = binned_groups(pc1, vals, N_PC1_BINS)
            draw_boxbins(ax, positions, groups, color, sym, sp_title)

        fig.suptitle(f'{g_mouse} / {g_human} — expression by PC1 bin (boxplots)')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)

print(f'Saved {OUT_PDF_BOX}')

# --- single-page PDF: gene x PC1-bin heatmap per species, shared clustered row order ---
print(f'Writing {OUT_PDF_HEAT} ({len(genes)} genes x {N_PC1_BINS} PC1 bins)...')
mouse_mat = binmean_matrix(mouse_pc1, mouse_expr, genes, 'mouse')
human_mat = binmean_matrix(human_pc1, human_expr, genes_human, 'human')

# per-gene z-scoring WITHIN each species (each row centered on its own species-specific mean)
def zscore_rows(mat, species):
    """Z-score each row across its bins (mean 0, sd 1); fail fast on flat rows."""
    sd  = mat.std(axis=1, keepdims=True)
    bad = np.where(sd[:, 0] == 0)[0]
    if bad.size:
        raise ValueError(f'{species} genes with flat expression across all PC1 bins '
                         f'(cannot z-score or cluster): {[genes[i] for i in bad]}')
    return (mat - mat.mean(axis=1, keepdims=True)) / sd


mouse_scaled = zscore_rows(mouse_mat, 'mouse')
human_scaled = zscore_rows(human_mat, 'human')
joint_scaled = np.hstack([mouse_scaled, human_scaled])
zlim = np.abs(joint_scaled).max()   # symmetric, shared color limits so 0 = each gene's mean

# one row order from the joint profile, applied to both panels so rows read across species
row_order = leaves_list(linkage(joint_scaled, method=CLUSTER_METHOD, metric=CLUSTER_METRIC,
                                optimal_ordering=True))
print(f'Row order ({CLUSTER_METHOD}/{CLUSTER_METRIC} clustering): {[genes[i] for i in row_order]}')

fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.34 * len(genes) + 2.0))
for ax, mat, gene_syms, sp_title, label_side in [
    (axes[0], mouse_scaled[row_order], [genes[i] for i in row_order],
     'Cheng22 mouse L2/3 IT', 'left'),
    (axes[1], human_scaled[row_order], [genes_human[i] for i in row_order],
     'Jorstad23 human L2/3 IT', 'right'),
]:
    im = ax.imshow(mat, aspect='auto', cmap=HEAT_CMAP, vmin=-zlim, vmax=zlim)
    ax.set_xticks(range(N_PC1_BINS))
    ax.set_xticklabels(range(1, N_PC1_BINS + 1))
    ax.set_xlabel(f'PC1 bin (1..{N_PC1_BINS}, low→high)')
    ax.set_yticks(range(len(gene_syms)))
    ax.set_yticklabels(gene_syms, fontsize=8)
    ax.yaxis.set_label_position(label_side)
    ax.yaxis.set_ticks_position(label_side)
    ax.set_title(sp_title)

fig.colorbar(im, ax=axes, label='z-scored mean expr\n(per gene, within each species)',
             shrink=0.6, pad=0.1)
fig.suptitle(f'{len(genes)} genes — mean expression over PC1 bins (rows: hierarchical clustering)')
fig.savefig(OUT_PDF_HEAT, bbox_inches='tight', dpi=DPI)
plt.close(fig)

print(f'Saved {OUT_PDF_HEAT}')
print('Done.')
