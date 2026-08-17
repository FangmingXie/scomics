"""CCA1-vs-gene-expression views — mouse Cheng22 & human Jorstad23 L2/3 IT.

Cross-species-axis counterpart of script 40: both views are identical, but cells are ordered
along the CONSERVED CCA1 axis instead of each species' own PC1. Two PDFs:
  1. per-bin boxplots, mouse | human                    (one gene per page)
  2. gene x CCA1-bin heatmap per species, all genes in one figure

Why CCA1 rather than PC1: PC1 is fit independently per species, so mouse PC1 and human PC1
are only loosely comparable. CCA1 is the pair of directions (one per species) whose ortholog
gene loadings are maximally correlated across species (r = 0.623, the only canonically stable
component for L2/3) — so bin k means the same thing in both panels, and mouse-vs-human shape
differences are interpretable rather than an artifact of two unrelated axes.

A cell's CCA1 coordinate follows script it_evo/18b exactly: `(VX coords - mean) . a_vx`, where
`a_vx` is the Gate-A VX canonical weight vector from it_evo/16 (normalized and sign-fixed
there). Both species are reflected along CCA1 (axis labelled CCA1'), matching 18b/18c, so the
two panels share one orientation. The cell sets are identical to script 40's (4044 mouse,
47125 human), so the expression pipeline below is unchanged.

Gene expression (both species normalized identically for comparability):
  - Mouse h5ad holds raw counts in X          -> log2(CP10k + 1) (as in script 21).
  - Human h5ad holds raw counts in .raw.X     -> log2(CP10k + 1), same formula.
  - Human gene symbol = mouse symbol upper-cased (Grm1 -> GRM1).

Heatmap panel details (same as script 40):
  - Cell value = that gene's mean log2(CP10k+1) over cells in one of N_CCA1_BINS equal-width
    bins spanning the species' own CCA1 min-max, so mouse and human columns are relative-CCA1
    comparable.
  - Each gene row is z-scored WITHIN its species across that species' bins (mean 0, sd 1), so
    every gene's CCA1 shape reads at full contrast. Mouse-vs-human amplitude is therefore NOT
    comparable across panels — only shape is.
  - Color limits are FIXED at +/- ZLIM (not data-driven), so panels are directly comparable
    across genes, across species and across scripts 40b/40c. |z| beyond ZLIM is clipped, which
    the colorbar marks with arrowheads.
  - Rows are ordered ONCE by hierarchical clustering (correlation distance, average linkage,
    optimal leaf ordering) of the concatenated mouse+human z-scored profile, and that single
    order is applied to both panels so rows can be read across species.

Reads:
  local_data/res/it/19.cheng22_L23_varimax_coords.tsv
  local_data/res/it_evo/02.human_L23_varimax_coords.tsv
  local_data/res/it_evo/16.L23_axis_cca_weights_mouse.tsv
  local_data/res/it_evo/16.L23_axis_cca_weights_human.tsv
  links/l23_evo/cheng22_mouse_IT_P28.h5ad
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  local_data/fig/l23_evo/40b.cca1_gene_expr_boxbins_mouse_human.pdf
  local_data/fig/l23_evo/40b.cca1_gene_expr_heatmap_mouse_human.pdf

Usage:
  python 40b.viz.cca1_gene_expr_boxbins_mouse_human.py [--genes Grm1 Kcnh5 ...]
"""

import os
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
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IT_EVO_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IN_MOUSE_VX    = os.path.join(IT_RES_DIR, '19.cheng22_L23_varimax_coords.tsv')
IN_HUMAN_VX    = os.path.join(IT_EVO_RES_DIR, '02.human_L23_varimax_coords.tsv')
IN_MOUSE_W     = os.path.join(IT_EVO_RES_DIR, '16.L23_axis_cca_weights_mouse.tsv')
IN_HUMAN_W     = os.path.join(IT_EVO_RES_DIR, '16.L23_axis_cca_weights_human.tsv')
INPUT_MOUSE    = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'cheng22_mouse_IT_P28.h5ad')
INPUT_HUMAN    = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_PDF_BOX    = os.path.join(OUT_FIG_DIR, '40b.cca1_gene_expr_boxbins_mouse_human.pdf')
OUT_PDF_HEAT   = os.path.join(OUT_FIG_DIR, '40b.cca1_gene_expr_heatmap_mouse_human.pdf')

# --- parameters ---
MOUSE_SUBCLASS = 'L2/3'
GENES          = ['Grm1', 'Dscaml1', 'Kcnh5', 'Ntng1', 'Cdh13', 'Nfia', 'Rorb',    # mouse symbols
                  'Epha6', 'Pde1a', 'Cntnap2', 'Tox', 'Astn2', 'Gpc6']
CCA_AXIS       = 'CCA1'
HUMAN_VX_COLS  = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']   # Gate-A human L2/3 (it_evo 04/16)
MOUSE_VX_COLS  = ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9']    # Gate-A mouse L2/3 (it_evo 05/16)
# Display-only CCA1 reflection, applied to BOTH species exactly as in it_evo 18b/18c, so the
# two panels share one orientation. Cosmetic: canonical correlations are unchanged.
MOUSE_CCA1_SIGN = -1.0
HUMAN_CCA1_SIGN = -1.0
AXIS_LABEL     = "CCA1'"     # primed: reflected for display (see signs above)
DPI            = 300
N_CCA1_BINS    = 10          # CCA1 bins for the boxplots and the heatmap
MOUSE_COLOR    = '#2166ac'   # mouse boxes
HUMAN_COLOR    = '#b2182b'   # human boxes
FILL_ALPHA     = 0.3         # alpha for the box faces
HEAT_CMAP      = 'RdBu_r'    # gene x CCA1-bin heatmap color (per-gene z, symmetric about 0)
ZLIM           = 2.5         # fixed heatmap color limits (+/-); |z| beyond this is clipped
CLUSTER_METRIC = 'correlation'   # row distance for hierarchical clustering (CCA1 shape)
CLUSTER_METHOD = 'average'       # linkage method

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--genes', nargs='+', default=GENES,
                    help='Gene symbols in mouse casing (human = upper-cased). Default: %(default)s')
args = parser.parse_args()
genes = list(args.genes)

os.makedirs(OUT_FIG_DIR, exist_ok=True)


def cca_axis(vx_path, weights_path, vx_cols, sign, species):
    """Project one species' cells onto CCA_AXIS: (VX coords - mean) . a_vx, as in it_evo 18b.

    Returns (cell_index, coord, canonical_r). The display sign is applied last; it is a
    cosmetic reflection and leaves the canonical correlation untouched.
    """
    vx_df   = pd.read_csv(vx_path, sep='\t', index_col=0)
    weights = pd.read_csv(weights_path, sep='\t', index_col=0)

    missing = [c for c in vx_cols if c not in vx_df.columns]
    if missing:
        raise ValueError(f'{species}: Gate-A VX columns {missing} absent from {vx_path}')

    # the weight row mixes floats with the bool `stable` column, so force float for the matmul
    w = weights.loc[CCA_AXIS, vx_cols].to_numpy(dtype=float)   # normalized + sign-fixed by 16
    r_cca = float(weights.loc[CCA_AXIS, 'canonical_r'])
    print(f'  {species}: {vx_df.shape[0]} cells, VX {vx_cols}, r({CCA_AXIS})={r_cca:.3f} '
          f'(stable={weights.loc[CCA_AXIS, "stable"]})')

    C = vx_df[vx_cols].values
    return vx_df.index, ((C - C.mean(axis=0)) @ w) * sign, r_cca


def binned_stats_relative(x, y, n_bins):
    """Bin y into n_bins equal-width bins over x's own [min, max] range.

    Returns (rel_centers, means, stds) for all n_bins. rel_centers are the bin centers
    mapped to [0, 1] ((b+0.5)/n_bins), so different-range inputs (mouse vs human CCA1)
    sit on exactly the same n_bins x-positions. Empty bins yield NaN mean/std.
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


def binmean_matrix(coord, expr_df, gene_list, species):
    """Rows = genes, cols = N_CCA1_BINS relative CCA1 bins; value = mean log2(CP10k+1) in bin.

    Uses the same relative binning as binned_stats_relative, so mouse and human matrices
    have column-comparable (relative-CCA1) bins. Fails fast if any bin is empty, since an
    empty bin would leave a NaN that hierarchical clustering cannot consume.
    """
    rows = []
    for g in gene_list:
        _, means, _ = binned_stats_relative(coord, expr_df[g].values, N_CCA1_BINS)
        empty = np.where(np.isnan(means))[0]
        if empty.size:
            raise ValueError(f'{species} {g}: CCA1 bins {(empty + 1).tolist()} are empty at '
                             f'N_CCA1_BINS={N_CCA1_BINS}; cannot build the heatmap.')
        rows.append(means)
    return np.vstack(rows)


def zscore_rows(mat, species):
    """Z-score each row across its bins (mean 0, sd 1); fail fast on flat rows."""
    sd  = mat.std(axis=1, keepdims=True)
    bad = np.where(sd[:, 0] == 0)[0]
    if bad.size:
        raise ValueError(f'{species} genes with flat expression across all {AXIS_LABEL} bins '
                         f'(cannot z-score or cluster): {[genes[i] for i in bad]}')
    return (mat - mat.mean(axis=1, keepdims=True)) / sd


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
        raise ValueError('Mouse expression has NaN after reindexing to CCA1 barcodes (barcode mismatch).')
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
        raise ValueError('Human expression has NaN after reindexing to CCA1 barcodes (barcode mismatch).')
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
    ax.set_xticks(range(1, N_CCA1_BINS + 1))
    ax.set_xticklabels(range(1, N_CCA1_BINS + 1))
    ax.set_xlabel(f'{AXIS_LABEL} bin (1..{N_CCA1_BINS}, low→high)')
    ax.set_ylabel(f'{sym} log2(CP10k+1) expr')
    ax.set_title(sp_title)
    sns.despine(ax=ax)


# --- project cells onto the conserved CCA1 axis (display-reflected) ---
print('Projecting cells onto the cross-species CCA1 axis...')
mouse_index, mouse_cca1, mouse_r = cca_axis(
    IN_MOUSE_VX, IN_MOUSE_W, MOUSE_VX_COLS, MOUSE_CCA1_SIGN, 'mouse')
human_index, human_cca1, human_r = cca_axis(
    IN_HUMAN_VX, IN_HUMAN_W, HUMAN_VX_COLS, HUMAN_CCA1_SIGN, 'human')
if not np.isclose(mouse_r, human_r):
    raise ValueError(f'Canonical correlations differ between species ({mouse_r} vs {human_r}); '
                     'the two weight tables must come from the same CCA fit.')
R_CCA1 = mouse_r

# --- gene symbols per species ---
genes_human = [g.upper() for g in genes]

# --- expression matrices (aligned to CCA1 barcode order) ---
mouse_expr = load_mouse_expr(genes, mouse_index)
human_expr = load_human_expr(genes_human, human_index)

MOUSE_TITLE = 'Cheng22 mouse L2/3 IT'
HUMAN_TITLE = 'Jorstad23 human L2/3 IT'
plt.rcParams['pdf.fonttype'] = 42   # editable vector text

# --- multi-page PDF: one gene per page, per-bin boxplots (mouse left, human right) ---
print(f'Writing {OUT_PDF_BOX} ({len(genes)} pages)...')
with PdfPages(OUT_PDF_BOX) as pdf:
    for g_mouse, g_human in zip(genes, genes_human):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

        for ax, coord, vals, color, sym, sp_title in [
            (axes[0], mouse_cca1, mouse_expr[g_mouse].values, MOUSE_COLOR, g_mouse, MOUSE_TITLE),
            (axes[1], human_cca1, human_expr[g_human].values, HUMAN_COLOR, g_human, HUMAN_TITLE),
        ]:
            positions, groups = binned_groups(coord, vals, N_CCA1_BINS)
            draw_boxbins(ax, positions, groups, color, sym, sp_title)

        fig.suptitle(f'{g_mouse} / {g_human} — expression by {AXIS_LABEL} bin '
                     f'(boxplots; r = {R_CCA1:.3f})')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)

print(f'Saved {OUT_PDF_BOX}')

# --- single-page PDF: gene x CCA1-bin heatmap per species, shared clustered row order ---
print(f'Writing {OUT_PDF_HEAT} ({len(genes)} genes x {N_CCA1_BINS} {AXIS_LABEL} bins)...')
mouse_mat = binmean_matrix(mouse_cca1, mouse_expr, genes, 'mouse')
human_mat = binmean_matrix(human_cca1, human_expr, genes_human, 'human')

# per-gene z-scoring WITHIN each species (each row centered on its own species-specific mean)
mouse_scaled = zscore_rows(mouse_mat, 'mouse')
human_scaled = zscore_rows(human_mat, 'human')
joint_scaled = np.hstack([mouse_scaled, human_scaled])
n_clipped = int((np.abs(joint_scaled) > ZLIM).sum())
if n_clipped:
    print(f'  NOTE: {n_clipped} of {joint_scaled.size} cells exceed |z| = {ZLIM} and are '
          f'clipped (max |z| = {np.abs(joint_scaled).max():.2f})')

# one row order from the joint profile, applied to both panels so rows read across species
row_order = leaves_list(linkage(joint_scaled, method=CLUSTER_METHOD, metric=CLUSTER_METRIC,
                                optimal_ordering=True))
print(f'Row order ({CLUSTER_METHOD}/{CLUSTER_METRIC} clustering): {[genes[i] for i in row_order]}')

fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.34 * len(genes) + 2.0))
for ax, mat, gene_syms, sp_title, label_side in [
    (axes[0], mouse_scaled[row_order], [genes[i] for i in row_order], MOUSE_TITLE, 'left'),
    (axes[1], human_scaled[row_order], [genes_human[i] for i in row_order], HUMAN_TITLE, 'right'),
]:
    im = ax.imshow(mat, aspect='auto', cmap=HEAT_CMAP, vmin=-ZLIM, vmax=ZLIM)
    ax.set_xticks(range(N_CCA1_BINS))
    ax.set_xticklabels(range(1, N_CCA1_BINS + 1))
    ax.set_xlabel(f'{AXIS_LABEL} bin (1..{N_CCA1_BINS}, low→high)')
    ax.set_yticks(range(len(gene_syms)))
    ax.set_yticklabels(gene_syms, fontsize=8)
    ax.yaxis.set_label_position(label_side)
    ax.yaxis.set_ticks_position(label_side)
    ax.set_title(sp_title)

fig.colorbar(im, ax=axes, label='z-scored mean expr\n(per gene, within each species)',
             shrink=0.6, pad=0.1, extend='both')
fig.suptitle(f'{len(genes)} genes — mean expression over conserved {AXIS_LABEL} bins '
             f'(r = {R_CCA1:.3f}; rows: hierarchical clustering)')
fig.savefig(OUT_PDF_HEAT, bbox_inches='tight', dpi=DPI)
plt.close(fig)

print(f'Saved {OUT_PDF_HEAT}')
print('Done.')
