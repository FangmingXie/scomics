"""CCA2-vs-gene-expression views for the top archetype-B genes — mouse Cheng22 & human Jorstad23 L2/3 IT.

Third view in the 40 family. Script 40 orders cells along each species' own PC1, 40b along the
conserved CCA1; this one orders cells along the conserved CCA2 and, instead of a hand-picked
gene list, DERIVES its genes from script it_evo/24b: the mouse archetype-B genes that
contribute most to CCA2.

Gene selection (mirrors it_evo/24b exactly):
  - Gene universe = 1-to-1 orthologs present in both species' expanded varimax loadings
    (26.{human,mouse}_L23_varimax_loadings_full.tsv), restricted per --universe. Default
    `hvg_union` = genes that are HVG in EITHER species (n = 3220), which is 24b's default.
  - Per-gene canonical CCA2 loading per species = centered VX loadings . the Gate-A canonical
    weight vector from it_evo/16 (`ms` for mouse, `hs` for human).
  - Archetype membership = mouse archetype markers (05.mouse_L23_archetype_markers.tsv),
    each gene assigned to its highest-log2FC archetype. Internal key B == displayed B' under
    24b's ARCH_RELABEL {A:C', B:B', C:A'}; 35 of the 3220 shared genes are B genes.
  - "Contributes most to CCA2" = largest |ms * hs|, the same product 24b ranks its labeled
    top genes by. The top N_GENES B genes by that product become the rows/pages here.

Cell coordinates follow it_evo/18b: `(VX coords - mean) . a_vx`. NOTE the universe caveat —
18b and script 40b use the `hvg_intersect` weights, whereas this script uses ONE universe for
both the gene ranking and the cell projection (default `hvg_union`, matching 24b), so its CCA
axes are NOT numerically identical to 40b's. Under `hvg_union` both CCA1 and CCA2 are
bootstrap-stable (r = 0.358 and 0.313); under `hvg_intersect` CCA2 is NOT stable, which is the
main reason the union is the default here. Pass --universe hvg_intersect to match 40b/18b.
CCA1 is display-reflected as in 18b/18c (label CCA1'); CCA2 is shown unflipped.

Gene expression (both species normalized identically for comparability):
  - Mouse h5ad holds raw counts in X          -> log2(CP10k + 1) (as in script 21).
  - Human h5ad holds raw counts in .raw.X     -> log2(CP10k + 1), same formula.
  - Human symbol comes from the ORTHOLOG TABLE (not upper-casing, as in 40/40b), since the
    gene list is derived in ortholog space.

Five PDFs, all along CCA2 (same views as 40/40b):
  1. CCA2 vs expression scatter + bin-mean overlay      (one gene per page)
  2. CCA2 x CCA1 embedding colored by expression        (one gene per page)
  3. mouse+human mean +/- std over 10 relative bins     (one gene per page)
  4. per-bin boxplots, mouse | human                    (one gene per page)
  5. gene x CCA2-bin heatmap per species, all genes in one figure — per-gene z-scored WITHIN
     each species, symmetric shared color limits, rows ordered once by hierarchical clustering
     (correlation distance, average linkage, optimal leaf ordering) of the concatenated
     mouse+human profile and applied to both panels.

Reads:
  local_data/res/it/19.cheng22_L23_varimax_coords.tsv
  local_data/res/it/19.cheng22_L23_varimax_loadings.tsv
  local_data/res/it_evo/02.human_L23_varimax_{coords,loadings}.tsv
  local_data/res/it_evo/26.{human,mouse}_L23_varimax_loadings_full.tsv
  local_data/res/it_evo/16.L23_axis_cca_weights_{human,mouse}<SUFFIX>.tsv
  local_data/res/it_evo/05.mouse_L23_archetype_markers.tsv
  data/human_mouse_orthologs.tsv
  links/l23_evo/cheng22_mouse_IT_P28.h5ad
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  local_data/fig/l23_evo/40c.cca2_vs_gene_expr_mouse_human<SUFFIX>.pdf
  local_data/fig/l23_evo/40c.cca2_cca1_gene_expr_mouse_human<SUFFIX>.pdf
  local_data/fig/l23_evo/40c.cca2_binmean_overlay_mouse_human<SUFFIX>.pdf
  local_data/fig/l23_evo/40c.cca2_gene_expr_boxbins_mouse_human<SUFFIX>.pdf
  local_data/fig/l23_evo/40c.cca2_gene_expr_heatmap_mouse_human<SUFFIX>.pdf
  local_data/res/l23_evo/40c.cca2_bgene_contributions<SUFFIX>.tsv   (all B genes, ranked)

Usage:
  python 40c.viz.cca2_bgene_expr_boxbins_mouse_human.py [--n-genes 15] [--universe hvg_union]
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

# --- config ---
UNIVERSE_SUFFIX = {'hvg_intersect': '', 'hvg_union': '_union'}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--n-genes', type=int, default=15,
                    help='Number of top archetype-B genes by |ms*hs| on CCA2. Default: %(default)s')
parser.add_argument('--universe', choices=list(UNIVERSE_SUFFIX), default='hvg_union',
                    help='Ortholog gene universe, for BOTH gene ranking and cell projection. '
                         'Default: %(default)s (matches it_evo/24b)')
args     = parser.parse_args()
N_GENES  = args.n_genes
UNIVERSE = args.universe
SUFFIX   = UNIVERSE_SUFFIX[UNIVERSE]

# --- file paths ---
OUT_FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
OUT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
IT_RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IT_EVO_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IN_MOUSE_VX    = os.path.join(IT_RES_DIR, '19.cheng22_L23_varimax_coords.tsv')
IN_HUMAN_VX    = os.path.join(IT_EVO_RES_DIR, '02.human_L23_varimax_coords.tsv')
IN_MOUSE_W     = os.path.join(IT_EVO_RES_DIR, f'16.L23_axis_cca_weights_mouse{SUFFIX}.tsv')
IN_HUMAN_W     = os.path.join(IT_EVO_RES_DIR, f'16.L23_axis_cca_weights_human{SUFFIX}.tsv')
IN_M_HVG       = os.path.join(IT_RES_DIR, '19.cheng22_L23_varimax_loadings.tsv')
IN_H_HVG       = os.path.join(IT_EVO_RES_DIR, '02.human_L23_varimax_loadings.tsv')
IN_M_FULL      = os.path.join(IT_EVO_RES_DIR, '26.mouse_L23_varimax_loadings_full.tsv')
IN_H_FULL      = os.path.join(IT_EVO_RES_DIR, '26.human_L23_varimax_loadings_full.tsv')
IN_MARKERS     = os.path.join(IT_EVO_RES_DIR, '05.mouse_L23_archetype_markers.tsv')
IN_ORTHOLOGS   = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
INPUT_MOUSE    = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'cheng22_mouse_IT_P28.h5ad')
INPUT_HUMAN    = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_PDF        = os.path.join(OUT_FIG_DIR, f'40c.cca2_vs_gene_expr_mouse_human{SUFFIX}.pdf')
OUT_PDF_EMB    = os.path.join(OUT_FIG_DIR, f'40c.cca2_cca1_gene_expr_mouse_human{SUFFIX}.pdf')
OUT_PDF_OVL    = os.path.join(OUT_FIG_DIR, f'40c.cca2_binmean_overlay_mouse_human{SUFFIX}.pdf')
OUT_PDF_BOX    = os.path.join(OUT_FIG_DIR, f'40c.cca2_gene_expr_boxbins_mouse_human{SUFFIX}.pdf')
OUT_PDF_HEAT   = os.path.join(OUT_FIG_DIR, f'40c.cca2_gene_expr_heatmap_mouse_human{SUFFIX}.pdf')
OUT_RANK_TSV   = os.path.join(OUT_RES_DIR, f'40c.cca2_bgene_contributions{SUFFIX}.tsv')

# --- parameters ---
MOUSE_SUBCLASS = 'L2/3'
CCA_AXES       = ['CCA1', 'CCA2']
HUMAN_VX_COLS  = ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10']   # Gate-A human L2/3 (it_evo 04/16)
MOUSE_VX_COLS  = ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9']    # Gate-A mouse L2/3 (it_evo 05/16)
MOUSE_NOC      = 3
ALPHABET       = ['A', 'B', 'C', 'D', 'E', 'F']
ARCH_KEY       = 'B'         # internal key; displayed B' under 24b's ARCH_RELABEL {A:C',B:B',C:A'}
ARCH_LABEL     = "B'"        # published label for ARCH_KEY
# Display-only CCA1 reflection, applied to BOTH species as in it_evo 18b/18c. CCA2 is not
# flipped (18b flips CCA1 only). Cosmetic: canonical correlations are unchanged.
MOUSE_CCA1_SIGN = -1.0
HUMAN_CCA1_SIGN = -1.0
MOUSE_CCA2_SIGN = 1.0
HUMAN_CCA2_SIGN = 1.0
CCA1_LABEL     = "CCA1'"     # primed: reflected for display
CCA2_LABEL     = 'CCA2'
POINT_SIZE     = 4
DPI            = 300
N_CCA2_BINS    = 10          # CCA2 bins for the mean-expression overlay line, boxplots, heatmap
EMB_CMAP       = 'RdBu_r'    # CCA2-vs-CCA1 expression colormap
EMB_PCTILE     = (2, 98)     # per-panel color-scale clip percentiles
MOUSE_COLOR    = '#2166ac'   # mouse curve/boxes
HUMAN_COLOR    = '#b2182b'   # human curve/boxes
FILL_ALPHA     = 0.3         # alpha for the +/- std fill bands and box faces
HEAT_CMAP      = 'RdBu_r'    # gene x CCA2-bin heatmap color (per-gene z, symmetric about 0)
CLUSTER_METRIC = 'correlation'   # row distance for hierarchical clustering (CCA2 shape)
CLUSTER_METHOD = 'average'       # linkage method

os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)


def select_bgenes():
    """Rank archetype-B genes by their CCA2 contribution |ms*hs|; mirrors it_evo/24b.

    Returns (ranked DataFrame over ALL B genes, list of mouse symbols, list of human symbols)
    for the top N_GENES that are also measurable in both h5ads.
    """
    ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
             .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
    hvg_h = pd.read_csv(IN_H_HVG, sep='\t', index_col=0)
    hvg_m = pd.read_csv(IN_M_HVG, sep='\t', index_col=0)
    if UNIVERSE == 'hvg_intersect':
        H, M = hvg_h, hvg_m
    else:
        H = pd.read_csv(IN_H_FULL, sep='\t', index_col=0)
        M = pd.read_csv(IN_M_FULL, sep='\t', index_col=0)
    shared = ortho[ortho['human_symbol'].isin(H.index) & ortho['mouse_symbol'].isin(M.index)]
    if UNIVERSE == 'hvg_union':
        shared = shared[shared['human_symbol'].isin(hvg_h.index)
                        | shared['mouse_symbol'].isin(hvg_m.index)]
    shared = shared.reset_index(drop=True)
    print(f'  {UNIVERSE}: {len(shared)} shared ortholog genes')

    Xc = H.loc[shared['human_symbol'].values, HUMAN_VX_COLS].values
    Yc = M.loc[shared['mouse_symbol'].values, MOUSE_VX_COLS].values
    Xc = Xc - Xc.mean(0)
    Yc = Yc - Yc.mean(0)

    wdf_h = pd.read_csv(IN_HUMAN_W, sep='\t', index_col=0)
    wdf_m = pd.read_csv(IN_MOUSE_W, sep='\t', index_col=0)
    hs = Xc @ wdf_h.loc[CCA2_LABEL, HUMAN_VX_COLS].to_numpy(dtype=float)
    ms = Yc @ wdf_m.loc[CCA2_LABEL, MOUSE_VX_COLS].to_numpy(dtype=float)

    # mouse archetype membership per gene (highest-log2FC archetype if multiple), as in 24b
    mk = pd.read_csv(IN_MARKERS, sep='\t')
    mk['letter'] = mk['archetype'].map({f'archetype_{i+1}': ALPHABET[i] for i in range(MOUSE_NOC)})
    mk = mk.sort_values('log2FC', ascending=False).drop_duplicates('gene')
    gene2arch = dict(zip(mk['gene'], mk['letter']))
    letters = np.array([gene2arch.get(g, '') for g in shared['mouse_symbol'].values])

    ranked = (pd.DataFrame({'mouse_symbol': shared['mouse_symbol'].values,
                            'human_symbol': shared['human_symbol'].values,
                            'cca2_loading_mouse': ms, 'cca2_loading_human': hs,
                            'contribution': np.abs(ms * hs)})[letters == ARCH_KEY]
              .sort_values('contribution', ascending=False)
              .reset_index(drop=True))
    print(f'  archetype {ARCH_KEY} ({ARCH_LABEL}) genes: {len(ranked)}')
    if len(ranked) < N_GENES:
        raise ValueError(f'Only {len(ranked)} archetype-{ARCH_KEY} genes in the {UNIVERSE} '
                         f'universe; --n-genes {N_GENES} cannot be satisfied.')

    # keep only genes measurable in both h5ads (a gene can carry a loading yet be absent from
    # a matrix); reported explicitly rather than silently skipped
    m_names = set(ad.read_h5ad(INPUT_MOUSE, backed='r').var_names.values)
    h_adata = ad.read_h5ad(INPUT_HUMAN, backed='r')
    h_names = set(h_adata.raw.var['feature_name'].values
                  if 'feature_name' in h_adata.raw.var.columns
                  else h_adata.raw.var_names.values)
    ok = ranked['mouse_symbol'].isin(m_names) & ranked['human_symbol'].isin(h_names)
    dropped = ranked.loc[~ok].head(N_GENES)
    if len(dropped):
        print(f'  NOTE: {int((~ok).sum())} of {len(ranked)} B genes are absent from one of the '
              f'expression matrices and cannot be plotted; highest-ranked among them: '
              f'{dropped["mouse_symbol"].tolist()}')
    top = ranked[ok].head(N_GENES)
    if len(top) < N_GENES:
        raise ValueError(f'Only {len(top)} archetype-{ARCH_KEY} genes are present in both '
                         f'expression matrices; --n-genes {N_GENES} cannot be satisfied.')
    return ranked, top['mouse_symbol'].tolist(), top['human_symbol'].tolist()


def cca_coords(vx_path, weights_path, vx_cols, cca1_sign, cca2_sign, species):
    """Project one species' cells onto CCA1/CCA2: (VX coords - mean) . a_vx, as in it_evo 18b.

    Returns (cell_index, cca1, cca2, canonical_r). Display signs are applied last; they are
    cosmetic reflections and leave the canonical correlations untouched.
    """
    vx_df   = pd.read_csv(vx_path, sep='\t', index_col=0)
    weights = pd.read_csv(weights_path, sep='\t', index_col=0)

    missing = [c for c in vx_cols if c not in vx_df.columns]
    if missing:
        raise ValueError(f'{species}: Gate-A VX columns {missing} absent from {vx_path}')

    W = weights.loc[CCA_AXES, vx_cols].values.T    # (n_vx x 2), normalized + sign-fixed by 16
    r_cca = weights.loc[CCA_AXES, 'canonical_r'].values
    stable = weights.loc[CCA_AXES, 'stable'].values
    print(f'  {species}: {vx_df.shape[0]} cells, VX {vx_cols}, '
          f'r(CCA1)={r_cca[0]:.3f} (stable={stable[0]}), r(CCA2)={r_cca[1]:.3f} (stable={stable[1]})')

    C = vx_df[vx_cols].values
    cca = (C - C.mean(axis=0)) @ W
    return vx_df.index, cca[:, 0] * cca1_sign, cca[:, 1] * cca2_sign, r_cca


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
    mapped to [0, 1] ((b+0.5)/n_bins), so different-range inputs (mouse vs human CCA2)
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


def binmean_matrix(cca2, expr_df, gene_list, species):
    """Rows = genes, cols = N_CCA2_BINS relative CCA2 bins; value = mean log2(CP10k+1) in bin.

    Uses the same relative binning as binned_stats_relative, so mouse and human matrices
    have column-comparable (relative-CCA2) bins. Fails fast if any bin is empty, since an
    empty bin would leave a NaN that hierarchical clustering cannot consume.
    """
    rows = []
    for g in gene_list:
        _, means, _ = binned_stats_relative(cca2, expr_df[g].values, N_CCA2_BINS)
        empty = np.where(np.isnan(means))[0]
        if empty.size:
            raise ValueError(f'{species} {g}: CCA2 bins {(empty + 1).tolist()} are empty at '
                             f'N_CCA2_BINS={N_CCA2_BINS}; cannot build the heatmap.')
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
        raise ValueError('Mouse expression has NaN after reindexing to CCA barcodes (barcode mismatch).')
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
        raise ValueError('Human expression has NaN after reindexing to CCA barcodes (barcode mismatch).')
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
    ax.set_xticks(range(1, N_CCA2_BINS + 1))
    ax.set_xticklabels(range(1, N_CCA2_BINS + 1))
    ax.set_xlabel(f'{CCA2_LABEL} bin (1..{N_CCA2_BINS}, low→high)')
    ax.set_ylabel(f'{sym} log2(CP10k+1) expr')
    ax.set_title(sp_title)
    sns.despine(ax=ax)


# --- derive the gene list: top archetype-B contributors to CCA2 (it_evo/24b) ---
print(f'Selecting the top {N_GENES} archetype-{ARCH_KEY} genes by CCA2 contribution...')
ranked, genes, genes_human = select_bgenes()
ranked.to_csv(OUT_RANK_TSV, sep='\t', index=False)
print(f'Saved {OUT_RANK_TSV}')
print(f'  genes: {genes}')

# --- project cells onto the conserved CCA axes ---
print('Projecting cells onto the cross-species CCA axes...')
mouse_index, mouse_cca1, mouse_cca2, mouse_r = cca_coords(
    IN_MOUSE_VX, IN_MOUSE_W, MOUSE_VX_COLS, MOUSE_CCA1_SIGN, MOUSE_CCA2_SIGN, 'mouse')
human_index, human_cca1, human_cca2, human_r = cca_coords(
    IN_HUMAN_VX, IN_HUMAN_W, HUMAN_VX_COLS, HUMAN_CCA1_SIGN, HUMAN_CCA2_SIGN, 'human')
if not np.allclose(mouse_r, human_r):
    raise ValueError(f'Canonical correlations differ between species ({mouse_r} vs {human_r}); '
                     'the two weight tables must come from the same CCA fit.')
R_CCA2 = float(mouse_r[1])

# --- expression matrices (aligned to CCA barcode order) ---
mouse_expr = load_mouse_expr(genes, mouse_index)
human_expr = load_human_expr(genes_human, human_index)

MOUSE_TITLE = 'Cheng22 mouse L2/3 IT'
HUMAN_TITLE = 'Jorstad23 human L2/3 IT'
SUB = f'{ARCH_LABEL} genes, {UNIVERSE}'

# --- multi-page PDF: one gene per page, mouse (left) + human (right) ---
plt.rcParams['pdf.fonttype'] = 42   # editable vector text
print(f'Writing {OUT_PDF} ({len(genes)} pages)...')
with PdfPages(OUT_PDF) as pdf:
    for g_mouse, g_human in zip(genes, genes_human):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

        for ax, cca2, vals, sym, sp_title in [
            (axes[0], mouse_cca2, mouse_expr[g_mouse].values, g_mouse, MOUSE_TITLE),
            (axes[1], human_cca2, human_expr[g_human].values, g_human, HUMAN_TITLE),
        ]:
            ax.scatter(cca2, vals, s=POINT_SIZE, linewidths=0, color='#bbbbbb', rasterized=True)
            bc, bm = binned_mean(cca2, vals, N_CCA2_BINS)   # bin-mean overlay (vector)
            ax.plot(bc, bm, '-o', color='#c0392b', linewidth=1.5, markersize=4,
                    zorder=3, label=f'mean over {N_CCA2_BINS} {CCA2_LABEL} bins')
            ax.set_xlabel(CCA2_LABEL)
            ax.set_ylabel(f'{sym} (log-norm expr)')
            ax.set_title(sp_title)
            ax.legend(frameon=False, fontsize=8, loc='upper right')
            sns.despine(ax=ax)

        fig.suptitle(f'{g_mouse} / {g_human} — {CCA2_LABEL} vs expression '
                     f'(r = {R_CCA2:.3f}; {SUB})')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)

print(f'Saved {OUT_PDF}')

# --- multi-page PDF: one gene per page, CCA2-vs-CCA1 colored by expression ---
print(f'Writing {OUT_PDF_EMB} ({len(genes)} pages)...')
with PdfPages(OUT_PDF_EMB) as pdf:
    for g_mouse, g_human in zip(genes, genes_human):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

        for ax, cca2, cca1, vals, sym, sp_title in [
            (axes[0], mouse_cca2, mouse_cca1, mouse_expr[g_mouse].values, g_mouse, MOUSE_TITLE),
            (axes[1], human_cca2, human_cca1, human_expr[g_human].values, g_human, HUMAN_TITLE),
        ]:
            vmin, vmax = np.nanpercentile(vals, EMB_PCTILE)
            sc = ax.scatter(cca2, cca1, c=vals, cmap=EMB_CMAP, vmin=vmin, vmax=vmax,
                            s=POINT_SIZE, linewidths=0, rasterized=True)
            ax.set_aspect('equal', adjustable='box')   # true CCA geometry
            ax.set_xlabel(CCA2_LABEL)
            ax.set_ylabel(CCA1_LABEL)
            ax.set_title(sp_title)
            fig.colorbar(sc, ax=ax, label=f'{sym} (log-norm expr)', shrink=0.8)
            sns.despine(ax=ax)

        fig.suptitle(f'{g_mouse} / {g_human} — {CCA2_LABEL} vs {CCA1_LABEL}, colored by expression')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)

print(f'Saved {OUT_PDF_EMB}')

# --- multi-page PDF: one gene per page, mouse+human mean +/- std over 10 relative CCA2 bins ---
print(f'Writing {OUT_PDF_OVL} ({len(genes)} pages)...')
with PdfPages(OUT_PDF_OVL) as pdf:
    for g_mouse, g_human in zip(genes, genes_human):
        fig, ax = plt.subplots(figsize=(5.4, 4.2))

        for cca2, vals, color, lbl in [
            (mouse_cca2, mouse_expr[g_mouse].values, MOUSE_COLOR, f'mouse {g_mouse}'),
            (human_cca2, human_expr[g_human].values, HUMAN_COLOR, f'human {g_human}'),
        ]:
            bc, bm, bs = binned_stats_relative(cca2, vals, N_CCA2_BINS)
            ax.fill_between(bc, bm - bs, bm + bs, color=color, alpha=FILL_ALPHA, linewidth=0)
            ax.plot(bc, bm, '-o', color=color, linewidth=1.5, markersize=4, label=lbl)

        ax.set_xlabel(f'{CCA2_LABEL} (relative, per-species min–max; {N_CCA2_BINS} bins)')
        ax.set_ylabel('log2(CP10k+1) expr')
        ax.set_title(f'{g_mouse} / {g_human} — mean ± std over {CCA2_LABEL} bins')
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

        for ax, cca2, vals, color, sym, sp_title in [
            (axes[0], mouse_cca2, mouse_expr[g_mouse].values, MOUSE_COLOR, g_mouse, MOUSE_TITLE),
            (axes[1], human_cca2, human_expr[g_human].values, HUMAN_COLOR, g_human, HUMAN_TITLE),
        ]:
            positions, groups = binned_groups(cca2, vals, N_CCA2_BINS)
            draw_boxbins(ax, positions, groups, color, sym, sp_title)

        fig.suptitle(f'{g_mouse} / {g_human} — expression by {CCA2_LABEL} bin (boxplots)')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)

print(f'Saved {OUT_PDF_BOX}')

# --- single-page PDF: gene x CCA2-bin heatmap per species, shared clustered row order ---
print(f'Writing {OUT_PDF_HEAT} ({len(genes)} genes x {N_CCA2_BINS} {CCA2_LABEL} bins)...')
mouse_mat = binmean_matrix(mouse_cca2, mouse_expr, genes, 'mouse')
human_mat = binmean_matrix(human_cca2, human_expr, genes_human, 'human')


# per-gene z-scoring WITHIN each species (each row centered on its own species-specific mean)
def zscore_rows(mat, species):
    """Z-score each row across its bins (mean 0, sd 1); fail fast on flat rows."""
    sd  = mat.std(axis=1, keepdims=True)
    bad = np.where(sd[:, 0] == 0)[0]
    if bad.size:
        raise ValueError(f'{species} genes with flat expression across all {CCA2_LABEL} bins '
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
    (axes[0], mouse_scaled[row_order], [genes[i] for i in row_order], MOUSE_TITLE, 'left'),
    (axes[1], human_scaled[row_order], [genes_human[i] for i in row_order], HUMAN_TITLE, 'right'),
]:
    im = ax.imshow(mat, aspect='auto', cmap=HEAT_CMAP, vmin=-zlim, vmax=zlim)
    ax.set_xticks(range(N_CCA2_BINS))
    ax.set_xticklabels(range(1, N_CCA2_BINS + 1))
    ax.set_xlabel(f'{CCA2_LABEL} bin (1..{N_CCA2_BINS}, low→high)')
    ax.set_yticks(range(len(gene_syms)))
    ax.set_yticklabels(gene_syms, fontsize=8)
    ax.yaxis.set_label_position(label_side)
    ax.yaxis.set_ticks_position(label_side)
    ax.set_title(sp_title)

fig.colorbar(im, ax=axes, label='z-scored mean expr\n(per gene, within each species)',
             shrink=0.6, pad=0.1)
fig.suptitle(f'Top {len(genes)} {ARCH_LABEL} genes by {CCA2_LABEL} contribution — mean expression '
             f'over {CCA2_LABEL} bins (r = {R_CCA2:.3f}; rows: hierarchical clustering)')
fig.savefig(OUT_PDF_HEAT, bbox_inches='tight', dpi=DPI)
plt.close(fig)

print(f'Saved {OUT_PDF_HEAT}')
print('Done.')
