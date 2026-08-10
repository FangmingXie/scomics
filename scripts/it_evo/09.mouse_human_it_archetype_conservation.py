"""Cross-species archetype conservation per IT subclass — mouse vs human.

The l23_evo/26 analysis applied to the four subclass pairs. For each (mouse archetype,
human archetype) pair:
  1. Marker-set overlap via 1-to-1 orthologs: Jaccard + hypergeometric p (background
     N_ORTHOLOGS = len(ortho), as in 26)
  2. Spearman correlation of log2FC profiles across all shared HVG orthologs

The matrices are rectangular — NOC_mouse != NOC_human in general (G10), so 26's
n_mouse/n_human parameterization is kept rather than a single NOC.

Three changes relative to l23_evo/26:

  * Top cells are read from the `*_top_cells.tsv` files written by scripts 04/05 rather
    than re-derived here. 26 re-derives them by Euclidean distance in XP (PCHA) space,
    whereas 04/05 select them by back-projection into VX space; these give different cell
    sets, and using the saved ones means the markers and the conservation matrix are built
    on the same cells.
  * Mouse expression is normalized from `.raw` integer counts, not `.X` (G2) — matching
    the it/ varimax record that scripts 05 and this script build on.
  * `archetype_log2fc` takes a divisor per species (G11): human `.X` is ln(CPM+1) so the
    difference is divided by ln(2), while the mouse `X_norm` is already log2(CP10k+1) and
    is not. Spearman is rank-based and so immune to this; the marker-set metrics are not,
    but they come from the already-corrected scripts 04/05.

COMPAT_MODE restores 26's two analysis choices — XP-space top cells and `.X` mouse
normalization — so the L23 control can be run both ways and a disagreement with
26.archetype_spearman.tsv attributed to one variable rather than three.

Reads (per TOKEN):
  local_data/res/it_evo/05.mouse_<TOKEN>_{archetype_markers,top_cells,pcha_xp,pcha_aa}.tsv
  local_data/res/it_evo/04.human_<TOKEN>_{archetype_markers,top_cells,pcha_xp,pcha_aa}.tsv
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_loadings.tsv   (mouse HVG list)
  local_data/res/it_evo/02.human_<TOKEN>_varimax_loadings.tsv            (human HVG list)
  data/human_mouse_orthologs.tsv
  links/it_evo/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  links/it_evo/jorstad23_human_WithinArea_<HTOKEN>.h5ad
Outputs (per TOKEN, suffixed `.compat` when COMPAT_MODE is on):
  local_data/res/it_evo/09.<TOKEN>_archetype_jaccard.tsv
  local_data/res/it_evo/09.<TOKEN>_archetype_pvals.tsv
  local_data/res/it_evo/09.<TOKEN>_archetype_spearman.tsv
  local_data/res/it_evo/09.<TOKEN>_archetype_gene_conservation.tsv
  local_data/fig/it_evo/09.<TOKEN>_conservation_heatmap.html
  local_data/fig/it_evo/09.<TOKEN>_conservation_scatter.html
"""

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import spearmanr
import anndata as ad
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compat', action='store_true',
                    help="restore l23_evo/26's XP-space top cells and .X mouse "
                         "normalization; writes to *.compat.tsv")
parser.add_argument('--tokens', nargs='*', default=None,
                    help='subset of tokens to run (default: all four)')
args = parser.parse_args()
COMPAT_MODE = args.compat

# --- file paths ---
LINK_DIR     = os.path.join(PROJECT_ROOT, 'links', 'it_evo')
IT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_ORTHOLOGS = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
IN_MOUSE_H5AD = os.path.join(LINK_DIR, 'superdupermegaRNA_cheng22_IT_P28NR.h5ad')

# `n_mouse` must match script 05's `noc`; `n_human` must match script 04's.
SUBCLASSES = [
    {'token': 'L23',  'mouse_subclass': 'L2/3', 'human_subclass': 'L2/3 IT',
     'h5ad': 'jorstad23_human_WithinArea_L23IT.h5ad',
     'mouse_hvg': '19.cheng22_L23_varimax_loadings.tsv',
     'n_mouse': 3, 'n_human': 4},
    {'token': 'L4',   'mouse_subclass': 'L4',   'human_subclass': 'L4 IT',
     'h5ad': 'jorstad23_human_WithinArea_L4IT.h5ad',
     'mouse_hvg': '21.cheng22_L4_varimax_loadings.tsv',
     'n_mouse': 3, 'n_human': 3},
    {'token': 'L5IT', 'mouse_subclass': 'L5IT', 'human_subclass': 'L5 IT',
     'h5ad': 'jorstad23_human_WithinArea_L5IT.h5ad',
     'mouse_hvg': '23.cheng22_L5IT_varimax_loadings.tsv',
     'n_mouse': 2, 'n_human': 4},
    {'token': 'L6IT', 'mouse_subclass': 'L6IT', 'human_subclass': 'L6 IT',
     'h5ad': 'jorstad23_human_WithinArea_L6IT.h5ad',
     'mouse_hvg': '25.cheng22_L6IT_varimax_loadings.tsv',
     'n_mouse': 3, 'n_human': 2},
]
if args.tokens:
    SUBCLASSES = [c for c in SUBCLASSES if c['token'] in args.tokens]

# --- parameters ---
SUBCLASS_COL     = 'Subclass'
GENE_NAME_COL    = 'feature_name'
COMPAT_N_TOP     = 300      # l23_evo/26's fixed N_TOP_CELLS, used only in COMPAT_MODE
ALPHABET         = ['A', 'B', 'C', 'D', 'E', 'F']
STATUS_COLORS    = {'conserved': '#e41a1c', 'mouse_only': '#aaaaaa'}
SUFFIX           = '.compat' if COMPAT_MODE else ''

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

if COMPAT_MODE:
    print('*** COMPAT_MODE: l23_evo/26 top-cell space (XP) and mouse .X normalization ***')

# --- orthologs ---
ortho = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol')
         .drop_duplicates('mouse_symbol'))
mouse_to_human = dict(zip(ortho['mouse_symbol'], ortho['human_symbol']))
human_to_mouse = dict(zip(ortho['human_symbol'], ortho['mouse_symbol']))
N_ORTHOLOGS = len(ortho)
print(f'Orthologs: {N_ORTHOLOGS} 1-to-1 pairs')

print(f'Loading {IN_MOUSE_H5AD}...')
m_adata_all = ad.read_h5ad(IN_MOUSE_H5AD)


def load_top_cells(path, cell_index, noc):
    """Read a scripts-04/05 top_cells.tsv back into positional index arrays."""
    df  = pd.read_csv(path, sep='\t')
    pos = pd.Series(np.arange(len(cell_index)), index=cell_index)
    out = []
    for k in range(noc):
        sub = df[df['archetype'] == f'archetype_{k+1}'].sort_values('rank')
        out.append(pos.loc[sub['cell'].values].values)
    return out


def top_cells_by_xp(xp, aa, noc, n_top):
    """l23_evo/26's rule: nearest cells to each vertex by Euclidean distance in XP space."""
    dists = np.stack([np.linalg.norm(xp - aa[k], axis=1) for k in range(noc)], axis=1)
    return [np.argsort(dists[:, k])[:n_top] for k in range(noc)]


def archetype_log2fc(X, top_cells_list, noc, divisor):
    """Mean log2FC per archetype: its top cells vs. every other cell.

    `divisor` is np.log(2) for a natural-log matrix and 1.0 for one already in log2 (G11).
    """
    n_cells = X.shape[0]
    fc = np.zeros((noc, X.shape[1]))
    for k in range(noc):
        in_idx = top_cells_list[k]
        out_mask = np.ones(n_cells, dtype=bool)
        out_mask[in_idx] = False
        fc[k] = (X[in_idx].mean(axis=0) - X[out_mask].mean(axis=0)) / divisor
    return fc


for cfg in SUBCLASSES:
    token          = cfg['token']
    mouse_subclass = cfg['mouse_subclass']
    human_subclass = cfg['human_subclass']
    n_mouse        = cfg['n_mouse']
    n_human        = cfg['n_human']
    if n_human is None:
        raise ValueError(
            f'{token}: n_human is unset — it must equal script 04\'s noc for this token '
            f'(Gate B)')

    mouse_names = ALPHABET[:n_mouse]
    human_names = ALPHABET[:n_human]
    mouse_labels = [f'mouse_{n}' for n in mouse_names]
    human_labels = [f'human_{n}' for n in human_names]

    in_h5ad          = os.path.join(LINK_DIR, cfg['h5ad'])
    in_mouse_markers = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_archetype_markers.tsv')
    in_human_markers = os.path.join(OUT_RES_DIR, f'04.human_{token}_archetype_markers.tsv')
    in_mouse_top     = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_top_cells.tsv')
    in_human_top     = os.path.join(OUT_RES_DIR, f'04.human_{token}_top_cells.tsv')
    in_mouse_xp      = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_xp.tsv')
    in_mouse_aa      = os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_aa.tsv')
    in_human_xp      = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_xp.tsv')
    in_human_aa      = os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_aa.tsv')
    in_mouse_hvg     = os.path.join(IT_RES_DIR, cfg['mouse_hvg'])
    in_human_hvg     = os.path.join(OUT_RES_DIR, f'02.human_{token}_varimax_loadings.tsv')
    out_jaccard      = os.path.join(OUT_RES_DIR, f'09.{token}_archetype_jaccard{SUFFIX}.tsv')
    out_pvals        = os.path.join(OUT_RES_DIR, f'09.{token}_archetype_pvals{SUFFIX}.tsv')
    out_corr         = os.path.join(OUT_RES_DIR, f'09.{token}_archetype_spearman{SUFFIX}.tsv')
    out_genes        = os.path.join(
        OUT_RES_DIR, f'09.{token}_archetype_gene_conservation{SUFFIX}.tsv')
    out_heatmap_html = os.path.join(
        OUT_FIG_DIR, f'09.{token}_conservation_heatmap{SUFFIX}.html')
    out_scatter_html = os.path.join(
        OUT_FIG_DIR, f'09.{token}_conservation_scatter{SUFFIX}.html')

    print(f'\n{"=" * 70}\n{token} — mouse {mouse_subclass} ({n_mouse}) vs '
          f'human {human_subclass} ({n_human})\n{"=" * 70}')

    # --- marker sets ---
    mouse_df = pd.read_csv(in_mouse_markers, sep='\t')
    human_df = pd.read_csv(in_human_markers, sep='\t')

    mouse_sets, mouse_log2fc = {}, {}
    for k in range(n_mouse):
        sub = mouse_df[mouse_df['archetype'] == f'archetype_{k+1}']
        mouse_sets[k], mouse_log2fc[k] = set(sub['gene']), dict(zip(sub['gene'], sub['log2FC']))
    human_sets, human_log2fc = {}, {}
    for j in range(n_human):
        sub = human_df[human_df['archetype'] == f'archetype_{j+1}']
        human_sets[j], human_log2fc[j] = set(sub['gene']), dict(zip(sub['gene'], sub['log2FC']))

    print('Mouse archetype sizes:', {mouse_names[k]: len(v) for k, v in mouse_sets.items()})
    print('Human archetype sizes:', {human_names[j]: len(v) for j, v in human_sets.items()})

    # --- overlap statistics ---
    jaccard_mat  = np.zeros((n_mouse, n_human))
    pval_mat     = np.zeros((n_mouse, n_human))
    n_shared_mat = np.zeros((n_mouse, n_human), dtype=int)

    for k in range(n_mouse):
        m_orth = {mouse_to_human[g] for g in mouse_sets[k] if g in mouse_to_human}
        K = len(m_orth)
        for j in range(n_human):
            h_set    = human_sets[j]
            n_draw   = len(h_set)
            n_shared = len(m_orth & h_set)
            union    = m_orth | h_set
            jaccard_mat[k, j]  = n_shared / len(union) if union else 0.0
            pval_mat[k, j]     = scipy.stats.hypergeom.sf(n_shared - 1, N_ORTHOLOGS, K, n_draw)
            n_shared_mat[k, j] = n_shared
            print(f'  mouse {mouse_names[k]} vs human {human_names[j]}: '
                  f'shared={n_shared}, J={jaccard_mat[k,j]:.3f}, p={pval_mat[k,j]:.2e}')

    pd.DataFrame(jaccard_mat, index=mouse_labels, columns=human_labels).to_csv(out_jaccard, sep='\t')
    pd.DataFrame(pval_mat,    index=mouse_labels, columns=human_labels).to_csv(out_pvals,   sep='\t')
    print(f'Saved {out_jaccard} and {out_pvals}')

    # --- per-pair gene conservation table ---
    rows = []
    for k in range(n_mouse):
        for j in range(n_human):
            for mg in mouse_sets[k]:
                hg = mouse_to_human.get(mg)
                if hg is None:
                    continue
                rows.append({'mouse_arch': mouse_names[k], 'human_arch': human_names[j],
                             'mouse_gene': mg, 'human_gene': hg,
                             'status': 'conserved' if hg in human_sets[j] else 'mouse_only',
                             'mouse_log2FC': mouse_log2fc[k].get(mg, np.nan),
                             'human_log2FC': human_log2fc[j].get(hg, np.nan)})
            for hg in human_sets[j]:
                mg = human_to_mouse.get(hg)
                if mg is None or mg in mouse_sets[k]:
                    continue
                rows.append({'mouse_arch': mouse_names[k], 'human_arch': human_names[j],
                             'mouse_gene': mg, 'human_gene': hg, 'status': 'human_only',
                             'mouse_log2FC': np.nan,
                             'human_log2FC': human_log2fc[j].get(hg, np.nan)})

    genes_df = pd.DataFrame(rows)
    genes_df.to_csv(out_genes, sep='\t', index=False)
    print(f'Saved {out_genes}  ({len(genes_df)} rows)')

    # --- Spearman of log2FC profiles over shared HVG orthologs ---
    print('\nComputing Spearman correlation of full log2FC profiles...')
    mouse_hvg = set(pd.read_csv(in_mouse_hvg, sep='\t', index_col=0).index)
    human_hvg = set(pd.read_csv(in_human_hvg, sep='\t', index_col=0).index)
    shared_ortho = ortho[ortho['mouse_symbol'].isin(mouse_hvg)
                         & ortho['human_symbol'].isin(human_hvg)]
    print(f'Shared HVG orthologs: {len(shared_ortho)}')

    mouse_xp = pd.read_csv(in_mouse_xp, sep='\t', index_col=0)
    human_xp = pd.read_csv(in_human_xp, sep='\t', index_col=0)

    if COMPAT_MODE:
        mouse_aa = pd.read_csv(in_mouse_aa, sep='\t', index_col=0).values
        human_aa = pd.read_csv(in_human_aa, sep='\t', index_col=0).values
        top_cells_mouse = top_cells_by_xp(mouse_xp.values, mouse_aa, n_mouse, COMPAT_N_TOP)
        top_cells_human = top_cells_by_xp(human_xp.values, human_aa, n_human, COMPAT_N_TOP)
    else:
        top_cells_mouse = load_top_cells(in_mouse_top, mouse_xp.index.values, n_mouse)
        top_cells_human = load_top_cells(in_human_top, human_xp.index.values, n_human)

    # --- mouse expression, aligned to XP order, shared-ortholog genes only ---
    print('Loading mouse expression...')
    m_adata = m_adata_all[m_adata_all.obs[SUBCLASS_COL] == mouse_subclass]
    m_adata = m_adata[mouse_xp.index]
    m_gene_names = m_adata.var_names.values
    m_idx = np.where(np.isin(m_gene_names, shared_ortho['mouse_symbol'].values))[0]

    if COMPAT_MODE:
        # l23_evo/26's choice: treat the already-normalized .X as raw counts
        X_mouse_raw = m_adata.X.toarray().astype(np.float32)
    else:
        X_mouse_raw = m_adata.raw[:, m_adata.var_names].X.toarray().astype(np.float32)  # G2
    depths = X_mouse_raw.sum(axis=1, keepdims=True)
    depths[depths == 0] = 1
    X_mouse = np.log2(X_mouse_raw / depths * 1e4 + 1)[:, m_idx]
    m_genes_used = m_gene_names[m_idx]
    del X_mouse_raw
    gc.collect()

    # --- human expression, aligned to XP order, shared-ortholog genes only ---
    print('Loading human expression...')
    h_adata = ad.read_h5ad(in_h5ad)
    h_adata = h_adata[human_xp.index]
    h_gene_names = h_adata.var[GENE_NAME_COL].values
    h_idx = np.where(np.isin(h_gene_names, shared_ortho['human_symbol'].values))[0]
    X_human = h_adata.X[:, h_idx].toarray().astype(np.float32)   # G7: slice before densify
    h_genes_used = h_gene_names[h_idx]

    # --- align mouse and human columns via ortholog pairs ---
    m_gene_to_col = {g: i for i, g in enumerate(m_genes_used)}
    h_gene_to_col = {g: i for i, g in enumerate(h_genes_used)}
    m_to_h = dict(zip(shared_ortho['mouse_symbol'], shared_ortho['human_symbol']))

    aligned_mouse = [g for g in m_genes_used if g in m_to_h and m_to_h[g] in h_gene_to_col]
    aligned_human = [m_to_h[g] for g in aligned_mouse]
    X_m = X_mouse[:, [m_gene_to_col[g] for g in aligned_mouse]]
    X_h = X_human[:, [h_gene_to_col[g] for g in aligned_human]]
    print(f'Aligned ortholog gene pairs: {len(aligned_mouse)}')

    # G11: mouse X_norm is already log2, human .X is natural-log
    mouse_fc = archetype_log2fc(X_m, top_cells_mouse, n_mouse, divisor=1.0)
    human_fc = archetype_log2fc(X_h, top_cells_human, n_human, divisor=np.log(2))

    corr_mat = np.zeros((n_mouse, n_human))
    for k in range(n_mouse):
        for j in range(n_human):
            corr_mat[k, j], _ = spearmanr(mouse_fc[k], human_fc[j])

    pd.DataFrame(corr_mat, index=mouse_labels, columns=human_labels).to_csv(out_corr, sep='\t')
    print(f'Saved {out_corr}')
    print('Spearman correlations:')
    print(pd.DataFrame(corr_mat, index=mouse_names, columns=human_names).round(3))
    print('mouse -> human argmax assignment:',
          {mouse_names[k]: human_names[int(np.argmax(corr_mat[k]))] for k in range(n_mouse)})

    del m_adata, h_adata, X_mouse, X_human, X_m, X_h
    gc.collect()

    # --- heatmap HTML: Jaccard | -log10(p) | Spearman ---
    annot_text = [[f"J={jaccard_mat[k,j]:.2f}<br>p={pval_mat[k,j]:.1e}<br>n={n_shared_mat[k,j]}"
                   for j in range(n_human)] for k in range(n_mouse)]
    corr_annot = [[f"r={corr_mat[k,j]:.3f}" for j in range(n_human)] for k in range(n_mouse)]

    panels = [
        (jaccard_mat,                  annot_text, 'Jaccard similarity',   'Jaccard'),
        (-np.log10(pval_mat + 1e-300), annot_text, '−log₁₀(p-value)',      '−log₁₀(p)'),
        (corr_mat,                     corr_annot, 'Spearman correlation', 'Spearman r'),
    ]
    colorbar_xs = [0.27, 0.63, 1.0]

    fig_hm = make_subplots(rows=1, cols=3, subplot_titles=[p[2] for p in panels],
                           horizontal_spacing=0.12)
    for col_idx, (zmat, ann, _, cblabel) in enumerate(panels, start=1):
        fig_hm.add_trace(go.Heatmap(
            z=zmat, x=human_names, y=mouse_names,
            colorscale='Blues' if col_idx < 3 else 'RdBu',
            zmid=0 if col_idx == 3 else None,
            colorbar=dict(title=cblabel, x=colorbar_xs[col_idx - 1], len=0.7),
            text=ann, texttemplate='%{text}', textfont=dict(size=10),
            showscale=True,
        ), row=1, col=col_idx)
        fig_hm.update_xaxes(title_text='Human archetype', row=1, col=col_idx)
    fig_hm.update_yaxes(title_text='Mouse archetype', row=1, col=1)
    fig_hm.update_layout(
        title=f'{token}: mouse {mouse_subclass} vs human {human_subclass} '
              f'archetype conservation' + (' (COMPAT)' if COMPAT_MODE else ''),
        height=380, width=1200)
    fig_hm.write_html(out_heatmap_html)
    print(f'Saved {out_heatmap_html}')

    # --- conservation scatter: one subplot per mouse archetype (best match by Spearman) ---
    best_human = [int(np.argmax(corr_mat[k])) for k in range(n_mouse)]
    fig_sc = make_subplots(
        rows=1, cols=n_mouse,
        subplot_titles=[f'Mouse {mouse_names[k]} → Human {human_names[best_human[k]]}'
                        f'  (r={corr_mat[k, best_human[k]]:.3f})' for k in range(n_mouse)],
        horizontal_spacing=0.08)

    for k in range(n_mouse):
        j = best_human[k]
        sub = genes_df[(genes_df['mouse_arch'] == mouse_names[k])
                       & (genes_df['human_arch'] == human_names[j])
                       & (genes_df['status'].isin(['conserved', 'mouse_only']))]
        for status, color in STATUS_COLORS.items():
            pts = sub[sub['status'] == status]
            fig_sc.add_trace(go.Scatter(
                x=pts['mouse_log2FC'], y=pts['human_log2FC'], mode='markers',
                marker=dict(color=color, size=7, opacity=0.8),
                name=status, showlegend=(k == 0), text=pts['mouse_gene'],
                hovertemplate='%{text}<br>mouse log2FC=%{x:.2f}<br>'
                              'human log2FC=%{y:.2f}<extra></extra>',
            ), row=1, col=k + 1)
        fig_sc.update_xaxes(title_text='Mouse log2FC (marker set)', row=1, col=k + 1)
        fig_sc.update_yaxes(title_text='Human log2FC (marker set)', row=1, col=k + 1)

    fig_sc.update_layout(
        title=f'{token}: archetype marker gene conservation (best match by Spearman)'
              + (' (COMPAT)' if COMPAT_MODE else ''),
        height=450, width=300 * n_mouse + 100)
    fig_sc.write_html(out_scatter_html)
    print(f'Saved {out_scatter_html}')

print('\nDone.')
