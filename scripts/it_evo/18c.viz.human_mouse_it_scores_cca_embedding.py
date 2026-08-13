"""Mouse archetype scores on the conserved CCA1xCCA2 embedding, per IT subclass (plots only).

Generalizes script 18b (L2/3 only) to all four IT subclasses (L2/3, L4, L5IT, L6IT). For each
subclass this writes one figure: a 2 x N grid where rows are species (top = human Jorstad23,
bottom = mouse Cheng22) and the N columns are the mouse archetype scores (score_A, score_B, and
score_C where a subclass has three archetypes; L5IT has only two). Every panel colors that
subclass's cells by one archetype score.

Both rows share one construction. A cell's CCA coordinate is `(VX coords - mean) . a_vx`, with
`a_vx` the Gate-A VX canonical weight vector from 16 (normalized + sign-fixed there); the species'
PCHA archetype vertices are carried into the same frame via `aa_vx = aa . inner_comps + inner_mean`
(scripts 04 human / 05 mouse) and the same projection, so cells and vertices share one coordinate
system per row. The human row uses the mouse-derived scores computed on human cells (06); the mouse
row uses the native mouse scores (05). Human PCHA vertices are labelled with primed letters (A'..)
to mark that they are the human archetypes, not the mouse archetypes that define the scores; mouse
vertices are labelled A/B/C, which coincide with the score letters (score_A = archetype_1). Unlike
18b, the CCA axes are shown in their raw orientation (no sign-flip / relabel). Color limits are
clipped per panel to the 5th-95th percentile of that panel's score, so the same colormap means the
same score range within a panel, not across species (imputed-on-human vs native-on-mouse).

Caveat: per script 21, only L2/3 shows clearly conserved CCA axes; the L4/L5IT/L6IT cross-species
canonical correlations are weak (CCA1 r ~ 0.36/0.33/0.38, CCA2 r ~ 0.27/0.30/0.31), so those
embeddings are exploratory. 18b remains the dedicated, hand-aligned L2/3 figure.

Reads (per TOKEN):
  local_data/res/it_evo/06.mouse_{TOKEN}_archetype_scores_on_human_cells.tsv
  local_data/res/it_evo/02.human_{TOKEN}_varimax_coords.tsv
  local_data/res/it_evo/16.{TOKEN}_axis_cca_weights_human.tsv
  local_data/res/it_evo/04.human_{TOKEN}_pcha_{aa,inner_components,inner_mean}.tsv
  local_data/res/it_evo/05.mouse_{TOKEN}_archetype_scores.tsv
  local_data/res/it/{19,21,23,25}.cheng22_{TOKEN}_varimax_coords.tsv
  local_data/res/it_evo/16.{TOKEN}_axis_cca_weights_mouse.tsv
  local_data/res/it_evo/05.mouse_{TOKEN}_pcha_{aa,inner_components,inner_mean}.tsv
Outputs:
  local_data/fig/it_evo/18c.human_mouse_{TOKEN}_scores_cca.pdf   (one per subclass)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- directories ---
RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')

# --- subclasses (Gate-A VX sets mirror script 21; mouse coords prefixes 19/21/23/25 differ) ---
SUBCLASSES = [
    {'token': 'L23',  'label': 'L2/3',
     'human_vx': ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9'],
     'mouse_coords': '19.cheng22_L23_varimax_coords.tsv'},
    {'token': 'L4',   'label': 'L4',
     'human_vx': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX4', 'VX5', 'VX6'],
     'mouse_coords': '21.cheng22_L4_varimax_coords.tsv'},
    {'token': 'L5IT', 'label': 'L5IT',
     'human_vx': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'],
     'mouse_coords': '23.cheng22_L5IT_varimax_coords.tsv'},
    {'token': 'L6IT', 'label': 'L6IT',
     'human_vx': ['VX6', 'VX7', 'VX8', 'VX9'],
     'mouse_vx': ['VX1', 'VX2', 'VX9', 'VX10'],
     'mouse_coords': '25.cheng22_L6IT_varimax_coords.tsv'},
]

# --- config ---
CCA_AXES     = ['CCA1', 'CCA2']
ALPHABET     = ['A', 'B', 'C', 'D', 'E', 'F']
SCORE_PCTILE = (5, 95)   # clip each panel's color range to the 5th-95th pct of its score
POINT_SIZE   = 3
# archetype-tinted sequential colormaps (gray -> archetype color), matching A->C0/B->C1/C->C2
ARCH_BASE    = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
ARCH_CMAPS   = {L: LinearSegmentedColormap.from_list(f'gray_{ARCH_BASE[i]}',
                                                     ['gainsboro', 'lightgray', ARCH_BASE[i]])
                for i, L in enumerate(ALPHABET)}

os.makedirs(FIG_DIR, exist_ok=True)


def embed(coords_path, vx_cols, weights_path, aa_path, inner_cmp_path, inner_mean_path, primed):
    """Project cells and PCHA archetype vertices into the shared CCA1xCCA2 frame for one species.

    Vertices are labelled A/B/C.. in archetype_1.. order (score_A = archetype_1); primed=True adds
    a prime (A'..) to mark the human archetypes. Returns (cell index, cell coords, ordered vertex
    coords, ordered vertex labels, [CCA1 r, CCA2 r]).
    """
    vx_df   = pd.read_csv(coords_path, sep='\t', index_col=0)
    weights = pd.read_csv(weights_path, sep='\t', index_col=0)
    aa_df   = pd.read_csv(aa_path, sep='\t', index_col=0)
    inner_cmp  = pd.read_csv(inner_cmp_path, sep='\t', index_col=0)
    inner_mean = pd.read_csv(inner_mean_path, sep='\t', index_col=0)[vx_cols].values.ravel()

    W = weights.loc[CCA_AXES, vx_cols].values.T          # (n_vx x 2 CCA), sign-fixed by 16
    r_cca = weights.loc[CCA_AXES, 'canonical_r'].values

    C = vx_df[vx_cols].values
    mean_vx = C.mean(axis=0)
    cca_cells = (C - mean_vx) @ W                        # cells x 2

    aa_vx = aa_df.values @ inner_cmp.loc[list(aa_df.columns), vx_cols].values + inner_mean
    cca_aa = (aa_vx - mean_vx) @ W                       # n_arch x 2

    # archetype_i -> A/B/C.. (optionally primed); order by angle for a clean (simple) polygon
    labels = np.array([ALPHABET[int(i.split('_')[1]) - 1] + ("'" if primed else '')
                       for i in aa_df.index])
    ang = np.arctan2(cca_aa[:, 1] - cca_aa[:, 1].mean(), cca_aa[:, 0] - cca_aa[:, 0].mean())
    order = np.argsort(ang)
    return vx_df.index, cca_cells, cca_aa[order], labels[order], r_cca


def draw_panel(ax, coords, vals, cmap, aa, aa_labels, r_cca, title, fig):
    vmin, vmax = np.percentile(vals, SCORE_PCTILE)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=POINT_SIZE, linewidths=0, rasterized=True)
    ax.plot(list(aa[:, 0]) + [aa[0, 0]], list(aa[:, 1]) + [aa[0, 1]], '-', color='black', lw=1.0)
    ax.scatter(aa[:, 0], aa[:, 1], marker='D', color='black', s=30, zorder=3)
    for (ax_, ay_), label in zip(aa[:, :2], aa_labels):
        ax.annotate(label, (ax_, ay_), textcoords='offset points', xytext=(5, 5),
                    fontsize=8, fontweight='bold', color='black', zorder=4)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'CCA1 (r={r_cca[0]:.3f})')
    ax.set_ylabel(f'CCA2 (r={r_cca[1]:.3f})')
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label='archetype score [0-1]', shrink=0.8)
    sns.despine(ax=ax)


for cfg in SUBCLASSES:
    tok = cfg['token']
    in_h_scores = os.path.join(RES_DIR, f'06.mouse_{tok}_archetype_scores_on_human_cells.tsv')
    in_h_coords = os.path.join(RES_DIR, f'02.human_{tok}_varimax_coords.tsv')
    in_h_weights = os.path.join(RES_DIR, f'16.{tok}_axis_cca_weights_human.tsv')
    in_h_aa = os.path.join(RES_DIR, f'04.human_{tok}_pcha_aa.tsv')
    in_h_inner_cmp = os.path.join(RES_DIR, f'04.human_{tok}_pcha_inner_components.tsv')
    in_h_inner_mean = os.path.join(RES_DIR, f'04.human_{tok}_pcha_inner_mean.tsv')
    in_m_scores = os.path.join(RES_DIR, f'05.mouse_{tok}_archetype_scores.tsv')
    in_m_coords = os.path.join(IT_RES_DIR, cfg['mouse_coords'])
    in_m_weights = os.path.join(RES_DIR, f'16.{tok}_axis_cca_weights_mouse.tsv')
    in_m_aa = os.path.join(RES_DIR, f'05.mouse_{tok}_pcha_aa.tsv')
    in_m_inner_cmp = os.path.join(RES_DIR, f'05.mouse_{tok}_pcha_inner_components.tsv')
    in_m_inner_mean = os.path.join(RES_DIR, f'05.mouse_{tok}_pcha_inner_mean.tsv')
    out_pdf = os.path.join(FIG_DIR, f'18c.human_mouse_{tok}_scores_cca.pdf')

    # native mouse scores define the archetype letters (score_A, score_B, ...)
    m_scores = pd.read_csv(in_m_scores, sep='\t', index_col=0)
    h_scores = pd.read_csv(in_h_scores, sep='\t', index_col=0)
    if list(h_scores.columns) != list(m_scores.columns):
        raise ValueError(f'{tok}: score columns differ between {in_h_scores} ({list(h_scores.columns)}) '
                         f'and {in_m_scores} ({list(m_scores.columns)}).')
    letters = [c.split('_')[1] for c in m_scores.columns]

    h_index, h_cells, h_aa, h_labels, h_r = embed(
        in_h_coords, cfg['human_vx'], in_h_weights, in_h_aa, in_h_inner_cmp, in_h_inner_mean,
        primed=True)
    if not h_scores.index.equals(h_index):
        raise ValueError(f'{tok}: human cell index mismatch: {in_h_scores} vs {in_h_coords} — '
                         f'scores and coords must cover the same cells in the same order.')

    m_index, m_cells, m_aa, m_labels, m_r = embed(
        in_m_coords, cfg['mouse_vx'], in_m_weights, in_m_aa, in_m_inner_cmp, in_m_inner_mean,
        primed=False)
    if not m_scores.index.equals(m_index):
        raise ValueError(f'{tok}: mouse cell index mismatch: {in_m_scores} vs {in_m_coords} — '
                         f'scores and coords must cover the same cells in the same order.')

    rows = [
        {'species': 'Human Jorstad23 (mouse scores)', 'cells': h_cells, 'aa': h_aa,
         'labels': h_labels, 'r': h_r, 'scores': h_scores},
        {'species': 'Mouse Cheng22 (native scores)', 'cells': m_cells, 'aa': m_aa,
         'labels': m_labels, 'r': m_r, 'scores': m_scores},
    ]

    print(f'--- {cfg["label"]}: mouse {letters} scores on conserved CCA1xCCA2 '
          f'(r = {h_r[0]:.3f}, {h_r[1]:.3f}) ---')
    print(f'  human cells {len(h_cells)}, mouse cells {len(m_cells)}')

    plt.rcParams['pdf.fonttype'] = 42
    ncols = len(letters)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(4.2 * ncols, 4 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        for j, L in enumerate(letters):
            draw_panel(axes[i][j], row['cells'], row['scores'][f'score_{L}'].values,
                       ARCH_CMAPS[L], row['aa'], row['labels'], row['r'],
                       f'{row["species"]} — Score {L}', fig)

    fig.suptitle(f'{cfg["label"]} mouse Cheng22 archetype scores on the conserved CCA1xCCA2 '
                 f'embedding — human vs mouse cells')
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  Saved {out_pdf}')

print('\nDone.')
