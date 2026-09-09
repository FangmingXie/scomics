"""One-panel-per-layer archetype-score blend on the two-dataset IT PCHA embeddings.

Scripts 33-36 (`{34,33,35,36}.follow.viz.two_{L23,L4,L5IT,L6IT}_archetype_scores.py`) each
render one panel PER ARCHETYPE score over the same cell embedding, so a 3-archetype layer costs
three panels that the reader has to fuse mentally. This script compresses each layer's panels
into ONE: every cell is drawn once, coloured by a continuous blend of the archetype colours
(A' -> C0, B' -> C1, C' -> C2) weighted by its own archetype scores. Four layers, four panels,
one PDF. No recomputation — the cached 33-36 results are read as-is.

Colour rule (score blend with a pale centre):
  1. Each score column is clipped at its own SCORE_PCTILE (the same 5-95 clip scripts 33-36 use
     for their per-panel colour scales) and rescaled to [0, 1], because the raw scores span only
     ~0.11-0.75 and would otherwise blend to a uniform mid-gray.
  2. The rescaled weights are normalised to sum to 1, giving each cell a point on the archetype
     simplex; the hue is that convex mix of the archetype colours. A cell dominated by A' is
     pure C0.
  3. Saturation follows the winning margin: `conf` runs 0 (perfectly uniform weights, no
     dominant archetype) to 1 (one-hot), and the hue is mixed toward PALE_RGB by
     MIN_MIX + (1 - MIN_MIX) * conf**GAMMA. Ambiguous cells therefore read as pale gray and only
     committed cells carry saturated colour. This is a second, purely visual channel — it
     encodes nothing the weights don't already contain.

L5IT is NOC=2 in this two-dataset fit, so its panel is an honest two-colour C0<->C1 blend
(`conf` is measured against the K=2 uniform point, 0.5); the other three layers are NOC=3.

Primed archetype letters are READ from the persisted depth arc (see ARCHETYPE_MAPPING.md and the
`load_relabel` idiom of scripts/it/52), never hard-coded, and colour follows the DISPLAYED label.
The archetype overlay (black diamonds + closing polygon + primed letters) matches 33-36. Those
scripts all leave their display FLIP at [1, 1], so no flip is applied here either.

Reads (per layer STEM):
  local_data/res/it/<STEM>_archetype_scores.tsv
  local_data/res/it/<STEM>_pcha_xp.tsv
  local_data/res/it/<STEM>_pcha_aa.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv   (primed relabel)
Outputs:
  local_data/fig/it/59.it_archetype_score_blend_embedding.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IT_EVO_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
FIG_DIR     = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
IN_ARCMAP   = os.path.join(IT_EVO_DIR, '15.mouse_IT_joint_archetype_arc_order.tsv')
OUT_PDF     = os.path.join(FIG_DIR, '59.it_archetype_score_blend_embedding.pdf')

# `token` keys into the depth-arc table; `stem` prefixes the three cached TSVs of scripts 33-36.
SUBCLASSES = [
    dict(subclass='L2/3', token='L23',  stem='34.follow.two_L23'),
    dict(subclass='L4',   token='L4',   stem='33.follow.two_L4'),
    dict(subclass='L5IT', token='L5IT', stem='35.follow.two_L5IT'),
    dict(subclass='L6IT', token='L6IT', stem='36.follow.two_L6IT'),
]

# --- parameters ---
ARCHETYPE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']   # archetype_1 -> A, archetype_2 -> B, ...
# Colour follows the DISPLAYED (primed) label, never the internal key -- ARCHETYPE_MAPPING.md.
ARCH_COLORS  = {"A'": 'C0', "B'": 'C1', "C'": 'C2', "D'": 'C3'}
SCORE_PCTILE = (5, 95)          # per-score clip before rescaling (mirrors 33-36's colour limits)
PALE_RGB     = np.array(mcolors.to_rgb('#e6e6e6'))   # colour of a cell with no dominant archetype
MIN_MIX      = 0.12             # floor on the pale->hue mix, so the palest cells still tint
GAMMA        = 0.7              # <1 lifts middling margins out of the gray
POINT_SIZE   = 3
FIG_PANEL_W  = 4.4
FIG_PANEL_H  = 4.8
DPI          = 300


def paths_for(cfg):
    """The three cached 33-36 inputs for one layer."""
    return {k: os.path.join(RES_DIR, f'{cfg["stem"]}_{k}.tsv')
            for k in ('archetype_scores', 'pcha_xp', 'pcha_aa')}


def load_relabel():
    """{token: {old_letter: new_letter}} from the persisted depth arc (read, never hard-coded)."""
    arc = pd.read_csv(IN_ARCMAP, sep='\t')
    missing = {'token', 'old_letter', 'new_letter'} - set(arc.columns)
    if missing:
        raise ValueError(f'{IN_ARCMAP} missing column(s): {sorted(missing)}')
    return {token: dict(zip(sub['old_letter'], sub['new_letter']))
            for token, sub in arc.groupby('token')}


def blend(scores, rgb_by_col):
    """Per-cell blend colour from raw archetype scores. -> (rgb (n,3), weights (n,K), conf (n,)).

    See the module docstring for the three steps. `rgb_by_col[k]` is the RGB of score column k.
    """
    lo, hi = np.percentile(scores, SCORE_PCTILE, axis=0)
    if np.any(hi <= lo):
        raise ValueError(f'score column with no spread at percentiles {SCORE_PCTILE}: '
                         f'lo={lo}, hi={hi}')
    w = np.clip((scores - lo) / (hi - lo), 0.0, 1.0)
    tot = w.sum(axis=1, keepdims=True)
    k = scores.shape[1]
    # A cell under the low percentile on EVERY archetype has no weight at all; it is maximally
    # ambiguous by construction, so it takes the uniform point and renders palest.
    weights = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 1.0 / k)
    hue = weights @ np.asarray(rgb_by_col)
    conf = (weights.max(axis=1) - 1.0 / k) / (1.0 - 1.0 / k)   # 0 = uniform, 1 = one-hot
    mix = MIN_MIX + (1.0 - MIN_MIX) * conf ** GAMMA
    return PALE_RGB + (hue - PALE_RGB) * mix[:, None], weights, conf


def prep(cfg, relabel_by_token):
    """Everything one panel needs from the cached TSVs, validated against the depth arc."""
    P = paths_for(cfg)
    scores_df = pd.read_csv(P['archetype_scores'], sep='\t', index_col=0)
    xp_df     = pd.read_csv(P['pcha_xp'], sep='\t', index_col=0)
    aa_df     = pd.read_csv(P['pcha_aa'], sep='\t', index_col=0)
    scores_df = scores_df.reindex(xp_df.index)          # defensive alignment to the PC coords
    if scores_df.isna().any().any():
        raise ValueError(f'{P["archetype_scores"]}: cells missing from the score table')

    letters = [ARCHETYPE_LETTERS[i] for i in range(len(aa_df))]
    expect = [f'score_{L}' for L in letters]
    if list(scores_df.columns) != expect:
        raise ValueError(f'{P["archetype_scores"]}: columns {list(scores_df.columns)} != {expect}')

    relabel = relabel_by_token[cfg['token']]            # internal letter -> primed figure letter
    if set(relabel) != set(letters):
        raise ValueError(f'{cfg["subclass"]}: arc-table letters {sorted(relabel)} != '
                         f'score-file letters {letters}')

    rgb_by_col = [np.array(mcolors.to_rgb(ARCH_COLORS[relabel[L]])) for L in letters]
    rgb, weights, conf = blend(scores_df.values, rgb_by_col)
    return dict(cfg, letters=letters, relabel=relabel, rgb_by_col=rgb_by_col,
                xp=xp_df[['PC1', 'PC2']].values, aa=aa_df[['PC1', 'PC2']].values,
                rgb=rgb, weights=weights, conf=conf, n=len(xp_df))


def draw(ax, S):
    """One layer's panel: blended cells, then the vector archetype overlay and colour key."""
    xp, aa = S['xp'], S['aa']
    ax.scatter(xp[:, 0], xp[:, 1], c=S['rgb'], s=POINT_SIZE, linewidths=0, rasterized=True)

    ax.plot(list(aa[:, 0]) + [aa[0, 0]], list(aa[:, 1]) + [aa[0, 1]],
            '-', color='black', linewidth=1.0, zorder=3)
    ax.scatter(aa[:, 0], aa[:, 1], marker='D', color='black', s=30, zorder=4)
    for (ax_, ay_), L in zip(aa, S['letters']):
        ax.annotate(S['relabel'][L], (ax_, ay_), textcoords='offset points', xytext=(5, 5),
                    fontsize=8, fontweight='bold', color='black', zorder=5)

    order = sorted(range(len(S['letters'])), key=lambda k: S['relabel'][S['letters'][k]])
    ax.legend(handles=[mpatches.Patch(facecolor=S['rgb_by_col'][k], edgecolor='none',
                                      label=S['relabel'][S['letters'][k]]) for k in order],
              title='archetype (cells are mixtures;\npale = no dominant archetype)',
              loc='upper left', fontsize=8, title_fontsize=6, framealpha=0.9)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{S["subclass"]}  (n={S["n"]} cells, NOC={len(S["letters"])})')
    sns.despine(ax=ax)


print('--- IT archetype-score blend embeddings (one panel per layer) ---')
os.makedirs(FIG_DIR, exist_ok=True)

for cfg in SUBCLASSES:                      # fail fast, before any plotting
    for path in paths_for(cfg).values():
        if not os.path.exists(path):
            raise FileNotFoundError(f'{cfg["subclass"]}: missing {path}')

relabel_by_token = load_relabel()
panels = [prep(cfg, relabel_by_token) for cfg in SUBCLASSES]

plt.rcParams['pdf.fonttype'] = 42           # editable vector text
fig, axes = plt.subplots(1, len(panels), squeeze=False,
                         figsize=(FIG_PANEL_W * len(panels), FIG_PANEL_H))
for ax, S in zip(axes[0], panels):
    draw(ax, S)
    shown = ', '.join(f'{L}->{S["relabel"][L]} (mean weight {S["weights"][:, k].mean():.2f})'
                      for k, L in enumerate(S['letters']))
    print(f'  {S["subclass"]:5s} n={S["n"]:5d} NOC={len(S["letters"])}  {shown}')
    print(f'        margin conf: median {np.median(S["conf"]):.2f}, '
          f'{(S["conf"] < 0.1).mean() * 100:.0f}% below 0.1 (pale), '
          f'{(S["conf"] > 0.5).mean() * 100:.0f}% above 0.5 (saturated)')

fig.suptitle('Two-dataset (cheng22+yoo25) IT — archetype scores blended into one panel per layer')
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=DPI)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
